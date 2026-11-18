"""Fused NKI ops + fast layouts for the Strands Decider torso on Trainium2 (LNC=2, one logical core).

Every function is a drop-in piece for `neuron/decider_model.py` (same math as its torch code) and is
meant to run inside a `torch.compile(backend="neuron_native_lite")` graph.  NKI kernels are launched
with `wrap_nki(kernel)[lnc]` (lnc = NEURON_LOGICAL_NC_CONFIG, default 2) and split their work over the
two physical cores.  Weights must first be re-laid-out once with the prepare_* helpers.

GDN layer  (gdn_layer_fast = the whole decider gdn_layer with the GDN core passed in)
    prepare_gdn_layer(sd, i)   pre-transposed GEMM weights, conv taps, packed (A, dt_bias), w_out_t
    rms_norm_nki(x, ln)        NKI RMSNorm [.., 2048] (keeps neuronx-cc from fusing the norm into the
                               feature-major einsum, which makes that GEMM 2-2.5x slower)
    gdn_in_proj(x, ln1, prep)  RMSNorm + 2 compiler GEMMs: proj_fm [B,48,128,T] (q|k|v, feature-major)
                               and zba [B,T,2080] (z | b | a, token-major)
    gdn_prologue(...)          NKI: causal depthwise conv1d (K=4) + SiLU + L2-norm(q, k) + beta/g
                               -> q, k, v [B,16,T,128] bf16 (+ optional feature-major q, k), g/beta fp32
    gated_rms_norm_nki(o, zba, gnorm) / gdn_epilogue(x, o, zba, gnorm, w_out_t)
                               NKI gated RMSNorm, token-major out -> plain [K,N] out-proj GEMM + residual
    gdn_prologue_ref(...)      torch reference of the prologue (decider math)
    gdn_out_proj(...)          gated norm -> out-proj WITHOUT residual (the TP partial sum to all-reduce);
                               gdn_epilogue / gdn_layer_fast / mlp_fast / attn_layer_fast take `reduce=`
    gdn_layer_prefix_fast / gdn_layer_suffix_fast / conv_tail_fm
                               shared-prefix path: prefix masks beta/g (torch) and returns the conv tail;
                               suffix feeds the tail to the prologue kernel as conv history
MLP / attention
    prepare_mlp_layer(sd, i), mlp_fast(x, prep)   pre-transposed [K, N] weights (compiler GEMMs)
    prepare_attn_layer(sd, i), attn_layer_fast(...) NKI RMSNorm + pre-transposed einsum + token-major o-proj
    sigmoid_gate_nki(...)      NKI o * sigmoid(gate) -> token-major (measured: no layer win; kept for reuse)

Tensor parallel: every prepare_* / kernel derives head counts from the (shard_state) weight shapes;
H = heads per rank in {16, 8, 4} (must be even: each of the 2 physical cores takes H/2 heads).
Layouts: B batch, T tokens (T % 128 == 0), hidden 2048, H heads x 128 for each of q/k/v/z.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F

import nki
import nki.isa as nisa
import nki.language as nl

P = 128          # partitions / head dim
K_CONV = 4       # causal conv taps
L2_EPS = 1e-6    # decider_model.l2norm eps


def _lnc() -> int:
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


# ===================================================================================== (A) kernel
@nki.jit
def gdn_prologue_kernel(proj, conv_diag, ba, gdn_ab, conv_hist=None, eps=L2_EPS, emit_qk_fm=False):
    """GDN prologue NKI kernel (traced once per physical core of the logical NC; see gdn_prologue).

    proj      [B, 3H, 128, T] bf16   conv input, FEATURE-major (channel on axis 2, token on axis 3);
                                     heads 0..H-1 = q, H..2H-1 = k, 2H..3H-1 = v (H = heads per TP rank).
    conv_diag [3H, 128, 4, 128] bf16 conv_diag[h, c, j, c'] = w[h, c, j] * (c == c')  (tap j).
    ba        [B, T, Z] bf16         token-major, [b | a] in the LAST 2H columns (h @ w_ba.T, or zba).
    gdn_ab    [H, 2] fp32            column 0 = A (= -exp(A_log)), column 1 = dt_bias.
    conv_hist [Bh, 3H, 3, 128] bf16  optional: the 3 pre-conv tokens preceding token 0 (head-major,
                                     DeciderNeuron.conv_tail layout); Bh = 1 (broadcast) or B. None = zeros.

    Each core takes H/2 of the q, k and v heads (and half the tokens for beta/g).
    The conv AND the feature->token transpose are ONE tensor-engine op per (128-token block, tap):
    stationary = y[:, t-3+j] (channels on partitions; for a full 512-token tile the 128 tokens are
    strided by 4 so output partition p owns the 4 consecutive tokens t0+4p..t0+4p+3 -> 1 KiB HBM
    rows), moving = diag(w_j); PSUM accumulates the 4 taps in fp32 (exact for the bf16 operands) and
    comes out token-major [tok, ch].  Each block's 4-matmul accumulation group gets its own PSUM bank
    (a bank holds one in-flight group).  ScalarE: SiLU straight out of PSUM.  VectorE: sum of squares
    (fused multiply + accumulate) and the scale.  GpSimd: rsqrt (ULP-1; ScalarE's is ~4e4 ULP).
    T <= 512 (suffix shapes): one tap / activation / history load and one store per head KIND covers
    all heads of this core and all B sequences (each dma_copy costs ~1 us of GpSimd descriptor work).
    T > 512: per (b, head) loads, per-tile stores (the TP1 prefill path, unchanged).
    (Measured alternatives -- feature-major conv with 4 shared tap loads + PE transposes, on-chip
    diag(w), per-head stores at long T, ScalarE squares -- were all 10-30% slower at T=512..4096.)
    """
    B, NH, D, T = proj.shape
    HQ = NH // 3
    n_shards = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    shard = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    bf16, fp32 = nl.bfloat16, nl.float32
    MT = 512                       # tokens per work tile
    NB = MT // P                   # 4 token blocks per tile
    BANK = 512                     # fp32 elements per PSUM bank partition
    NBT = T // P                   # 128-token blocks per sequence
    KH = K_CONV - 1                # history columns
    has_hist = conv_hist is not None
    Bh = conv_hist.shape[0] if has_hist else 1

    q = nl.ndarray((B, HQ, T, D), dtype=bf16, buffer=nl.shared_hbm)
    k = nl.ndarray((B, HQ, T, D), dtype=bf16, buffer=nl.shared_hbm)
    v = nl.ndarray((B, HQ, T, D), dtype=bf16, buffer=nl.shared_hbm)
    g = nl.ndarray((B, HQ, T), dtype=fp32, buffer=nl.shared_hbm)
    beta = nl.ndarray((B, HQ, T), dtype=fp32, buffer=nl.shared_hbm)
    if emit_qk_fm:
        q_fm = nl.ndarray((B, HQ, D, T), dtype=bf16, buffer=nl.shared_hbm)
        k_fm = nl.ndarray((B, HQ, D, T), dtype=bf16, buffer=nl.shared_hbm)
    outs = (q, k, v)

    hps = HQ // n_shards                    # heads of each kind per core
    n_tt = (T + MT - 1) // MT
    # T <= 512 and the per-kind working set fits: one load / one store per head kind for all B
    nsub1 = NB if T == MT else NBT
    batched = n_tt == 1 and hps * B * ((T + K_CONV - 1) + nsub1 * D) * 2 <= 64 * 1024

    def load_hist(dst, c0, nch, b0, nb):
        """dst [P, nch, nb, 3] <- conv_hist[b0:b0+nb (or 0, broadcast), c0:c0+nch] with d on partitions."""
        nbh = nb if Bh > 1 else 1
        hrow = nl.ndarray((nch * KH, nbh, D), dtype=bf16, buffer=nl.sbuf)       # [(c, j), b, d]
        nisa.dma_copy(dst=hrow, src=conv_hist.ap(pattern=[[D, nch * KH], [NH * KH * D, nbh], [1, D]],
                                                 offset=((b0 if Bh > 1 else 0) * NH + c0) * KH * D))
        hps_ = nl.ndarray((P, nbh, nch * KH), dtype=bf16, buffer=nl.psum)
        for bb in range(nbh):
            nisa.nc_transpose(dst=hps_[:, bb, :], data=hrow[:, bb, :], engine=nisa.tensor_engine)
        src = hps_.reshape((P, nbh, nch, KH)).permute((0, 2, 1, 3))           # [P, nch, nbh, 3]
        if nbh != nb:
            src = src.broadcast(2, nb)
        nisa.tensor_copy(dst=dst, src=src, engine=nisa.engine.vector)

    def conv_tile(y_b, diag_sb, t0, tw, kind, b, h, o_dst):
        """conv + SiLU (+ L2 for q/k) of tokens [t0, t0+tw) of one sequence; writes o_dst [P, nsub, D]."""
        strided = tw == MT              # full tile: partition p owns tokens t0+4p+i, i < 4
        nsub = NB if strided else tw // P  # otherwise contiguous 128-token blocks
        pc = nl.ndarray((P, NB, BANK), dtype=fp32, buffer=nl.psum)   # one bank per block
        for i in range(nsub):
            for j in range(K_CONV):
                if strided:
                    base = t0 + i + j   # y column of token (t - 3 + j), t = t0 + 4p + i
                    stat = y_b[:, base:base + NB * (P - 1) + 1:NB]
                else:
                    base = t0 + i * P + j
                    stat = y_b[:, base:base + P]
                nisa.nc_matmul(dst=pc[:, i, 0:P], stationary=stat, moving=diag_sb[:, j, :], accumulate=(j > 0))
        if kind == 2:
            nisa.activation(dst=o_dst, op=nl.silu, data=pc[:, 0:nsub, 0:P])
            return
        s_sb = nl.ndarray((P, NB, P), dtype=bf16, buffer=nl.sbuf)
        nisa.activation(dst=s_sb[:, 0:nsub, :], op=nl.silu, data=pc[:, 0:nsub, 0:P])
        # L2 norm over the 128 channels of each token (free axis here).
        ss = nl.ndarray((P, NB), dtype=fp32, buffer=nl.sbuf)
        junk = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        for i in range(nsub):
            nisa.scalar_tensor_tensor(dst=junk, data=s_sb[:, i, :], op0=nl.multiply, operand0=1.0,
                                      op1=nl.multiply, operand1=s_sb[:, i, :],
                                      reduce_op=nl.add, reduce_res=ss[:, i:i + 1])
        sse = nl.ndarray((P, NB), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sse[:, 0:nsub], data=ss[:, 0:nsub], op0=nl.add, operand0=eps)
        rn = nl.ndarray((P, NB), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rn[:, 0:nsub], data=sse[:, 0:nsub], op0=nl.rsqrt, operand0=None,
                           engine=nisa.engine.gpsimd)
        for i in range(nsub):
            nisa.tensor_scalar(dst=o_dst[:, i, :], data=s_sb[:, i, :], op0=nl.multiply, operand0=rn[:, i:i + 1])
        if emit_qk_fm:
            # feature-major copy [D, tw]: PE transposes of the normalised tile.
            fm_ps = nl.ndarray((P, NB, P), dtype=bf16, buffer=nl.psum)
            for i in range(nsub):
                nisa.nc_transpose(dst=fm_ps[:, i, :], data=o_dst[:, i, :], engine=nisa.tensor_engine)
            fdst = (q_fm, k_fm)[kind][b, h, :, t0:t0 + tw]
            if strided:
                # fm_ps[d, i, p] is token t0 + 4p + i -> fm_sb[d, p, i] is contiguous in t.
                fm_sb = nl.ndarray((P, P, NB), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=fm_sb, src=fm_ps.permute((0, 2, 1)))
                nisa.dma_copy(dst=fdst, src=fm_sb.reshape((P, tw)))
            else:
                fm2 = nl.ndarray((P, NB, P), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=fm2[:, 0:nsub, :], src=fm_ps[:, 0:nsub, :])
                nisa.dma_copy(dst=fdst, src=fm2[:, 0:nsub, :].reshape((P, tw)))

    if batched:
        # T <= 512: every DMA covers all hps heads of a kind on this core and all B sequences
        # (3 tap loads, 3 activation loads, 3 stores per core instead of per (head, batch)).
        strided = T == MT
        nsub = NB if strided else NBT
        for kind in range(3):               # 0 = q, 1 = k, 2 = v
            c0 = kind * HQ + shard * hps
            diag_k = nl.ndarray((P, hps, K_CONV, P), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=diag_k, src=conv_diag.ap(pattern=[[K_CONV * P, P], [P * K_CONV * P, hps], [1, K_CONV * P]],
                                                         offset=c0 * P * K_CONV * P))
            # y_k[:, hh, b, 3 + t] = proj[b, c0 + hh, :, t]; columns 0..2 = conv history (or zeros).
            y_k = nl.ndarray((P, hps, B, T + KH), dtype=bf16, buffer=nl.sbuf)
            if has_hist:
                load_hist(y_k[:, :, :, 0:KH], c0, hps, 0, B)
            else:
                nisa.memset(dst=y_k[:, :, :, 0:KH], value=0.0)
            nisa.dma_copy(dst=y_k[:, :, :, KH:KH + T],
                          src=proj.ap(pattern=[[T, P], [D * T, hps], [NH * D * T, B], [1, T]], offset=c0 * D * T))
            o_k = nl.ndarray((P, hps, B, nsub, D), dtype=bf16, buffer=nl.sbuf)
            for hh in range(hps):
                h = shard * hps + hh
                for b in range(B):
                    conv_tile(y_k[:, hh, b, :], diag_k[:, hh, :, :], 0, T, kind, b, h, o_k[:, hh, b, 0:nsub, :])
            h0 = shard * hps
            if strided:
                nisa.dma_copy(dst=outs[kind].ap(pattern=[[NB * D, P], [T * D, hps], [HQ * T * D, B], [1, NB * D]],
                                                offset=h0 * T * D),
                              src=o_k.reshape((P, hps, B, nsub * D)))
            elif nsub == 1:
                nisa.dma_copy(dst=outs[kind].ap(pattern=[[D, P], [T * D, hps], [HQ * T * D, B], [1, D]], offset=h0 * T * D),
                              src=o_k[:, :, :, 0, :])
            else:
                for hh in range(hps):
                    nisa.dma_copy(dst=outs[kind].ap(pattern=[[D, P], [HQ * T * D, B], [P * D, nsub], [1, D]],
                                                    offset=(h0 + hh) * T * D),
                                  src=o_k[:, hh, :, 0:nsub, :])
    else:
        for b in range(B):
            for kind in range(3):           # 0 = q, 1 = k, 2 = v
                for hh in range(hps):
                    h = shard * hps + hh
                    ch = kind * HQ + h
                    diag_sb = nl.ndarray((P, K_CONV, P), dtype=bf16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=diag_sb, src=conv_diag[ch])
                    # y_sb[:, 3 + t] = proj[b, ch, :, t]; columns 0..2 = conv history (or zeros).
                    y_sb = nl.ndarray((P, T + KH), dtype=bf16, buffer=nl.sbuf)
                    if has_hist:
                        load_hist(y_sb[:, 0:KH].reshape((P, 1, 1, KH)), ch, 1, b, 1)
                    else:
                        nisa.memset(dst=y_sb[:, 0:KH], value=0.0)
                    nisa.dma_copy(dst=y_sb[:, KH:KH + T], src=proj[b, ch])
                    for ti in range(n_tt):
                        t0 = ti * MT
                        tw = min(MT, T - t0)
                        strided = tw == MT
                        nsub = NB if strided else tw // P
                        o_sb = nl.ndarray((P, NB, P), dtype=bf16, buffer=nl.sbuf)
                        conv_tile(y_sb, diag_sb, t0, tw, kind, b, h, o_sb[:, 0:nsub, :])
                        if strided:
                            nisa.dma_copy(dst=outs[kind][b, h, t0:t0 + tw, :].reshape((P, NB * D)),
                                          src=o_sb.reshape((P, NB * D)))
                        else:
                            nisa.dma_copy(dst=outs[kind].ap(pattern=[[D, P], [P * D, nsub], [1, D]],
                                                            offset=((b * HQ + h) * T + t0) * D),
                                          src=o_sb[:, 0:nsub, :])

    # ---- beta / g: this core takes a contiguous token range (whole 128-blocks) of every sequence.
    ZB = ba.shape[2]                        # b|a are the LAST 2H columns (ba [B,T,2H] or zba)
    ab_sb = nl.ndarray((HQ, 2), dtype=fp32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ab_sb, src=gdn_ab)
    bps = (NBT + n_shards - 1) // n_shards
    bb0 = shard * bps
    nbb = min(NBT, bb0 + bps) - bb0
    tb0 = bb0 * P
    tps = nbb * P
    if nbb > 0:
        ba_sb = nl.ndarray((P, B, bps, 2 * HQ), dtype=bf16, buffer=nl.sbuf)   # token = tb0 + 128*i + p
        nisa.dma_copy(dst=ba_sb[:, :, 0:nbb, :],
                      src=ba.ap(pattern=[[ZB, P], [T * ZB, B], [P * ZB, nbb], [1, 2 * HQ]],
                                offset=tb0 * ZB + ZB - 2 * HQ))
        beta_sb = nl.ndarray((HQ, B, bps * P), dtype=fp32, buffer=nl.sbuf)
        g_sb = nl.ndarray((HQ, B, bps * P), dtype=fp32, buffer=nl.sbuf)
        for b in range(B):
            for i0 in range(0, nbb, NB):
                ni = min(NB, nbb - i0)
                pb = nl.ndarray((HQ, MT), dtype=bf16, buffer=nl.psum)
                pa = nl.ndarray((HQ, MT), dtype=bf16, buffer=nl.psum)
                for i in range(ni):
                    nisa.nc_transpose(dst=pb[:, i * P:(i + 1) * P], data=ba_sb[:, b, i0 + i, 0:HQ],
                                      engine=nisa.tensor_engine)
                    nisa.nc_transpose(dst=pa[:, i * P:(i + 1) * P], data=ba_sb[:, b, i0 + i, HQ:2 * HQ],
                                      engine=nisa.tensor_engine)
                c0 = i0 * P
                cw = ni * P
                nisa.activation(dst=beta_sb[:, b, c0:c0 + cw], op=nl.sigmoid, data=pb[:, 0:cw])
                sp_sb = nl.ndarray((HQ, MT), dtype=fp32, buffer=nl.sbuf)
                nisa.activation(dst=sp_sb[:, 0:cw], op=nl.softplus, data=pa[:, 0:cw], bias=ab_sb[:, 1:2])
                nisa.tensor_scalar(dst=g_sb[:, b, c0:c0 + cw], data=sp_sb[:, 0:cw], op0=nl.multiply,
                                   operand0=ab_sb[:, 0:1], engine=nisa.engine.vector)
        nisa.dma_copy(dst=beta.ap(pattern=[[T, HQ], [HQ * T, B], [1, tps]], offset=tb0), src=beta_sb[:, :, 0:tps])
        nisa.dma_copy(dst=g.ap(pattern=[[T, HQ], [HQ * T, B], [1, tps]], offset=tb0), src=g_sb[:, :, 0:tps])

    if emit_qk_fm:
        return q, k, v, g, beta, q_fm, k_fm
    return q, k, v, g, beta


@functools.lru_cache(maxsize=None)
def _wrapped_prologue():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(gdn_prologue_kernel)


def gdn_prologue(proj_fm, conv_diag, ba, gdn_ab, eps: float = L2_EPS, emit_qk_fm: bool = False, conv_hist=None):
    """Fused GDN prologue (NKI): causal depthwise conv1d (K=4, zero history) + SiLU + L2-norm(q, k) + beta/g.

    H = heads per tensor-parallel rank (16 at TP1, 8 at TP2, 4 at TP4; shard_state head order).
    Args:
        proj_fm: [B, 3H, 128, T] bf16, pre-conv q|k|v projection, FEATURE-major per head
                 (= einsum("btc,chd->bhdt", rms_norm(x), w_qkv_t), see gdn_in_proj).
        conv_diag: [3H, 128, 4, 128] bf16 diagonal tap matrices (prepare_gdn_layer / conv_diag_from_conv_w).
        ba:      [B, T, Z] bf16, token-major; the LAST 2H columns are [b(H) | a(H)] -- pass h @ w_ba.T
                 ([B, T, 2H]) or gdn_in_proj's zba ([B, T, 128H + 2H]) directly.
        gdn_ab:  [H, 2] fp32, (A, dt_bias) per head (prepare_gdn_layer).
        emit_qk_fm: also return feature-major copies q_fm, k_fm [B, H, 128, T].
        conv_hist: optional [Bh, 3H, 3, 128] bf16 (Bh = 1 broadcasts): pre-conv values of the 3 tokens
                 before token 0, head-major (= DeciderNeuron.conv_tail / conv_tail_fm of the prefix).
                 The conv then runs over [history | proj]; None = zero history.

    Returns:
        q, k: [B, H, T, 128] bf16, L2-normalised over the head dim: x * rsqrt(sum x^2 + 1e-6).
        v:    [B, H, T, 128] bf16.
        g:    [B, H, T] fp32 = A * softplus(a + dt_bias).
        beta: [B, H, T] fp32 = sigmoid(b).
        (+ q_fm, k_fm: [B, H, 128, T] bf16 when emit_qk_fm)
    Requires T % 128 == 0 and H even.  Matches decider_model.gdn_layer to rel-L2 ~3e-3 (q, k: SiLU output is
    rounded to bf16 before the norm), ~1e-5 (v), ~1e-7 (g, beta).
    """
    kw = {} if conv_hist is None else {"conv_hist": conv_hist.contiguous()}
    return _wrapped_prologue()[_lnc()](proj_fm, conv_diag, ba, gdn_ab, eps=eps, emit_qk_fm=emit_qk_fm, **kw)


# ===================================================================================== RMSNorm kernel
@nki.jit
def rmsnorm_kernel(x, ln, eps=1e-6):
    """x [N, H] (bf16), ln [H] fp32 -> [N, H] bf16 = bf16((x * rsqrt(mean(x^2) + eps)) * ln), fp32 math.

    Token tiles of 128 rows are split over the cores of the logical NC. ScalarE: square+row-sum,
    GpSimd: rsqrt (ULP-1), VectorE: one fused (x * rstd) * ln pass.
    """
    N, H = x.shape
    n_shards = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    shard = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    out = nl.ndarray((N, H), dtype=x.dtype, buffer=nl.shared_hbm)
    ln_sb = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ln_sb, src=ln.ap(pattern=[[0, P], [1, H]], offset=0))
    n_tiles = (N + P - 1) // P
    per = (n_tiles + n_shards - 1) // n_shards
    for ti in range(shard * per, min(n_tiles, (shard + 1) * per)):
        r0 = ti * P
        rw = min(P, N - r0)
        x_sb = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb[0:rw], src=x[r0:r0 + rw, :])
        junk = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=junk[0:rw], op=nl.square, data=x_sb[0:rw], reduce_op=nl.add,
                               reduce_res=ss[0:rw])
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms[0:rw], data=ss[0:rw], op0=nl.multiply, operand0=1.0 / H, op1=nl.add,
                           operand1=eps)
        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rstd[0:rw], data=ms[0:rw], op0=nl.rsqrt, operand0=None, engine=nisa.engine.gpsimd)
        y_sb = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=y_sb[0:rw], data=x_sb[0:rw], op0=nl.multiply, operand0=rstd[0:rw],
                                  op1=nl.multiply, operand1=ln_sb[0:rw])
        nisa.dma_copy(dst=out[r0:r0 + rw, :], src=y_sb[0:rw])
    return out


@functools.lru_cache(maxsize=None)
def _wrapped_rmsnorm():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(rmsnorm_kernel)


def rms_norm_nki(x, ln, eps: float = 1e-6):
    """NKI RMSNorm, same math as decider_model.rms_norm: x [..., H] bf16, ln [H] fp32 (folded 1+w) -> bf16."""
    shp = x.shape
    y = _wrapped_rmsnorm()[_lnc()](x.reshape(-1, shp[-1]), ln.float(), eps=eps)
    return y.reshape(shp)


# ===================================================================================== gated RMSNorm (GDN epilogue)
@nki.jit
def gated_rmsnorm_kernel(o, zba, gnorm, eps=1e-6):
    """o [B, H, T, D] bf16 (GDN core output), zba [B, T, Z>=H*D] bf16 (z in columns 0:H*D), gnorm [D] bf16
    -> y [B, T, H*D] bf16, token-major (ready for the out-projection GEMM):
        y = bf16( bf16(gnorm * bf16(o * rsqrt(mean_d o^2 + eps))) * silu(z) )   (decider gated_rms_norm)

    128-token tiles are split over the cores.  ScalarE: o^2, silu(z); VectorE: row sums, the three
    multiplies (rounded like the reference); GpSimd: rsqrt.
    """
    B, H, T, D = o.shape
    Z = zba.shape[2]
    HD = H * D
    n_shards = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    shard = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    bf16, fp32 = nl.bfloat16, nl.float32
    y = nl.ndarray((B, T, HD), dtype=bf16, buffer=nl.shared_hbm)
    w_sb = nl.ndarray((P, D), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=gnorm.ap(pattern=[[0, P], [1, D]], offset=0))
    n_tiles = B * ((T + P - 1) // P)
    per = (n_tiles + n_shards - 1) // n_shards
    tpb = (T + P - 1) // P
    for ti in range(shard * per, min(n_tiles, (shard + 1) * per)):
        b = ti // tpb
        t0 = (ti % tpb) * P
        tw = min(P, T - t0)
        o_sb = nl.ndarray((P, H, D), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=o_sb[0:tw], src=o.ap(pattern=[[D, tw], [T * D, H], [1, D]], offset=(b * H * T + t0) * D))
        z_sb = nl.ndarray((P, HD), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=z_sb[0:tw], src=zba.ap(pattern=[[Z, tw], [1, HD]], offset=(b * T + t0) * Z))
        sq = nl.ndarray((P, H, D), dtype=bf16, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:tw], op=nl.square, data=o_sb[0:tw])
        ss = nl.ndarray((P, H), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ss[0:tw], op=nl.add, data=sq[0:tw], axis=2)
        ms = nl.ndarray((P, H), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ms[0:tw], data=ss[0:tw], op0=nl.multiply, operand0=1.0 / D, op1=nl.add, operand1=eps,
                           engine=nisa.engine.vector)
        rn = nl.ndarray((P, H), dtype=fp32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rn[0:tw], data=ms[0:tw], op0=nl.rsqrt, operand0=None, engine=nisa.engine.gpsimd)
        y1 = nl.ndarray((P, H, D), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y1[0:tw], data1=o_sb[0:tw], data2=rn[0:tw].expand_dim(2).broadcast(2, D),
                           op=nl.multiply, engine=nisa.engine.vector)
        y2 = nl.ndarray((P, H, D), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y2[0:tw], data1=y1[0:tw], data2=w_sb[0:tw].expand_dim(1).broadcast(1, H),
                           op=nl.multiply, engine=nisa.engine.vector)
        sz = nl.ndarray((P, HD), dtype=fp32, buffer=nl.sbuf)
        nisa.activation(dst=sz[0:tw], op=nl.silu, data=z_sb[0:tw])
        y_sb = nl.ndarray((P, HD), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_sb[0:tw], data1=y2[0:tw].reshape((tw, HD)), data2=sz[0:tw], op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=y[b, t0:t0 + tw, :], src=y_sb[0:tw])
    return y


@functools.lru_cache(maxsize=None)
def _wrapped_gated():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(gated_rmsnorm_kernel)


def gated_rms_norm_nki(o, zba, gnorm, eps: float = 1e-6):
    """NKI gated RMSNorm (decider_model.gated_rms_norm math), token-major output for the out-projection.

    o:     [B, 16, T, 128] bf16, GDN core output (head-major).
    zba:   [B, T, Z] bf16 with z in columns 0:2048 (pass gdn_in_proj's zba output directly, Z = 2080).
    gnorm: [128] bf16.
    Returns [B, T, 2048] bf16 = gated_rms_norm(o, z, gnorm).transpose(1, 2).reshape(B, T, 2048).
    """
    return _wrapped_gated()[_lnc()](o, zba, gnorm, eps=eps)


def gdn_out_proj(o, zba, gnorm, w_out_t, eps: float = 1e-6):
    """out_proj(gated_rms_norm(o, z)) WITHOUT the residual: the TP partial sum ([B, T, 2048] bf16) the
    caller all-reduces.  o [B, H, T, 128], zba [B, T, 128H + 2H], w_out_t [128H, 2048]."""
    return gated_rms_norm_nki(o, zba, gnorm, eps) @ w_out_t


def gdn_epilogue(x, o, zba, gnorm, w_out_t, eps: float = 1e-6, reduce=None):
    """x + reduce(out_proj(gated_rms_norm(o, z))):  x [B, T, 2048] bf16, o [B, H, T, 128] (H heads per
    rank), zba [B, T, 128H + 2H], w_out_t [128H, 2048] bf16 (prepare_gdn_layer).  `reduce` (e.g. the
    TP all-reduce, DeciderNeuron._reduce) is applied to the projection before the residual add."""
    y = gdn_out_proj(o, zba, gnorm, w_out_t, eps)
    return x + (reduce(y) if reduce is not None else y)


# ===================================================================================== attention gate (epilogue)
@nki.jit
def sigmoid_gate_kernel(o, gsrc, goff=0):
    """o [B, H, T, D] bf16 (attention core output, head-major), gsrc [B, T, Z] bf16 token-major with the
    gate (H x D) at columns goff : goff + H*D  ->  y [B, T, H*D] bf16 = o * sigmoid(gate), token-major
    (ready for a [K, N] o-projection GEMM).  128-token tiles split over the cores."""
    B, H, T, D = o.shape
    Z = gsrc.shape[2]
    HD = H * D
    n_shards = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1
    shard = nl.program_id(axis=0) if nl.program_ndim() != 0 else 0
    y = nl.ndarray((B, T, HD), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    tpb = (T + P - 1) // P
    n_tiles = B * tpb
    per = (n_tiles + n_shards - 1) // n_shards
    for ti in range(shard * per, min(n_tiles, (shard + 1) * per)):
        b = ti // tpb
        t0 = (ti % tpb) * P
        tw = min(P, T - t0)
        o_sb = nl.ndarray((P, H, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=o_sb[0:tw], src=o.ap(pattern=[[D, tw], [T * D, H], [1, D]], offset=(b * H * T + t0) * D))
        g_sb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_sb[0:tw], src=gsrc.ap(pattern=[[Z, tw], [1, HD]], offset=(b * T + t0) * Z + goff))
        sg = nl.ndarray((P, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg[0:tw], op=nl.sigmoid, data=g_sb[0:tw])
        y_sb = nl.ndarray((P, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=y_sb[0:tw], data1=o_sb[0:tw].reshape((tw, HD)), data2=sg[0:tw], op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=y[b, t0:t0 + tw, :], src=y_sb[0:tw])
    return y


@functools.lru_cache(maxsize=None)
def _wrapped_sgate():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(sigmoid_gate_kernel)


def sigmoid_gate_nki(o, gsrc, goff: int):
    """[B, T, H*D] = (o * sigmoid(gate)) token-major; o [B, H, T, D] head-major, gate = gsrc[..., goff:goff+H*D]."""
    return _wrapped_sgate()[_lnc()](o, gsrc, goff=goff)


# ===================================================================================== (A) torch side
def conv_diag_from_conv_w(conv_w: torch.Tensor) -> torch.Tensor:
    """conv_w [3H, 128, 4] -> [3H, 128, 4, 128] with out[h, c, j, c'] = conv_w[h, c, j] * (c == c')."""
    H, D, K = conv_w.shape
    eye = torch.eye(D, dtype=conv_w.dtype, device=conv_w.device)
    return (conv_w[:, :, :, None] * eye[None, :, None, :]).contiguous()


def prepare_gdn_layer(sd: dict, i: int, dtype=torch.bfloat16) -> dict:
    """One-time weight re-layout for GDN layer i of a decider_model.load_state() state dict, or of a
    shard_state() shard (H = heads per rank = sd["w_qkvz"].shape[0] // 4: 16 / 8 / 4 at TP1 / 2 / 4).

    Returns (all contiguous, CPU; move to device once):
        ln1       [2048]            fp32  folded (1 + w) RMSNorm weight (unchanged)
        w_qkv_t   [2048, 3H, 128]   bf16  in-proj q|k|v heads, K-major (pre-transposed) -> feature-major out
        w_zba_t   [2048, 128H + 2H] bf16  z (H x 128) | b (H) | a (H) columns, K-major -> token-major out
        conv_diag [3H, 128, 4, 128] bf16  conv taps as diagonal matrices diag(w[h, :, j]) (6 MB/layer at TP1)
        gdn_ab    [H, 2]            fp32  (A, dt_bias)
        gnorm     [128]             bf16  RMSNormGated weight
        w_out_t   [128H, 2048]      bf16  out-proj, rows = (h, d) input features, K-major
    """
    p = f"layers.{i}."
    w_qkvz = sd[p + "w_qkvz"].to(dtype)                        # [4H, 128, 2048]
    H = w_qkvz.shape[0] // 4
    C = w_qkvz.shape[2]
    w_qkv_t = w_qkvz[:3 * H].permute(2, 0, 1).contiguous()     # [2048, 3H, 128]
    w_z = w_qkvz[3 * H:].reshape(H * w_qkvz.shape[1], C)       # [(h d), c]
    w_zba_t = torch.cat([w_z, sd[p + "w_ba"].to(dtype)], 0).t().contiguous()  # [2048, 128H + 2H]
    w_out = sd[p + "w_out"].to(dtype)                          # [2048, H, 128]
    return {
        "ln1": sd[p + "ln1"].float().contiguous(),
        "w_qkv_t": w_qkv_t,
        "w_zba_t": w_zba_t,
        "conv_diag": conv_diag_from_conv_w(sd[p + "conv_w"].to(dtype)),
        "gdn_ab": torch.stack([sd[p + "A"].float(), sd[p + "dt_bias"].float()], 1).contiguous(),
        "gnorm": sd[p + "gnorm"].to(dtype).contiguous(),
        "w_out_t": w_out.reshape(w_out.shape[0], -1).t().contiguous(),
    }


def prepare_mlp_layer(sd: dict, i: int, dtype=torch.bfloat16) -> dict:
    """Pre-transposed (K-major) MLP weights: w_gate_up_t [2048, 2I], w_down_t [I, 2048]; ln2 [2048] fp32.
    I = per-rank intermediate size (6144 / TP).  neuronx-cc GEMMs read [K, N] weights 5-15% faster
    than the [N, K] layout used with `x @ w.t()`.
    """
    p = f"layers.{i}."
    return {
        "ln2": sd[p + "ln2"].float().contiguous(),
        "w_gate_up_t": sd[p + "w_gate_up"].to(dtype).t().contiguous(),
        "w_down_t": sd[p + "w_down"].to(dtype).t().contiguous(),
    }


def mlp_fast(x, prep, eps: float = 1e-6, inter: int | None = None, reduce=None):
    """Drop-in for DeciderNeuron.mlp: x + reduce((silu(g) * u) @ W_down) with pre-transposed weights.

    x [B, T, 2048] bf16 -> [B, T, 2048] bf16.  inter defaults to the (per-rank) weight shape.  RMSNorm
    stays in torch: in front of a token-major GEMM neuronx-cc fuses it well.
    """
    inter = prep["w_gate_up_t"].shape[1] // 2 if inter is None else inter
    h = _rms_norm(x, prep["ln2"], eps)
    gu = h @ prep["w_gate_up_t"]
    y = (F.silu(gu[..., :inter]) * gu[..., inter:]) @ prep["w_down_t"]
    return x + (reduce(y) if reduce is not None else y)


def gdn_layer_fast(x, prep, core, eps: float = 1e-6, reduce=None):
    """Drop-in for DeciderNeuron.gdn_layer (same math), with the GDN core passed in.

    x [B, T, 2048] bf16; prep = prepare_gdn_layer(...) on device; core(q, k, v, g, beta) -> o [B, H, T, 128]
    with q, k, v [B, H, T, 128] bf16 (q, k L2-normalised) and g, beta [B, H, T] fp32 (H heads per rank).
    reduce: TP all-reduce applied to the out-projection partial sum before the residual.
    Pipeline: NKI RMSNorm -> 2 compiler GEMMs (feature-major q|k|v, token-major z|b|a) -> NKI prologue
    (conv + SiLU + L2 + beta/g) -> core -> NKI gated RMSNorm (token-major) -> out-proj GEMM + residual.
    """
    proj_fm, zba = gdn_in_proj(x, prep["ln1"], prep, eps)
    q, k, v, g, beta = gdn_prologue(proj_fm, prep["conv_diag"], zba, prep["gdn_ab"])
    o = core(q, k, v, g, beta)
    return gdn_epilogue(x, o.to(x.dtype), zba, prep["gnorm"], prep["w_out_t"], eps, reduce)


def conv_tail_fm(proj_fm, s_len, K: int = K_CONV):
    """Last K-1 pre-conv tokens before position s_len, from the feature-major projection:
    proj_fm [1, 3H, 128, S], s_len [1] int -> [1, 3H, K-1, 128] (zeros before position 0).
    Same values/layout as DeciderNeuron.conv_tail(qkvz[:, :3H], s_len, K)."""
    idx = s_len.reshape(1).long() - (K - 1) + torch.arange(K - 1, device=proj_fm.device)
    ok = (idx >= 0).to(proj_fm.dtype)[None, None, None, :]
    t = torch.index_select(proj_fm, 3, idx.clamp_min(0)) * ok        # [1, 3H, 128, K-1]
    return t.transpose(2, 3).contiguous()


def gdn_layer_prefix_fast(x, prep, core, valid, s_len, eps: float = 1e-6, reduce=None):
    """Drop-in for DeciderNeuron.gdn_layer_prefix: x [1, S, 2048] (right-padded), valid [1, S] bool,
    s_len [1] int.  beta and g are zeroed for pad positions (in torch: 2 x [1, H, S] multiplies).
    core(q, k, v, g, beta, return_state=True) -> (o, state).  Returns (y, state, conv tail [1, 3H, 3, 128])."""
    proj_fm, zba = gdn_in_proj(x, prep["ln1"], prep, eps)
    tail = conv_tail_fm(proj_fm, s_len)
    q, k, v, g, beta = gdn_prologue(proj_fm, prep["conv_diag"], zba, prep["gdn_ab"])
    vm = valid[:, None, :].float()
    o, S = core(q, k, v, g * vm, beta * vm, return_state=True)
    return gdn_epilogue(x, o.to(x.dtype), zba, prep["gnorm"], prep["w_out_t"], eps, reduce), S, tail


def gdn_layer_suffix_fast(x, prep, core, S0, tail, eps: float = 1e-6, reduce=None):
    """Drop-in for DeciderNeuron.gdn_layer_suffix: x [B, L, 2048], S0 [1, H, 128, 128] fp32 prefix state,
    tail [1 or B, 3H, 3, 128] bf16 prefix conv tail (conv history of the suffix, consumed in-kernel).
    core(q, k, v, g, beta, S0=[B, H, 128, 128]) -> o."""
    B = x.shape[0]
    proj_fm, zba = gdn_in_proj(x, prep["ln1"], prep, eps)
    q, k, v, g, beta = gdn_prologue(proj_fm, prep["conv_diag"], zba, prep["gdn_ab"], conv_hist=tail.to(x.dtype))
    o = core(q, k, v, g, beta, S0=S0.expand(B, -1, -1, -1).contiguous())
    return gdn_epilogue(x, o.to(x.dtype), zba, prep["gnorm"], prep["w_out_t"], eps, reduce)


def prepare_attn_layer(sd: dict, i: int, dtype=torch.bfloat16) -> dict:
    """Pre-transposed attention weights for attn_layer_fast (works on shard_state() shards too):
        ln1 [2048] fp32; w_qgkv_t [2048, nq+nq+2*nkv, 256] bf16 (K-major head-major einsum);
        w_o_t [nq*256, 2048] bf16 (K-major, rows = (h, d)); q_norm, k_norm [256] fp32."""
    p = f"layers.{i}."
    w_o = sd[p + "w_o"].to(dtype)                               # [2048, nq, 256]
    return {
        "ln1": sd[p + "ln1"].float().contiguous(),
        "w_qgkv_t": sd[p + "w_qgkv"].to(dtype).permute(2, 0, 1).contiguous(),
        "w_o_t": w_o.reshape(w_o.shape[0], -1).t().contiguous(),
        "q_norm": sd[p + "q_norm"].float().contiguous(),
        "k_norm": sd[p + "k_norm"].float().contiguous(),
    }


def attn_layer_fast(x, prep, core, cos, sin, nq: int, nkv: int, eps: float = 1e-6, reduce=None, **core_kw):
    """Drop-in for DeciderNeuron.attn_layer (same math): NKI RMSNorm -> pre-transposed head-major einsum ->
    q/k RMSNorm + partial RoPE (torch) -> core(q, k, v, scale, **core_kw) -> sigmoid gate -> token-major
    [K, N] o-proj -> reduce -> + x.  nq / nkv = per-rank query / kv heads.  Measured (core stubbed):
    T=512 0.44 -> 0.30 ms, T=2048 1.08 -> 1.07 ms (the T>=2048 non-GEMM cost is the q/k norm + RoPE)."""
    from decider_model import apply_partial_rope

    B, T, _ = x.shape
    h = rms_norm_nki(x, prep["ln1"], eps)
    qgkv = torch.einsum("btc,chd->bhtd", h, prep["w_qgkv_t"])
    q = apply_partial_rope(_rms_norm(qgkv[:, :nq], prep["q_norm"], eps), cos, sin)
    gate = qgkv[:, nq:2 * nq]
    k = apply_partial_rope(_rms_norm(qgkv[:, 2 * nq:2 * nq + nkv], prep["k_norm"], eps), cos, sin)
    v = qgkv[:, 2 * nq + nkv:]
    o = core(q, k, v, q.shape[-1] ** -0.5, **core_kw) * torch.sigmoid(gate)
    y = o.transpose(1, 2).reshape(B, T, -1) @ prep["w_o_t"]
    return x + (reduce(y) if reduce is not None else y)


def _rms_norm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def gdn_in_proj(x, ln1, prep, eps: float = 1e-6, nki_norm: bool = True):
    """RMSNorm + GDN input projections in the kernel's layouts (compiler GEMMs, pre-transposed weights).

    x [B, T, 2048] bf16, ln1 [2048] fp32 (folded 1 + w).  H = heads per rank.
    Returns proj_fm [B, 3H, 128, T] bf16 and zba [B, T, 128H + 2H] bf16 (columns 0:128H = z token-major
    (h, d), then b (H), then a (H)).  Pass zba to gdn_prologue (it reads the last 2H columns) and to
    gated_rms_norm_nki / gdn_epilogue (they read the z columns).

    nki_norm: run the RMSNorm as the NKI kernel (a custom-call boundary in front of the GEMMs).  In
    some graph contexts neuronx-cc fuses the torch rms_norm into the feature-major einsum and the
    pair runs 2-2.5x slower (measured 1.13 ms vs 0.58 ms at T=2048); in this exact function the two
    are within ~5% (torch slightly faster at T<=1024, NKI faster at T=4096).
    """
    h = rms_norm_nki(x, ln1, eps) if nki_norm else _rms_norm(x, ln1, eps)
    proj_fm = torch.einsum("btc,chd->bhdt", h, prep["w_qkv_t"])
    zba = h @ prep["w_zba_t"]
    return proj_fm, zba


def gdn_prologue_ref(proj_hm, conv_w, ba, A, dt_bias, eps: float = L2_EPS, conv_hist=None):
    """Exact decider_model.gdn_layer(_suffix) math for the prologue (torch, any device).

    proj_hm [B, 3H, T, 128] (head-major token-major, like qkvz[:, :3H]), conv_w [3H, 128, 4], ba [B, T, 2H],
    conv_hist optional [Bh, 3H, 3, 128] (pre-conv history, prepended along T as in gdn_layer_suffix).
    Returns q, k, v (bf16 [B,H,T,128]), g, beta (fp32 [B,H,T]).
    """
    Kc = conv_w.shape[-1]
    B, C, T, _ = proj_hm.shape
    H = C // 3
    if conv_hist is None:
        xf = F.pad(proj_hm.float(), (0, 0, Kc - 1, 0))
    else:
        xf = torch.cat([conv_hist.float().expand(B, -1, -1, -1), proj_hm.float()], 2)
    wf = conv_w.float()
    out = xf[:, :, Kc - 1:Kc - 1 + T] * wf[None, :, None, :, Kc - 1]
    for kk in range(Kc - 1):
        out = out + xf[:, :, kk:kk + T] * wf[None, :, None, :, kk]
    conv = F.silu(out)

    def l2(t):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + eps)

    q = l2(conv[:, :H]).to(torch.bfloat16)
    k = l2(conv[:, H:2 * H]).to(torch.bfloat16)
    v = conv[:, 2 * H:3 * H].to(torch.bfloat16)
    baf = ba.float().transpose(1, 2)
    beta = torch.sigmoid(baf[:, :H])
    g = A.float()[None, :, None] * F.softplus(baf[:, H:2 * H] + dt_bias.float()[None, :, None])
    return q, k, v, g, beta
