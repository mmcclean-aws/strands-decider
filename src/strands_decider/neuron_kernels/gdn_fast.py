"""Fast Gated DeltaNet (GDN) chunked-prefill NKI kernel for trn2 (LNC=2), + torch wrapper.

Math (chunk C=128, per lane = (batch, head); g = per-token log decay, G = within-chunk cumsum):
    A[r,s]  = beta_r * exp(G_r - G_s) * k_r.k_s            (s < r)      intra-chunk interaction
    T       = (I + A)^-1                                     unit lower triangular
    u       = T (beta v),  w = T (beta e^G k)               WY form, state independent
    delta   = u - w S                                        per chunk, S = state entering the chunk
    out     = (scale e^G q) S + W delta,  W[r,s] = scale exp(G_r-G_s) q_r.k_s (s <= r)
    S_next  = e^{G_last} S + (e^{G_last - G} k)^T delta
Same semantics as neuron_model.gdn_chunk_ref / HF torch_chunk_gated_delta_rule (incl. optional S0).

Design (vs nkilib gdn_cte schur; see the timing/profile notes in test_gdn_fast.py):
  * WY form: the Schur-complement inverse is applied to [beta v | beta e^G k] once per chunk OFF the
    recurrence, so the sequential (state) path per chunk is 4 matmuls + 2 vector ops.
  * Lanes are processed 4 at a time with every vector/scalar instruction batched over the 4 lanes
    ([128, 4*128] operands = one PSUM bank), amortising the ~65 ns (DVE) / ~290 ns (ACT) fixed cost per
    instruction that dominates [128,128] ops; identity-stationary matmuls are batched to N=512.
  * Jobs (4 lanes x 1 chunk) are emitted level-major in batches of `jobs_per_batch`, with the previous
    batch's state updates interleaved between the inverse levels, so every engine has independent work
    (B=1 has only 8 lanes per NeuronCore).
  * Schur inverse with the off-block mask applied on the PSUM evacuation of A^T Inv^T (no per-level masked
    copies of A); Inv^T accumulates in fp32 PSUM across the 7 levels.
  * Decay matrices via a K=3 broadcast matmul (exact bf16 hi/mid/lo split of G) + mask matmul and ONE
    masked exp per lane (beta and the query scale folded into the exp bias); no fp32 matmuls per chunk.
  * All DMAs are DGE-generated (no dge_mode=none): no per-execution static descriptor rewrite, which is
    what costs nkilib gdn_cte ~1.8 ms per call when NEURON_RT_IO_RING_CACHE_SIZE is not set.

Layouts / dtypes (all contiguous):
    q, k, v      [B, H, T, 128]   compute dtype (bf16 default; fp16 = same speed, ~8x lower error;
                                   fp32 = 1.7x slower, ~1e-6 error). q, k L2-normalised, q NOT pre-scaled.
    qt, kt       [B, H, 128, T]   OPTIONAL feature-major q / k (then q is not read; k token-major still is):
                                   skips the in-kernel DMA transposes (~2% faster).
    g, beta      [B, H, T]        fp32 (g = log decay <= 0, beta in (0, 1])
    S0           [B, H, 128, 128] fp32, optional initial state (zero and no load when omitted)
    consts       gdn_fast_consts(device): [9, 128, 128] fp32 host-built masks (create once, outside compile)
    -> out       [B, H, T, 128]   out_dtype (default: compute dtype), head-major token-major
    -> S_final   [B, H, 128, 128] fp32 final state
T must be a multiple of 128 (pad the tail with g=0, beta=0: padded tokens then leave the state unchanged).
"""

from __future__ import annotations

import functools

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

C = 128  # chunk length == partition count
D = 128  # head dim (Dk == Dv == 128)
U = 4  # lanes per batched instruction (one PSUM bank of fp32 [128, 4*128])
N_LEVELS = 7  # log2(C)
BIG = 30000.0  # mask magnitude inside exp()

# constant pack planes (host built, fp32)
I_EYE = 0
I_MASK_INCL = 1  # +BIG where f > p   (exp(-(G_f + MASK) + G_p) -> lower triangle incl. diagonal)
I_NEGOFFT = 2  # 7 planes: -offT_l[s, r'] (s in lower half, r' in upper half of a 2^(l+1) block)
N_CONST = I_NEGOFFT + N_LEVELS


def build_consts() -> np.ndarray:
    eye = np.eye(C, dtype=np.float32)
    p = np.arange(C)[:, None]
    f = np.arange(C)[None, :]
    planes = [eye, np.where(f > p, BIG, 0.0).astype(np.float32)]
    bs = 1
    while bs < C:
        m = 2 * bs
        # off_l[r, s] = 1 where r in q-half, s in p-half of the same m-block; offT_l[s, r'] = off_l[r', s]
        offT = ((p // m == f // m) & ((p % m) < bs) & ((f % m) >= bs)).astype(np.float32)
        planes.append(-offT)
        bs = m
    return np.ascontiguousarray(np.stack(planes, 0))


@nki.jit
def gdn_fast_kernel(
    q,
    k,
    v,
    g,
    beta,
    consts,
    s0=None,
    qt_in=None,
    kt_in=None,
    scale=1.0,
    out_dtype=None,
    jobs_per_batch=4,
):
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    assert Dk == D and Dv == D and T % C == 0
    assert (q is not None) or (qt_in is not None and kt_in is not None)
    cdt = k.dtype  # matmul operand dtype
    odt = cdt if out_dtype is None else out_dtype
    n = T // C
    L = B * H
    n_shards = nl.num_programs(axes=0)
    shard = nl.program_id(axis=0)
    per, extra = divmod(L, n_shards)
    nl_here = per + (1 if shard < extra else 0)
    l_start = shard * per + min(shard, extra)

    out = nl.ndarray((B, H, T, Dv), dtype=odt, buffer=nl.shared_hbm)
    s_out = nl.ndarray((B, H, Dk, Dv), dtype=nl.float32, buffer=nl.shared_hbm)

    q_f = q.reshape((L, n, C, D)) if q is not None else None
    k_f = k.reshape((L, n, C, D))
    v_f = v.reshape((L, n, C, D))
    o_f = out.reshape((L, n, C, D))
    g_rows = g.reshape((L * n, C))
    b_rows = beta.reshape((L * n, C))
    s_out_f = s_out.reshape((L, D, D))
    s0_f = s0.reshape((L, D, D)) if s0 is not None else None

    # ---------------- constants
    cst = nl.ndarray((C, N_CONST, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cst, src=consts.ap(pattern=[[C, C], [C * C, N_CONST], [1, C]], offset=0))
    eye_f = cst[:, I_EYE, :]
    eye_c = nl.ndarray((C, C), dtype=cdt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eye_c, src=eye_f, engine=nisa.engine.vector)
    mask_c = nl.ndarray((C, C), dtype=cdt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mask_c, src=cst[:, I_MASK_INCL, :], engine=nisa.engine.vector)
    negoff = nl.ndarray((C, N_LEVELS, C), dtype=cdt, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=negoff, src=cst[:, I_NEGOFFT : I_NEGOFFT + N_LEVELS, :], engine=nisa.engine.vector
    )
    ln_scale = float(np.log(scale))

    # lane groups of U (last may be partial)
    groups = []
    li = 0
    while li < nl_here:
        groups.append((li, min(U, nl_here - li)))
        li += U

    # ---------------- per-shard prologue: gates (rows = (lane, chunk) pairs, <=128 per block)
    rows_total = nl_here * n
    # G rows, split exactly into 3 bf16 parts (hi + mid + lo == G to fp32 precision), go to HBM so each job can
    # fetch its lanes' rows onto partitions 0..2 for ONE K=3 bf16 broadcast matmul (ones^T @ parts = G).
    g_scr = nl.ndarray((3, rows_total, C), dtype=nl.bfloat16, buffer=nl.private_hbm)
    ones3 = nl.ndarray((3, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones3, value=1.0)
    n_rb = (rows_total + C - 1) // C
    # column-form per-token quantities: colq[t, quantity, row]
    NQ = 7  # 0: G (row form only), 1: beta, 2: e^{GL-G}, 3: beta e^G, 4: unused, 5: G + ln(beta), 6: G + ln(scale)
    colq = nl.ndarray((C, NQ, n_rb * C), dtype=nl.float32, buffer=nl.sbuf)
    glb = nl.ndarray(
        (C, n_rb * C), dtype=nl.float32, buffer=nl.sbuf
    )  # e^{G_last} broadcast to all partitions
    for rb in range(n_rb):
        r0 = rb * C
        nr = min(C, rows_total - r0)
        grow = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        brow = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=grow[0:nr, :], src=g_rows[l_start * n + r0 : l_start * n + r0 + nr, :])
        nisa.dma_copy(dst=brow[0:nr, :], src=b_rows[l_start * n + r0 : l_start * n + r0 + nr, :])
        qrow = nl.ndarray((C, NQ, C), dtype=nl.float32, buffer=nl.sbuf)
        Gr = qrow[0:nr, 0, :]
        nisa.tensor_scalar_cumulative(
            dst=Gr, src=grow[0:nr, :], op0=nl.multiply, imm0=1.0, op1=nl.add
        )
        nisa.tensor_copy(dst=qrow[0:nr, 1, :], src=brow[0:nr, :], engine=nisa.engine.vector)
        # e^{GL - G}
        nisa.activation(
            dst=qrow[0:nr, 2, :], op=nl.exp, data=Gr, bias=qrow[0:nr, 0, C - 1 : C], scale=-1.0
        )
        eG = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=eG[0:nr, :], op=nl.exp, data=Gr)
        nisa.tensor_tensor(
            dst=qrow[0:nr, 3, :], data1=eG[0:nr, :], data2=brow[0:nr, :], op=nl.multiply
        )
        lnb = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=lnb[0:nr, :], op=nl.log, data=brow[0:nr, :])
        nisa.tensor_tensor(dst=qrow[0:nr, 5, :], data1=Gr, data2=lnb[0:nr, :], op=nl.add)
        nisa.tensor_scalar(
            dst=qrow[0:nr, 6, :], data=Gr, op0=nl.add, operand0=ln_scale, engine=nisa.engine.vector
        )
        gsp = nl.ndarray((C, 3, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        gres = nl.ndarray((C, 2, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gsp[0:nr, 0, :], src=Gr, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=gres[0:nr, 0, :], data1=Gr, data2=gsp[0:nr, 0, :], op=nl.subtract)
        nisa.tensor_copy(dst=gsp[0:nr, 1, :], src=gres[0:nr, 0, :], engine=nisa.engine.vector)
        nisa.tensor_tensor(
            dst=gres[0:nr, 1, :], data1=gres[0:nr, 0, :], data2=gsp[0:nr, 1, :], op=nl.subtract
        )
        nisa.tensor_copy(dst=gsp[0:nr, 2, :], src=gres[0:nr, 1, :], engine=nisa.engine.vector)
        for part in range(3):
            nisa.dma_copy(dst=g_scr[part, r0 : r0 + nr, :], src=gsp[0:nr, part, :])
        for qi in (1, 2, 3, 5, 6):
            tp = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[:, 0:nr], data=qrow[0:nr, qi, :], engine=nisa.engine.tensor)
            nisa.tensor_copy(
                dst=colq[:, qi, r0 : r0 + nr], src=tp[:, 0:nr], engine=nisa.engine.vector
            )
        # e^{G_last} per row broadcast over partitions: transpose of a column broadcast
        tpb = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=tpb[:, 0:nr], data=eG[0:nr, C - 1 : C].broadcast(1, C), engine=nisa.engine.tensor
        )
        nisa.tensor_copy(dst=glb[:, r0 : r0 + nr], src=tpb[:, 0:nr], engine=nisa.engine.vector)

    def row(li_, c_):
        return li_ * n + c_

    # ---------------- state init
    S32 = []
    Sb = []
    for g0, gn in groups:
        s32 = nl.ndarray((D, U, D), dtype=nl.float32, buffer=nl.sbuf)
        sb = nl.ndarray((D, U, D), dtype=cdt, buffer=nl.sbuf)
        if s0 is not None:
            nisa.dma_copy(
                dst=s32[:, 0:gn, :],
                src=s0_f.ap(pattern=[[D, D], [D * D, gn], [1, D]], offset=(l_start + g0) * D * D),
            )
            nisa.tensor_copy(dst=sb[:, 0:gn, :], src=s32[:, 0:gn, :], engine=nisa.engine.scalar)
        else:
            nisa.memset(dst=s32, value=0.0)
            nisa.memset(dst=sb, value=0.0)
        S32.append(s32)
        Sb.append(sb)
    has_state = s0 is not None

    # ---------------- phase A: state-independent per (lane group, chunk)
    def phase_a(gi, c, res):
        g0, gn = groups[gi]
        lanes = range(gn)
        # loads
        qt = nl.ndarray((D, U, C), dtype=cdt, buffer=nl.sbuf)
        kt = nl.ndarray((D, U, C), dtype=cdt, buffer=nl.sbuf)
        if qt_in is not None:
            fpat = [[T, D], [D * T, gn], [1, C]]
            fbase = (l_start + g0) * D * T + c * C
            nisa.dma_copy(dst=qt[:, 0:gn, :], src=qt_in.ap(pattern=fpat, offset=fbase))
            nisa.dma_copy(dst=kt[:, 0:gn, :], src=kt_in.ap(pattern=fpat, offset=fbase))
        else:
            for j in lanes:
                nisa.dma_transpose(dst=qt[:, j, :], src=q_f[l_start + g0 + j, c, :, :])
                nisa.dma_transpose(dst=kt[:, j, :], src=k_f[l_start + g0 + j, c, :, :])
        ktok = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        vtok = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        base = (l_start + g0) * T * D + c * C * D
        pat = [[D, C], [T * D, gn], [1, D]]
        nisa.dma_copy(dst=ktok[:, 0:gn, :], src=k_f.ap(pattern=pat, offset=base))
        nisa.dma_copy(dst=vtok[:, 0:gn, :], src=v_f.ap(pattern=pat, offset=base))
        grow0 = nl.ndarray((3, U, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=grow0[0:3, 0:gn, :],
            src=g_scr.ap(
                pattern=[[rows_total * C, 3], [n * C, gn], [1, C]], offset=(g0 * n + c) * C
            ),
        )

        rows = [row(g0 + j, c) for j in lanes]

        def colv(qi, j):
            r = rows[j]
            return colq[:, qi, r : r + 1]

        # decay exponent: E[p, f] = G_f + MASK[p, f]  (fp32 PSUM)
        pe = nl.ndarray((C, U, C), dtype=nl.float32, buffer=nl.psum)
        # K=3 bf16 matmul: ones^T @ [G_hi; G_mid; G_lo] = G_f broadcast to all partitions (one instruction)
        nisa.nc_matmul(dst=pe[:, 0:gn, :], stationary=ones3, moving=grow0[0:3, 0:gn, :])
        EGb = nl.ndarray((D, U, C), dtype=cdt, buffer=nl.sbuf)  # scale e^{G_r}, all partitions
        nisa.activation(dst=EGb[:, 0:gn, :], op=nl.exp, data=pe[:, 0:gn, :], bias=ln_scale)
        # + MASK (identity stationary shared by the lanes -> one instruction), read by the exps below
        nisa.nc_matmul(
            dst=pe[:, 0:gn, :], stationary=eye_c, moving=mask_c.expand_dim(1).broadcast(1, gn)
        )
        Mb = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)  # beta_r exp(G_r - G_s), s <= r
        Ms = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)  # scale exp(G_r - G_s), s <= r
        for j in lanes:
            nisa.activation(
                dst=Mb[:, j, :], op=nl.exp, data=pe[:, j, :], bias=colv(5, j), scale=-1.0
            )
            nisa.activation(
                dst=Ms[:, j, :], op=nl.exp, data=pe[:, j, :], bias=colv(6, j), scale=-1.0
            )
        # Gram (k k^T) and QK^T
        pg = nl.ndarray((C, U, C), dtype=nl.float32, buffer=nl.psum)
        pq = nl.ndarray((C, U, C), dtype=nl.float32, buffer=nl.psum)
        for j in lanes:
            nisa.nc_matmul(dst=pg[:, j, :], stationary=kt[:, j, :], moving=kt[:, j, :])
            nisa.nc_matmul(dst=pq[:, j, :], stationary=qt[:, j, :], moving=kt[:, j, :])
        A = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
        W = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=A[:, 0:gn, :],
            data1=pg[:, 0:gn, :],
            data2=Mb[:, 0:gn, :],
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        nisa.tensor_tensor(
            dst=W[:, 0:gn, :],
            data1=pq[:, 0:gn, :],
            data2=Ms[:, 0:gn, :],
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        # W^T via PE transposes; (scale e^G q)^T = q^T * EGb
        ptw = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.psum)
        for j in lanes:
            nisa.nc_transpose(dst=ptw[:, j, :], data=W[:, j, :], engine=nisa.engine.tensor)
        Wt = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
        qGt = nl.ndarray((D, U, C), dtype=cdt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Wt[:, 0:gn, :], src=ptw[:, 0:gn, :], engine=nisa.engine.vector)
        nisa.tensor_tensor(
            dst=qGt[:, 0:gn, :],
            data1=qt[:, 0:gn, :],
            data2=EGb[:, 0:gn, :],
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        # token-major scaled operands
        kd = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        kbg = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        vb = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        for j in lanes:
            nisa.tensor_scalar(
                dst=kd[:, j, :],
                data=ktok[:, j, :],
                op0=nl.multiply,
                operand0=colv(2, j),
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=kbg[:, j, :],
                data=ktok[:, j, :],
                op0=nl.multiply,
                operand0=colv(3, j),
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=vb[:, j, :],
                data=vtok[:, j, :],
                op0=nl.multiply,
                operand0=colv(1, j),
                engine=nisa.engine.vector,
            )

        yield
        # ---- Schur inverse: InvT = (I + A)^-T, accumulated in fp32 PSUM
        invT = None
        inv = None
        pinv = nl.ndarray((C, U, C), dtype=nl.float32, buffer=nl.psum)
        for lvl in range(N_LEVELS):
            pz = nl.ndarray((C, U, C), dtype=nl.float32, buffer=nl.psum)
            for j in lanes:
                nisa.nc_matmul(
                    dst=pz[:, j, :],
                    stationary=A[:, j, :],
                    moving=(eye_c if lvl == 0 else invT[:, j, :]),
                )
            mid = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=mid[:, 0:gn, :],
                data1=pz[:, 0:gn, :],
                data2=negoff[:, lvl : lvl + 1, :].broadcast(1, gn),
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            if lvl > 0:
                ptt = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.psum)
                for j in lanes:
                    nisa.nc_transpose(
                        dst=ptt[:, j, :], data=invT[:, j, :], engine=nisa.engine.tensor
                    )
                inv = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=inv[:, 0:gn, :], src=ptt[:, 0:gn, :], engine=nisa.engine.vector
                )
            if lvl == 0:
                nisa.nc_matmul(
                    dst=pinv[:, 0:gn, :],
                    stationary=eye_c,
                    moving=eye_c.expand_dim(1).broadcast(1, gn),
                )
                nisa.nc_matmul(dst=pinv[:, 0:gn, :], stationary=eye_c, moving=mid[:, 0:gn, :])
            else:
                for j in lanes:
                    nisa.nc_matmul(dst=pinv[:, j, :], stationary=inv[:, j, :], moving=mid[:, j, :])
            invT = nl.ndarray((C, U, C), dtype=cdt, buffer=nl.sbuf)
            nisa.activation(dst=invT[:, 0:gn, :], op=nl.copy, data=pinv[:, 0:gn, :])
            yield
        # ---- u = T vb, w^T = (T kbg)^T
        pu = nl.ndarray((C, U, D), dtype=nl.float32, buffer=nl.psum)
        pw = nl.ndarray((D, U, C), dtype=nl.float32, buffer=nl.psum)
        for j in lanes:
            nisa.nc_matmul(dst=pu[:, j, :], stationary=invT[:, j, :], moving=vb[:, j, :])
            nisa.nc_matmul(dst=pw[:, j, :], stationary=kbg[:, j, :], moving=invT[:, j, :])
        u = nl.ndarray((C, U, D), dtype=nl.float32, buffer=nl.sbuf)
        wT = nl.ndarray((D, U, C), dtype=cdt, buffer=nl.sbuf)
        nisa.activation(dst=u[:, 0:gn, :], op=nl.copy, data=pu[:, 0:gn, :])
        nisa.activation(dst=wT[:, 0:gn, :], op=nl.copy, data=pw[:, 0:gn, :])
        res.update(u=u, wT=wT, Wt=Wt, qGt=qGt, kd=kd)

    # ---------------- phase B: recurrence for one (lane group, chunk)
    def phase_b(gi, c, a):
        g0, gn = groups[gi]
        lanes = range(gn)
        s32, sb = S32[gi], Sb[gi]
        first = (c == 0) and not has_state
        delta = nl.ndarray((C, U, D), dtype=cdt, buffer=nl.sbuf)
        if first:
            nisa.tensor_copy(
                dst=delta[:, 0:gn, :], src=a["u"][:, 0:gn, :], engine=nisa.engine.vector
            )
        else:
            p1 = nl.ndarray((C, U, D), dtype=nl.float32, buffer=nl.psum)
            for j in lanes:
                nisa.nc_matmul(dst=p1[:, j, :], stationary=a["wT"][:, j, :], moving=sb[:, j, :])
            nisa.tensor_tensor(
                dst=delta[:, 0:gn, :],
                data1=a["u"][:, 0:gn, :],
                data2=p1[:, 0:gn, :],
                op=nl.subtract,
                engine=nisa.engine.vector,
            )
        p2 = nl.ndarray((C, U, D), dtype=nl.float32, buffer=nl.psum)
        for j in lanes:
            if not first:
                nisa.nc_matmul(dst=p2[:, j, :], stationary=a["qGt"][:, j, :], moving=sb[:, j, :])
            nisa.nc_matmul(dst=p2[:, j, :], stationary=a["Wt"][:, j, :], moving=delta[:, j, :])
        p3 = nl.ndarray((D, U, D), dtype=nl.float32, buffer=nl.psum)
        for j in lanes:
            nisa.nc_matmul(dst=p3[:, j, :], stationary=a["kd"][:, j, :], moving=delta[:, j, :])
        ot = nl.ndarray((C, U, D), dtype=odt, buffer=nl.sbuf)
        nisa.activation(dst=ot[:, 0:gn, :], op=nl.copy, data=p2[:, 0:gn, :])
        base = (l_start + g0) * T * D + c * C * D
        nisa.dma_copy(
            dst=o_f.ap(pattern=[[D, C], [T * D, gn], [1, D]], offset=base), src=ot[:, 0:gn, :]
        )
        for j in lanes:
            r = row(g0 + j, c)
            nisa.scalar_tensor_tensor(
                dst=s32[:, j, :],
                data=s32[:, j, :],
                op0=nl.multiply,
                operand0=glb[:, r : r + 1],
                op1=nl.add,
                operand1=p3[:, j, :],
            )
        nisa.activation(dst=sb[:, 0:gn, :], op=nl.copy, data=s32[:, 0:gn, :])

    # ---------------- main schedule
    # Phase-A jobs (lane group, chunk) in chunk-major order are cut into batches of `jobs_per_batch`; a batch
    # is emitted LEVEL-MAJOR (stage k of every job, then stage k+1) and the previous batch's phase-B (state)
    # steps are interleaved between the stages, so every engine always has independent work queued.
    order = [(gi, c) for c in range(n) for gi in range(len(groups))]  # chunk-major
    jpb = max(1, min(jobs_per_batch, 4))  # >4 jobs in flight over-subscribes PSUM (8 banks)
    batches = [order[i : i + jpb] for i in range(0, len(order), jpb)]
    pending = []
    for batch in batches:
        results = []
        gens = []
        for gi, c in batch:
            res = {}
            results.append((gi, c, res))
            gens.append(phase_a(gi, c, res))
        n_slots = N_LEVELS + 2
        per_slot = (len(pending) + n_slots - 1) // n_slots if pending else 0
        active = list(gens)
        while active:
            for gen in list(active):
                try:
                    next(gen)
                except StopIteration:
                    active.remove(gen)
            for _ in range(per_slot):
                if pending:
                    gi, c, res = pending.pop(0)
                    phase_b(gi, c, res)
        while pending:
            gi, c, res = pending.pop(0)
            phase_b(gi, c, res)
        pending = results
    for gi, c, res in pending:
        phase_b(gi, c, res)

    for gi, (g0, gn) in enumerate(groups):
        nisa.dma_copy(
            dst=s_out_f.ap(pattern=[[D, D], [D * D, gn], [1, D]], offset=(l_start + g0) * D * D),
            src=S32[gi][:, 0:gn, :],
        )
    return out, s_out


# ----------------------------------------------------------------------------- torch wrapper
@functools.cache
def _consts_cpu():
    import torch

    return torch.from_numpy(build_consts())


@functools.cache
def _wrapped():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(gdn_fast_kernel)


def gdn_fast_consts(device):
    """Constant pack ([N_CONST,128,128] fp32) the kernel needs; create once per device OUTSIDE torch.compile
    (e.g. as a module buffer) and pass it as `consts`."""
    return _consts_cpu().to(device)


def gdn_fast(
    q,
    k,
    v,
    g,
    beta,
    consts,
    initial_state=None,
    scale=None,
    compute_dtype=None,
    out_dtype=None,
    return_state=False,
    qt=None,
    kt=None,
    **knobs,
):
    """Chunked GDN prefill on Neuron (callable inside torch.compile). See module docstring for layouts.

    q, k, v: [B, H, T, 128] (q, k L2-normalised); g, beta: [B, H, T] fp32; T % 128 == 0.
    consts: gdn_fast_consts(device).  initial_state: optional [B, H, 128, 128] fp32.
    compute_dtype: matmul operand dtype: torch.bfloat16 (default), torch.float16 (same speed, ~8x lower
        error), torch.float32 (~1.7x slower, fp32-reference accurate). Inputs are cast to it.
    out_dtype: output dtype (default compute dtype; torch.float32 avoids the final rounding).
    qt, kt: optional feature-major [B, H, 128, T] q / k (q may then be None).
    Returns out [B, H, T, 128] [, S_final [B, H, 128, 128] fp32 if return_state].
    """
    import torch

    cdt = compute_dtype or torch.bfloat16
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    odt = out_dtype or cdt
    nki_odt = {torch.bfloat16: nl.bfloat16, torch.float16: nl.float16, torch.float32: nl.float32}[
        odt
    ]
    q_arg = None if (q is None or qt is not None) else q.to(cdt).contiguous()
    args = (
        q_arg,
        k.to(cdt).contiguous(),
        v.to(cdt).contiguous(),
        g.float().contiguous(),
        beta.float().contiguous(),
        consts,
    )
    s0 = None if initial_state is None else initial_state.float().contiguous()
    if qt is not None:
        knobs = dict(knobs, qt_in=qt.to(cdt).contiguous(), kt_in=kt.to(cdt).contiguous())
    out, st = _wrapped()(*args, s0=s0, scale=float(scale), out_dtype=nki_odt, **knobs)
    return (out, st) if return_state else out
