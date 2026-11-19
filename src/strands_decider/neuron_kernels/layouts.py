"""The weight layouts of the fused Neuron kernels, and their torch references.

fused_ops.py's kernels read the decider's weights in layouts of their own: GEMM weights
pre-transposed to [K, N], the Gated DeltaNet in-projection feature-major, the depthwise
convolution's taps as diagonal matrices. The `prepare_*` functions build them once from a
`neuron_model.relayout` state dict, on the CPU. This module needs only torch, so the layouts and
the references here are tested without Trainium (tests/test_neuron_kernels.py).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

P = 128  # partitions, and the Gated DeltaNet head dim
K_CONV = 4  # causal conv taps
L2_EPS = 1e-6  # neuron_model.l2norm's eps


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def conv_diag_from_conv_w(conv_w: torch.Tensor) -> torch.Tensor:
    """conv_w [3H, 128, 4] -> [3H, 128, 4, 128] with out[h, c, j, c'] = conv_w[h, c, j] * (c == c')."""
    D = conv_w.shape[1]
    eye = torch.eye(D, dtype=conv_w.dtype, device=conv_w.device)
    return (conv_w[:, :, :, None] * eye[None, :, None, :]).contiguous()


def prepare_gdn_layer(
    sd: dict[str, torch.Tensor], i: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """One-time weight re-layout for GDN layer i of a neuron_model.load_state() state dict, or of a
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
    w_qkvz = sd[p + "w_qkvz"].to(dtype)  # [4H, 128, 2048]
    H = w_qkvz.shape[0] // 4
    C = w_qkvz.shape[2]
    w_qkv_t = w_qkvz[: 3 * H].permute(2, 0, 1).contiguous()  # [2048, 3H, 128]
    w_z = w_qkvz[3 * H :].reshape(H * w_qkvz.shape[1], C)  # [(h d), c]
    w_zba_t = torch.cat([w_z, sd[p + "w_ba"].to(dtype)], 0).t().contiguous()  # [2048, 128H + 2H]
    w_out = sd[p + "w_out"].to(dtype)  # [2048, H, 128]
    return {
        "ln1": sd[p + "ln1"].float().contiguous(),
        "w_qkv_t": w_qkv_t,
        "w_zba_t": w_zba_t,
        "conv_diag": conv_diag_from_conv_w(sd[p + "conv_w"].to(dtype)),
        "gdn_ab": torch.stack([sd[p + "A"].float(), sd[p + "dt_bias"].float()], 1).contiguous(),
        "gnorm": sd[p + "gnorm"].to(dtype).contiguous(),
        "w_out_t": w_out.reshape(w_out.shape[0], -1).t().contiguous(),
    }


def prepare_mlp_layer(
    sd: dict[str, torch.Tensor], i: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
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


def prepare_attn_layer(
    sd: dict[str, torch.Tensor], i: int, dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """Pre-transposed attention weights for attn_layer_fast (works on shard_state() shards too):
    ln1 [2048] fp32; w_qgkv_t [2048, nq+nq+2*nkv, 256] bf16 (K-major head-major einsum);
    w_o_t [nq*256, 2048] bf16 (K-major, rows = (h, d)); q_norm, k_norm [256] fp32."""
    p = f"layers.{i}."
    w_o = sd[p + "w_o"].to(dtype)  # [2048, nq, 256]
    return {
        "ln1": sd[p + "ln1"].float().contiguous(),
        "w_qgkv_t": sd[p + "w_qgkv"].to(dtype).permute(2, 0, 1).contiguous(),
        "w_o_t": w_o.reshape(w_o.shape[0], -1).t().contiguous(),
        "q_norm": sd[p + "q_norm"].float().contiguous(),
        "k_norm": sd[p + "k_norm"].float().contiguous(),
    }


def conv_tail_fm(proj_fm: torch.Tensor, s_len: torch.Tensor, K: int = K_CONV) -> torch.Tensor:
    """Last K-1 pre-conv tokens before position s_len, from the feature-major projection:
    proj_fm [1, 3H, 128, S], s_len [1] int -> [1, 3H, K-1, 128] (zeros before position 0).
    Same values/layout as NeuronDeciderModel.conv_tail(qkvz[:, :3H], s_len, K)."""
    idx = s_len.reshape(1).long() - (K - 1) + torch.arange(K - 1, device=proj_fm.device)
    ok = (idx >= 0).to(proj_fm.dtype)[None, None, None, :]
    t = torch.index_select(proj_fm, 3, idx.clamp_min(0)) * ok  # [1, 3H, 128, K-1]
    return t.transpose(2, 3).contiguous()


def mlp_fast(
    x: torch.Tensor,
    prep: dict[str, torch.Tensor],
    eps: float = 1e-6,
    inter: int | None = None,
    reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Drop-in for NeuronDeciderModel.mlp: x + reduce((silu(g) * u) @ W_down) with pre-transposed weights.

    x [B, T, 2048] bf16 -> [B, T, 2048] bf16.  inter defaults to the (per-rank) weight shape.  RMSNorm
    stays in torch: in front of a token-major GEMM neuronx-cc fuses it well.
    """
    inter = prep["w_gate_up_t"].shape[1] // 2 if inter is None else inter
    h = _rms_norm(x, prep["ln2"], eps)
    gu = h @ prep["w_gate_up_t"]
    y = (F.silu(gu[..., :inter]) * gu[..., inter:]) @ prep["w_down_t"]
    return x + (reduce(y) if reduce is not None else y)


def gdn_prologue_ref(
    proj_hm: torch.Tensor,
    conv_w: torch.Tensor,
    ba: torch.Tensor,
    A: torch.Tensor,
    dt_bias: torch.Tensor,
    eps: float = L2_EPS,
    conv_hist: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact neuron_model.gdn_layer(_suffix) math for the prologue (torch, any device).

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
    out = xf[:, :, Kc - 1 : Kc - 1 + T] * wf[None, :, None, :, Kc - 1]
    for kk in range(Kc - 1):
        out = out + xf[:, :, kk : kk + T] * wf[None, :, None, :, kk]
    conv = F.silu(out)

    def l2(t: torch.Tensor) -> torch.Tensor:
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + eps)

    q = l2(conv[:, :H]).to(torch.bfloat16)
    k = l2(conv[:, H : 2 * H]).to(torch.bfloat16)
    v = conv[:, 2 * H : 3 * H].to(torch.bfloat16)
    baf = ba.float().transpose(1, 2)
    beta = torch.sigmoid(baf[:, :H])
    g = A.float()[None, :, None] * F.softplus(baf[:, H : 2 * H] + dt_bias.float()[None, :, None])
    return q, k, v, g, beta


def prepare_layers(
    layer_types: list[str], sd: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> list[dict[str, torch.Tensor]]:
    """Every layer's weights in the fused kernels' layouts: the MLP's, and a Gated DeltaNet
    layer's own or an attention layer's own, the latter under an "attn_" prefix."""
    out = []
    for i, lt in enumerate(layer_types):
        prep = prepare_mlp_layer(sd, i, dtype)
        if lt == "linear_attention":
            prep.update(prepare_gdn_layer(sd, i, dtype))
        else:
            prep.update({"attn_" + k: v for k, v in prepare_attn_layer(sd, i, dtype).items()})
        out.append(prep)
    return out
