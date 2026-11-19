"""The decider's Qwen3.5 torso and pointer readout as one static-shape module, for `--device neuron`.

AWS Trainium runs a model as graphs compiled ahead of time for fixed shapes
(`torch.compile(backend="neuron_native_lite")`, neuron_engine.py). transformers' Qwen3.5 does not
compile that way: its Gated DeltaNet layers branch on the cache, its LoRA adapter adds a second
matmul to every projection, and the readout runs on the host. `NeuronDeciderModel` computes the
same function as `StrandsDeciderModel` (transformers' Qwen3_5TextModel, the PEFT adapter and the
PointerHead) with every shape fixed by its inputs:

  * The LoRA adapter is merged into the base weights at load, in fp32, then rounded once to the
    compute dtype (`from_decider_model`).
  * The zero-centred RMSNorm weights are folded to (1 + w).
  * Each Gated DeltaNet layer's q/k/v/z projections are one head-major weight, so one einsum
    gives [B, heads, T, head_dim] with no transpose; the attention layer's q/gate/k/v likewise.
  * Batches are right-padded. Causal attention and the causal recurrence keep padding invisible
    to every real position, so no padding mask is needed.
  * The readout runs inside the graph in fp32: gather `<answer>` and the option positions,
    LayerNorm, the q/k projections, temperature and the masked softmax. A forward returns
    probabilities [B, K].
  * A request's state can be encoded once (`prefix_forward`, batch 1, padded to a bucket) and
    every question continued from its caches (`suffix_forward`): per Gated DeltaNet layer the
    final recurrent state and the last K-1 pre-convolution tokens, per attention layer K and V.

The two token mixers are pluggable (`gdn_impl`, `attn_impl`). The defaults here are the torch
references, which CPU tests run; the engine swaps in NKI kernels on Trainium.

Only the 2B torso's layout is implemented: Gated DeltaNet with as many key heads as value heads
and equal key and value head dims, default RoPE, no attention bias. Other layouts are refused at
load.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from .modeling import StrandsDeciderModel, refuse_token_readout

GdnImpl = Callable[..., Any]
AttnImpl = Callable[..., torch.Tensor]

# The masked softmax's fill. -inf would give NaN for a row with no valid option, which a padded
# batch row is; exp(-1e4) is 0 in fp32.
_MASKED = -1e4


@dataclass
class TorsoShape:
    """The dimensions `NeuronDeciderModel` needs, read from the torso's Qwen3_5TextConfig."""

    hidden: int
    layer_types: list[str]
    inter: int
    eps: float
    vocab: int
    # attention layers
    nq: int
    nkv: int
    hd: int
    rope_dim: int
    rope_theta: float
    # Gated DeltaNet layers: `gh` heads of `gd` dims for each of q, k, v and z
    gh: int
    gd: int
    conv_k: int
    # readout
    pointer_dim: int
    temperature: float = 1.0
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    max_length: int = 4096

    @classmethod
    def from_text_config(cls, cfg: Any, pointer_dim: int) -> TorsoShape:
        """Read and check a transformers Qwen3_5TextConfig."""
        rope = dict(getattr(cfg, "rope_parameters", None) or {})
        problems = []
        if cfg.linear_num_key_heads != cfg.linear_num_value_heads:
            problems.append(
                f"{cfg.linear_num_key_heads} Gated DeltaNet key heads and "
                f"{cfg.linear_num_value_heads} value heads"
            )
        if cfg.linear_key_head_dim != cfg.linear_value_head_dim:
            problems.append(
                f"Gated DeltaNet key dim {cfg.linear_key_head_dim} and value dim "
                f"{cfg.linear_value_head_dim}"
            )
        if rope.get("rope_type", "default") != "default":
            problems.append(f"rope_type {rope.get('rope_type')!r}")
        if getattr(cfg, "attention_bias", False):
            problems.append("attention bias")
        if getattr(cfg, "hidden_act", "silu") != "silu":
            problems.append(f"hidden_act {cfg.hidden_act!r}")
        if problems:
            raise ValueError(
                "the neuron backend implements the Qwen3.5-2B torso layout; this torso has "
                + ", ".join(problems)
            )
        hd = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
        return cls(
            hidden=cfg.hidden_size,
            layer_types=list(cfg.layer_types),
            inter=cfg.intermediate_size,
            eps=cfg.rms_norm_eps,
            vocab=cfg.vocab_size,
            nq=cfg.num_attention_heads,
            nkv=cfg.num_key_value_heads,
            hd=hd,
            rope_dim=int(hd * rope.get("partial_rotary_factor", 1.0)),
            rope_theta=float(rope["rope_theta"]),
            gh=cfg.linear_num_value_heads,
            gd=cfg.linear_value_head_dim,
            conv_k=cfg.linear_conv_kernel_dim,
            pointer_dim=pointer_dim,
        )


# ------------------------------------------------------------------ weights


def _merged_torso(model: StrandsDeciderModel) -> nn.Module:
    """The torso in fp32 with its adapter merged into the base weights.

    Upcasting first makes the merge W + (alpha / r) B A run in fp32 on bf16-rounded base
    weights, which is what the unmerged torch path computes. This modifies `model`.
    """
    torso: nn.Module = model.torso.float()
    merge = getattr(torso, "merge_and_unload", None)  # a PeftModel
    return cast(nn.Module, merge()) if callable(merge) else torso


def from_decider_model(
    model: StrandsDeciderModel, dtype: torch.dtype = torch.bfloat16
) -> tuple[TorsoShape, dict[str, torch.Tensor]]:
    """The shape and the weights of `NeuronDeciderModel` for a loaded checkpoint.

    Consumes `model`: its torso is upcast and its adapter merged in place.
    """
    refuse_token_readout(model.config)
    if model.config.expert_lora:
        raise ValueError("the neuron backend has no mixture-of-experts layers (expert_lora)")
    torso = _merged_torso(model)
    shape = TorsoShape.from_text_config(torso.config, model.config.pointer_dim)
    shape.temperature = float(model.config.temperature)
    shape.temperature_by_kind = dict(model.config.temperature_by_kind or {})
    shape.max_length = model.config.max_length
    raw = {k: v.detach() for k, v in torso.state_dict().items()}
    sd = relayout(shape, raw, dtype)
    head = model.head.state_dict()
    sd.update(
        {
            "head.ln_w": head["norm.weight"].float(),
            "head.ln_b": head["norm.bias"].float(),
            "head.wq": head["q.weight"].float(),
            "head.bq": head["q.bias"].float(),
            "head.wk": head["k.weight"].float(),
            "head.bk": head["k.bias"].float(),
        }
    )
    return shape, sd


def relayout(
    c: TorsoShape, raw: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """transformers' Qwen3.5 weight names and layouts -> this module's (see the module docstring)."""
    H, gh, gd, hd, nq, nkv = c.hidden, c.gh, c.gd, c.hd, c.nq, c.nkv
    sd: dict[str, torch.Tensor] = {
        "embed": raw["embed_tokens.weight"].to(dtype),
        "final_norm": 1.0 + raw["norm.weight"].float(),
    }
    for i, lt in enumerate(c.layer_types):
        p = f"layers.{i}."
        sd[p + "ln1"] = 1.0 + raw[p + "input_layernorm.weight"].float()
        sd[p + "ln2"] = 1.0 + raw[p + "post_attention_layernorm.weight"].float()
        sd[p + "w_gate_up"] = torch.cat(
            [raw[p + "mlp.gate_proj.weight"], raw[p + "mlp.up_proj.weight"]], 0
        ).to(dtype)
        sd[p + "w_down"] = raw[p + "mlp.down_proj.weight"].to(dtype)
        if lt == "linear_attention":
            q = p + "linear_attn."
            # [4 * gh, gd, H]: q, k, v and z heads, so one einsum gives [B, 4 * gh, T, gd]
            qkv = raw[q + "in_proj_qkv.weight"].reshape(3 * gh, gd, H)
            z = raw[q + "in_proj_z.weight"].reshape(gh, gd, H)
            sd[p + "w_qkvz"] = torch.cat([qkv, z], 0).to(dtype)
            sd[p + "w_ba"] = torch.cat(
                [raw[q + "in_proj_b.weight"], raw[q + "in_proj_a.weight"]], 0
            ).to(dtype)
            sd[p + "conv_w"] = raw[q + "conv1d.weight"][:, 0, :].reshape(3 * gh, gd, -1).to(dtype)
            sd[p + "A"] = -raw[q + "A_log"].float().exp()  # g = A * softplus(a + dt_bias)
            sd[p + "dt_bias"] = raw[q + "dt_bias"].float()
            sd[p + "gnorm"] = raw[q + "norm.weight"].to(dtype)  # the gated norm: a plain weight
            sd[p + "w_out"] = raw[q + "out_proj.weight"].reshape(H, gh, gd).to(dtype)
        else:
            a = p + "self_attn."
            # [2 * nq + 2 * nkv, hd, H]: q, gate, k and v heads. q_proj's rows are
            # [q_h | gate_h] per head.
            qg = raw[a + "q_proj.weight"].reshape(nq, 2, hd, H)
            sd[p + "w_qgkv"] = torch.cat(
                [
                    qg[:, 0],
                    qg[:, 1],
                    raw[a + "k_proj.weight"].reshape(nkv, hd, H),
                    raw[a + "v_proj.weight"].reshape(nkv, hd, H),
                ],
                0,
            ).to(dtype)
            sd[p + "q_norm"] = 1.0 + raw[a + "q_norm.weight"].float()
            sd[p + "k_norm"] = 1.0 + raw[a + "k_norm.weight"].float()
            sd[p + "w_o"] = raw[a + "o_proj.weight"].reshape(H, nq, hd).to(dtype)
    return sd


# ------------------------------------------------------------------ math


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5's RMSNorm with (1 + w) already folded into `w`, computed in fp32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def gated_rms_norm(
    x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """Qwen3_5RMSNormGated: w * norm(x), rounded to x's dtype, times silu(gate) in fp32."""
    dt = x.dtype
    xf = x.float()
    y = w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dt)
    return (y.float() * F.silu(gate.float())).to(dt)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def causal_conv1d_silu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [B, C, T, D], w [C, D, K]: a depthwise causal convolution along T from zero history,
    then SiLU, in fp32."""
    K = w.shape[-1]
    T = x.shape[2]
    xf = F.pad(x.float(), (0, 0, K - 1, 0))
    wf = w.float()
    out = xf[:, :, K - 1 : K - 1 + T] * wf[None, :, None, :, K - 1]
    for k in range(K - 1):
        out = out + xf[:, :, k : k + T] * wf[None, :, None, :, k]
    return F.silu(out)


def rope_cos_sin(
    T: int,
    dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
    offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin [T, dim] for positions 0..T-1, or offset..offset+T-1 with `offset` [1]."""
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    if offset is not None:
        pos = pos + offset.reshape(1).float()
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first cos.shape[-1] dims of x [B, H, T, D] (the rotate_half convention).

    Qwen3.5's multimodal RoPE gives text the same position on all three axes, where it is
    this one-axis RoPE.
    """
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


def gdn_chunk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk: int = 64,
    S0: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """The chunked gated delta rule in fp32, with transformers' torch_chunk_gated_delta_rule's
    semantics, in the WY form: per chunk, u = T (beta v) and w = T (beta e^G k) with
    T = (I + A)^-1, then the state recurrence.

    q, k [B, H, T, D] (L2-normalised), v [B, H, T, Dv], g and beta [B, H, T]. S0 [B, H, D, Dv]
    is the initial state (zero by default). Returns the output [B, H, T, Dv] in fp32, and with
    `return_state` the final state too.
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    q = q.float() * Dk**-0.5
    k, v, g, beta = k.float(), v.float(), g.float(), beta.float()
    pad = (chunk - T % chunk) % chunk
    if pad:  # g = beta = 0 on the padding: it leaves the state unchanged
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    n = (T + pad) // chunk
    vb = v * beta[..., None]
    kb = k * beta[..., None]
    q, k, kb, vb = (x.reshape(B, H, n, chunk, -1) for x in (q, k, kb, vb))
    g = g.reshape(B, H, n, chunk).cumsum(-1)
    upper = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    dec = (g[..., :, None] - g[..., None, :]).masked_fill(upper, float("-inf")).exp()
    A = (kb @ k.transpose(-1, -2)) * dec
    attn = (q @ k.transpose(-1, -2)) * dec
    strictly_lower = torch.ones_like(upper).tril(-1)
    Lm = A.masked_fill(~strictly_lower, 0.0) + torch.eye(chunk, device=q.device)
    u = torch.linalg.solve_triangular(Lm, vb, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(Lm, kb * g.exp()[..., None], upper=False, unitriangular=True)
    S = torch.zeros(B, H, Dk, Dv, device=q.device) if S0 is None else S0.float().clone()
    qd = q * g.exp()[..., None]
    kd = k * (g[..., -1:] - g).exp()[..., None]
    cd = g[..., -1].exp()[..., None, None]
    out = torch.zeros(B, H, n, chunk, Dv, device=q.device)
    for i in range(n):
        vn = u[:, :, i] - w[:, :, i] @ S
        out[:, :, i] = qd[:, :, i] @ S + attn[:, :, i] @ vn
        S = S * cd[:, :, i] + kd[:, :, i].transpose(-1, -2) @ vn
    o = out.reshape(B, H, n * chunk, Dv)[:, :, :T]
    return (o, S) if return_state else o


def attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    k_prior: torch.Tensor | None = None,
    v_prior: torch.Tensor | None = None,
    prior_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal grouped-query attention, q [B, Hq, T, D] and k, v [B, Hkv, T, D], softmax in fp32.

    With a cached prefix, k_prior and v_prior [B, Hkv, S, D] come first, of which the first
    `prior_len` ([1]) positions are real and the rest padding.
    """
    rep = q.shape[1] // k.shape[1]
    T = q.shape[2]
    masked = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    if k_prior is not None:
        assert v_prior is not None and prior_len is not None
        S = k_prior.shape[2]
        k = torch.cat([k_prior, k], 2)
        v = torch.cat([v_prior, v], 2)
        pad = (torch.arange(S, device=q.device) >= prior_len.reshape(1))[None, :].expand(T, S)
        masked = torch.cat([pad, masked], 1)
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    p = torch.softmax(s.masked_fill(masked, float("-inf")), -1).to(q.dtype)
    return p @ v


# ------------------------------------------------------------------ the model

# Weights kept in fp32 whatever the compute dtype: the norms, the decay parameters, the readout.
_FP32 = ("A", "dt_bias", "ln1", "ln2", "q_norm", "k_norm")


class NeuronDeciderModel(nn.Module):
    """The decider with static shapes; see the module docstring.

    Inputs of `forward` (all int32 unless noted): input_ids [B, T], right-padded; ans_idx [B],
    the `<answer>` position; opt_idx [B, K], the option positions, padded with any value;
    n_slots [B], each row's option count; inv_temp [B] fp32, 1 / temperature.
    """

    def __init__(
        self,
        shape: TorsoShape,
        sd: dict[str, torch.Tensor],
        *,
        compute_dtype: torch.dtype = torch.bfloat16,
        gdn_impl: GdnImpl | None = None,
        attn_impl: AttnImpl | None = None,
    ) -> None:
        super().__init__()
        self.cfg = shape
        self.dt = compute_dtype
        self.gdn_impl: GdnImpl = gdn_impl or gdn_chunk_ref
        self.attn_impl: AttnImpl = attn_impl or attn_ref
        for k, v in sd.items():
            fp32 = k.startswith("head.") or k == "final_norm" or k.rsplit(".", 1)[-1] in _FP32
            self.register_buffer(
                k.replace(".", "__"), v.float() if fp32 else v.to(compute_dtype), persistent=False
            )

    def w(self, name: str) -> torch.Tensor:
        t: torch.Tensor = getattr(self, name.replace(".", "__"))
        return t

    def _reduce(self, y: torch.Tensor) -> torch.Tensor:
        """A layer's output projection, summed over tensor-parallel ranks (none here)."""
        return y

    # --- layers

    def _gdn_proj(self, i: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c, p = self.cfg, f"layers.{i}."
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        qkvz = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qkvz"))  # [B, 4 * gh, T, gd]
        ba = (h @ self.w(p + "w_ba").t()).float().transpose(1, 2)  # [B, 2 * gh, T]
        beta = torch.sigmoid(ba[:, : c.gh])
        g = self.w(p + "A")[None, :, None] * F.softplus(
            ba[:, c.gh :] + self.w(p + "dt_bias")[None, :, None]
        )
        return qkvz, beta, g

    def _qkv(self, conv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gh, dt = self.cfg.gh, self.dt
        return (
            l2norm(conv[:, :gh]).to(dt),
            l2norm(conv[:, gh : 2 * gh]).to(dt),
            conv[:, 2 * gh : 3 * gh].to(dt),
        )

    def _gdn_out(self, i: int, x: torch.Tensor, o: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = f"layers.{i}."
        o = gated_rms_norm(o.to(self.dt), z, self.w(p + "gnorm"), self.cfg.eps)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(p + "w_out")))

    def gdn_layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        gh = self.cfg.gh
        qkvz, beta, g = self._gdn_proj(i, x)
        conv = causal_conv1d_silu(qkvz[:, : 3 * gh], self.w(f"layers.{i}.conv_w"))
        o = self.gdn_impl(*self._qkv(conv), g, beta)
        return self._gdn_out(i, x, o, qkvz[:, 3 * gh :])

    def _attn_qkv(
        self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c, p = self.cfg, f"layers.{i}."
        nq, nkv = c.nq, c.nkv
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        qgkv = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qgkv"))  # [B, 2nq + 2nkv, T, hd]
        q = apply_partial_rope(rms_norm(qgkv[:, :nq], self.w(p + "q_norm"), c.eps), cos, sin)
        gate = qgkv[:, nq : 2 * nq]
        k = rms_norm(qgkv[:, 2 * nq : 2 * nq + nkv], self.w(p + "k_norm"), c.eps)
        return q, gate, apply_partial_rope(k, cos, sin), qgkv[:, 2 * nq + nkv :]

    def _attn_out(
        self, i: int, x: torch.Tensor, o: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        o = o * torch.sigmoid(gate)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(f"layers.{i}.w_o")))

    def attn_layer(
        self, i: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q, gate, k, v = self._attn_qkv(i, x, cos, sin)
        return self._attn_out(i, x, self.attn_impl(q, k, v, self.cfg.hd**-0.5), gate)

    def mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p, inter = f"layers.{i}.", self.cfg.inter
        h = rms_norm(x, self.w(p + "ln2"), self.cfg.eps)
        gu = h @ self.w(p + "w_gate_up").t()
        return x + self._reduce(
            (F.silu(gu[..., :inter]) * gu[..., inter:]) @ self.w(p + "w_down").t()
        )

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.w("embed"))

    def torso(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [B, T] -> the final-norm hidden states [B, T, hidden]."""
        c = self.cfg
        x = self._embed(input_ids)
        cos, sin = rope_cos_sin(input_ids.shape[1], c.rope_dim, c.rope_theta, x.device, x.dtype)
        for i, lt in enumerate(c.layer_types):
            x = (
                self.gdn_layer(i, x)
                if lt == "linear_attention"
                else self.attn_layer(i, x, cos, sin)
            )
            x = self.mlp(i, x)
        return rms_norm(x, self.w("final_norm"), c.eps)

    def readout(
        self,
        hidden: torch.Tensor,
        ans_idx: torch.Tensor,
        opt_idx: torch.Tensor,
        n_slots: torch.Tensor,
        inv_temp: torch.Tensor,
    ) -> torch.Tensor:
        """The pointer head and the masked softmax: hidden [B, T, hidden] -> probabilities [B, K],
        zero past each row's n_slots."""
        B, _, Hd = hidden.shape
        K = opt_idx.shape[1]
        idx = torch.cat([ans_idx.long()[:, None], opt_idx.long().clamp(0, hidden.shape[1] - 1)], 1)
        g = torch.gather(hidden, 1, idx[..., None].expand(B, 1 + K, Hd)).float()
        g = F.layer_norm(g, (Hd,), self.w("head.ln_w"), self.w("head.ln_b"), 1e-5)
        qv = g[:, 0] @ self.w("head.wq").t() + self.w("head.bq")  # [B, P]
        kv = g[:, 1:] @ self.w("head.wk").t() + self.w("head.bk")  # [B, K, P]
        logits = (kv @ qv[..., None]).squeeze(-1) * self.cfg.pointer_dim**-0.5
        logits = logits * inv_temp.float()[:, None]
        valid = torch.arange(K, device=logits.device)[None, :] < n_slots[:, None]
        probs = torch.softmax(logits.masked_fill(~valid, _MASKED), -1)
        return probs.masked_fill(~valid, 0.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        ans_idx: torch.Tensor,
        opt_idx: torch.Tensor,
        n_slots: torch.Tensor,
        inv_temp: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout(self.torso(input_ids), ans_idx, opt_idx, n_slots, inv_temp)

    # --- shared prefix
    #
    # Pad positions of the prefix are no-ops for the recurrence (beta = g = 0) and are masked
    # out of attention (prior_len = s_len), so padding the prefix to a bucket is exact.

    @staticmethod
    def conv_tail(pre: torch.Tensor, s_len: torch.Tensor, K: int) -> torch.Tensor:
        """The K - 1 pre-convolution tokens before position s_len: pre [1, C, S, D] ->
        [1, C, K - 1, D], zero before position 0."""
        idx = s_len.reshape(1).long() - (K - 1) + torch.arange(K - 1, device=pre.device)
        ok = (idx >= 0).to(pre.dtype)[None, None, :, None]
        return torch.index_select(pre, 2, idx.clamp_min(0)) * ok

    def gdn_layer_prefix(
        self, i: int, x: torch.Tensor, valid: torch.Tensor, s_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gh = self.cfg.gh
        qkvz, beta, g = self._gdn_proj(i, x)
        tail = self.conv_tail(qkvz[:, : 3 * gh], s_len, self.cfg.conv_k)
        vm = valid[:, None, :].float()
        conv = causal_conv1d_silu(qkvz[:, : 3 * gh], self.w(f"layers.{i}.conv_w"))
        o, S = self.gdn_impl(*self._qkv(conv), g * vm, beta * vm, return_state=True)
        return self._gdn_out(i, x, o, qkvz[:, 3 * gh :]), S, tail

    def gdn_layer_suffix(
        self, i: int, x: torch.Tensor, S0: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        gh, K = self.cfg.gh, self.cfg.conv_k
        B = x.shape[0]
        qkvz, beta, g = self._gdn_proj(i, x)
        pre = torch.cat([tail.expand(B, -1, -1, -1).to(qkvz.dtype), qkvz[:, : 3 * gh]], 2)
        conv = causal_conv1d_silu(pre, self.w(f"layers.{i}.conv_w"))[:, :, K - 1 :]
        o = self.gdn_impl(*self._qkv(conv), g, beta, S0=S0.expand(B, -1, -1, -1).contiguous())
        return self._gdn_out(i, x, o, qkvz[:, 3 * gh :])

    def prefix_forward(
        self, input_ids: torch.Tensor, s_len: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """input_ids [1, S], right-padded, with s_len [1] real tokens -> the caches, two per
        layer in layer order: a Gated DeltaNet layer's final state [1, gh, gd, gd] (fp32) and
        conv tail [1, 3 gh, K - 1, gd]; an attention layer's K and V [1, nkv, S, hd]."""
        c = self.cfg
        x = self._embed(input_ids)
        S = input_ids.shape[1]
        valid = torch.arange(S, device=x.device)[None, :] < s_len.reshape(1, 1)
        cos, sin = rope_cos_sin(S, c.rope_dim, c.rope_theta, x.device, x.dtype)
        caches: list[torch.Tensor] = []
        for i, lt in enumerate(c.layer_types):
            if lt == "linear_attention":
                x, state, tail = self.gdn_layer_prefix(i, x, valid, s_len)
                caches += [state, tail]
            else:
                q, gate, k, v = self._attn_qkv(i, x, cos, sin)
                x = self._attn_out(i, x, self.attn_impl(q, k, v, c.hd**-0.5), gate)
                caches += [k, v]
            x = self.mlp(i, x)
        return tuple(caches)

    def suffix_forward(
        self,
        input_ids: torch.Tensor,
        s_len: torch.Tensor,
        caches: Sequence[torch.Tensor],
        ans_idx: torch.Tensor,
        opt_idx: torch.Tensor,
        n_slots: torch.Tensor,
        inv_temp: torch.Tensor,
    ) -> torch.Tensor:
        """The question suffixes [B, L], right-padded, continued from `prefix_forward`'s caches.
        ans_idx and opt_idx are positions within the suffix. Returns probabilities [B, K]."""
        c = self.cfg
        x = self._embed(input_ids)
        B = x.shape[0]
        cos, sin = rope_cos_sin(
            input_ids.shape[1], c.rope_dim, c.rope_theta, x.device, x.dtype, offset=s_len
        )
        for i, lt in enumerate(c.layer_types):
            a, b = caches[2 * i], caches[2 * i + 1]
            if lt == "linear_attention":
                x = self.gdn_layer_suffix(i, x, a, b)
            else:
                q, gate, k, v = self._attn_qkv(i, x, cos, sin)
                kp, vp = (t.expand(B, -1, -1, -1).contiguous() for t in (a, b))
                o = self.attn_impl(q, k, v, c.hd**-0.5, k_prior=kp, v_prior=vp, prior_len=s_len)
                x = self._attn_out(i, x, o, gate)
            x = self.mlp(i, x)
        hidden = rms_norm(x, self.w("final_norm"), c.eps)
        return self.readout(hidden, ans_idx, opt_idx, n_slots, inv_temp)
