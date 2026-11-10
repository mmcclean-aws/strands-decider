"""Strands Decider 2B as a static-shape, compile-friendly PyTorch module.

Same math as transformers' Qwen3_5TextModel (5.17) + strands_decider.PointerHead, rewritten
for an ahead-of-time compiled prefill graph:
  * LoRA (r=48, alpha=96) merged into the base weights at load: zero runtime cost.
  * Zero-centered RMSNorm weights folded to (1 + w).
  * GDN input projections fused into one [2048 -> 6144+2048+16+16] GEMM.
  * Right-padded batches [B, T]: causal attention and the causal GDN recurrence make padded
    tail tokens invisible to real positions, so no padding mask is needed.
  * The readout (gather <answer> + option positions, LayerNorm, q/k projections, scaled dot
    product, temperature, masked softmax) runs on device in fp32; the graph returns [B, K].
  * The two heavy token mixers are pluggable (`gdn_impl`, `attn_impl`) so NKI kernels can be
    swapped in on Neuron while CPU uses the torch reference.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeciderConfig:
    hidden: int = 2048
    n_layers: int = 24
    layer_types: list[str] = field(default_factory=lambda: (["linear_attention"] * 3 + ["full_attention"]) * 6)
    inter: int = 6144
    eps: float = 1e-6
    vocab: int = 248320
    # attention
    nq: int = 8
    nkv: int = 2
    hd: int = 256
    rope_dim: int = 64
    rope_theta: float = 1e7
    # GDN
    gk: int = 16
    gv: int = 16
    dk: int = 128
    dv: int = 128
    conv_k: int = 4
    # head
    pointer_dim: int = 768
    temperature_by_kind: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0
    max_length: int = 4096


# ------------------------------------------------------------------ weight loading
def _snapshot(repo: str, hf_home: str | None = None) -> str:
    hf_home = hf_home or os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    name = "models--" + repo.replace("/", "--")
    snaps = glob.glob(os.path.join(hf_home, "hub", name, "snapshots", "*"))
    if not snaps:
        raise FileNotFoundError(f"{repo} not in {hf_home}")
    return sorted(snaps)[-1]


def load_state(base_dir: str, decider_dir: str, dtype=torch.bfloat16) -> tuple[DeciderConfig, dict[str, torch.Tensor]]:
    """Merged state dict in this module's naming. LoRA merge done in fp32, then cast."""
    from safetensors import safe_open

    dcfg = json.load(open(os.path.join(decider_dir, "strands_decider_config.json")))
    bcfg = json.load(open(os.path.join(base_dir, "config.json")))["text_config"]
    cfg = DeciderConfig(layer_types=bcfg["layer_types"], pointer_dim=dcfg["pointer_dim"],
                        temperature_by_kind=dcfg.get("temperature_by_kind", {}),
                        temperature=dcfg.get("temperature", 1.0), max_length=dcfg.get("max_length", 4096))
    raw: dict[str, torch.Tensor] = {}
    for f in glob.glob(os.path.join(base_dir, "*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith("model.language_model."):
                    # HF loads the torso in bf16 (torch_dtype), rounding the fp32-stored norm weights:
                    # the model was trained and is served with bf16-rounded base params.
                    raw[k[len("model.language_model."):]] = s.get_tensor(k).to(torch.bfloat16).float()
    # LoRA merge: W += (alpha / r) * B @ A
    acfg = json.load(open(os.path.join(decider_dir, "lora", "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    assert not acfg.get("use_rslora") and not acfg.get("use_dora")
    with safe_open(os.path.join(decider_dir, "lora", "adapter_model.safetensors"), "pt") as s:
        keys = list(s.keys())
        for k in keys:
            if ".lora_A." not in k:
                continue
            tgt = k.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
            A = s.get_tensor(k).float()
            B = s.get_tensor(k.replace(".lora_A.", ".lora_B.")).float()
            raw[tgt] = raw[tgt] + scale * (B @ A)
    with safe_open(os.path.join(decider_dir, "head.safetensors"), "pt") as s:
        head = {k: s.get_tensor(k).float() for k in s.keys()}

    sd: dict[str, torch.Tensor] = {}
    sd["embed"] = raw["embed_tokens.weight"].to(dtype)
    sd["final_norm"] = (1.0 + raw["norm.weight"]).float()
    for i, lt in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        sd[p + "ln1"] = (1.0 + raw[p + "input_layernorm.weight"]).float()
        sd[p + "ln2"] = (1.0 + raw[p + "post_attention_layernorm.weight"]).float()
        sd[p + "w_gate_up"] = torch.cat([raw[p + "mlp.gate_proj.weight"], raw[p + "mlp.up_proj.weight"]], 0).to(dtype)
        sd[p + "w_down"] = raw[p + "mlp.down_proj.weight"].to(dtype)
        if lt == "linear_attention":
            q = p + "linear_attn."
            # head-major projections: one einsum -> [B, 64, T, 128] = q(16) k(16) v(16) z(16) heads
            qkv_w = raw[q + "in_proj_qkv.weight"].reshape(48, 128, 2048)
            z_w = raw[q + "in_proj_z.weight"].reshape(16, 128, 2048)
            sd[p + "w_qkvz"] = torch.cat([qkv_w, z_w], 0).to(dtype)
            sd[p + "w_ba"] = torch.cat([raw[q + "in_proj_b.weight"], raw[q + "in_proj_a.weight"]], 0).to(dtype)
            sd[p + "conv_w"] = raw[q + "conv1d.weight"][:, 0, :].reshape(48, 128, 4).to(dtype)
            sd[p + "A"] = (-raw[q + "A_log"].exp()).float()  # [16]  g = A * softplus(a + dt_bias)
            sd[p + "dt_bias"] = raw[q + "dt_bias"].float()
            sd[p + "gnorm"] = raw[q + "norm.weight"].to(dtype)  # RMSNormGated: plain weight
            sd[p + "w_out"] = raw[q + "out_proj.weight"].reshape(2048, 16, 128).to(dtype)
        else:
            q = p + "self_attn."
            # head-major: [20, 256, 2048] = q(8) gate(8) k(2) v(2); q_proj rows are [q_h | gate_h] per head
            qg = raw[q + "q_proj.weight"].reshape(8, 2, 256, 2048)
            sd[p + "w_qgkv"] = torch.cat([qg[:, 0], qg[:, 1], raw[q + "k_proj.weight"].reshape(2, 256, 2048),
                                          raw[q + "v_proj.weight"].reshape(2, 256, 2048)], 0).to(dtype)
            sd[p + "q_norm"] = (1.0 + raw[q + "q_norm.weight"]).float()
            sd[p + "k_norm"] = (1.0 + raw[q + "k_norm.weight"]).float()
            sd[p + "w_o"] = raw[q + "o_proj.weight"].reshape(2048, 8, 256).to(dtype)
    sd["head.ln_w"] = head["norm.weight"]
    sd["head.ln_b"] = head["norm.bias"]
    sd["head.wq"] = head["q.weight"]
    sd["head.bq"] = head["q.bias"]
    sd["head.wk"] = head["k.weight"]
    sd["head.bk"] = head["k.bias"]
    return cfg, sd


def shard_state(cfg: DeciderConfig, sd: dict[str, torch.Tensor], rank: int, world: int):
    """Megatron-style TP shard: heads (GDN, attention q/gate) and MLP intermediate split over
    `world` ranks; KV heads split when world <= nkv, else replicated (rank r reads kv head
    r * nkv // world). Embedding, norms and the readout are replicated."""
    import copy

    if world == 1:
        return cfg, sd
    c = copy.deepcopy(cfg)
    assert cfg.gv % world == 0 and cfg.nq % world == 0 and cfg.inter % world == 0
    hl, ql, il = cfg.gv // world, cfg.nq // world, cfg.inter // world
    kvl = max(1, cfg.nkv // world)
    kv0 = (rank * cfg.nkv) // world if world > cfg.nkv else rank * kvl
    c.gk = c.gv = hl
    c.nq, c.nkv, c.inter = ql, kvl, il
    out = {}
    for k, v in sd.items():
        name = k.rsplit(".", 1)[-1]
        hs = slice(rank * hl, (rank + 1) * hl)
        if name == "w_qkvz":  # [64,128,2048] = q16 k16 v16 z16
            out[k] = torch.cat([v[g * cfg.gv:(g + 1) * cfg.gv][hs] for g in range(4)], 0)
        elif name == "conv_w":  # [48,128,4]
            out[k] = torch.cat([v[g * cfg.gv:(g + 1) * cfg.gv][hs] for g in range(3)], 0)
        elif name == "w_ba":  # [32,2048] = b16 a16
            out[k] = torch.cat([v[:cfg.gv][hs], v[cfg.gv:][hs]], 0)
        elif name in ("A", "dt_bias"):
            out[k] = v[hs]
        elif name == "w_out":  # [2048,16,128]
            out[k] = v[:, hs]
        elif name == "w_qgkv":  # [20,256,2048] = q8 gate8 k2 v2
            qs = slice(rank * ql, (rank + 1) * ql)
            kvs = slice(kv0, kv0 + kvl)
            n = cfg.nq
            out[k] = torch.cat([v[:n][qs], v[n:2 * n][qs], v[2 * n:2 * n + cfg.nkv][kvs], v[2 * n + cfg.nkv:][kvs]], 0)
        elif name == "w_o":  # [2048,8,256]
            out[k] = v[:, rank * ql:(rank + 1) * ql]
        elif name == "w_gate_up":  # [12288,2048] = gate6144 up6144
            out[k] = torch.cat([v[:cfg.inter][rank * il:(rank + 1) * il], v[cfg.inter:][rank * il:(rank + 1) * il]], 0)
        elif name == "w_down":  # [2048,6144]
            out[k] = v[:, rank * il:(rank + 1) * il]
        else:
            out[k] = v
    return c, {k: v.contiguous() for k, v in out.items()}


# ------------------------------------------------------------------ math helpers
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3.5 RMSNorm with the (1+w) already folded into w: (x*rsqrt(mean x^2)) * w in fp32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(x.dtype)


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated: w * norm(x) (cast to x.dtype) * silu(gate) (fp32)."""
    dt = x.dtype
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(dt)
    y = w * y
    return (y.float() * F.silu(gate.float())).to(dt)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def causal_conv1d_silu(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [B, T, C], w [C, K]: depthwise causal conv (zero history) then SiLU. Channels-last."""
    K = w.shape[1]
    T = x.shape[1]
    xf = F.pad(x.float(), (0, 0, K - 1, 0))
    wf = w.float()
    out = xf[:, K - 1:K - 1 + T] * wf[:, K - 1]
    for k in range(K - 1):
        out = out + xf[:, k:k + T] * wf[:, k]
    return F.silu(out.to(x.dtype).float()).to(x.dtype)


def causal_conv1d_silu_hm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D], w [H, D, K]: depthwise causal conv along T (zero history), SiLU, fp32 out."""
    K = w.shape[-1]
    T = x.shape[2]
    xf = F.pad(x.float(), (0, 0, K - 1, 0))
    wf = w.float()
    out = xf[:, :, K - 1:K - 1 + T] * wf[None, :, None, :, K - 1]
    for k in range(K - 1):
        out = out + xf[:, :, k:k + T] * wf[None, :, None, :, k]
    return F.silu(out)


def rope_cos_sin(T: int, dim: int, theta: float, device, dtype, offset=None) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    pos = torch.arange(T, dtype=torch.float32, device=device)
    if offset is not None:  # [1] int tensor: positions start after a cached prefix
        pos = pos + offset.reshape(1).float()
    f = torch.outer(pos, inv)
    emb = torch.cat([f, f], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, H, T, D]; rotate the first cos.shape[-1] dims (rotate_half convention)."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


# ------------------------------------------------------------------ token mixers (reference)
def gdn_chunk_ref(q, k, v, g, beta, chunk: int = 64, S0=None, return_state: bool = False):
    """Reference chunked gated delta rule (fp32), HF torch_chunk_gated_delta_rule semantics.

    q, k: [B, H, T, Dk] (already L2-normalised), v: [B, H, T, Dv], g, beta: [B, H, T] fp32.
    Returns o [B, H, T, Dv] fp32. Initial state zero. Uses solve_triangular (CPU reference).
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    q = q.float() * Dk ** -0.5
    k, v, g, beta = k.float(), v.float(), g.float(), beta.float()
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    n = (T + pad) // chunk
    vb = v * beta[..., None]
    kb = k * beta[..., None]
    q, k, kb, vb = (x.reshape(B, H, n, chunk, -1) for x in (q, k, kb, vb))
    g = g.reshape(B, H, n, chunk).cumsum(-1)
    mask = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).triu(1)
    dec = (g[..., :, None] - g[..., None, :]).masked_fill(mask, float("-inf")).exp()
    A = (kb @ k.transpose(-1, -2)) * dec
    attn = (q @ k.transpose(-1, -2)) * dec
    eye = torch.eye(chunk, device=q.device)
    Lm = (A.masked_fill(~torch.ones_like(mask).tril(-1), 0.0)) + eye  # unit lower triangular
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


def attn_ref(q, k, v, scale: float, k_prior=None, v_prior=None, prior_len=None):
    """q [B, Hq, T, D], k/v [B, Hkv, T, D]; causal GQA attention, softmax in fp32.
    Optional prefix: k_prior/v_prior [B, Hkv, S, D] of which the first prior_len ([1] int) are valid."""
    rep = q.shape[1] // k.shape[1]
    T = q.shape[2]
    m = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    if k_prior is not None:
        S = k_prior.shape[2]
        k = torch.cat([k_prior, k], 2)
        v = torch.cat([v_prior, v], 2)
        pm = (torch.arange(S, device=q.device) >= prior_len.reshape(1))[None, :].expand(T, S)
        m = torch.cat([pm, m], 1)
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    s = s.masked_fill(m, float("-inf"))
    p = torch.softmax(s, -1).to(q.dtype)
    return p @ v


# ------------------------------------------------------------------ the model
class DeciderNeuron(nn.Module):
    def __init__(self, cfg: DeciderConfig, sd: dict[str, torch.Tensor], gdn_impl=None, attn_impl=None,
                 compute_dtype=torch.bfloat16, use_fast: bool = False):
        super().__init__()
        self.use_fast = use_fast
        self.cfg = cfg
        self.dt = compute_dtype
        self.gdn_impl = gdn_impl or gdn_chunk_ref
        self.attn_impl = attn_impl or attn_ref
        self.tp_group = None  # set to a process group for tensor parallel (partial sums all-reduced)
        for k, v in sd.items():
            fp32 = (k.startswith("head.") or k == "final_norm"
                    or k.rsplit(".", 1)[-1] in ("A", "dt_bias", "ln1", "ln2", "q_norm", "k_norm"))
            if fp32:
                t = v.float()
            else:
                t = v.to(compute_dtype)
            self.register_buffer(k.replace(".", "__"), t, persistent=False)
        self.fast_keys: dict[int, list[str]] = {}
        if use_fast:  # fused NKI prologue/epilogue + pre-transposed GEMM weights (kernels/fused_ops.py)
            import os as _os
            import sys as _sys

            _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "kernels"))
            import fused_ops as fo

            self._fo = fo
            for i, lt in enumerate(cfg.layer_types):
                prep = fo.prepare_mlp_layer(sd, i, compute_dtype)
                if lt == "linear_attention":
                    prep.update(fo.prepare_gdn_layer(sd, i, compute_dtype))
                else:
                    a = fo.prepare_attn_layer(sd, i, compute_dtype)
                    a.pop("ln1")
                    prep.update({("attn_" + k2): v2 for k2, v2 in a.items()})
                    prep["attn_ln1"] = sd[f"layers.{i}.ln1"].float().contiguous()
                self.fast_keys[i] = list(prep)
                for k2, v2 in prep.items():
                    self.register_buffer(f"fast__{i}__{k2}", v2, persistent=False)

    def fprep(self, i: int) -> dict:
        return {k: getattr(self, f"fast__{i}__{k}") for k in self.fast_keys[i]}

    def aprep(self, i: int) -> dict:
        d = self.fprep(i)
        return {k[5:]: v for k, v in d.items() if k.startswith("attn_")}

    tp_chunks: int = 1

    @staticmethod
    def _chunks(T: int, n: int):
        step = T // n
        return [(j * step, (j + 1) * step if j < n - 1 else T) for j in range(n)]

    @property
    def red(self):
        return self._reduce if self.tp_group is not None else None

    def w(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "__"))

    def _reduce(self, y: torch.Tensor) -> torch.Tensor:
        if self.tp_group is None or getattr(self, "_noreduce", False):  # _noreduce: timing experiments only
            return y
        import torch.distributed._functional_collectives as funcol

        return funcol.all_reduce(y, "sum", self.tp_group)

    # --- layers
    def gdn_layer(self, i: int, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        p = f"layers.{i}."
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        Hl = c.gv  # local (per-rank) heads; gk == gv for this model
        qkvz = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qkvz"))  # [B, 4*Hl, T, 128]
        conv = causal_conv1d_silu_hm(qkvz[:, : 3 * Hl], self.w(p + "conv_w"))  # fp32
        q = l2norm(conv[:, :Hl])
        k = l2norm(conv[:, Hl:2 * Hl])
        v = conv[:, 2 * Hl:3 * Hl]
        z = qkvz[:, 3 * Hl:]
        ba = (h @ self.w(p + "w_ba").t()).float().transpose(1, 2)  # [B, 2*Hl, T]
        beta = torch.sigmoid(ba[:, : c.gv])
        g = self.w(p + "A")[None, :, None] * F.softplus(ba[:, c.gv:] + self.w(p + "dt_bias")[None, :, None])
        dt = self.dt
        o = self.gdn_impl(q.to(dt), k.to(dt), v.to(dt), g, beta)  # [B, H, T, Dv]
        o = gated_rms_norm(o.to(dt), z, self.w(p + "gnorm"), c.eps)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(p + "w_out")))

    def attn_layer(self, i: int, x: torch.Tensor, cos, sin) -> torch.Tensor:
        c = self.cfg
        p = f"layers.{i}."
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        qgkv = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qgkv"))  # [B, 20, T, 256]
        q = rms_norm(qgkv[:, : c.nq], self.w(p + "q_norm"), c.eps)
        gate = qgkv[:, c.nq: 2 * c.nq]
        k = rms_norm(qgkv[:, 2 * c.nq: 2 * c.nq + c.nkv], self.w(p + "k_norm"), c.eps)
        v = qgkv[:, 2 * c.nq + c.nkv:]
        q = apply_partial_rope(q, cos, sin)
        k = apply_partial_rope(k, cos, sin)
        o = self.attn_impl(q, k, v, c.hd ** -0.5)  # [B, Hq, T, D]
        o = o * torch.sigmoid(gate)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(p + "w_o")))

    def mlp(self, i: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{i}."
        h = rms_norm(x, self.w(p + "ln2"), self.cfg.eps)
        gu = h @ self.w(p + "w_gate_up").t()
        gte, up = gu[..., : self.cfg.inter], gu[..., self.cfg.inter:]
        return x + self._reduce((F.silu(gte) * up) @ self.w(p + "w_down").t())

    def torso(self, input_ids: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        x = F.embedding(input_ids.long(), self.w("embed"))
        T = input_ids.shape[1]
        cos, sin = rope_cos_sin(T, c.rope_dim, c.rope_theta, x.device, x.dtype)
        for i, lt in enumerate(c.layer_types):
            if self.use_fast:
                prep = self.fprep(i)
                nch = self.tp_chunks if self.tp_group is not None else 1
                if lt == "linear_attention":
                    if nch > 1:  # independent per-chunk out-proj -> all-reduce chains (comm/compute overlap)
                        fo = self._fo
                        proj_fm, zba = fo.gdn_in_proj(x, prep["ln1"], prep, c.eps)
                        q, k, v, g, beta = fo.gdn_prologue(proj_fm, prep["conv_diag"], zba, prep["gdn_ab"])
                        o = self.gdn_impl(q, k, v, g, beta).to(x.dtype)
                        x = torch.cat([x[:, a:b] + self.red(fo.gdn_out_proj(o[:, :, a:b].contiguous(), zba[:, a:b].contiguous(),
                                                                             prep["gnorm"], prep["w_out_t"], c.eps))
                                       for a, b in self._chunks(T, nch)], 1)
                    else:
                        x = self._fo.gdn_layer_fast(x, prep, self.gdn_impl, c.eps, reduce=self.red)
                else:
                    x = self._fo.attn_layer_fast(x, self.aprep(i), self.attn_impl, cos, sin, c.nq, c.nkv, c.eps, reduce=self.red)
                if nch > 1:
                    x = torch.cat([self._fo.mlp_fast(x[:, a:b], prep, c.eps, reduce=self.red) for a, b in self._chunks(T, nch)], 1)
                else:
                    x = self._fo.mlp_fast(x, prep, c.eps, reduce=self.red)
            else:
                x = self.gdn_layer(i, x) if lt == "linear_attention" else self.attn_layer(i, x, cos, sin)
                x = self.mlp(i, x)
        return rms_norm(x, self.w("final_norm"), c.eps)

    # ------------------------------------------------------------------ shared prefix
    # A request's state is encoded once (prefix, B=1, right-padded to a bucket, real length
    # s_len), then every question suffix runs from its caches: per GDN layer the final
    # recurrent state and the last K-1 pre-conv tokens, per attention layer K and V.
    # Pad positions of the prefix are made no-ops for the recurrence (beta = 0, g = 0) and are
    # masked out of attention (prior_len = s_len), so bucketing the prefix is exact.
    def _gdn_proj(self, i, x):
        c = self.cfg
        p = f"layers.{i}."
        Hl = c.gv
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        qkvz = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qkvz"))
        ba = (h @ self.w(p + "w_ba").t()).float().transpose(1, 2)
        beta = torch.sigmoid(ba[:, :Hl])
        g = self.w(p + "A")[None, :, None] * F.softplus(ba[:, Hl:] + self.w(p + "dt_bias")[None, :, None])
        return qkvz, beta, g

    def _gdn_out(self, i, x, o, z):
        p = f"layers.{i}."
        o = gated_rms_norm(o.to(self.dt), z, self.w(p + "gnorm"), self.cfg.eps)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(p + "w_out")))

    def gdn_layer_prefix(self, i, x, valid, s_len):
        c = self.cfg
        Hl, K = c.gv, c.conv_k
        qkvz, beta, g = self._gdn_proj(i, x)
        tail = self.conv_tail(qkvz[:, : 3 * Hl], s_len, K)
        vm = valid[:, None, :].float()  # [1,1,S]
        beta, g = beta * vm, g * vm
        conv = causal_conv1d_silu_hm(qkvz[:, : 3 * Hl], self.w(f"layers.{i}.conv_w"))
        q, k, v = l2norm(conv[:, :Hl]), l2norm(conv[:, Hl:2 * Hl]), conv[:, 2 * Hl:3 * Hl]
        o, S = self.gdn_impl(q.to(self.dt), k.to(self.dt), v.to(self.dt), g, beta, return_state=True)
        return self._gdn_out(i, x, o, qkvz[:, 3 * Hl:]), S, tail

    @staticmethod
    def conv_tail(pre, s_len, K):
        """Last K-1 pre-conv tokens before position s_len: pre [1,C,S,D] -> [1,C,K-1,D] (zeros before 0)."""
        idx = s_len.reshape(1).long() - (K - 1) + torch.arange(K - 1, device=pre.device)
        ok = (idx >= 0).to(pre.dtype)[None, None, :, None]
        t = torch.index_select(pre, 2, idx.clamp_min(0))
        return t * ok

    def gdn_layer_suffix(self, i, x, S0, tail):
        c = self.cfg
        Hl, K = c.gv, c.conv_k
        B = x.shape[0]
        qkvz, beta, g = self._gdn_proj(i, x)
        pre = torch.cat([tail.expand(B, -1, -1, -1).to(qkvz.dtype), qkvz[:, : 3 * Hl]], 2)
        conv = causal_conv1d_silu_hm(pre, self.w(f"layers.{i}.conv_w"))[:, :, K - 1:]
        q, k, v = l2norm(conv[:, :Hl]), l2norm(conv[:, Hl:2 * Hl]), conv[:, 2 * Hl:3 * Hl]
        o = self.gdn_impl(q.to(self.dt), k.to(self.dt), v.to(self.dt), g, beta, S0=S0.expand(B, -1, -1, -1).contiguous())
        return self._gdn_out(i, x, o, qkvz[:, 3 * Hl:])

    def _attn_qkv(self, i, x, cos, sin):
        c = self.cfg
        p = f"layers.{i}."
        h = rms_norm(x, self.w(p + "ln1"), c.eps)
        qgkv = torch.einsum("btc,hdc->bhtd", h, self.w(p + "w_qgkv"))
        q = apply_partial_rope(rms_norm(qgkv[:, : c.nq], self.w(p + "q_norm"), c.eps), cos, sin)
        gate = qgkv[:, c.nq: 2 * c.nq]
        k = apply_partial_rope(rms_norm(qgkv[:, 2 * c.nq: 2 * c.nq + c.nkv], self.w(p + "k_norm"), c.eps), cos, sin)
        v = qgkv[:, 2 * c.nq + c.nkv:]
        return q, gate, k, v

    def _attn_out(self, i, x, o, gate):
        o = o * torch.sigmoid(gate)
        return x + self._reduce(torch.einsum("bhtd,chd->btc", o, self.w(f"layers.{i}.w_o")))

    def attn_layer_prefix(self, i, x, cos, sin):
        q, gate, k, v = self._attn_qkv(i, x, cos, sin)
        return self._attn_out(i, x, self.attn_impl(q, k, v, self.cfg.hd ** -0.5), gate), (k, v)

    def attn_layer_suffix(self, i, x, cos, sin, kv, s_len):
        q, gate, k, v = self._attn_qkv(i, x, cos, sin)
        B = x.shape[0]
        kp, vp = (t.expand(B, -1, -1, -1).contiguous() for t in kv)
        o = self.attn_impl(q, k, v, self.cfg.hd ** -0.5, k_prior=kp, v_prior=vp, prior_len=s_len)
        return self._attn_out(i, x, o, gate)

    def prefix_forward(self, input_ids, s_len):
        """input_ids [1, S] (right-padded), s_len [1] int -> flat tuple of caches, layer order:
        GDN layer: (state [1,H,Dk,Dv] fp32, conv tail [1,3H,K-1,D]); attention layer: (K, V)."""
        c = self.cfg
        x = F.embedding(input_ids.long(), self.w("embed"))
        S = input_ids.shape[1]
        valid = torch.arange(S, device=x.device)[None, :] < s_len.reshape(1, 1)
        cos, sin = rope_cos_sin(S, c.rope_dim, c.rope_theta, x.device, x.dtype)
        caches = []
        for i, lt in enumerate(c.layer_types):
            if lt == "linear_attention":
                if self.use_fast:
                    x, St, tail = self._fo.gdn_layer_prefix_fast(x, self.fprep(i), self.gdn_impl, valid, s_len, c.eps, reduce=self.red)
                else:
                    x, St, tail = self.gdn_layer_prefix(i, x, valid, s_len)
                caches += [St, tail]
            else:
                x, (k, v) = self.attn_layer_prefix(i, x, cos, sin)
                caches += [k, v]
            x = self._fo.mlp_fast(x, self.fprep(i), c.eps, reduce=self.red) if self.use_fast else self.mlp(i, x)
        return tuple(caches)

    def suffix_forward(self, input_ids, s_len, caches, ans_idx, opt_idx, n_slots, inv_temp):
        c = self.cfg
        x = F.embedding(input_ids.long(), self.w("embed"))
        L = input_ids.shape[1]
        cos, sin = rope_cos_sin(L, c.rope_dim, c.rope_theta, x.device, x.dtype, offset=s_len)
        for i, lt in enumerate(c.layer_types):
            a, b = caches[2 * i], caches[2 * i + 1]
            if lt == "linear_attention":
                if self.use_fast:
                    x = self._fo.gdn_layer_suffix_fast(x, self.fprep(i), self.gdn_impl, a, b, c.eps, reduce=self.red)
                else:
                    x = self.gdn_layer_suffix(i, x, a, b)
            else:
                x = self.attn_layer_suffix(i, x, cos, sin, (a, b), s_len)
            x = self._fo.mlp_fast(x, self.fprep(i), c.eps, reduce=self.red) if self.use_fast else self.mlp(i, x)
        hidden = rms_norm(x, self.w("final_norm"), c.eps)
        probs, _ = self.readout(hidden, ans_idx, opt_idx, n_slots, inv_temp)
        return probs

    def readout(self, hidden, ans_idx, opt_idx, n_slots, inv_temp):
        """hidden [B,T,H]; ans_idx [B]; opt_idx [B,K]; n_slots [B]; inv_temp [B] -> probs [B,K]."""
        B, T, Hd = hidden.shape
        K = opt_idx.shape[1]
        idx = torch.cat([ans_idx.long()[:, None], opt_idx.long().clamp_min(0)], 1)  # [B, 1+K]
        g = torch.gather(hidden, 1, idx[..., None].expand(B, 1 + K, Hd)).float()
        g = F.layer_norm(g, (Hd,), self.w("head.ln_w"), self.w("head.ln_b"), 1e-5)
        qv = g[:, 0] @ self.w("head.wq").t() + self.w("head.bq")  # [B, P]
        kv = g[:, 1:] @ self.w("head.wk").t() + self.w("head.bk")  # [B, K, P]
        logits = (kv @ qv[..., None]).squeeze(-1) * self.cfg.pointer_dim ** -0.5
        logits = logits * inv_temp[:, None]
        valid = torch.arange(K, device=logits.device)[None, :] < n_slots[:, None]
        logits = logits.masked_fill(~valid, -1e4)
        probs = torch.softmax(logits, -1)
        return probs.masked_fill(~valid, 0.0), logits

    def forward(self, input_ids, ans_idx, opt_idx, n_slots, inv_temp):
        hidden = self.torso(input_ids)
        probs, _ = self.readout(hidden, ans_idx, opt_idx, n_slots, inv_temp)
        return probs
