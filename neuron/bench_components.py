"""Per-component device timing at a given (B, T): which part of a layer is slow?"""

import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import torch.nn.functional as F
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.compile.native_backend import register

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decider_model import (DeciderNeuron, _snapshot, causal_conv1d_silu, gated_rms_norm, l2norm, load_state,  # noqa: E402
                           rms_norm)
from nki_ops import NkiMixers  # noqa: E402

BACKEND = register()
DEV = torch.device("neuron", 0)
B, T = int(sys.argv[1]), int(sys.argv[2])
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None


def timeit(name, fn, *args, iters=20, flops=0.0):
    if only and name not in only:
        return
    f = torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
    t0 = time.time()
    with torch.inference_mode():
        y = f(*args)
        (y[0] if isinstance(y, tuple) else y).cpu()
        tc = time.time() - t0
        for _ in range(3):
            y = f(*args)
        (y[0] if isinstance(y, tuple) else y).cpu()
        t0 = time.perf_counter()
        for i in range(iters):
            y = f(*args)
        (y[0] if isinstance(y, tuple) else y).cpu()
        dt = (time.perf_counter() - t0) / iters
    tf = f"{flops / dt / 1e12:6.1f} TF/s ({flops / dt / (667e12 / 4):5.1%} of LNC2 peak)" if flops else ""
    print(f"{name:28s} {dt * 1e3:8.3f} ms   compile {tc:5.1f}s  {tf}", flush=True)


cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"))
mix = NkiMixers(DEV)
dt = torch.bfloat16
W = {k: v.to(DEV) for k, v in sd.items() if k.startswith("layers.0.") or k.startswith("layers.3.")}
x = (torch.randn(B, T, 2048) * 0.5).to(dt).to(DEV)
H = 2048


def mlp(x, ln, wgu, wd):
    h = rms_norm(x, ln, 1e-6)
    gu = h @ wgu.t()
    return x + (F.silu(gu[..., :6144]) * gu[..., 6144:]) @ wd.t()


timeit("mlp(norm+swiglu+resid)", mlp, x, W["layers.0.ln2"], W["layers.0.w_gate_up"], W["layers.0.w_down"],
       flops=2 * B * T * 3 * 2048 * 6144)


def gemm_only(x, wgu):
    return x @ wgu.t()


timeit("gemm 2048x12288", gemm_only, x, W["layers.0.w_gate_up"], flops=2 * B * T * 2048 * 12288)


def gdn_pre(x, ln, w_in, conv_w, A, dtb):
    h = rms_norm(x, ln, 1e-6)
    proj = h @ w_in.t()
    qkv = causal_conv1d_silu(proj[..., :6144], conv_w)
    q = l2norm(qkv[..., :2048].reshape(B, T, 16, 128).transpose(1, 2).float()).to(dt)
    k = l2norm(qkv[..., 2048:4096].reshape(B, T, 16, 128).transpose(1, 2).float()).to(dt)
    v = qkv[..., 4096:].reshape(B, T, 16, 128).transpose(1, 2)
    beta = torch.sigmoid(proj[..., 8192:8208].float()).transpose(1, 2)
    g = (A * F.softplus(proj[..., 8208:].float() + dtb)).transpose(1, 2)
    return q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), proj[..., 6144:8192].contiguous()


p0 = "layers.0."
timeit("gdn_pre(norm+inproj+conv+..)", gdn_pre, x, W[p0 + "ln1"], W[p0 + "w_in"], W[p0 + "conv_w"], W[p0 + "A"], W[p0 + "dt_bias"],
       flops=2 * B * T * 2048 * 8224)


def inproj(x, w_in):
    return x @ w_in.t()


timeit("gemm 2048x8224 (in_proj)", inproj, x, W[p0 + "w_in"], flops=2 * B * T * 2048 * 8224)


def conv(x, conv_w):
    return causal_conv1d_silu(x, conv_w)


xc = (torch.randn(B, T, 6144) * 0.5).to(dt).to(DEV)
timeit("conv1d+silu (torch)", conv, xc, W[p0 + "conv_w"])

q = l2norm(torch.randn(B, 16, T, 128)).to(dt).to(DEV)
k = l2norm(torch.randn(B, 16, T, 128)).to(dt).to(DEV)
v = (torch.randn(B, 16, T, 128) * 0.5).to(dt).to(DEV)
g = (-torch.rand(B, 16, T) * 0.1).to(DEV)
beta = torch.rand(B, 16, T).to(DEV)


def gdn_core(q, k, v, g, beta):
    return mix.gdn(q, k, v, g, beta)


timeit("gdn_cte NKI (16 heads)", gdn_core, q, k, v, g, beta)


def gdn_post(o, z, x, gw, wout):
    o = o.transpose(1, 2)
    o = gated_rms_norm(o, z.reshape(B, T, 16, 128), gw, 1e-6)
    return x + o.reshape(B, T, 2048) @ wout.t()


z = (torch.randn(B, T, 2048) * 0.5).to(dt).to(DEV)
timeit("gdn_post(gnorm+outproj)", gdn_post, v, z, x, W[p0 + "gnorm"], W[p0 + "w_out"], flops=2 * B * T * 2048 * 2048)

qa = (torch.randn(B, 8, T, 256) * 0.1).to(dt).to(DEV)
ka = (torch.randn(B, 2, T, 256) * 0.1).to(dt).to(DEV)
va = (torch.randn(B, 2, T, 256) * 0.1).to(dt).to(DEV)


def attn_core(q, k, v):
    return mix.attn(q, k, v, 256 ** -0.5)


timeit("attention_cte NKI d=256", attn_core, qa, ka, va, flops=4 * 8 * 256 * B * T * T / 2)

m = DeciderNeuron(cfg, {k: v for k, v in sd.items()}, gdn_impl=mix.gdn, attn_impl=mix.attn).eval()
m.cfg.layer_types = m.cfg.layer_types[:4]
m = m.to(DEV)
cos_sin = None


def gdn_layer_full(x):
    return m.mlp(0, m.gdn_layer(0, x))


timeit("full GDN layer + MLP", gdn_layer_full, x)


def attn_layer_full(x):
    from decider_model import rope_cos_sin
    cos, sin = rope_cos_sin(T, 64, 1e7, x.device, x.dtype)
    return m.mlp(3, m.attn_layer(3, x, cos, sin))


timeit("full attn layer + MLP", attn_layer_full, x)
