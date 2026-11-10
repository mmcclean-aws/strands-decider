"""GDN pre-processing layout variants + gdn_cte launch variants."""

import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import torch.nn.functional as F
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.compile.native_backend import register

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decider_model import causal_conv1d_silu, l2norm, rms_norm  # noqa: E402

BACKEND = register()
DEV = torch.device("neuron", 0)
B, T = int(sys.argv[1]), int(sys.argv[2])
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
dt = torch.bfloat16


def timeit(name, fn, *args, iters=20):
    if only and not any(name.startswith(o) for o in only):
        return
    f = torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
    t0 = time.time()
    with torch.inference_mode():
        y = f(*args)
        (y[0] if isinstance(y, (tuple, list)) else y).cpu()
        tc = time.time() - t0
        for _ in range(3):
            y = f(*args)
        (y[0] if isinstance(y, (tuple, list)) else y).cpu()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = f(*args)
        (y[0] if isinstance(y, (tuple, list)) else y).cpu()
        d = (time.perf_counter() - t0) / iters
    print(f"{name:40s} {d * 1e3:8.3f} ms  (compile {tc:4.1f}s)", flush=True)


x = (torch.randn(B, T, 2048) * 0.5).to(dt).to(DEV)
ln = torch.ones(2048).to(DEV)
w_in = (torch.randn(8224, 2048) * 0.02).to(dt).to(DEV)
conv_w = (torch.randn(6144, 4) * 0.3).to(dt).to(DEV)
A = (-torch.rand(16)).to(DEV)
dtb = torch.zeros(16).to(DEV)


# v1: current (token-major GEMM, then reshape/transpose to head-major)
def v1(x, ln, w_in, conv_w, A, dtb):
    h = rms_norm(x, ln, 1e-6)
    proj = h @ w_in.t()
    qkv = causal_conv1d_silu(proj[..., :6144], conv_w)
    q = l2norm(qkv[..., :2048].reshape(B, T, 16, 128).transpose(1, 2).float()).to(dt)
    k = l2norm(qkv[..., 2048:4096].reshape(B, T, 16, 128).transpose(1, 2).float()).to(dt)
    v = qkv[..., 4096:].reshape(B, T, 16, 128).transpose(1, 2)
    beta = torch.sigmoid(proj[..., 8192:8208].float()).transpose(1, 2)
    g = (A * F.softplus(proj[..., 8208:].float() + dtb)).transpose(1, 2)
    return q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous()


timeit("v1 token-major + transpose", v1, x, ln, w_in, conv_w, A, dtb)

# pieces of v1
def v1_noconv(x, ln, w_in):
    h = rms_norm(x, ln, 1e-6)
    proj = h @ w_in.t()
    q = l2norm(proj[..., :2048].reshape(B, T, 16, 128).transpose(1, 2).float()).to(dt)
    return q.contiguous()


timeit("v1a norm+gemm+q-l2norm+transpose", v1_noconv, x, ln, w_in)


def v1_transpose_only(x, ln, w_in):
    proj = x @ w_in.t()
    return proj[..., :6144].reshape(B, T, 48, 128).transpose(1, 2).contiguous()


timeit("v1b gemm+transpose(48 heads)", v1_transpose_only, x, ln, w_in)


def v1_l2_only(x, ln, w_in):
    proj = x @ w_in.t()
    return l2norm(proj[..., :2048].float()).to(dt)


timeit("v1c gemm+l2norm(token-major,fp32)", v1_l2_only, x, ln, w_in)

# v2: head-major GEMM via einsum (weights pre-shaped [48, 128, 2048]); conv along T in [B,H,T,D]
w_hm = w_in[:6144].reshape(48, 128, 2048).contiguous()
w_ab = w_in[8192:].contiguous()
conv_hm = conv_w.reshape(48, 128, 4).contiguous()


def conv_t(x, w):  # x [B,H,T,D], w [H,D,K]: causal conv along T
    K = w.shape[-1]
    xf = F.pad(x.float(), (0, 0, K - 1, 0))
    wf = w.float()
    out = xf[:, :, K - 1:K - 1 + T] * wf[None, :, None, :, K - 1]
    for k in range(K - 1):
        out = out + xf[:, :, k:k + T] * wf[None, :, None, :, k]
    return F.silu(out)


def v2(x, ln, w_hm, w_ab, conv_hm, A, dtb):
    h = rms_norm(x, ln, 1e-6)
    qkv = torch.einsum("btc,hdc->bhtd", h, w_hm)  # [B,48,T,128]
    qkv = conv_t(qkv, conv_hm)  # fp32
    q = l2norm(qkv[:, :16]).to(dt)
    k = l2norm(qkv[:, 16:32]).to(dt)
    v = qkv[:, 32:].to(dt)
    ab = (h @ w_ab.t()).float().transpose(1, 2)  # [B,32,T]
    beta = torch.sigmoid(ab[:, :16])
    g = A[None, :, None] * F.softplus(ab[:, 16:] + dtb[None, :, None])
    return q, k, v, g.contiguous(), beta.contiguous()


timeit("v2 head-major einsum + conv(T)", v2, x, ln, w_hm, w_ab, conv_hm, A, dtb)


def v2_gemm(x, w_hm):
    return torch.einsum("btc,hdc->bhtd", x, w_hm)


timeit("v2a head-major einsum only", v2_gemm, x, w_hm)

# v3: token-major everything, transpose done as one [B,T,6144]->[B,6144,T]-style permute at the end in bf16
def v3(x, ln, w_in, conv_w, A, dtb):
    h = rms_norm(x, ln, 1e-6)
    proj = h @ w_in.t()
    qkv = causal_conv1d_silu(proj[..., :6144], conv_w)  # bf16 [B,T,6144]
    qk = qkv[..., :4096].reshape(B, T, 32, 128).float()
    qk = (qk * torch.rsqrt((qk * qk).sum(-1, keepdim=True) + 1e-6)).to(dt)
    allh = torch.cat([qk.reshape(B, T, 4096), qkv[..., 4096:]], -1).reshape(B, T, 48, 128)
    allh = allh.permute(0, 2, 1, 3).contiguous()
    beta = torch.sigmoid(proj[..., 8192:8208].float()).transpose(1, 2)
    g = (A * F.softplus(proj[..., 8208:].float() + dtb)).transpose(1, 2)
    return allh[:, :16], allh[:, 16:32], allh[:, 32:], g.contiguous(), beta.contiguous()


timeit("v3 token-major, one bf16 permute", v3, x, ln, w_in, conv_w, A, dtb)

# gdn_cte launch variants
import nki  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402
from nkilib.experimental.gdn import build_mask_pack, gdn_cte  # noqa: E402

mp = torch.as_tensor(build_mask_pack(), dtype=torch.float32).to(DEV)
kern = wrap_nki(gdn_cte)
for H in (16,):
    q = l2norm(torch.randn(B, H, T, 128)).to(dt).to(DEV)
    k = l2norm(torch.randn(B, H, T, 128)).to(dt).to(DEV)
    v = (torch.randn(B, H, T, 128) * 0.5).to(dt).to(DEV)
    g = (-torch.rand(B, H, T) * 0.1).to(DEV)
    be = torch.rand(B, H, T).to(DEV)
    for launch in ("grid2", "plain"):
        for gs in (8, 4, 2):
            def core(q, k, v, g, be, launch=launch, gs=gs):
                kk = kern[2] if launch == "grid2" else kern
                return kk(q, k, v, g, be, scale=128 ** -0.5, mask_pack=mp, group_size=gs)[0]
            try:
                timeit(f"gdn_cte H={H} {launch} group={gs}", core, q, k, v, g, be)
            except Exception as e:
                print(f"gdn_cte H={H} {launch} group={gs} FAILED {type(e).__name__}", flush=True)
