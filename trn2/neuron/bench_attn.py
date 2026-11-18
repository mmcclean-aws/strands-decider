"""Causal GQA attention (8 q heads, 2 kv heads, d=256) variants on one LNC2 core."""

import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import libtorch_neuronx_lite  # noqa: F401
import nki
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte

BACKEND = register()
DEV = torch.device("neuron", 0)
K = wrap_nki(nki.jit()(attention_cte))
dt = torch.bfloat16


def timeit(name, fn, args, flops, ref=None, iters=10):
    f = torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
    try:
        with torch.inference_mode():
            t0 = time.time()
            y = f(*args).cpu()
            tc = time.time() - t0
            for _ in range(2):
                f(*args)
            f(*args).cpu()
            t0 = time.perf_counter()
            for _ in range(iters):
                o = f(*args)
            o.cpu()
            d = (time.perf_counter() - t0) / iters
    except Exception as e:
        print(f"{name:34s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None
    err = ""
    if ref is not None:
        err = f"rel={((y.float() - ref).norm() / ref.norm()).item():.2e}"
    print(f"{name:34s} {d * 1e3:8.3f} ms {flops / d / 1e12:6.1f} TF/s  compile {tc:5.1f}s {err}", flush=True)
    return y


for T in [int(x) for x in sys.argv[1].split(",")]:
    B = 1
    torch.manual_seed(0)
    qc = torch.randn(B, 8, T, 256) * 0.5
    kc = torch.randn(B, 2, T, 256) * 0.5
    vc = torch.randn(B, 2, T, 256)
    sc = 256 ** -0.5
    ref = None
    if T <= 2048:
        kr, vr = kc.repeat_interleave(4, 1), vc.repeat_interleave(4, 1)
        s = (qc @ kr.transpose(-1, -2)) * sc
        s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        ref = torch.softmax(s, -1) @ vr
    q, k, v = (x.to(dt).to(DEV) for x in (qc, kc, vc))
    fl = 4 * 8 * 256 * T * T / 2
    print(f"--- T={T}", flush=True)

    def nki_gqa(q, k, v):  # native GQA: q [8,T,D] kv [2,T,D]
        return K(q=(q * sc).reshape(B * 8, T, 256), k=k.reshape(B * 2, T, 256), v=v.reshape(B * 2, T, 256), scale=1.0,
                 causal_mask=True, tp_q=True, tp_k=True, tp_out=False).reshape(B, 8, T, 256)

    timeit("nki attention_cte GQA tp_k=True", nki_gqa, (q, k, v), fl, ref)

    def nki_gqa_kT(q, k, v):  # k as [B, D, S] (tp_k=False)
        return K(q=(q * sc).reshape(B * 8, T, 256), k=k.reshape(B * 2, T, 256).transpose(1, 2).contiguous(), v=v.reshape(B * 2, T, 256),
                 scale=1.0, causal_mask=True, tp_q=True, tp_k=False, tp_out=False).reshape(B, 8, T, 256)

    timeit("nki attention_cte GQA tp_k=False", nki_gqa_kT, (q, k, v), fl, ref)

    def nki_rep(q, k, v):  # KV replicated per q head
        kk = k.repeat_interleave(4, 1).reshape(B * 8, T, 256)
        vv = v.repeat_interleave(4, 1).reshape(B * 8, T, 256)
        return K(q=(q * sc).reshape(B * 8, T, 256), k=kk, v=vv, scale=1.0, causal_mask=True, tp_q=True, tp_k=True,
                 tp_out=False).reshape(B, 8, T, 256)

    timeit("nki attention_cte KV-replicated", nki_rep, (q, k, v), fl, ref)

    def nki_split_d(q, k, v):
        """d=256 as two d=128 halves? Not exact for softmax -> skip; instead q-heads grouped per kv head:
        treat the 4 q heads sharing a kv head as 4*T queries against T keys (non-causal-safe only with mask)."""
        return None

    def torch_attn(q, k, v):
        kr, vr = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
        s = (q @ kr.transpose(-1, -2)).float() * sc
        m = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(m, -30000.0)
        p = torch.softmax(s, -1).to(dt)
        return p @ vr

    if T <= 2048:
        timeit("torch matmul+softmax (compiler)", torch_attn, (q, k, v), fl, ref)

    def torch_sdpa(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), is_causal=True, scale=sc)

    timeit("torch SDPA (lite lowering)", torch_sdpa, (q, k, v), fl, ref)
