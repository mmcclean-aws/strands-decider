"""Run the decider on one NeuronCore (LNC=2) with the SDK 2.33 PyTorch-native backend.

modes:
  layers   reduced-layer bring-up: first N layers on device vs the same layers on CPU fp32
  goldens  full model vs CPU fp32 goldens + labeled eval accuracy (neuron/accuracy.py gate)
  bench    latency / throughput / MFU per (T, B) bucket

env (set by run scripts): VLLM_PLUGINS=none VLLM_NEURON_BACKEND=neuron_native
  NEURON_LOGICAL_NC_CONFIG=2 NEURON_RT_ROOT_COMM_ID=127.0.0.1:<port> NEURON_RT_VISIBLE_CORES=<core>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch  # noqa: E402
import libtorch_neuronx_lite  # noqa: E402,F401  (registers the "neuron" device)
from libtorch_neuronx_lite.compile.native_backend import register  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decider_model import DeciderNeuron, _snapshot, load_state  # noqa: E402

BACKEND = register()
DEV = torch.device("neuron", 0)


def fence():
    """Block until all queued device work is done (execution is async and in order)."""
    if not hasattr(fence, "fn"):
        fence.t = torch.zeros(1, device=DEV)
        fence.fn = torch.compile(lambda t: t + 1, backend=BACKEND, fullgraph=True, dynamic=False)
    fence.fn(fence.t).cpu()


def build(n_layers: int | None, use_nki_gdn: bool, use_nki_attn: bool, dtype=torch.bfloat16, use_fast: bool | None = None):
    if use_fast is None:
        use_fast = bool(int(os.environ.get("DECIDER_FAST", "1")))
    cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"), dtype=dtype)
    if n_layers:
        cfg.layer_types = cfg.layer_types[:n_layers]
        cfg.n_layers = n_layers
    from nki_ops import NkiMixers

    mix = NkiMixers(DEV)
    gdn = mix.gdn if use_nki_gdn else None
    attn = mix.attn if use_nki_attn else None
    if os.environ.get("DECIDER_GDN", "fast") == "fast" and use_nki_gdn:
        from nki_ops import FastGdn

        gdn = FastGdn(DEV)
    m = DeciderNeuron(cfg, sd, gdn_impl=gdn, attn_impl=attn, compute_dtype=dtype, use_fast=use_fast).eval()
    return cfg, sd, m.to(DEV)


def compiled(m):
    return torch.compile(m.forward, backend=BACKEND, fullgraph=True, dynamic=False)


def make_prefix_runner(fn):
    def run(ids, s_len):
        with torch.inference_mode():
            return fn(ids.to(DEV), s_len.to(DEV))  # caches stay on device
    return run


def make_suffix_runner(fn):
    def run(ids, s_len, caches, ans, opt, ns, it):
        with torch.inference_mode():
            return fn(ids.to(DEV), s_len.to(DEV), caches, ans.to(DEV), opt.to(DEV), ns.to(DEV), it.to(DEV)).cpu()
    return run


def make_runner(fn):
    def run(ids, ans, opt, ns, it):
        with torch.inference_mode():
            out = fn(ids.to(DEV), ans.to(DEV), opt.to(DEV), ns.to(DEV), it.to(DEV))
            return out.cpu()
    return run


def mode_layers(a):
    """Device torso (first N layers, NKI mixers) vs CPU fp32 reference on random-ish real tokens."""
    import torch.nn.functional as F
    from decider_model import rope_cos_sin

    cfg, sd, m = build(a.layers, not a.no_nki_gdn, not a.no_nki_attn)
    cpu_cfg, cpu_sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"), dtype=torch.float32)
    cpu_cfg.layer_types = cpu_cfg.layer_types[: a.layers]
    ref = DeciderNeuron(cpu_cfg, cpu_sd, compute_dtype=torch.float32).eval()
    g = torch.load(a.golden) if a.golden else None
    T = a.T
    ids = (g["input_ids"][:T] if g is not None else torch.randint(0, 248000, (T,)))[None].repeat(a.B, 1).to(torch.int32)

    def torso_fn(x):
        return m.torso(x)

    f = torch.compile(torso_fn, backend=BACKEND, fullgraph=True, dynamic=False)
    t0 = time.time()
    with torch.inference_mode():
        out = f(ids.to(DEV)).cpu().float()
    print(f"compile+first run {time.time() - t0:.1f}s", flush=True)
    with torch.inference_mode():
        r = ref.torso(ids)
    rel = ((out - r).norm() / r.norm()).item()
    cos = F.cosine_similarity(out.flatten(1), r.flatten(1), dim=1)
    print(f"layers={a.layers} T={T} B={a.B} rel={rel:.3e} cos={cos.min().item():.6f}", flush=True)
    # timing
    for _ in range(3):
        y = f(ids.to(DEV))
    fence()
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        y = f(ids.to(DEV))
    y.cpu()
    print(f"torso[{a.layers}] latency {1e3 * (time.perf_counter() - t0) / n:.2f} ms/forward", flush=True)


def mode_goldens(a):
    from accuracy import evaluate

    _, _, m = build(None, not a.no_nki_gdn, not a.no_nki_attn)
    fn = compiled(m)
    buckets = [int(x) for x in a.buckets.split(",")]
    batch = {int(k): int(v) for k, v in (x.split(":") for x in a.batch.split(","))} if a.batch else {}
    r = evaluate(make_runner(fn), a.gdir, buckets, batch, k_pad=a.k_pad, limit=a.limit,
                 log=lambda s: print(s, flush=True))
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_task", "per_example")}, indent=1))
    print("per_example(first 12):", sorted(r["per_example"])[:12])
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)


def mode_bench(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    _, _, m = build(None, not a.no_nki_gdn, not a.no_nki_attn)
    fn = compiled(m)
    from cost_model import Model

    model = Model()
    res = {}
    for spec in a.shapes.split(","):
        T, B = (int(x) for x in spec.split("x"))
        ids = torch.randint(0, 248000, (B, T), dtype=torch.int32)
        ans = torch.full((B,), T - 1, dtype=torch.int32)
        opt = (torch.arange(a.k_pad, dtype=torch.int32)[None].repeat(B, 1) * 7 + 10).clamp_max(T - 2)
        ns = torch.full((B,), 4, dtype=torch.int32)
        it = torch.ones(B)
        dev_in = [x.to(DEV) for x in (ids, ans, opt, ns, it)]
        t0 = time.time()
        with torch.inference_mode():
            fn(*dev_in).cpu()
        tc = time.time() - t0
        with torch.inference_mode():
            for _ in range(3):
                fn(*dev_in).cpu()
            lat = []
            for _ in range(a.iters):
                t1 = time.perf_counter()
                fn(*dev_in).cpu()  # blocking request latency incl. the [B,K] read-back
                lat.append(time.perf_counter() - t1)
            # pipelined throughput (async queue)
            t1 = time.perf_counter()
            for i in range(a.iters):
                y = fn(*dev_in)
                if i % 8 == 7:
                    y.cpu()
            y.cpu()
            thr = (time.perf_counter() - t1) / a.iters
        lat.sort()
        p50 = lat[len(lat) // 2]
        flops = model.flops(T)["total"] * B
        mfu = flops / (thr * 667e12 / 4)
        res[spec] = {"compile_s": tc, "p50_ms": 1e3 * p50, "p99_ms": 1e3 * lat[min(len(lat) - 1, int(0.99 * len(lat)))],
                     "pipelined_ms": 1e3 * thr, "mfu_pipelined": mfu, "mfu_latency": flops / (p50 * 667e12 / 4),
                     "tok_per_s_core": T * B / thr}
        print(f"T={T} B={B} compile {tc:.0f}s p50 {1e3 * p50:.2f} ms pipelined {1e3 * thr:.2f} ms "
              f"MFU {mfu:.1%} ({T * B / thr:.0f} tok/s/core)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["layers", "goldens", "bench"])
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--golden", default="")
    ap.add_argument("--gdir", default="")
    ap.add_argument("--buckets", default="512,1024,2048,4096")
    ap.add_argument("--batch", default="")
    ap.add_argument("--k-pad", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shapes", default="512x1,1024x1,2048x1,4096x1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-nki-gdn", action="store_true")
    ap.add_argument("--no-nki-attn", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    {"layers": mode_layers, "goldens": mode_goldens, "bench": mode_bench}[a.mode](a)
