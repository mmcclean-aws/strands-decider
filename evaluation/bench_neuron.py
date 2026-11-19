"""Latency, throughput and MFU of the compiled forward on one logical NeuronCore (--device neuron).

`bench_local.py` times whole requests through the engine, on any device. This times the device
graph alone, for each (tokens, rows) shape asked for: the first call (compile, or a load from
the Neuron SDK's cache), the blocking latency of a forward including its [B, K] read-back
(median and p99), and the pipelined time per forward with up to 8 forwards in flight, which is
what a busy server sees. MFU is model FLOPs (the projections and MLP at 2 FLOPs per weight per
token, causal attention, the chunked Gated DeltaNet) over the bf16 peak of one logical core at
LNC=2, 667 TFLOP/s per Trainium2 chip over its 4 logical cores.

    NEURON_RT_VISIBLE_CORES=0 python evaluation/bench_neuron.py \\
        StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --shapes 512x1,2048x1,512x4 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from strands_decider.infer import EngineConfig
from strands_decider.neuron_engine import NeuronConfig, load_neuron_engine
from strands_decider.neuron_model import TorsoShape

PEAK_BF16_PER_CORE = 667e12 / 4


def flops(c: TorsoShape, T: int) -> float:
    """Model FLOPs of one T-token row."""
    H, gh, gd = c.hidden, c.gh, c.gd
    n_attn = sum(lt == "full_attention" for lt in c.layer_types)
    n_gdn = len(c.layer_types) - n_attn
    gdn_proj = (4 * gh * gd + 2 * gh) * H + gh * gd * H  # q, k, v, z, b, a; out
    attn_proj = (2 * c.nq + 2 * c.nkv) * c.hd * H + c.nq * c.hd * H  # q, gate, k, v; o
    mlp = 3 * H * c.inter
    dense = 2 * T * (n_gdn * gdn_proj + n_attn * attn_proj + len(c.layer_types) * mlp)
    attention = n_attn * 4 * c.nq * c.hd * T * T / 2  # QK^T and PV over the causal half
    chunk = 128  # per head per token: five C x D products, three D x D
    gdn_core = n_gdn * T * gh * (10 * chunk * gd + 6 * gd * gd)
    return float(dense + attention + gdn_core)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--shapes", default="512x1,1024x1,2048x1,4096x1", help="TxB,...")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    buckets = sorted({t for t, _ in shapes} | {4096})
    batch = {t: sorted({b for tt, b in shapes if tt == t}) for t in buckets}
    started = time.perf_counter()
    engine = load_neuron_engine(
        args.checkpoint, EngineConfig(), NeuronConfig(buckets=buckets, batch=batch)
    )
    print(f"loaded in {time.perf_counter() - started:.0f} s", flush=True)
    runtime = engine.runtime
    shape = runtime.model.cfg
    k_pad = engine.neuron.k_pad
    results = {}
    for T, B in shapes:
        cpu = [
            torch.randint(0, shape.vocab, (B, T), dtype=torch.int32),
            torch.full((B,), T - 1, dtype=torch.int32),
            (torch.arange(k_pad, dtype=torch.int32)[None].repeat(B, 1) * 7 + 10).clamp_max(T - 2),
            torch.full((B,), 4, dtype=torch.int32),
            torch.ones(B),
        ]
        inputs = [x.to(runtime.device) for x in cpu]
        fn = runtime._forward
        with torch.inference_mode():
            t0 = time.perf_counter()
            fn(*inputs).cpu()
            first = time.perf_counter() - t0
            for _ in range(3):
                fn(*inputs).cpu()
            lat = []
            for _ in range(args.iters):
                t1 = time.perf_counter()
                fn(*inputs).cpu()
                lat.append(time.perf_counter() - t1)
            t1 = time.perf_counter()
            for i in range(args.iters):
                y = fn(*inputs)
                if i % 8 == 7:  # the runtime's queue holds a limited number of executions
                    y.cpu()
            y.cpu()
            pipelined = (time.perf_counter() - t1) / args.iters
        lat.sort()
        work = flops(shape, T) * B
        results[f"{T}x{B}"] = row = {
            "first_call_s": round(first, 1),
            "p50_ms": round(1e3 * lat[len(lat) // 2], 2),
            "p99_ms": round(1e3 * lat[min(len(lat) - 1, int(0.99 * len(lat)))], 2),
            "pipelined_ms": round(1e3 * pipelined, 2),
            "mfu": round(work / (pipelined * PEAK_BF16_PER_CORE), 4),
            "tokens_per_s": round(T * B / pipelined),
        }
        print(f"T={T} B={B}: {row}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
