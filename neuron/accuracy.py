"""Accuracy gate for the Neuron port: golden tensors + the repo's labeled eval sets.

Golden files (tools/make_goldens.py, CPU fp32 HF+PEFT reference) carry per-example token ids,
option positions, labels, the calibrated probabilities and the readout hidden states.
`evaluate(runner, ...)` buckets the examples, runs `runner(input_ids, ans_idx, opt_idx, n_slots,
inv_temp) -> probs [B, K]` and reports, against the golden:
  * max / p99 / mean |dp|, argmax agreement, flips with golden top-2 margin (near-ties),
and against the labels (the repo's eval framework metrics, training/evaluate.py semantics):
  * accuracy overall and per task, ECE (10 bins, server confidence), NLL,
for both the device run and the CPU golden, so the gate is a like-for-like comparison.

PASS criteria (bf16 device vs fp32 CPU reference), based on the bf16 numerics floor measured
by running the SAME code in bf16 on CPU (max |dp| up to 0.016 on 24 rows; device and CPU-bf16
drift agree row by row, so these bounds separate "bf16" from "bug"):
  mean|dp| <= 0.005, p99|dp| <= 0.02, max|dp| <= 0.05,
  every argmax flip has golden top-2 margin < 0.05,
  |accuracy(device) - accuracy(golden)| <= 0.005, |ECE diff| <= 0.01, |NLL diff| <= 0.005.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from collections import defaultdict

import torch

TOL_MEAN_DP = 0.005
TOL_P99_DP = 0.02
TOL_DP = 0.05
TOL_FLIP_MARGIN = 0.05
TOL_ACC = 0.005
TOL_ECE = 0.01
TOL_NLL = 0.005


def derive_confidence(p: list[float], kind: str, ordinal_smoothing: float = 0.1) -> float:
    """The server's confidence (strands_decider.schema); local fallback mirrors it."""
    try:
        from strands_decider.schema import derive_confidence as dc, derive_score_confidence as dsc
        return dsc(p, ordinal_smoothing=ordinal_smoothing) if kind == "score" else dc(p)
    except ImportError:
        pass
    n = len(p)
    if kind == "score":
        mean = sum(i * x for i, x in enumerate(p))
        sigma = math.sqrt(sum(x * (i - mean) ** 2 for i, x in enumerate(p)))
        smax = (n - 1) / 2
        floor = math.sqrt(ordinal_smoothing) if ordinal_smoothing else 0.0
        denom = max(smax - floor, 1e-9)
        return max(0.0, min(1.0, 1 - max(0.0, sigma - floor) / denom))
    return (n * max(p) - 1) / (n - 1) if n > 1 else 1.0


def ece(rows: list[tuple[float, bool]], bins: int = 10) -> float:
    tot = len(rows)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in rows if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if sel:
            e += len(sel) / tot * abs(sum(ok for _, ok in sel) / len(sel) - sum(c for c, _ in sel) / len(sel))
    return e


def bucket_len(L: int, buckets: list[int]) -> int:
    for b in buckets:
        if L <= b:
            return b
    raise ValueError(f"length {L} exceeds largest bucket {buckets[-1]}")


def load_goldens(gdir: str, limit: int = 0) -> tuple[dict, list[dict]]:
    meta = json.load(open(os.path.join(gdir, "meta.json")))
    files = sorted(glob.glob(os.path.join(gdir, "ex*.pt")))
    if limit:
        files = files[:limit]
    recs = []
    for f in files:
        g = torch.load(f)
        g.pop("layers", None)
        g.pop("final_hidden", None)
        g["id"] = int(os.path.basename(f)[2:7])
        recs.append(g)
    return meta, recs


def evaluate(runner, gdir: str, buckets: list[int], batch: dict[int, int], k_pad: int = 32,
             pad_id: int = 248044, limit: int = 0, log=print) -> dict:
    meta, recs = load_goldens(gdir, limit)
    temps = meta["temperature_by_kind"]
    by_b: dict[int, list[dict]] = defaultdict(list)
    for g in recs:
        by_b[bucket_len(len(g["input_ids"]), buckets)].append(g)
    results = []
    lat = defaultdict(list)
    for T, items in sorted(by_b.items()):
        B = batch.get(T, 1)
        for s in range(0, len(items), B):
            chunk = items[s:s + B]
            n = len(chunk)
            ids = torch.full((B, T), pad_id, dtype=torch.int32)
            ans = torch.zeros(B, dtype=torch.int32)
            opt = torch.zeros(B, k_pad, dtype=torch.int32)
            ns = torch.full((B,), 2, dtype=torch.int32)
            it = torch.ones(B, dtype=torch.float32)
            for j, g in enumerate(chunk):
                L = len(g["input_ids"])
                ids[j, :L] = g["input_ids"]
                ans[j] = L - 1
                K = len(g["opt_idx"])
                assert K <= k_pad, f"{K} options > k_pad {k_pad}"
                opt[j, :K] = g["opt_idx"]
                ns[j] = g["n_slots"]
                it[j] = 1.0 / temps[g["kind"]]
            t0 = time.perf_counter()
            probs = runner(ids, ans, opt, ns, it)
            lat[(T, B)].append(time.perf_counter() - t0)
            probs = probs.float().cpu()
            for j, g in enumerate(chunk):
                k = g["n_slots"]
                results.append((g, probs[j, :k].clone()))
        log(f"bucket T={T} B={B}: {len(items)} examples, median {1e3 * sorted(lat[(T, B)])[len(lat[(T, B)]) // 2]:.1f} ms/forward")
    return summarise(results, lat, meta)


def summarise(results, lat, meta) -> dict:
    dps, flips, near_flips = [], [], 0
    dev_rows, ref_rows = [], []
    per_task = defaultdict(lambda: [0, 0, 0])  # n, dev correct, ref correct
    nll_dev = nll_ref = 0.0
    for g, p in results:
        ref = g["probs"][: g["n_slots"]]
        dp = (p - ref).abs().max().item()
        dps.append(dp)
        a_dev, a_ref = int(p.argmax()), int(ref.argmax())
        if a_dev != a_ref:
            top2 = ref.sort(descending=True).values
            margin = float(top2[0] - top2[1])
            flips.append((g["id"], margin))
            near_flips += margin < TOL_FLIP_MARGIN
        lab = g["label"]
        ok_d, ok_r = a_dev == lab, a_ref == lab
        dev_rows.append((derive_confidence(p.tolist(), g["kind"], meta.get("ordinal_smoothing", 0.1)), ok_d))
        ref_rows.append((derive_confidence(ref.tolist(), g["kind"], meta.get("ordinal_smoothing", 0.1)), ok_r))
        t = per_task[g["task"]]
        t[0] += 1
        t[1] += ok_d
        t[2] += ok_r
        nll_dev -= math.log(max(float(p[lab]), 1e-12))
        nll_ref -= math.log(max(float(ref[lab]), 1e-12))
    n = len(results)
    dps_sorted = sorted(dps)
    acc_d = sum(ok for _, ok in dev_rows) / n
    acc_r = sum(ok for _, ok in ref_rows) / n
    out = {
        "n": n,
        "max_dp": dps_sorted[-1],
        "p99_dp": dps_sorted[min(n - 1, int(0.99 * n))],
        "mean_dp": sum(dps) / n,
        "argmax_agreement": 1 - len(flips) / n,
        "flips": flips,
        "flips_not_near_tie": len(flips) - near_flips,
        "accuracy_device": acc_d,
        "accuracy_golden": acc_r,
        "ece_device": ece(dev_rows),
        "ece_golden": ece(ref_rows),
        "nll_device": nll_dev / n,
        "nll_golden": nll_ref / n,
        "per_example": [(g["id"], round((p - g["probs"][: g["n_slots"]]).abs().max().item(), 5)) for g, p in results],
        "per_task": {k: {"n": v[0], "acc_device": v[1] / v[0], "acc_golden": v[2] / v[0]} for k, v in sorted(per_task.items())},
        "latency_ms": {f"T{T}_B{B}": {"median": 1e3 * sorted(v)[len(v) // 2], "p99": 1e3 * sorted(v)[min(len(v) - 1, int(0.99 * len(v)))], "n": len(v)}
                       for (T, B), v in lat.items()},
    }
    out["PASS"] = bool(out["max_dp"] <= TOL_DP and out["p99_dp"] <= TOL_P99_DP and out["mean_dp"] <= TOL_MEAN_DP
                       and out["flips_not_near_tie"] == 0
                       and abs(acc_d - acc_r) <= TOL_ACC and abs(out["ece_device"] - out["ece_golden"]) <= TOL_ECE
                       and abs(out["nll_device"] - out["nll_golden"]) <= TOL_NLL)
    return out


if __name__ == "__main__":
    # CPU self-check: the bf16 re-implementation against the fp32 golden.
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from decider_model import DeciderNeuron, _snapshot, load_state

    ap = argparse.ArgumentParser()
    ap.add_argument("gdir")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--dtype", default="bf16")
    a = ap.parse_args()
    torch.set_num_threads(16)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"), dtype=dt)
    m = DeciderNeuron(cfg, sd, compute_dtype=dt).eval()

    def run(ids, ans, opt, ns, it):
        with torch.inference_mode():
            return m(ids, ans, opt, ns, it)

    r = evaluate(run, a.gdir, [512, 1024, 2048, 4096], {}, limit=a.limit)
    print(json.dumps({k: v for k, v in r.items() if k != "per_task"}, indent=1))
