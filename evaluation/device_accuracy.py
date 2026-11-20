"""Does a device keep the model's accuracy? The labeled eval sets, answered on a reference device
and on each other device, then compared row by row and against the labels.

`device_parity.py` checks a few requests for changed answers. This checks the committed
labeled eval sets (1,019 rows: generated documents v16 and v18, answer adequacy, the v20 flips),
one question per request through the serving engine, so each device answers exactly as
`serve` would. For each device it reports, against the reference device's answers:

  - mean, p99 and max |dp|, the largest change in any option's probability on a row;
  - the rows whose chosen option changed, with the reference's margin between its top two
    options (a near-tie flips on rounding alone);

and, against the labels, for both devices: accuracy, ECE (10 bins, over the confidence the
server returns) and NLL.

A device PASSES when it is within the bf16 floor of a CPU fp32 reference: mean |dp| <= 0.005,
p99 <= 0.02, max <= 0.05, every changed answer at a reference margin under 0.05, and accuracy,
ECE and NLL within 0.005, 0.01 and 0.005. The floor was measured by running the same model in
bf16 on the CPU against fp32 (max |dp| 0.016 on 24 rows); a device outside it has a bug, not
rounding.

    python evaluation/device_accuracy.py StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \\
        --device cpu --device neuron --out accuracy.json
    # reuse a reference run: its answers are in the --out file
    python evaluation/device_accuracy.py CKPT --reference accuracy.json --device neuron

The reference is the first --device, or the answers saved by an earlier run (--reference).
Devices load one after another, and each engine is released before the next loads.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # training.data

from strands_decider.confidence import derive_confidence  # noqa: E402
from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.schema import Answer, SystemOneRequest  # noqa: E402
from training.data.format import Example, load_examples  # noqa: E402

EVAL_FILES = [
    "data/synthetic/generated_v18_eval.jsonl",
    "data/synthetic/generated_v16_eval.jsonl",
    "data/synthetic/adequacy_gen_eval.jsonl",
    "data/synthetic/flips_v20_eval.jsonl",
]
GATE = {
    "mean_dp": 0.005,
    "p99_dp": 0.02,
    "max_dp": 0.05,
    "flip_margin": 0.05,
    "accuracy": 0.005,
    "ece": 0.01,
    "nll": 0.005,
}
# Answers are rounded to 4 decimals, so a probability of 0 means "below 5e-5".
_P_FLOOR = 5e-5


def probabilities(answer: Answer, example: Example) -> list[float]:
    """An answer's probabilities in the example's canonical option order (its label's order):
    noul [false, true], a choice by option name, a score by level."""
    if answer.type == "noul":
        return [1 - answer.noul, answer.noul]
    if answer.type == "choice":
        return [answer.probabilities[name] for name, _ in example.options]
    return [answer.probabilities[str(level)] for level in range(len(example.options))]


def requests(examples: list[Example]) -> list[SystemOneRequest]:
    return [SystemOneRequest(state=ex.state, questions={"q": ex.to_question()}) for ex in examples]


def answer_rows(checkpoint: str, device: str, examples: list[Example]) -> list[list[float]]:
    engine = load_engine(checkpoint, device=device)
    rows, started = [], time.perf_counter()
    for i, (ex, request) in enumerate(zip(examples, requests(examples), strict=True)):
        rows.append(probabilities(engine.evaluate(request).answers["q"], ex))
        if (i + 1) % 100 == 0:
            print(f"[{device}] {i + 1}/{len(examples)} in {time.perf_counter() - started:.0f} s")
    del engine
    gc.collect()
    return rows


def ece(rows: list[tuple[float, bool]], bins: int = 10) -> float:
    total, err = len(rows), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in rows if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if sel:
            mean_c = sum(c for c, _ in sel) / len(sel)
            acc = sum(ok for _, ok in sel) / len(sel)
            err += len(sel) / total * abs(acc - mean_c)
    return err


def label_metrics(examples: list[Example], rows: list[list[float]]) -> dict[str, float]:
    """Accuracy, ECE over the confidence the server returns, NLL; against the labels."""
    graded, nll = [], 0.0
    for ex, p in zip(examples, rows, strict=True):
        ok = max(range(len(p)), key=p.__getitem__) == ex.label
        graded.append((derive_confidence(p), ok))
        nll -= math.log(max(p[ex.label], _P_FLOOR))
    n = len(rows)
    return {"accuracy": sum(ok for _, ok in graded) / n, "ece": ece(graded), "nll": nll / n}


def compare(examples: list[Example], reference: list[list[float]], rows: list[list[float]]) -> dict:
    """One device's rows against the reference's: |dp|, changed answers, label metrics, PASS."""
    dps, flips = [], []
    for i, (ref, got) in enumerate(zip(reference, rows, strict=True)):
        dps.append(max(abs(a - b) for a, b in zip(ref, got, strict=True)))
        if max(range(len(ref)), key=ref.__getitem__) != max(range(len(got)), key=got.__getitem__):
            top = sorted(ref, reverse=True)
            flips.append({"row": i, "task": examples[i].task, "margin": round(top[0] - top[1], 4)})
    n, ordered = len(dps), sorted(dps)
    ref_m, dev_m = label_metrics(examples, reference), label_metrics(examples, rows)
    out = {
        "rows": n,
        "mean_dp": sum(dps) / n,
        "p99_dp": ordered[min(n - 1, int(0.99 * n))],
        "max_dp": ordered[-1],
        "changed": flips,
        "reference": ref_m,
        "device": dev_m,
    }
    out["pass"] = bool(
        out["mean_dp"] <= GATE["mean_dp"]
        and out["p99_dp"] <= GATE["p99_dp"]
        and out["max_dp"] <= GATE["max_dp"]
        and all(f["margin"] < GATE["flip_margin"] for f in flips)
        and all(abs(dev_m[k] - ref_m[k]) <= GATE[k] for k in ("accuracy", "ece", "nll"))
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--device", action="append", default=[], help="repeatable")
    ap.add_argument("--reference", default=None, help="an earlier --out file: its reference rows")
    ap.add_argument("--limit", type=int, default=0, help="the first N rows only")
    ap.add_argument("--out", default=None, help="write every device's rows and the comparison here")
    args = ap.parse_args()
    examples = load_examples([os.path.join(REPO, f) for f in EVAL_FILES])
    if args.limit:
        examples = examples[: args.limit]
    devices = list(args.device)
    if args.reference:
        with open(args.reference, encoding="utf-8") as fh:
            saved = json.load(fh)
        ref_device = saved["reference_device"]
        results = {ref_device: saved["rows"][ref_device][: len(examples)]}
    else:
        if len(devices) < 2:
            ap.error("give at least two --device values, or --reference and one --device")
        ref_device = devices.pop(0)
        results = {ref_device: answer_rows(args.checkpoint, ref_device, examples)}

    def save(report: dict) -> None:
        # After every device, so a device that fails leaves the reference's rows to reuse.
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(
                    {"reference_device": ref_device, "report": report, "rows": results},
                    fh,
                    indent=1,
                )

    save({})
    for device in devices:
        results[device] = answer_rows(args.checkpoint, device, examples)
        save({})
    report = {d: compare(examples, results[ref_device], results[d]) for d in devices}
    for device, r in report.items():
        print(
            f"{device} vs {ref_device} on {r['rows']} rows: |dp| mean {r['mean_dp']:.4f} "
            f"p99 {r['p99_dp']:.4f} max {r['max_dp']:.4f}; {len(r['changed'])} changed "
            f"(margins {[f['margin'] for f in r['changed']]}); accuracy "
            f"{r['device']['accuracy']:.4f} / {r['reference']['accuracy']:.4f}, ECE "
            f"{r['device']['ece']:.4f} / {r['reference']['ece']:.4f}, NLL "
            f"{r['device']['nll']:.4f} / {r['reference']['nll']:.4f}: "
            + ("PASS" if r["pass"] else "FAIL")
        )
    save(report)


if __name__ == "__main__":
    main()
