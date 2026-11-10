"""CPU fp32 golden reference for the Strands Decider port.

Uses the repo's own evaluation path (training.evaluate: SystemOneCollator(train=False), no
option shuffling) on the committed labeled eval sets, one example per forward (no padding),
and saves for every example:
  input_ids, opt_idx, n_slots, kind, task, label,
  raw logits, calibrated probs (temperature_by_kind), h_answer [2048], h_options [K, 2048]
(the final-norm torso output at the readout positions, fp32).
For a few examples it also saves every layer's output (layer-wise debugging goldens).

usage: python make_goldens.py OUT_DIR [--limit N] [--layerwise-ids 0,5,9]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "ext", "strands-decider")
sys.path.insert(0, os.path.abspath(REPO))
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "src")))

from strands_decider import cpu_kernels  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel, gather_options, masked_log_softmax  # noqa: E402
from training.data.collate import CollatorConfig, SystemOneCollator  # noqa: E402
from training.data.format import load_examples  # noqa: E402

EVAL_FILES = [
    "data/synthetic/generated_v18_eval.jsonl",
    "data/synthetic/generated_v16_eval.jsonl",
    "data/synthetic/adequacy_gen_eval.jsonl",
    "data/synthetic/flips_v20_eval.jsonl",
]
CKPT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layerwise-ids", default="")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th example from i")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    cpu_kernels.install()

    model = StrandsDeciderModel.load(CKPT)
    model.torso.to(torch.float32)
    model.eval()
    cfg = model.config
    temps = {k: float(cfg.temperature_by_kind.get(k, cfg.temperature)) for k in ("noul", "choice", "score")}
    examples = load_examples([os.path.join(REPO, f) for f in EVAL_FILES])
    if args.limit:
        examples = examples[: args.limit]
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=cfg.max_length), train=False)
    layerwise = {int(x) for x in args.layerwise_ids.split(",") if x}
    si, sn = (int(x) for x in args.shard.split("/"))

    meta = {"ckpt": CKPT, "max_length": cfg.max_length, "temperature_by_kind": temps,
            "ordinal_smoothing": cfg.ordinal_smoothing, "n_examples": len(examples)}
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    layer_outs: list[torch.Tensor] = []

    def hook(_m, _i, o):
        layer_outs.append((o[0] if isinstance(o, tuple) else o).detach().float().clone())

    base = model.torso.get_base_model() if hasattr(model.torso, "get_base_model") else model.torso
    t0 = time.time()
    for idx, ex in enumerate(examples):
        if idx % sn != si:
            continue
        path = os.path.join(args.out, f"ex{idx:05d}.pt")
        if os.path.exists(path):
            continue
        batch = coll([ex])
        ids, mask = batch["input_ids"], batch["attention_mask"]
        assert int(mask.sum()) == ids.shape[1], "batch of one must be unpadded"
        handles = []
        if idx in layerwise:
            layer_outs.clear()
            handles = [l.register_forward_hook(hook) for l in base.layers]
        with torch.inference_mode():
            hidden = model.torso(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state.float()
            h_answer = hidden[:, -1]
            h_opts = gather_options(hidden, batch["opt_idx"])
            logits = model.head(h_answer, h_opts)
            n_slots = batch["n_slots"]
            probs = masked_log_softmax(logits / temps[ex.kind], n_slots).exp()
        for h in handles:
            h.remove()
        rec = {
            "input_ids": ids[0].to(torch.int32),
            "opt_idx": batch["opt_idx"][0].to(torch.int32),
            "n_slots": int(n_slots[0]),
            "kind": ex.kind,
            "task": ex.task,
            "label": int(batch["labels"][0]),
            "logits": logits[0].float().clone(),
            "probs": probs[0].float().clone(),
            "h_answer": h_answer[0].float().clone(),
            "h_options": h_opts[0].float().clone(),
        }
        if idx in layerwise:
            rec["layers"] = torch.stack([x[0] for x in layer_outs])  # [24, L, 2048] pre-final-norm
            rec["final_hidden"] = hidden[0].clone()
        torch.save(rec, path)
        n = ids.shape[1]
        print(f"[{idx}] L={n} kind={ex.kind} pred={int(probs[0].argmax())} label={rec['label']} "
              f"t={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
