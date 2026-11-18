# 12B-gemma4-v1-2610

The dense Gemma 4 12B v1 checkpoint: a LoRA adapter on `google/gemma-4-12B-it`
plus a small readout head. Highest text-only accuracy of the v1 line. The
checkpoint is a soup of three seeds, each distilled from a `Qwen/Qwen3.5-4B`
teacher and anchored to `strands-decider-2B-hobson-v19`.

- **Base model:** google/gemma-4-12B-it
- **Parameters:** 12B (rank-48 LoRA adapter + head)
- **Image support:** no (text decoder only)
- **License:** Apache-2.0
- **Hugging Face:** [StrandsAgents/strands-decider-12B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench public, window 4096 | 200/231 |
| Brier | 0.173 |
| ECE | 0.061 |
| Decision Index 0.2.1 (window 32,768) | 47.14 |
| Held-out short tasks (n=6,000) | 0.708 |
| Held-out MuSiQue (n=1,199) | 0.936 |
| Held-out HotpotQA (n=959) | 0.911 |

Full per-task table in the [model card](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610)
and in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

Gemma 4 support lives on `main`, not yet on PyPI:

```bash
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
strands-decider serve StrandsAgents/strands-decider-12B-gemma4-v1-2610 --port 8000
```

Split across multiple GPUs if needed:

```bash
strands-decider serve StrandsAgents/strands-decider-12B-gemma4-v1-2610 \
  --device cuda --device-map auto --port 8000
```

See [Inference server](../inference.md) for the request shape.

## When to pick it

- You have a 24 GB GPU (A10G, L4, 3090/4090, A100 24 GB) or larger.
- You want the best dense-model accuracy without stepping up to a mixture of
  experts.
- You're on text-only inputs — this checkpoint drops Gemma 4's vision and
  audio encoders at load.

## Limitations

- Long multi-step documents are the weak spot: JevBench's hard tier scores
  far below its easy tier.
- `score` / `noul` transfer poorly to rubrics and yes/no tasks unlike the
  training mix — train on a rubric like yours.
- Calibration is one temperature per primitive, fitted on held-out short
  classification; measure on your own traffic before trusting a threshold.
