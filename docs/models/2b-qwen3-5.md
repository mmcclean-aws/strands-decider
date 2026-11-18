# 2B-qwen3.5-v1-2610

The Qwen3.5-based v1 checkpoint. It is the only v1 model that keeps its base
model's vision tower, so it can answer questions about images. This checkpoint
is a soup of three training runs (seeds 0, 1, 2) of the same recipe.

- **Base model:** Qwen/Qwen3.5-2B-Base (LoRA adapter, rank 48)
- **Parameters:** 2B
- **Image support:** yes (load with `--vision`; no image training, uses the base's vision tower)
- **License:** Apache-2.0
- **Hugging Face:** [StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610) (also mirrored at [amazon/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/amazon/strands-decider-2B-qwen3.5-v1-2610))

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench public (window 4096) | 180/231 tasks |
| JevBench Brier | 0.280 |
| JevBench ECE | 0.072 |
| Decision Index (balanced skill) | 30.82 |
| HotpotQA (never trained on) | 0.832 |
| Held-out short tasks (n=6,000) | 0.653 |
| GPU memory (bf16) | ~6 GB |

Vision (NaturalBench first 300 groups, POPE-adv first 600): 0.776 / 0.872 accuracy, matching hobson-v21 within paired bootstrap 95% CIs.

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md#v1-the-released-model).

## Run it

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
```

With images (needs `transformers >= 5.18`):

```bash
pip install "strands-decider[vision]"
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --vision --port 8000
```

See [Inference server](../inference.md) for the request shape and
[designs/images](../designs/images.md) for the vision request format and
benchmark results.

## When to pick it

- You need image input.
- You want the smallest possible checkpoint.
- You are running on a laptop or on a small CPU/MPS host.
