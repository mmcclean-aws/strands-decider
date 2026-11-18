# 26B-A4B-gemma4-v1-2610

The largest v1 checkpoint. A mixture of experts: 26B total parameters with
~4B active per token. Highest JevBench score in the lineup. A **soup** of
three seeds trained with `Qwen/Qwen3.5-4B` as teacher and anchored to
`StrandsAgents/strands-decider-2B-hobson-v19`; each layer's LoRA update is
the exact mean of the three seeds (rank-48 adapter), then calibrated. Loads
the text decoder of `gemma-4-26B-A4B-it` only — this model reads text.

- **Base model:** google/gemma-4-26B-A4B-it (MoE)
- **Parameters:** 26B total, 4B active per token
- **Image support:** no
- **License:** Apache-2.0
- **Hugging Face:** [StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610) (also mirrored at [amazon/strands-decider-26B-A4B-gemma4-v1-2610](https://huggingface.co/amazon/strands-decider-26B-A4B-gemma4-v1-2610))

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench public (231, window 4096) | 207/231 |
| JevBench Brier | 0.162 |
| JevBench ECE | 0.051 |
| Decision Index 0.2.1, balanced skill | 48.69 |
| Held-out short tasks (n=6,000) | 0.708 |
| GPU memory (bf16) | ~51 GB |
| Latency, H100 (median) | 245 ms |

Full per-skill numbers (including 12 code sub-tasks ranging 0.562–0.989) are
on the [Hugging Face card](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610#results)
and in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610 \
  --device cuda --device-map auto --port 8000
```

Needs multiple GPUs for a single instance in bf16. See
[Inference server](../inference.md) for the request shape and
[Deployment](../deployment.md) for multi-GPU hosting.

## When to pick it

- You want the highest JevBench score in the lineup.
- You can host a 50 GB+ model (multi-GPU, H100, or quantization).
- Your tasks resemble short-task classification, boardgame reasoning, or
  code review — where the held-out numbers are strongest.
