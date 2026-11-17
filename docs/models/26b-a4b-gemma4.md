# 26B-A4B-gemma4-v1-2610

The largest v1 checkpoint. A mixture of experts: 26B total parameters with
~4B active per token. Highest JevBench score in the lineup. Loads the text
decoder of `gemma-4-26B-A4B-it`; the vision and audio encoders are dropped
at load, so this model reads text only.

- **Base model:** gemma-4-26B-A4B-it (MoE)
- **Parameters:** 26B total, 4B active per token
- **Image support:** no
- **Hugging Face:** [StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench (231) | 207 |
| Brier | 0.162 |
| Held-out short tasks | 0.708 |
| Held-out code tasks | 0.859 |
| Decision Index | 48.69 |
| GPU memory (bf16) | ~51 GB |
| Latency, H100 (median) | 245 ms |

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

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
