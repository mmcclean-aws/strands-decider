# 12B-gemma4-v1-2610

The dense Gemma 4 12B v1 checkpoint. Highest text-only accuracy on commodity
GPUs. Loads the text decoder of `gemma-4-12B-it`; the vision and audio
encoders are dropped at load, so this model reads text only.

- **Base model:** gemma-4-12B-it
- **Parameters:** 12B
- **Image support:** no
- **Hugging Face:** [StrandsAgents/strands-decider-12B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench (231) | 200 |
| Brier | 0.173 |
| Held-out short tasks | 0.708 |
| Held-out code tasks | 0.870 |
| Decision Index | 47.14 |
| GPU memory (bf16) | ~23 GB |
| Latency, H100 (median) | 136 ms |

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

```bash
pip install strands-decider
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
