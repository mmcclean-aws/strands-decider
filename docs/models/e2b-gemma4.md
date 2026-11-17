# E2B-gemma4-v1-2610

The smallest Gemma 4 v1 checkpoint. Loads the text decoder of
`gemma-4-E2B-it`; the vision and audio encoders are dropped at load, so this
model reads text only.

- **Base model:** gemma-4-E2B-it
- **Parameters:** 2B effective
- **Image support:** no
- **Hugging Face:** [StrandsAgents/strands-decider-E2B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench (231) | 180 |
| Brier | 0.313 |
| Held-out short tasks | 0.652 |
| Held-out code tasks | 0.773 |
| Decision Index | 29.78 |
| GPU memory (bf16) | ~9 GB |
| Latency, H100 (median) | 102 ms |

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --port 8000
```

E2B uses eager attention on CUDA; the server picks the right kernel
automatically. See [Inference server](../inference.md).

## When to pick it

- You want a small Gemma 4 checkpoint and do not need images.
- You are fine with the lowest per-token cost in the Gemma family and can
  accept the lowest JevBench score of the four.
