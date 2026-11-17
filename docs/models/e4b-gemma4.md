# E4B-gemma4-v1-2610

The mid-size Gemma 4 v1 checkpoint. Loads the text decoder of
`gemma-4-E4B-it`; the vision and audio encoders are dropped at load, so this
model reads text only.

- **Base model:** gemma-4-E4B-it
- **Parameters:** 4B effective
- **Image support:** no
- **Hugging Face:** [StrandsAgents/strands-decider-E4B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench (231) | 187 |
| Brier | 0.243 |
| Held-out short tasks | 0.684 |
| Held-out code tasks | 0.815 |
| Decision Index | 39.42 |
| GPU memory (bf16) | ~15 GB |
| Latency, H100 (median) | 139 ms |

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --port 8000
```

See [Inference server](../inference.md) for the request shape.

## When to pick it

- You want the best accuracy-per-GB in the Gemma family.
- You have a single mid-size GPU (16 – 24 GB).
