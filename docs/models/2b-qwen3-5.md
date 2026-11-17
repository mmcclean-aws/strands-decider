# 2B-qwen3.5-v1-2610

The Qwen3.5-based v1 checkpoint. It is the only v1 model that keeps its base
model's vision tower, so it can answer questions about images.

- **Base model:** Qwen3.5-2B-Base
- **Parameters:** 2B
- **Image support:** yes (load with `--vision`)
- **Hugging Face:** [StrandsAgents/strands-decider-2B-qwen3.5-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench (231) | 180 |
| Brier | 0.280 |
| Expected calibration error | 0.072 |
| GPU memory (bf16) | ~6 GB |

Full numbers in [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md#v1-the-released-model).

## Run it

```bash
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8000
```

With images:

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
