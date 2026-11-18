# E2B-gemma4-v1-2610

The smallest Gemma 4 v1 checkpoint: a LoRA adapter (rank-48, three-seed
soup) on `google/gemma-4-E2B-it` plus a typed readout head. Text only — the
base's vision and audio encoders are dropped at load.

- **Base model:** `google/gemma-4-E2B-it` (text decoder only)
- **Parameters:** 2B effective
- **Image support:** no
- **Hugging Face:** [StrandsAgents/strands-decider-E2B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610) (also mirrored at [amazon/strands-decider-E2B-gemma4-v1-2610](https://huggingface.co/amazon/strands-decider-E2B-gemma4-v1-2610))
- **License:** Apache-2.0

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench public, window 4096 | 180 / 231 |
| Brier (JevBench) | 0.313 |
| ECE (JevBench) | 0.053 |
| Decision Index 0.2.1 (window 32,768) | 29.78 |
| Held-out short tasks (n=6,000) | 0.652 |
| ContractNLI (n=1,026) | 0.856 |
| MuSiQue (n=1,199) | 0.913 |
| GPU memory (bf16) | ~9 GB |

Decision Index was measured before the E2B attention fix; multi-question
yes/no answers went through a faulty GPU kernel, so that figure may be off.
Single-question numbers (JevBench, internal sets) are unaffected. Full
per-skill results: [evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

Gemma 4 support lives on `main`; the PyPI package does not have it yet.

```bash
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --port 8000
```

E2B uses eager attention on CUDA; the server picks the right kernel
automatically. See [Inference server](../inference.md).

## When to pick it

- You want the smallest Gemma 4 checkpoint and do not need images.
- You can accept the lowest JevBench score in the Gemma family in exchange
  for the lowest per-token cost.
- Long, multi-step documents and off-distribution rubrics are the known
  weak spots — measure on your own traffic before trusting a threshold.
