# E4B-gemma4-v1-2610

The mid-size Gemma 4 v1 checkpoint. A LoRA adapter on `google/gemma-4-E4B-it`
plus a small readout head; the base's vision and audio encoders are dropped at
load, so this model reads text only. The release is a **soup of three seeds**
of the same recipe, then calibrated.

- **Base model:** google/gemma-4-E4B-it (LoRA rank-48 adapter + head)
- **Parameters:** 4B effective
- **Image support:** no
- **Teacher:** Qwen/Qwen3.5-4B, with a frozen-KL anchor to
  `strands-decider-2B-hobson-v19`
- **License:** Apache-2.0
- **Hugging Face:** [StrandsAgents/strands-decider-E4B-gemma4-v1-2610](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610)
  (mirrored at [amazon/strands-decider-E4B-gemma4-v1-2610](https://huggingface.co/amazon/strands-decider-E4B-gemma4-v1-2610))

## Benchmarks

| Metric | Value |
| --- | --- |
| JevBench public (window 4096) | 187 / 231 |
| Brier | 0.243 |
| ECE | 0.037 |
| Decision Index 0.2.1, balanced skill (window 32,768) | 39.42 |
| Held-out short tasks (n=6,000) | 0.684 |
| Held-out code tasks (avg over the [6] code/ set) | ~0.815 |

Full per-skill numbers in the HF card and in
[evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).

## Run it

```bash
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --port 8000
```

Gemma 4 support lives on the code repository's `main` — the PyPI package
doesn't have it yet. See [Inference server](../inference.md) for the request
shape.

## When to pick it

- You want the best accuracy-per-GB in the Gemma 4 family.
- You have a single mid-size GPU (16 – 24 GB).
- Text-only is fine. Pick [2B-qwen3.5](2b-qwen3-5.md) if you need images.

## Known limitations

- Questions are read less than documents: with the state and options fixed, a
  changed question often gets the same answer.
- Long, multi-step documents are the weak spot (JevBench hard tier lags its
  easy tier).
- `score` and `noul` transfer poorly to rubrics and yes/no tasks unlike the
  training mix.
- Calibration is one temperature per primitive, fitted on held-out short
  classification; measure on your own traffic before trusting a threshold.
