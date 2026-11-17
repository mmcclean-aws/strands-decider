# Models

Strands Decider v1 publishes five checkpoints: one Qwen3.5 base (with a vision
tower) and four Gemma 4 bases spanning 2B – 26B parameters. All five share the
same pointer-head architecture, training recipe, and public API; they differ
in base model, size, and image support.

## v1 model matrix

| Model | Base | Params | JevBench (231) | Brier | GPU memory (bf16) | Latency (H100, median) | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [2B-qwen3.5-v1-2610](models/2b-qwen3-5.md) | Qwen3.5-2B-Base | 2B | 180 | 0.280 | ~6 GB | — | ✓ |
| [E2B-gemma4-v1-2610](models/e2b-gemma4.md) | gemma-4-E2B-it | 2B eff. | 180 | 0.313 | ~9 GB | 102 ms | — |
| [E4B-gemma4-v1-2610](models/e4b-gemma4.md) | gemma-4-E4B-it | 4B eff. | 187 | 0.243 | ~15 GB | 139 ms | — |
| [12B-gemma4-v1-2610](models/12b-gemma4.md) | gemma-4-12B-it | 12B | 200 | 0.173 | ~23 GB | 136 ms | — |
| [26B-A4B-gemma4-v1-2610](models/26b-a4b-gemma4.md) | gemma-4-26B-A4B-it | 26B MoE (4B active) | 207 | 0.162 | ~51 GB | 245 ms | — |

JevBench scores and latency come from
[evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).
Lower Brier is better; higher JevBench is better.

## Which model should I pick?

- **Fit on a laptop or small instance, or need image input:** 2B-qwen3.5.
- **Best accuracy-per-GB on a single GPU:** E4B-gemma4.
- **Best overall accuracy on commodity GPUs:** 12B-gemma4.
- **Best accuracy, mixture-of-experts serving:** 26B-A4B-gemma4.
- **Smallest Gemma option:** E2B-gemma4.

Every checkpoint speaks the same `--state` / `--choice` / `--noul` /
`--score` interface and the same HTTP contract, so switching models is a
one-line change in [`strands-decider serve`](inference.md).

## Naming

Model IDs follow `strands-decider-{size}-{base}-v{N}-{YYMM}`. See
[designs/naming](designs/naming.md) for the full rule, including what
changes when `N` increments.
