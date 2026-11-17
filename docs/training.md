# Training a Decider model

Strands Decider ships a single-command recipe that reproduces the published
v1 checkpoint from source. You can run it as-is to get a byte-equivalent
model, or point it at a different backbone or data mix to produce your own.

New to the architecture? Start with [Understanding Strands
Decider](understanding.md).

## What the recipe does

`training/recipe.sh all` runs the six steps the published models go through:

1. **Build the corpora** — convert ~24 public classification datasets into
   `(state, question, answer)` triples. Deterministic; the same build on a
   Mac and on AWS produces the same SHA-256.
2. **Teach-force the teacher** — record the teacher model's answer
   distributions on every row so later training can match them.
3. **Fit the head** — train the small pointer head over the frozen backbone.
   Three seeds in parallel.
4. **Soup** — average the LoRA updates and head scores of the three seeds
   exactly. No test-set selection.
5. **Calibrate** — rescale the raw probabilities against a held-out set so
   the reported confidence matches the true frequency.
6. **Evaluate** — run JevBench and the held-out short / code tasks.

Running all six end-to-end takes about 11 hours on one RTX 3090 or 1 h 10
min on 8× H100.

## Prerequisites

- One Linux or WSL2 host with 1 – 8 NVIDIA GPUs.
- Hugging Face access for the backbone (Qwen3.5 or Gemma 4) and the teacher.
- About 100 GB of free disk for data, intermediate artefacts, and the final
  checkpoint.

The recipe downloads the backbone and teacher automatically. On a clean
box, plan for a few hundred megabytes of network transfer at first use.

## Quickstart

Clone the repository and run the full recipe:

```bash
git clone https://github.com/strands-labs/strands-decider.git
cd strands-decider
./training/recipe.sh all
```

To use all 8 GPUs on an AWS-sized host:

```bash
NGPU=8 FAST=1 ./training/recipe.sh all
```

The output is a Hugging Face-compatible directory under `training/out/`.
Serve it the same way you would a published checkpoint:

```bash
strands-decider serve ./training/out --port 8000
```

## Running one stage at a time

Each step of the recipe is addressable by name, so you can rerun a single
stage without redoing the previous ones:

```bash
./training/recipe.sh data        # Step 1
./training/recipe.sh teach       # Step 2
./training/recipe.sh train       # Step 3
./training/recipe.sh soup        # Step 4
./training/recipe.sh calibrate   # Step 5
./training/recipe.sh eval        # Step 6
```

`training/steps.md` in the repository documents every command, its inputs,
and what it writes.

## Training on your own data

If you want a model tuned to your own classification tasks, replace the
build step's dataset list with your own rows in the same
`(state, question, answer)` shape. See
[`training/data/`](https://github.com/strands-labs/strands-decider/tree/main/data)
for the loader conventions and what the build script expects.

The rest of the recipe is unchanged: teacher-force your corpus, fit the
head, soup, calibrate, evaluate.

## What a "reproduces v1" checkpoint means

A retrain reproduces v1 when it:

- scores between 157 and 177 on the JevBench public set;
- is not significantly different from v1 per task (McNemar, `p ≥ 0.05`);
- reaches at least 0.689 on HelpSteer2.

Numbers from
[`evaluation/results.md`](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md#retraining-on-aws).

## Where to go next

- [`training/README.md`](https://github.com/strands-labs/strands-decider/blob/main/training/README.md) — the full training guide, including hardware notes.
- [`training/steps.md`](https://github.com/strands-labs/strands-decider/blob/main/training/steps.md) — every command with its inputs and outputs.
- [`training/hardware.md`](https://github.com/strands-labs/strands-decider/blob/main/training/hardware.md) — memory, throughput, and length-grouped batching.
- [`training/aws/`](https://github.com/strands-labs/strands-decider/tree/main/training/aws) — one EC2 host over SSM with measured stage timings.
