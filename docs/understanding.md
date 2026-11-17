# Understanding Strands Decider

Strands Decider is a decision model built on top of a frozen language-model
backbone. This page explains the shape of the system — what it answers, how
the head works, and what the training recipe produces — without going into
the full operator reference. For that, see the design notes in
[docs/designs/](https://github.com/strands-labs/strands-decider/tree/main/docs/designs)
on GitHub.

## A decision, not a generation

A language model generates text token by token. Decider does not. It reads a
**state** (the thing you want a decision about) and a **question** (the
decision to make), then produces a single answer drawn from a space you
defined on the request. The space is one of three types:

- **noul** — yes or no, returned as the probability that the answer is true.
- **choice** — one of a labelled set of options, returned with per-option
  probabilities.
- **score** — a level on an ordered scale, returned with the full
  distribution over levels and an expected level.

Because the output space is fixed by the request, Decider produces exactly
one token. Latency is a single forward pass; cost per decision is a single
output token.

## The pointer head

Under the hood, a Decider checkpoint is a frozen backbone (Qwen3.5-2B or one
of the Gemma 4 sizes) plus a small trained *pointer head*. The head reads
the backbone's final hidden state and scores each option in the question.
Importantly:

- The head holds **no per-option parameters**. Options are defined by the
  request, not baked into the weights, so the same checkpoint can answer a
  new taxonomy the day you add one — no retraining to add a category.
- All three question types come out of the same masked softmax; `noul`,
  `choice`, and `score` differ in how the probabilities are read back, not
  in the forward pass.
- A shared prefix (the state) is encoded once per request, then reused for
  every question in that request. Asking many questions about the same
  state is nearly free.

Calibration is applied on top: the raw probabilities are rescaled so that a
reported confidence of 0.8 means "right about 80% of the time on
held-out data." Calibration is why you can act on a threshold like "accept
above 0.9, confirm 0.6–0.9, escalate below 0.6" without reasoning about the
specific question type.

## Where the training recipe matters

v1 uses a balanced mix of 246,678 rows across 24 public classification
datasets — topic, intent, sentiment, entailment, ordinal ratings, yes/no,
rule chaining — plus new code tasks and policy-application rows. The mix is
deliberate: a corpus drawn from one task teaches memorisation, not binding
to the question. The teacher is `gemma-4-31B-it` for the Gemma family and
`Qwen/Qwen3.5-4B` for the Qwen family; the final checkpoint is a soup of
three seeds, with their LoRA updates and head scores averaged exactly.

If you want to train your own decider — on your own data mix, or against a
different backbone — see [Training a Decider model](training.md).

## Further reading

The full design notes live in the GitHub repository. These are deeper than
the site surface and change as the research moves:

- [docs/designs/architecture.md](https://github.com/strands-labs/strands-decider/blob/main/docs/designs/architecture.md) — the pointer head, question-type read-back, calibration, and shared-prefix cache.
- [docs/designs/inference.md](https://github.com/strands-labs/strands-decider/blob/main/docs/designs/inference.md) — the full operator reference, device kernels, and Hub artefact format.
- [docs/designs/naming.md](https://github.com/strands-labs/strands-decider/blob/main/docs/designs/naming.md) — released model identifiers and version fields.
