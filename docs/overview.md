System One models understand natural language and make a decision based on
that information. They do not generate text — they return a typed answer from a
known output space along with calibrated confidence. Because the output space
is fixed before the request and the model produces a single output token,
Decider is cheaper and faster than asking an LLM to decide.

Strands Decider is a System One model. It supports three question types:
**Choice**, **Noul** (yes/no), and **Score**.

## When to use Decider

Use Decider when the output space is known before the request and the same
kind of decision happens repeatedly. Good fits include:

- routing a request to a model, tool, team, or queue;
- checking tool arguments, grounding, safety, or policy compliance;
- scoring model responses for evaluation;
- classifying documents, support tickets, or images into fixed categories;
- handling routine decisions in a hybrid agent while an LLM handles harder,
  open-ended work.

Use an LLM when the task needs generated text, conversation, coding,
summarization, or complex reasoning. Decider classifies and scores; it does
not generate a response.

## Question types

### Choice

Given a list of options, pick one.

> **Example**
>
> _State:_ "What color is this fruit: Banana"
>
> _Choices:_ Red, Yellow, or Blue?
>
> _Strands Decider:_ **Yellow**

Try it yourself:

```bash
pip install strands-decider

strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "What color is this fruit: Banana" \
  --choice "color=red,yellow,blue"
```

### Noul (yes / no)

Ask a yes-or-no question. Decider returns the probability that the answer is
true.

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "urgent=is this request urgent?"
```

### Score

Rate something on an ordered scale. Decider returns the expected score, the
distribution across levels, and confidence.

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "The checkout button is missing on mobile." \
  --score "severity=low,medium,high,critical"
```

## Image support

Strands Decider can also answer questions about images when loaded with
`--vision`. See [Images](images.md) for the request format and supported
models.

```bash
pip install "strands-decider[vision]"
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --vision --port 8000
```

Example — classify a screenshot:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --image screen.png \
  --choice "status=paid,retry,cart,empty"
```

## Good use cases for System One models

- **Replacing decision making from LLMs with Decider models.** LLMs are great
  at making decisions based on some information, but they can be costly and
  slow. Switching to a System One model reduces token cost and response
  latency. You no longer pay for idle compute while waiting for an LLM —
  Decider often responds in under one second, and because it produces a
  single output token, you do not pay for output tokens. See [Amazon saves
  millions replacing Sonnet with Strands
  Decider](https://strandsagents.com/blog/amazon-saves-millions-replacing-sonnet-with-strands-decider/).
- **Model routing.** Use a System One model to route an incoming question to
  the correctly scoped model — Haiku for easy problems, Opus for hard ones.
  See [Decision models in the Strands
  SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/).
- **Interventions.** Use a System One model to decide whether a human should
  intervene in an agent's action. If an agent might perform an action that
  cannot be deterministically identified as harmful, Decider can flag it for
  human review.
- **Filtering.** Use a System One model inside a graph to decide whether a
  request should be processed by an LLM (expensive) or skipped to save cost.
- **Non-agentic classification.** Score support tickets, triage incoming
  email, or tag documents into fixed categories without wiring up an agent.

## Interested in learning or trying more?

- [Strands Decider on GitHub](https://github.com/strands-labs/strands-decider) — design and inference notes
- [Train a model yourself](https://github.com/strands-labs/strands-decider/tree/main/training)
