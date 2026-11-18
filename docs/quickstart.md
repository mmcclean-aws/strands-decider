# Quickstart

This walkthrough takes you from install to using Strands Decider as a model
router inside a Strands agent.

## Walkthrough

**1. Install the CLI**

```bash
pip install strands-decider
```

**2. Ask a simple choice question**

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --choice "team=billing,technical,sales"
```

<details>
<summary>Sample output</summary>

```
choice_0 -> technical (confidence 0.697)
    technical                0.798
    billing                  0.195
    sales                    0.007
```

</details>

Decider returns the selected option, a probability for every option, and a
confidence score.

**3. Serve the model on a local HTTP server**

```bash
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --port 8000
```

**4. Curl the server for a response**

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "billing and payments",
        "technical": "bugs and outages",
        "sales": "pre-sales questions"
      }
    }
  }
}'
```

<details>
<summary>Sample output</summary>

```json
{
  "model": "strands-decider-E2B-gemma4-v1-2610",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.8505,
        "technical": 0.1469,
        "sales": 0.0026
      },
      "confidence": 0.7757
    }
  },
  "usage": { "input_tokens": 82, "output_tokens": 1 },
  "latency_ms": 367.92
}
```

</details>

**5. Use Decider inside a Strands agent**

See [Decision models in the Strands SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/)
for the model-routing agent walkthrough.

## Ask about an image

Install the vision extra and load a vision-capable model with `--vision`:

```bash
pip install "strands-decider[vision]"
strands-decider ask StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --state "What is the state of this checkout screen?" \
  --image screen.png \
  --choice "status=paid,retry,cart,empty"
```

<details>
<summary>Sample output</summary>

Run against this empty-cart screenshot:

![Empty cart screenshot](/decider-assets/screen.png)

```
choice_0 -> empty (confidence 0.887)
    empty                    0.915
    cart                     0.081
    paid                     0.003
    retry                    0.001
```

</details>

From the HTTP API, pass base64 images in the `images` field. See
[designs/images](designs/images.md) for the request shape, limits, and the
vision benchmark results.

## More examples

**Score a support ticket**

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Order #4421 never arrived and the tracking link is broken." \
  --score "severity=low,medium,high,critical"
```

<details>
<summary>Sample output</summary>

```
score_0 score = 1.71 (confidence 0.357)
    0: low                                      0.185
    1: medium                                   0.216
    2: high                                     0.307
    3: critical                                 0.292
```

</details>

**Ask a yes-or-no (noul) question**

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "urgent=is this request urgent?"
```

<details>
<summary>Sample output</summary>

```
noul_0 noul = 0.821
```

</details>

**Classify an incoming request (non-agentic)**

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Refund request from customer in EU for a damaged item." \
  --choice "queue=refunds,returns,complaints,general"
```

<details>
<summary>Sample output</summary>

```
choice_0 -> refunds (confidence 0.840)
    refunds                  0.880
    returns                  0.080
    complaints               0.035
    general                  0.005
```

</details>

**Ask multiple questions about the same state at once**

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Refund request from customer in EU for a damaged item.",
  "questions": {
    "queue":    {"type": "choice", "instructions": "Which queue handles this?",
                 "criteria": {"refunds": "refund request", "returns": "return request",
                              "complaints": "complaint", "general": "other"}},
    "urgent":   {"type": "noul",   "instructions": "Is this time-sensitive?"},
    "severity": {"type": "score",  "instructions": "How severe is the complaint?",
                 "criteria": ["minor", "moderate", "major"]}
  }
}'
```

<details>
<summary>Sample output</summary>

```json
{
  "model": "strands-decider-E2B-gemma4-v1-2610",
  "answers": {
    "queue": {
      "type": "choice",
      "choice": "refunds",
      "probabilities": {
        "refunds": 0.8508,
        "returns": 0.1098,
        "complaints": 0.0333,
        "general": 0.006
      },
      "confidence": 0.8011
    },
    "urgent": { "type": "noul", "noul": 0.4361 },
    "severity": {
      "type": "score",
      "score": 1.0653,
      "legend": { "0": "minor", "1": "moderate", "2": "major" },
      "probabilities": { "0": 0.2589, "1": 0.4168, "2": 0.3243 },
      "confidence": 0.3497
    }
  },
  "usage": { "input_tokens": 210, "output_tokens": 3 },
  "latency_ms": 1531.41
}
```

</details>

## Next steps

Wire Decider into a Strands agent as the decision model: see
[Decision models in the Strands SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/)
for the SDK getting-started walkthrough.
