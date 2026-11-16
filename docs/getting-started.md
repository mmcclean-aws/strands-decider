This walkthrough takes you from install to using Strands Decider as a model
router inside a Strands agent.

## Walkthrough

### 1. Install the CLI

```bash
pip install strands-decider
```

### 2. Ask a simple choice question

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days." \
  --choice "team=billing,technical,sales"
```

Decider returns the selected option, a probability for every option, and a
confidence score.

### 3. Serve the model on a local HTTP server

```bash
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --port 8000
```

### 4. Curl the server for a response

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

### 5. Use Decider inside a Strands agent

See [Decision models in the Strands SDK](https://strandsagents.com/docs/user-guide/sdk/decision-models/)
for the model-routing agent walkthrough.

## More examples

### Score a support ticket

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Order #4421 never arrived and the tracking link is broken." \
  --score "severity=low,medium,high,critical"
```

### Classify an incoming request (non-agentic)

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v1-2610 \
  --state "Refund request from customer in EU for a damaged item." \
  --choice "queue=refunds,returns,complaints,general"
```

### Ask multiple questions about the same state at once

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
