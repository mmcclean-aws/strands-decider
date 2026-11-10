"""Reference SystemOneEngine (CPU fp32, HF+PEFT) answers on the repo's device-parity request set."""
import json, os, sys
REPO = os.path.join(os.path.dirname(__file__), "..", "ext", "strands-decider")
sys.path.insert(0, os.path.abspath(os.path.join(REPO, "src"))); sys.path.insert(0, os.path.abspath(os.path.join(REPO, "evaluation")))
import torch
torch.set_num_threads(16)
from device_parity import requests
from strands_decider.infer import load_engine
eng = load_engine("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", device="cpu")
out = {}
for name, req in requests().items():
    out[name] = {"request": req.model_dump(), "response": eng.evaluate(req).model_dump()}
    print(name, out[name]["response"]["usage"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
