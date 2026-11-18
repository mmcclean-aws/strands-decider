"""Request-level parity: NeuronDeciderEngine vs the reference SystemOneEngine answers (goldens/requests.json).

usage: python test_engine.py REQUEST_GOLDENS.json [--device cpu|neuron]
Prints max |dp| over every answer probability and the number of changed answers.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("goldens")
ap.add_argument("--device", default="cpu")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--no-prefix", action="store_true")
a = ap.parse_args()

if a.device == "neuron":
    os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from decider_model import DeciderNeuron, _snapshot, load_state  # noqa: E402
from engine import EngineConfig, NeuronDeciderEngine  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402

dec = _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
tok = AutoTokenizer.from_pretrained(dec)
gold = json.load(open(a.goldens))
cfg_e = EngineConfig()

if a.device == "cpu":
    torch.set_num_threads(16)
    cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), dec, dtype=torch.float32)
    m = DeciderNeuron(cfg, sd, compute_dtype=torch.float32).eval()

    def mk(T, B):
        def run(*args):
            with torch.inference_mode():
                return m(*args)
        return run

    def pre_run(ids, s_len):
        with torch.inference_mode():
            return m.prefix_forward(ids, s_len)

    def suf_run(*args):
        with torch.inference_mode():
            return m.suffix_forward(*args)
else:
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.recompile_limit = 64
    import device_runner as dr

    cfg, _, m = dr.build(None, True, True)
    fn = dr.compiled(m)
    run_dev = dr.make_runner(fn)

    def mk(T, B):
        return run_dev

    pre_run = dr.make_prefix_runner(torch.compile(m.prefix_forward, backend=dr.BACKEND, fullgraph=True, dynamic=False))
    suf_run = dr.make_suffix_runner(torch.compile(m.suffix_forward, backend=dr.BACKEND, fullgraph=True, dynamic=False))

runners = {(T, B): mk(T, B) for T, bs in cfg_e.batch.items() for B in bs}
cfg_e.use_prefix_cache = not a.no_prefix
eng = NeuronDeciderEngine(tok, runners, {k: cfg.temperature_by_kind.get(k, cfg.temperature) for k in ("noul", "choice", "score")},
                          ordinal_smoothing=0.1, cfg=cfg_e, prefix_runner=pre_run, suffix_runner=suf_run)


def probs(ans):
    if ans["type"] == "noul":
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return dict(ans["probabilities"])


worst, changed, total = 0.0, 0, 0
lat = {}
for name, g in gold.items():
    req = SystemOneRequest.model_validate(g["request"])
    for rep in range(a.repeat):
        tm = {}
        resp = eng.evaluate(req, timings=tm).model_dump()
    lat[name] = tm
    for q, ref in g["response"]["answers"].items():
        p, r = probs(resp["answers"][q]), probs(ref)
        worst = max(worst, max(abs(p[k] - r[k]) for k in r))
        changed += max(p, key=p.get) != max(r, key=r.get)
        total += 1
    print(f"{name:14s} tokens={resp['usage']['input_tokens']:6d} host {tm['host_ms']:6.1f} ms device {tm['device_ms']:7.1f} ms", flush=True)
print(json.dumps({"device": a.device, "max_abs_dp": worst, "changed_answers": changed, "answers": total}))
