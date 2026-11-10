"""Request-level parity + latency of the engine on a TP group (every rank runs the same request stream).

CORE_BASE=12 torchrun --nproc_per_node=2 tp_engine_test.py ~/decider/goldens/requests.json
"""
import json
import os
import sys

sys.argv, goldens = sys.argv[:1] + ["bench"], sys.argv[1]  # tp_runner parses argv on import of main(); avoid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_runner as tr  # noqa: E402  (pins the core, inits the process group, defines build/BACKEND/DEV)
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from decider_model import _snapshot  # noqa: E402
from engine import EngineConfig, NeuronDeciderEngine  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402
import device_runner as dr  # noqa: E402

cfg, m = tr.build("fast")
m.tp_chunks = int(os.environ.get("DECIDER_TP_CHUNKS", "4"))
comp = lambda f: torch.compile(f, backend=tr.BACKEND, fullgraph=True, dynamic=False)  # noqa: E731
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 64
full = dr.make_runner(comp(m.forward))
pre = dr.make_prefix_runner(comp(m.prefix_forward))
suf = dr.make_suffix_runner(comp(m.suffix_forward))
ec = EngineConfig(buckets=[512, 1024, 2048, 3072, 4096], batch={512: [1], 1024: [1], 2048: [1], 3072: [1], 4096: [1]},
                  prefix_buckets=[512, 1024, 2048, 2560, 3072, 4096])
runners = {(T, B): full for T, bs in ec.batch.items() for B in bs}
tok = AutoTokenizer.from_pretrained(_snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"))
eng = NeuronDeciderEngine(tok, runners, {k: cfg.temperature_by_kind[k] for k in cfg.temperature_by_kind}, 0.1, ec,
                          prefix_runner=pre, suffix_runner=suf)
gold = json.load(open(goldens))


def probs(a):
    return {"true": a["noul"], "false": 1 - a["noul"]} if a["type"] == "noul" else dict(a["probabilities"])


worst, changed, total = 0.0, 0, 0
for name, g in gold.items():
    if "p30" not in name and "p4" not in name:
        continue
    req = SystemOneRequest.model_validate(g["request"])
    for _ in range(3):
        tm = {}
        resp = eng.evaluate(req, timings=tm).model_dump()
    for q, ref in g["response"]["answers"].items():
        p, r = probs(resp["answers"][q]), probs(ref)
        worst = max(worst, max(abs(p[k] - r[k]) for k in r))
        changed += max(p, key=p.get) != max(r, key=r.get)
        total += 1
    tr.log(f"TP{tr.world} {name:12s} tokens={resp['usage']['input_tokens']:6d} device {tm['device_ms']:7.1f} ms")
tr.log(json.dumps({"tp": tr.world, "max_abs_dp": worst, "changed_answers": changed, "answers": total}))
