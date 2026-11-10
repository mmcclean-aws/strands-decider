"""CPU fp32: shared-prefix path (bucketed, right-padded prefix) == full prompt forward, and == reference engine."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(16)
from transformers import AutoTokenizer
from decider_model import DeciderNeuron, _snapshot, load_state
from engine import NeuronDeciderEngine, EngineConfig
from strands_decider.schema import SystemOneRequest
dec = _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), dec, dtype=torch.float32)
m = DeciderNeuron(cfg, sd, compute_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(dec)
eng = NeuronDeciderEngine(tok, {}, {k: cfg.temperature_by_kind[k] for k in cfg.temperature_by_kind})
gold = json.load(open(sys.argv[1]))
temps = cfg.temperature_by_kind
worst = 0.0
for name in [n for n in gold if n.endswith("q5")][: int(sys.argv[2]) if len(sys.argv) > 2 else 3]:
    req = SystemOneRequest.model_validate(gold[name]["request"])
    rows, n_state = eng.rows(req)
    S_b = ((n_state + 127) // 128) * 128 + 128  # padded prefix bucket (deliberately with padding)
    pid = torch.full((1, S_b), 248044, dtype=torch.int32); pid[0, :n_state] = torch.tensor(rows[0]["ids"][:n_state])
    s_len = torch.tensor([n_state], dtype=torch.int32)
    L_b = max(len(r["ids"]) - n_state for r in rows); L_b = ((L_b + 63) // 64) * 64
    B = len(rows)
    ids = torch.full((B, L_b), 248044, dtype=torch.int32); ans = torch.zeros(B, dtype=torch.int32)
    opt = torch.zeros(B, 32, dtype=torch.int32); ns = torch.zeros(B, dtype=torch.int32); it = torch.zeros(B)
    for j, r in enumerate(rows):
        suf = r["ids"][n_state:]; ids[j, :len(suf)] = torch.tensor(suf); ans[j] = len(suf) - 1
        opt[j, :len(r["opt"])] = torch.tensor([o - n_state for o in r["opt"]]); ns[j] = r["rq"].n_slots; it[j] = 1 / temps[r["rq"].kind]
    with torch.inference_mode():
        caches = m.prefix_forward(pid, s_len)
        p_pre = m.suffix_forward(ids, s_len, caches, ans, opt, ns, it)
        # full-prompt path, one row at a time
        p_full = []
        for j, r in enumerate(rows):
            x = torch.tensor([r["ids"]], dtype=torch.int32)
            p_full.append(m(x, torch.tensor([len(r["ids"]) - 1]), torch.tensor([r["opt"] + [0] * (32 - len(r["opt"]))]), ns[j:j+1], it[j:j+1])[0])
        p_full = torch.stack(p_full)
    ref = gold[name]["response"]["answers"]
    d_full = (p_pre - p_full).abs().max().item()
    d_ref = 0.0
    for j, r in enumerate(rows):
        a = ref[r["name"]]
        pr = [1 - a["noul"], a["noul"]] if a["type"] == "noul" else [a["probabilities"][lab] for lab in r["rq"].slot_labels]
        d_ref = max(d_ref, max(abs(float(p_pre[j, k]) - pr[k]) for k in range(len(pr))))
    worst = max(worst, d_full, d_ref)
    print(f"{name}: state={n_state} S_b={S_b} L_b={L_b} | prefix vs full {d_full:.2e} | prefix vs reference engine {d_ref:.2e}", flush=True)
print("WORST", worst)
