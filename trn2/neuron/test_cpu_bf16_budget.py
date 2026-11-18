"""How far does a bf16 forward (same code) drift from the fp32 golden? Sets Neuron tolerances."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from decider_model import DeciderNeuron, _snapshot, load_state
torch.set_num_threads(int(os.environ.get("THREADS", "8")))
cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"))
m = DeciderNeuron(cfg, sd, compute_dtype=torch.bfloat16).eval()
temps = {k: cfg.temperature_by_kind.get(k, cfg.temperature) for k in ("noul", "choice", "score")}
for i in [int(x) for x in sys.argv[2:]]:
    g = torch.load(os.path.join(sys.argv[1], f"ex{i:05d}.pt")); x = g["input_ids"][None]; T = x.shape[1]
    with torch.inference_mode():
        h = m.torso(x)
        p, _ = m.readout(h, torch.tensor([T - 1]), g["opt_idx"][None], torch.tensor([g["n_slots"]]), torch.tensor([1 / temps[g["kind"]]]))
    ha = h[0, T - 1].float()
    cos = torch.nn.functional.cosine_similarity(ha, g["h_answer"], 0).item()
    print(f"ex{i} T={T} cos(h_ans)={cos:.5f} max|dp|={(p[0,:g['n_slots']]-g['probs'][:g['n_slots']]).abs().max():.4f} argmax {int(p.argmax())}/{int(g['probs'].argmax())}", flush=True)
