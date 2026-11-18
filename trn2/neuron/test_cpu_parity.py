"""CPU fp32 parity: DeciderNeuron (re-implementation, merged LoRA) vs HF+PEFT goldens.

usage: python test_cpu_parity.py GOLDEN_DIR [ids...]
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from decider_model import DeciderNeuron, _snapshot, load_state  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "16")))
gdir = sys.argv[1]
ids = [int(x) for x in sys.argv[2:]] or [0]
cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"),
                     dtype=torch.float32)
model = DeciderNeuron(cfg, sd, compute_dtype=torch.float32).eval()
temps = {k: cfg.temperature_by_kind.get(k, cfg.temperature) for k in ("noul", "choice", "score")}
worst = 0.0
for i in ids:
    g = torch.load(os.path.join(gdir, f"ex{i:05d}.pt"))
    x = g["input_ids"][None]
    T = x.shape[1]
    with torch.inference_mode():
        t0 = time.time()
        if "layers" in g:
            outs = []
            hook_x = model.torso
            # layer-wise: replicate torso loop to capture per-layer outputs
            c = model.cfg
            from decider_model import rms_norm, rope_cos_sin
            import torch.nn.functional as F
            h = F.embedding(x.long(), model.w("embed"))
            cos, sin = rope_cos_sin(T, c.rope_dim, c.rope_theta, h.device, h.dtype)
            for li, lt in enumerate(c.layer_types):
                h = model.gdn_layer(li, h) if lt == "linear_attention" else model.attn_layer(li, h, cos, sin)
                h = model.mlp(li, h)
                ref = g["layers"][li]
                rel = ((h[0] - ref).norm() / ref.norm()).item()
                outs.append(rel)
            hidden = rms_norm(h, model.w("final_norm"), c.eps)
            print(f"ex{i}: per-layer rel err: " + " ".join(f"{e:.1e}" for e in outs))
        else:
            hidden = model.torso(x)
        K = g["opt_idx"].shape[0]
        probs, logits = model.readout(hidden, torch.tensor([T - 1]), g["opt_idx"][None], torch.tensor([g["n_slots"]]),
                                      torch.tensor([1.0 / temps[g["kind"]]]))
        dt = time.time() - t0
    ha = hidden[0, T - 1]
    e_h = ((ha - g["h_answer"]).norm() / g["h_answer"].norm()).item()
    e_p = (probs[0, : g["n_slots"]] - g["probs"][: g["n_slots"]]).abs().max().item()
    worst = max(worst, e_p)
    print(f"ex{i}: T={T} h_answer rel={e_h:.2e} max|dp|={e_p:.2e} argmax {int(probs[0].argmax())} vs {int(g['probs'].argmax())}  ({dt:.1f}s)")
print("WORST max|dp|", worst)
