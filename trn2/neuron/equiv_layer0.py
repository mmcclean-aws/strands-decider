"""Per-op equivalence for one GDN layer + MLP and one attention layer on real activations.

Each op runs (a) on device in bf16, (b) on CPU in bf16 with the same code, against (c) CPU fp32,
feeding every implementation the SAME fp32-reference input of that op (no error propagation),
so each line isolates one op's error.
"""

import os
import sys

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import torch.nn.functional as F
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.compile.native_backend import register

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decider_model import (DeciderNeuron, _snapshot, apply_partial_rope, causal_conv1d_silu_hm, gated_rms_norm,  # noqa: E402
                           gdn_chunk_ref, l2norm, load_state, rms_norm, rope_cos_sin)
from nki_ops import NkiMixers  # noqa: E402

BACKEND = register()
DEV = torch.device("neuron", 0)
torch.set_num_threads(16)
g = torch.load(sys.argv[1])
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ids = g["input_ids"][:T][None]
cfg32, sd32 = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"), dtype=torch.float32)
m32 = DeciderNeuron(cfg32, sd32, compute_dtype=torch.float32).eval()
W32 = {k: m32.w(k) for k in sd32 if k.startswith("layers.0.") or k.startswith("layers.3.")}
W16 = {k: (v if v.dtype == torch.float32 and (k.endswith(("ln1", "ln2", "q_norm", "k_norm", ".A", "dt_bias"))) else v.to(torch.bfloat16)) for k, v in W32.items()}
WD = {k: v.to(DEV) for k, v in W16.items()}
mix = NkiMixers(DEV)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def comp(name, fn, inputs32, cast_bf16=(), dev_fn=None):
    """fn(W, *inputs): run fp32 CPU, bf16 CPU, bf16 device; report rel errors vs fp32."""
    torch._dynamo.reset()  # closure guards are dropped under caching_precompile: never reuse a graph
    with torch.inference_mode():
        r32 = fn(W32, *inputs32)
        i16 = [x.to(torch.bfloat16) if j in cast_bf16 else x for j, x in enumerate(inputs32)]
        r16 = fn(W16, *i16)
        f = torch.compile(lambda *a: (dev_fn or fn)(WD, *a), backend=BACKEND, fullgraph=True, dynamic=False)
        rd = f(*[x.to(DEV) for x in i16])
        rd = [t.cpu() for t in rd] if isinstance(rd, (tuple, list)) else rd.cpu()
    if not isinstance(r32, (tuple, list)):
        r32, r16, rd = [r32], [r16], [rd]
    for j, (a, b, c) in enumerate(zip(r32, r16, rd)):
        print(f"{name}[{j}]: cpu-bf16 {rel(b, a):.2e}   device {rel(c, a):.2e}", flush=True)
    return r32


with torch.inference_mode():
    x0 = F.embedding(ids.long(), m32.w("embed"))  # fp32 residual input to layer 0
p = "layers.0."
(h,) = comp("rms_norm", lambda W, x: [rms_norm(x, W[p + "ln1"], 1e-6)], [x0], cast_bf16=(0,))
(qkvz,) = comp("einsum qkvz", lambda W, h: [torch.einsum("btc,hdc->bhtd", h, W[p + "w_qkvz"])], [h], cast_bf16=(0,))
(conv,) = comp("conv+silu", lambda W, z: [causal_conv1d_silu_hm(z[:, :48], W[p + "conv_w"])], [qkvz], cast_bf16=(0,))
q, k = comp("l2norm q,k", lambda W, c: [l2norm(c[:, :16]), l2norm(c[:, 16:32])], [conv])
with torch.inference_mode():
    v = conv[:, 32:48]
    ba32 = (h @ W32[p + "w_ba"].t()).transpose(1, 2)


def gb(W, ba):
    beta = torch.sigmoid(ba[:, :16].float())
    gg = W[p + "A"][None, :, None] * F.softplus(ba[:, 16:].float() + W[p + "dt_bias"][None, :, None])
    return [beta, gg]


beta, gg = comp("sigmoid/softplus beta,g", gb, [ba32], cast_bf16=(0,))
(o,) = comp("gdn core", lambda W, q, k, v, g_, b_: [gdn_chunk_ref(q, k, v, g_, b_)], [q, k, v, gg, beta], cast_bf16=(0, 1, 2),
            dev_fn=lambda W, q, k, v, g_, b_: [mix.gdn(q, k, v, g_, b_)])
z = qkvz[:, 48:]
(on,) = comp("gated rmsnorm", lambda W, o, z: [gated_rms_norm(o, z, W[p + "gnorm"], 1e-6)], [o, z], cast_bf16=(0, 1))
(y,) = comp("out einsum", lambda W, o: [torch.einsum("bhtd,chd->btc", o, W[p + "w_out"])], [on], cast_bf16=(0,))
x1 = x0 + y


def mlp(W, x):
    hh = rms_norm(x, W[p + "ln2"], 1e-6)
    gu = hh @ W[p + "w_gate_up"].t()
    return [(F.silu(gu[..., :6144]) * gu[..., 6144:]) @ W[p + "w_down"].t()]


comp("mlp (norm+swiglu)", mlp, [x1], cast_bf16=(0,))


def silu_only(W, x):
    return [F.silu(x)]


comp("silu alone", silu_only, [torch.randn(1, T, 6144) * 3], cast_bf16=(0,))
comp("sigmoid alone", lambda W, x: [torch.sigmoid(x)], [torch.randn(1, T, 2048) * 3], cast_bf16=())
comp("softplus alone (fp32)", lambda W, x: [F.softplus(x)], [torch.randn(1, 16, T) * 3], cast_bf16=())
comp("exp alone (fp32)", lambda W, x: [torch.exp(x)], [torch.randn(1, 16, T)], cast_bf16=())
comp("rsqrt alone (fp32)", lambda W, x: [torch.rsqrt(x)], [torch.rand(1, 16, T) + 0.1], cast_bf16=())

# attention layer 3 on the same x (isolated)
p = "layers.3."
cos32, sin32 = rope_cos_sin(T, 64, 1e7, "cpu", torch.float32)


def attn(W, x, cos, sin):
    h = rms_norm(x, W[p + "ln1"], 1e-6)
    qg = torch.einsum("btc,hdc->bhtd", h, W[p + "w_qgkv"])
    q = apply_partial_rope(rms_norm(qg[:, :8], W[p + "q_norm"], 1e-6), cos, sin)
    k = apply_partial_rope(rms_norm(qg[:, 16:18], W[p + "k_norm"], 1e-6), cos, sin)
    return [q, k, qg[:, 18:], qg[:, 8:16]]


q, k, v, gate = comp("attn proj+qknorm+rope", attn, [x1, cos32, sin32], cast_bf16=(0, 1, 2))
from decider_model import attn_ref  # noqa: E402

comp("attention core", lambda W, q, k, v: [attn_ref(q, k, v, 256 ** -0.5)], [q, k, v], cast_bf16=(0, 1, 2),
     dev_fn=lambda W, q, k, v: [mix.attn(q, k, v, 256 ** -0.5)])
