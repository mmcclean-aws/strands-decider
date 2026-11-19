"""Does the default compile auto-cast fp32 matmuls to bf16? Run once per flag set (separate processes:
the backend refuses the same graph with different compiler_args in one process)."""
import os, sys, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
BACKEND = register(); dev = torch.device("neuron", 0)
VLLM_NEURON_FLAGS = ["--auto-cast=none", "-O1",
                     "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
                     "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]
flags = {"default": None, "autocast_none": ["--auto-cast=none"], "vllm_neuron": VLLM_NEURON_FLAGS}[sys.argv[1]]
opts = {} if flags is None else {"compiler_args": flags}
torch.manual_seed(0)
x = torch.randn(512, 2048); w = torch.randn(2048, 2048)
ref = (x.double() @ w.double())
cf = torch.compile(lambda x, w: x @ w, backend=BACKEND, fullgraph=True, dynamic=False, options=opts)
t0 = time.perf_counter(); y = cf(x.to(dev), w.to(dev)).cpu().double(); tc = time.perf_counter() - t0
err = ((y - ref).abs().max() / ref.abs().max()).item()
print(f"flags={sys.argv[1]:14s} fp32 matmul maxrel={err:.2e} ({'bf16-like => auto-cast ON' if err > 1e-4 else 'fp32-accurate'}) compile {tc:.1f}s", flush=True)
