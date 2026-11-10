"""Q4: one independent replica per logical core. Launched N times in parallel by 04_dp_replicas.sh."""
import os, sys, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.nn.functional as F
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
BACKEND = register(); dev = torch.device("neuron", 0)
compile_ = lambda fn: torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
_ft = torch.zeros(1, device=dev); _ff = compile_(lambda t: t + 1)
def fence(): _ff(_ft).cpu()
t_start = time.perf_counter()
fence()
W = [(torch.randn(2048, 6144, dtype=torch.bfloat16) * 0.02).to(dev) for _ in range(2)]
x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
def f(x, w0, w1):                                  # ~MLP-sized work: 2 x 51.5 GFLOP
    return F.silu(x @ w0) @ w1.T
cf = compile_(f); y = cf(x, *W).cpu()
t_ready = time.perf_counter() - t_start
iters, chunk = 64, 8
for _ in range(3): cf(x, *W)
fence(); t0 = time.perf_counter()
for _ in range(iters // chunk):
    for _ in range(chunk): cf(x, *W)
    fence()
dt = (time.perf_counter() - t0) / iters
print(f"core={os.environ['NEURON_RT_VISIBLE_CORES']} comm={os.environ.get('NEURON_RT_ROOT_COMM_ID')} "
      f"ready_in={t_ready:.1f}s  {dt*1e6:.0f} us/iter  {2*2*2048*2048*6144/dt/1e12:.1f} TF/s", flush=True)
