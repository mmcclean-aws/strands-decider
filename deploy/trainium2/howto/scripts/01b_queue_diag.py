"""Diagnose async execution: queue depth and what torch.neuron.synchronize() waits for."""
import os, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
BACKEND = register(); dev = torch.device("neuron", 0)
print("MAX_QUEUED env =", os.environ.get("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"))
w = (torch.randn(2048, 6144, dtype=torch.bfloat16) * 0.02).to(dev)
x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
def heavy(x, w):
    acc = x @ w
    for i in range(1, 16):
        acc = acc + (x + i) @ w
    return acc
f = torch.compile(heavy, backend=BACKEND, fullgraph=True, dynamic=False)
f(x, w).cpu()
t0 = time.perf_counter(); f(x, w).cpu(); print(f"one blocking call: {(time.perf_counter()-t0)*1e3:.2f} ms")
# sync semantics
y = f(x, w)
t0 = time.perf_counter(); torch.neuron.synchronize(); t_sync = time.perf_counter() - t0
t0 = time.perf_counter(); y.cpu(); t_cpu = time.perf_counter() - t0
print(f"after 1 dispatch: synchronize() took {t_sync*1e3:.3f} ms, then .cpu() took {t_cpu*1e3:.3f} ms")
ys = [f(x, w) for _ in range(4)]
t0 = time.perf_counter(); torch.neuron.synchronize(); t_sync = time.perf_counter() - t0
t0 = time.perf_counter(); ys[-1].cpu(); t_cpu = time.perf_counter() - t0
print(f"after 4 dispatches: synchronize() took {t_sync*1e3:.3f} ms, then .cpu() took {t_cpu*1e3:.3f} ms")
# max in flight
for k in range(1, 70):
    try:
        ys = []
        for i in range(k):
            ys.append(f(x, w))
        ys[-1].cpu()
    except RuntimeError as e:
        print(f"queue full at k={k} (i={i}): {str(e)[:80]}")
        try:
            ys[-1].cpu()
        except Exception as e2:
            print("recovery .cpu() failed:", e2)
        break
else:
    print("no queue-full up to 69 in flight")
