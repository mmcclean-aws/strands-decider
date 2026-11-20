"""Q1: single-core placement, torch.compile with the native backend, sync + timing."""
import os, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")   # must precede the lite import
import torch
import libtorch_neuronx_lite                      # registers PrivateUse1 device "neuron"
from libtorch_neuronx_lite.compile.native_backend import register

BACKEND = register()                              # -> "neuron_native_lite"
print("backend:", BACKEND, "torch", torch.__version__, "lite", libtorch_neuronx_lite.__version__)
print("compile cache state:", libtorch_neuronx_lite.COMPILE_CACHE_STATE)
dev = torch.device("neuron", 0)                   # always neuron:0 inside a process (cores narrowed by NEURON_RT_VISIBLE_CORES)

H, N = 2048, 6144
REPEAT = 8
torch.manual_seed(0)
w_cpu = torch.randn(H, N, dtype=torch.bfloat16) * 0.02
w = w_cpu.to(dev)
print("w:", w.device, w.dtype, w.shape)

def mm(x, w):
    return x @ w

def mm_repeat(x, w):
    # REPEAT independent matmuls in ONE graph -> amortizes the ~80us per-call dispatch
    acc = x @ w
    for i in range(1, REPEAT):
        acc = acc + (x + i) @ w
    return acc

cmm = torch.compile(mm, backend=BACKEND, fullgraph=True, dynamic=False)
cmm_r = torch.compile(mm_repeat, backend=BACKEND, fullgraph=True, dynamic=False)

# Device fence: executions on a core run in submission order, so a tiny graph
# queued after the work + .cpu() of its 4-byte output waits for everything before it.
_fence_t = torch.zeros(1, device=dev)
_fence_fn = torch.compile(lambda t: t + 1, backend=BACKEND, fullgraph=True, dynamic=False)
def neuron_fence():
    _fence_fn(_fence_t).cpu()
neuron_fence()   # compile the fence once, outside any timed region

def bench(fn, args, iters=96, chunk=8):
    """Pipelined timing. Execution is ASYNC and in-order; torch.neuron.synchronize()
    does NOT wait for it, so block by reading back the last output (.cpu()).
    Keep <= NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS (default 8) calls in flight."""
    for _ in range(3):
        y = fn(*args)
    y.cpu()
    t0 = time.perf_counter()
    for c in range(iters // chunk):
        for _ in range(chunk):
            y = fn(*args)
        neuron_fence()   # blocks until the chunk finished (no big D2H)
    return (time.perf_counter() - t0) / iters

for T in (128, 512, 2048):
    x_cpu = torch.randn(T, H, dtype=torch.bfloat16)
    x = x_cpu.to(dev)
    t0 = time.perf_counter(); y_cpu = cmm(x, w).cpu(); t_first = time.perf_counter() - t0
    ref = x_cpu.float() @ w_cpu.float()
    rel = ((y_cpu.float() - ref).abs().max() / ref.abs().max()).item()
    t0 = time.perf_counter(); cmm_r(x, w).cpu(); t_first_r = time.perf_counter() - t0
    t_one = bench(cmm, (x, w))
    t_rep = bench(cmm_r, (x, w)) / REPEAT
    # sanity: fence really waits (time 8 heavy calls + fence vs 8 heavy calls + .cpu())
    t0 = time.perf_counter(); [cmm_r(x, w) for _ in range(8)]; neuron_fence(); t_f = time.perf_counter() - t0
    t0 = time.perf_counter(); ys = [cmm_r(x, w) for _ in range(8)]; ys[-1].cpu(); t_c = time.perf_counter() - t0
    print(f"   fence check: 8x{REPEAT}-mm + fence {t_f*1e3:.2f} ms vs + .cpu() {t_c*1e3:.2f} ms")
    # blocking latency: one call then .cpu() of the output (includes D2H of T x 6144 bf16)
    t0 = time.perf_counter()
    for _ in range(20):
        cmm(x, w).cpu()
    t_lat = (time.perf_counter() - t0) / 20
    flops = 2 * T * H * N
    print(f"T={T:5d} first_call={t_first:6.2f}s/{t_first_r:6.2f}s maxrel={rel:.1e} | "
          f"1-mm/graph {t_one*1e6:7.1f}us {flops/t_one/1e12:6.1f} TF/s | "
          f"{REPEAT}-mm/graph {t_rep*1e6:7.1f}us/mm {flops/t_rep/1e12:6.1f} TF/s | "
          f"call+.cpu() {t_lat*1e6:7.1f}us", flush=True)

rt = torch.classes.neuron.Runtime()
print("Runtime methods:", [m for m in dir(rt) if not m.startswith("_")])
print("Executor methods:", [m for m in dir(torch.classes.neuron.Executor) if not m.startswith("__")])
