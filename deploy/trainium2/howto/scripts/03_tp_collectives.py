"""Q3: multi-rank tensor parallel on the native backend.

Launch (one process per logical NeuronCore):
    torchrun --nproc_per_node=4 03_tp_collectives.py          # cores CORE_BASE..CORE_BASE+3
Each rank pins itself to ONE logical core BEFORE importing torch/lite.
"""
import os, sys, time
local_rank = int(os.environ["LOCAL_RANK"])
core_base = int(os.environ.get("CORE_BASE", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + local_rank)    # one core per process
os.environ["NEURON_RT_NUM_CORES"] = "1"
# ALL ranks of one collective group must share the same CCOM root (host:port).
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{61234 + core_base}")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register

dist.init_process_group("gloo")          # world PG is gloo; device collectives are traced into the NEFF
rank, world = dist.get_rank(), dist.get_world_size()
BACKEND = register()
dev = torch.device("neuron", 0)          # every process sees exactly one core: neuron:0
G = dist.group.WORLD
def cpu_barrier():
    # dist.barrier() raises NotImplementedError (PrivateUse1HooksInterface) -> use a gloo CPU all_reduce
    dist.all_reduce(torch.zeros(1))
def log(*a):
    if rank == 0:
        print(*a, flush=True)
log(f"world={world} backend={BACKEND} cores {core_base}..{core_base+world-1} "
    f"EXEC_BARRIER_DISABLED={os.environ.get('NEURON_RT_DISABLE_EXECUTION_BARRIER','0')}")

def compile_(fn):
    return torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)

_fence_t = torch.zeros(1, device=dev)
_fence = compile_(lambda t: t + 1)
def fence():
    _fence(_fence_t).cpu()
fence()

# ---------------- 1. functional collectives inside a compiled graph ----------------
def f_all_reduce(x):     return funcol.all_reduce(x, "sum", G)
def f_all_gather(x):     return funcol.all_gather_tensor(x, 0, G)                 # cat on dim 0
def f_all_gather_d1(x):  return funcol.all_gather_tensor(x, 1, G)                 # cat on dim 1
def f_reduce_scatter(x): return funcol.reduce_scatter_tensor(x, "sum", 0, G)
def f_all_to_all(x):     return funcol.all_to_all_single(x, None, None, G)

torch.manual_seed(1234 + rank)
x_cpu = torch.randn(1024, 2048, dtype=torch.bfloat16)
x = x_cpu.to(dev)

def ref_all_reduce(t):
    t = t.float().clone(); dist.all_reduce(t); return t
def ref_all_gather(t, dim):
    out = [torch.empty_like(t) for _ in range(world)]; dist.all_gather(out, t.contiguous()); return torch.cat(out, dim).float()
def ref_reduce_scatter(t):
    return ref_all_reduce(t).chunk(world, 0)[rank]
def ref_all_to_all(t):
    chunks = list(t.chunk(world, 0)); out = [torch.empty_like(c) for c in chunks]
    # gloo has no all_to_all: emulate via all_gather of every rank's chunks
    allc = [torch.empty_like(t) for _ in range(world)]; dist.all_gather(allc, t.contiguous())
    return torch.cat([allc[r].chunk(world, 0)[rank] for r in range(world)], 0).float()

tests = [
    ("all_reduce", f_all_reduce, lambda: ref_all_reduce(x_cpu)),
    ("all_gather(dim0)", f_all_gather, lambda: ref_all_gather(x_cpu, 0)),
    ("all_gather(dim1)", f_all_gather_d1, lambda: ref_all_gather(x_cpu, 1)),
    ("reduce_scatter", f_reduce_scatter, lambda: ref_reduce_scatter(x_cpu)),
    ("all_to_all_single", f_all_to_all, lambda: ref_all_to_all(x_cpu)),
]
only = os.environ.get("ONLY_TESTS")
for name, fn, ref_fn in tests:
    if only and name not in only.split(","):
        continue
    ref = ref_fn()
    try:
        t0 = time.perf_counter()
        y = compile_(fn)(x).cpu().float()
        tc = time.perf_counter() - t0
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        ok = torch.tensor([1.0 if (y.shape == ref.shape and err < 2e-2) else 0.0])
    except Exception as e:  # noqa
        print(f"[rank{rank}] {name} FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        ok = torch.tensor([0.0]); err = float("nan"); tc = 0; y = torch.empty(0)
    dist.all_reduce(ok)   # gloo, CPU
    log(f"{name:18s} out{tuple(y.shape)} maxrel={err:.2e} all_ranks_ok={int(ok.item())}/{world} first_call={tc:.1f}s")

# ---------------- 2. latency of all_reduce [1024,2048] bf16 ----------------
REP = 8
def ar_chain(x):
    for _ in range(REP):
        x = funcol.all_reduce(x, "sum", G) * (1.0 / world)
    return x
def scale_only(x):
    for _ in range(REP):
        x = x * (1.0 / world)
    return x
c_ar = compile_(f_all_reduce); c_chain = compile_(ar_chain); c_scale = compile_(scale_only)
c_ar(x).cpu(); c_chain(x).cpu(); c_scale(x).cpu()

def bench(fn, iters=64, chunk=8):
    for _ in range(4):
        fn(x)
    fence(); cpu_barrier()
    t0 = time.perf_counter()
    for _ in range(iters // chunk):
        for _ in range(chunk):
            fn(x)
        fence()
    return (time.perf_counter() - t0) / iters
t1 = bench(c_ar); t8 = bench(c_chain); ts = bench(c_scale)
nbytes = x.numel() * 2
log(f"all_reduce [1024,2048] bf16 ({nbytes/2**20:.0f} MiB), world={world}: "
    f"1 AR/graph {t1*1e6:.1f} us/call | {REP} AR/graph {(t8-ts)/REP*1e6:.1f} us/AR "
    f"(chain {t8*1e6:.0f} us minus {REP} scales {ts*1e6:.0f} us) | algbw {nbytes/((t8-ts)/REP)/1e9:.1f} GB/s")

# ---------------- 3. vllm_neuron.nn ColumnParallelLinear -> RowParallelLinear MLP ----------------
from vllm_neuron.nn import ColumnParallelLinear, RowParallelLinear
H, I = 2048, 6144
torch.manual_seed(0)
w1_full = torch.randn(I, H, dtype=torch.bfloat16) * 0.02     # nn.Linear layout [out, in]
w2_full = torch.randn(H, I, dtype=torch.bfloat16) * 0.02
xin = torch.randn(256, H, dtype=torch.bfloat16)
cpl = ColumnParallelLinear(H, I, bias=False, gather_output=False, dtype=torch.bfloat16)   # tp_group=WORLD
rpl = RowParallelLinear(I, H, bias=False, input_is_parallel=True, dtype=torch.bfloat16)
k = I // world
with torch.no_grad():
    cpl.weight.copy_(w1_full[rank * k:(rank + 1) * k, :])
    rpl.weight.copy_(w2_full[:, rank * k:(rank + 1) * k])
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.up = cpl; self.down = rpl
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.up(x)))   # RPL does funcol.all_reduce
mlp = MLP().to(dev).eval()
cmlp = torch.compile(mlp, backend=BACKEND, fullgraph=True, dynamic=False)
with torch.no_grad():
    y = cmlp(xin.to(dev)).cpu().float()
ref = (torch.nn.functional.silu(xin.float() @ w1_full.float().T) @ w2_full.float().T)
err = ((y - ref).abs().max() / ref.abs().max()).item()
log(f"CPL->silu->RPL TP={world}: maxrel vs full CPU MLP = {err:.2e}")

# a CPL with gather_output=True (all_gather on last dim, uses allow_in_graph'd chunk_cat)
cplg = ColumnParallelLinear(H, I, bias=False, gather_output=True, dtype=torch.bfloat16)
with torch.no_grad():
    cplg.weight.copy_(w1_full[rank * k:(rank + 1) * k, :])
cplg = cplg.to(dev)
ccplg = torch.compile(cplg, backend=BACKEND, fullgraph=True, dynamic=False)
with torch.no_grad():
    y = ccplg(xin.to(dev)).cpu().float()
ref = xin.float() @ w1_full.float().T
err = ((y - ref).abs().max() / ref.abs().max()).item()
log(f"CPL(gather_output=True) TP={world}: out{tuple(y.shape)} maxrel = {err:.2e}")
cpu_barrier()
dist.destroy_process_group()
