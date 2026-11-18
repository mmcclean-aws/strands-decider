"""In-graph collective cost at world=P: AR vs RS+AG on [T, 2048] bf16 (chain of 8 per graph)."""
import os, sys, time
lr = int(os.environ["LOCAL_RANK"]); cb = int(os.environ.get("CORE_BASE", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(cb + lr); os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{62500 + cb}"); os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.distributed as dist, torch.distributed._functional_collectives as fc
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
dist.init_process_group("gloo"); r, W = dist.get_rank(), dist.get_world_size(); BK = register(); dev = torch.device("neuron", 0); G = dist.group.WORLD
REP = 8
def ar(x):
    for _ in range(REP): x = fc.all_reduce(x, "sum", G) * 0.5
    return x
def rsag(x):
    for _ in range(REP): x = fc.all_gather_tensor(fc.reduce_scatter_tensor(x, "sum", 0, G) * 0.5, 0, G)
    return x
def sc(x):
    for _ in range(REP): x = x * 0.5
    return x
for T in [int(t) for t in sys.argv[1].split(",")]:
    x = torch.randn(T, 2048, dtype=torch.bfloat16).to(dev)
    res = {}
    for name, f in (("AR", ar), ("RS+AG", rsag), ("scale", sc)):
        cf = torch.compile(f, backend=BK, fullgraph=True, dynamic=False)
        cf(x).cpu(); cf(x).cpu(); dist.all_reduce(torch.zeros(1))
        t0 = time.perf_counter()
        for _ in range(10): y = cf(x)
        y.cpu(); res[name] = (time.perf_counter() - t0) / 10
    if r == 0:
        mb = T * 2048 * 2 / 2**20
        print(f"W={W} {mb:.0f} MiB: AR {(res['AR'] - res['scale']) / REP * 1e6:.0f} us  RS+AG {(res['RS+AG'] - res['scale']) / REP * 1e6:.0f} us", flush=True)
