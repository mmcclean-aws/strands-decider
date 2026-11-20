"""Why is host-timed gdn_cte (3.2 ms/call) >> device total_time (1.08 ms)?  Try runtime knobs + in-graph repeat."""
import os, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.nn.functional as F
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.gdn import gdn_cte, build_mask_pack
BACKEND = register(); dev = torch.device("neuron", 0)
compile_ = lambda fn: torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
_ft = torch.zeros(1, device=dev); _ff = compile_(lambda t: t + 1)
def fence(): _ff(_ft).cpu()
fence()
def bench(fn, *args, iters=32, chunk=8):
    for _ in range(3): fn(*args)
    fence(); t0 = time.perf_counter()
    for _ in range(iters // chunk):
        for _ in range(chunk): fn(*args)
        fence()
    return (time.perf_counter() - t0) / iters
cte = wrap_nki(gdn_cte)
S = 2048
q = F.normalize(torch.randn(1, 16, S, 128), dim=-1).bfloat16().to(dev)
k = F.normalize(torch.randn(1, 16, S, 128), dim=-1).bfloat16().to(dev)
v = torch.randn(1, 16, S, 128).bfloat16().to(dev)
g = (-F.softplus(torch.randn(1, 16, S)) * 0.1).to(dev); b = torch.sigmoid(torch.randn(1, 16, S)).to(dev)
mp = torch.from_numpy(build_mask_pack()).to(dev)
def one(q, k, v, g, b, mp):
    return cte(query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp)
def four(q, k, v, g, b, mp):
    o = None
    for i in range(4):
        oi, st = cte(query=q, key=k, value=v * (i + 1), g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp)
        o = oi if o is None else o + oi
    return o
c1, c4 = compile_(one), compile_(four)
t1 = bench(c1, q, k, v, g, b, mp); t4 = bench(c4, q, k, v, g, b, mp)
# blocking single-call latency
c1(q, k, v, g, b, mp)[0].cpu(); t0 = time.perf_counter()
for _ in range(8): c1(q, k, v, g, b, mp)[1].cpu()
tl = (time.perf_counter() - t0) / 8
w = (torch.randn(2048, 6144, dtype=torch.bfloat16) * 0.02).to(dev); x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
cm = compile_(lambda x, w: x @ w); tm = bench(cm, x, w)
print(f"IO_RING_CACHE={os.environ.get('NEURON_RT_IO_RING_CACHE_SIZE','-')} NO_EXEC_BARRIER={os.environ.get('NEURON_RT_DISABLE_EXECUTION_BARRIER','-')}: "
      f"gdn_cte 1/graph {t1*1e6:.0f} us/call | 4/graph {t4*1e6/4:.0f} us/kernel | blocking call {tl*1e6:.0f} us | matmul {tm*1e6:.0f} us", flush=True)
