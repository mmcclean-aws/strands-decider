"""Q6: device profiling of compiled graphs on the native backend.

(1) find the NEFF a compiled function runs (execute hook -> ExecuteMetadata.artifact_dir)
(2) in-process capture with NRT inspect: torch.classes.neuron.Runtime().start_profiling(...)
    (same call vllm-neuron's NeuronProfiler / MPExecutor.start_profile make)
The shell wrapper then post-processes with neuron-explorer.
"""
import os, sys, glob, json, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.nn.functional as F
import libtorch_neuronx_lite
from libtorch_neuronx_lite import envs as lenvs
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context, clear_execute_context
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.experimental.gdn import gdn_cte, build_mask_pack

OUT = sys.argv[1]
BACKEND = register(); dev = torch.device("neuron", 0)
compile_ = lambda fn: torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
_ft = torch.zeros(1, device=dev); _ff = compile_(lambda t: t + 1)
def fence(): _ff(_ft).cpu()
fence()

# graph 1: matmul T=2048
w = (torch.randn(2048, 6144, dtype=torch.bfloat16) * 0.02).to(dev)
x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
mm = compile_(lambda x, w: x @ w)
# graph 2: gdn_cte S=2048, 16 heads
cte = wrap_nki(gdn_cte)
S = 2048
q = F.normalize(torch.randn(1, 16, S, 128), dim=-1).bfloat16().to(dev)
k = F.normalize(torch.randn(1, 16, S, 128), dim=-1).bfloat16().to(dev)
v = torch.randn(1, 16, S, 128).bfloat16().to(dev)
g = (-F.softplus(torch.randn(1, 16, S)) * 0.1).to(dev); b = torch.sigmoid(torch.randn(1, 16, S)).to(dev)
mp = torch.from_numpy(build_mask_pack()).to(dev)
gdn = compile_(lambda q, k, v, g, b, mp: cte(query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp))
for _ in range(3):
    mm(x, w); gdn(q, k, v, g, b, mp)
fence()

# (1) which NEFF does each compiled callable run?
seen = {}
def hook(inputs, md):
    seen.setdefault(md.neff_id, md.artifact_dir)
for name, fn in (("matmul", lambda: mm(x, w)), ("gdn_cte", lambda: gdn(q, k, v, g, b, mp))):
    seen.clear(); set_execute_context(ExecuteContext(pre_execute_hook=hook)); fn(); clear_execute_context()
    (cache_key, adir), = seen.items()
    neffs = glob.glob(os.path.join(adir, "*.neff"))
    print(f"NEFF[{name}] = {neffs}", flush=True)
    with open(os.path.join(OUT, f"{name}.neffpath"), "w") as f:
        f.write(neffs[0])
fence()

# (2) in-process NRT inspect capture of a few executions
neff_cache_dir = lenvs.get_neff_cache_dir(BACKEND)
rt = torch.classes.neuron.Runtime()
rt.start_profiling(os.path.join(OUT, "inproc"), ["device_profile", "system_profile"], None, None, neff_cache_dir)
for _ in range(2):
    mm(x, w); fence()
    gdn(q, k, v, g, b, mp); fence()
rt.stop_profiling()
print("in-process profile written to", os.path.join(OUT, "inproc"), flush=True)
