"""Record the NEFF path of the mock 4-layer hybrid graph (T=2048) via the execute hook."""
import os, sys, glob
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context, clear_execute_context
sys.path.insert(0, os.path.dirname(__file__))
from mock_hybrid import HybridStack
BACKEND = register(); dev = torch.device("neuron", 0)
torch.manual_seed(0)
cm = torch.compile(HybridStack(4).to(dev).eval(), backend=BACKEND, fullgraph=True, dynamic=False)
x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
with torch.no_grad():
    cm(x).cpu()
    seen = []
    set_execute_context(ExecuteContext(pre_execute_hook=lambda inp, md: seen.append(md.artifact_dir)))
    cm(x).cpu(); clear_execute_context()
print("NEFF", glob.glob(os.path.join(seen[0], "*.neff"))[0])
