"""Q3/Q4: TP sub-groups inside one world (DP x TP layout): world=4 -> TP groups [[0,1],[2,3]].
Every rank must create every group (dist.new_group is collective); declare the FULL mesh so all
ranks trace the same SPMD replica groups (this is what vllm_neuron's NeuronGroupCoordinator does)."""
import os, time
local_rank = int(os.environ["LOCAL_RANK"]); core_base = int(os.environ.get("CORE_BASE", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + local_rank)
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{61234 + core_base}")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.distributed as dist, torch.distributed._functional_collectives as funcol
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.parallel.replica_mesh import active_mesh
dist.init_process_group("gloo"); rank, world = dist.get_rank(), dist.get_world_size()
BACKEND = register(); dev = torch.device("neuron", 0)
TP = int(os.environ.get("TP", "2"))
mesh = [list(range(i, i + TP)) for i in range(0, world, TP)]
my_group = None
with active_mesh(mesh):
    for ranks in mesh:
        g = dist.new_group(ranks)
        if rank in ranks:
            my_group = g
x = torch.full((1024, 2048), float(rank + 1), dtype=torch.bfloat16)
f = torch.compile(lambda t: funcol.all_reduce(t, "sum", my_group), backend=BACKEND, fullgraph=True, dynamic=False)
y = f(x.to(dev)).cpu()
expect = float(sum(r + 1 for r in mesh[rank // TP]))
ok = torch.tensor([1.0 if torch.all(y == expect) else 0.0]); dist.all_reduce(ok)
if rank == 0:
    print(f"world={world} TP groups {mesh}: sub-group all_reduce correct on {int(ok.item())}/{world} ranks", flush=True)
dist.destroy_process_group()
