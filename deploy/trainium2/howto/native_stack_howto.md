# Standalone models on the PyTorch-native Neuron backend (Neuron SDK 2.33, trn2)

Verified on 2026-11-10 on a trn2.48xlarge host, image
`pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`
(torch 2.13.0, libtorch-neuronx-lite 2.13.0.1.0.3112, neuronx-cc 2.28.7631, nki 0.7.0,
vllm 0.29.0, vllm-neuron 0.29.0.1.2.0, NRT 2.35.30, neuron-explorer 2.33.0), LNC=2.
No vLLM engine, no NxDI, no torch-xla tracing. Every snippet below comes from a script in
`howto_scripts/` that ran on the device. Numbers are from the same runs.

All numbers are for **one logical NeuronCore at LNC=2** (two physical NeuronCore-v3).
Peak: **158 TF/s bf16** and **716 GB/s HBM** per logical core. There are 64 logical cores per
trn2.48xlarge (4 per chip, 24 GiB HBM each).

---------------------------------------------------------------------------------------------

## 0. TL;DR: the environment every process needs

`howto_scripts/common_env.sh` (source it before every run):

```bash
export VLLM_PLUGINS=none                      # don't load the vLLM platform plugin
export VLLM_NEURON_BACKEND=neuron_native      # REQUIRED before importing libtorch_neuronx_lite (see gotcha G1)
export NEURON_LOGICAL_NC_CONFIG=2             # LNC=2 (lite sets this default too)
export NEURON_LIBTORCH_CACHE_ROOT=/tmp/howto_cache   # NEFF cache root (default falls back to /tmp if ~ is NFS)
export TORCHINDUCTOR_CACHE_DIR=...            # optional: Dynamo precompile + AOTAutograd cache (default /tmp/torchinductor_$USER)
export NEURON_RT_LOG_LEVEL=ERROR
export NEURON_RT_ROOT_COMM_ID=127.0.0.1:61234 # REQUIRED even on ONE core (G2). Per TP group: same value on all ranks
export NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=32   # async exec queue depth, default 8 (G4)
export NEURON_RT_IO_RING_CACHE_SIZE=32        # reuse per-exec DMA rings: 2.6x faster per call for gdn_cte (G5)
export NEURON_RT_VISIBLE_CORES=<logical core id>     # one core per process
```

Minimal program (pattern used by every script):

```python
import os
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")   # before the lite import
import torch
import libtorch_neuronx_lite                                      # registers PrivateUse1 device "neuron"
from libtorch_neuronx_lite.compile.native_backend import register

BACKEND = register()            # -> "neuron_native_lite" (idempotent)
dev = torch.device("neuron", 0) # always neuron:0; the physical core comes from NEURON_RT_VISIBLE_CORES

w = (torch.randn(2048, 6144, dtype=torch.bfloat16) * 0.02).to(dev)   # make/cast on CPU, then .to(dev)
x = torch.randn(2048, 2048, dtype=torch.bfloat16).to(dev)
f = torch.compile(lambda x, w: x @ w, backend=BACKEND, fullgraph=True, dynamic=False)
y = f(x, w)          # asynchronous: returns at once
print(y.cpu())       # .cpu() blocks until the result is ready
```

Backend names: `register()` returns **`neuron_native_lite`**, which is the same name vllm-neuron's
`envs.get_compile_backend_name()` returns on the native route. `"neuron"` is reserved for a separately
installed torch-neuronx. `"neuron_libtorch"` / `"vllm_neuron"` are the old **XLA** route, registered at
import. Do not use those.

---------------------------------------------------------------------------------------------

## 1. Single core: placement, compile, run, sync, timing (`01_matmul.py`, `01b_queue_diag.py`)

* **Device:** `torch.device("neuron", 0)`. `tensor.to(dev)`, `module.to(dev)`, `torch.empty/zeros/randn(device=dev)`
  and `.cpu()` all work. Keep dtype changes on the CPU or inside the graph (`x.to(dev, dtype=...)` fails, G11).
* **Compile:** `torch.compile(fn_or_module, backend="neuron_native_lite", fullgraph=True, dynamic=False)`.
  Optional `options={"compiler_args": [...]}` passes neuronx-cc flags (vllm-neuron uses
  `["--auto-cast=none", "-O1", "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
  "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"]`). The default
  compile does **not** auto-cast fp32 matmuls (verified: fp32 matmul maxrel 2.7e-7, `08_autocast.py`).
* **Execution is asynchronous and in-order per core.** A compiled call enqueues a NEFF execution and returns
  output tensors as futures.
  * **`torch.neuron.synchronize()` does NOT wait for device execution.** It returned in 0.02 ms while the
    queued work took 8.7 ms (`01b_queue_diag.py`). It only synchronizes the compile service.
  * To block, read back an output (`y.cpu()`), or use a **fence**: a tiny compiled graph queued after the work,
    with its 4-byte output read back. This works because executions on a core run in submission order:

```python
_fence_t = torch.zeros(1, device=dev)
_fence_fn = torch.compile(lambda t: t + 1, backend=BACKEND, fullgraph=True, dynamic=False)
def neuron_fence():
    _fence_fn(_fence_t).cpu()
neuron_fence()                      # compile it once, outside any timed region

def bench(fn, args, iters=96, chunk=8):
    for _ in range(3): y = fn(*args)
    y.cpu()                          # warm-up (first call = trace + compile or cache load)
    t0 = time.perf_counter()
    for _ in range(iters // chunk):
        for _ in range(chunk):       # keep <= NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS in flight
            y = fn(*args)
        neuron_fence()
    return (time.perf_counter() - t0) / iters
```

  The fence matched `.cpu()` timing (5.56 ms vs 5.59 ms for 8 heavy calls).
  * Without a fence, more than 8 (default) queued executions raise
    `RuntimeError: Failed to schedule neff execution. status=7 message=Execution Queue Full`.
    The limit equals `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (measured: failure at the 9th with the default
    and at the 33rd with 32).

**Measured: bf16 `[T,2048] x [2048,6144]` on 1 logical core** (pipelined with the fence, final env):

| T | 1 matmul per graph | TF/s | 8 matmuls in one graph (per mm) | TF/s | blocking `f(x).cpu()` (incl. D2H of T x 6144) |
|---:|---:|---:|---:|---:|---:|
| 128  | 110 us | 29 | 86 us | 37 | 301 us |
| 512  | 160 us | 81 | 131 us | 98 | 722 us |
| 2048 | 411 us | 125 | 425 us | 121 | 2342 us |

* Device-side (profiler) time for T=2048 is **382 us, MFU 86% (~135 TF/s)** (section 6).
* Per-call dispatch floor is about **85-90 us**, so T=128 is dispatch-bound. Put the whole model forward in
  **one** graph.
* Weight streaming at T=128 reaches only about 290 GB/s.
* H2D copy: 12.8 GB/s. D2H copy: 3.6 GB/s (64 MiB tensor).

---------------------------------------------------------------------------------------------

## 2. NKI kernels inside a compiled graph (`02_nki_kernels.py`, `09_nf_gates.py`)

**Pattern (what vllm_neuron's `NF.*` ops do):**

```python
import nki
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.experimental.gdn import gdn_cte, gdn_conv1d_prefill, build_mask_pack

attn_k = wrap_nki(nki.jit()(attention_cte))   # works on @nki.jit kernels too: wrap_nki(gdn_cte)
cte_k  = wrap_nki(gdn_cte)
conv_k = wrap_nki(gdn_conv1d_prefill)

def attn_nki(q, k, v):        # q [Hq,S,D] k [Hkv,S,D] v [Hkv,S,D]; GQA is native (Hq % Hkv == 0)
    return attn_k[2](q=q * q.shape[-1] ** -0.5, k=k, v=v, scale=1.0, causal_mask=True,
                     tp_q=True, tp_k=True, tp_out=False)            # -> [Hq,S,D]

mask_pack = torch.from_numpy(build_mask_pack()).to(dev)              # host-built constant, upload once
def gdn(q, k, v, g, b, mp):   # q,k [B,Hk,S,Dk] L2-normalised bf16; v [B,Hv,S,Dv]; g,b [B,Hv,S] fp32
    return cte_k(query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp)  # (out, state)

def gdn_cont(q, k, v, g, b, mp, state):  # state [B,Hv,Dk,Dv] fp32 is aliased to output 1, updated IN PLACE
    return cte_k(query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp,
                 recurrent_state=state)

conv = lambda m, w: conv_k(mixed=m, conv_weight=w)    # m [B,C,T] raw qkv, w [C,K] -> (silu(conv), last-K state)

f = torch.compile(gdn, backend=BACKEND, fullgraph=True, dynamic=False)
```

* `wrap_nki(kernel)` returns an `NKIHOPCaller`. Calling it inside a traced function emits one
  `torch.ops.higher_order.nki_kernel_wrapper` node. Output shapes come from compiling the kernel on meta
  tensors at trace time. Non-tensor kwargs (scale, flags, strings) are trace-time constants. Kernels that
  mutate an input (e.g. `recurrent_state`) are functionalised and written back in place (verified).
* **Grid / LNC:** `kernel[2](...)` launches on both physical cores of the LNC=2 pair. Calling
  `kernel(...)` without brackets takes the grid from `NEURON_LOGICAL_NC_CONFIG` (2). The compiler says
  brackets are being deprecated. **`kernel[1]` fails at LNC=2** (`NCC_EVRF066: NKI kernel ... is configured
  with LNC=1, but neuronx-cc is using LNC=2`), even for kernels such as `gdn_conv1d_prefill` that do not
  shard. Kernels that do shard (gdn_cte, attention_cte) split work over `nl.program_id(0)`.
* NKI kernels and ordinary torch ops fuse into the same NEFF. The 4-layer mock in section 5 has 10 kernel
  calls plus matmuls and norms in one graph.

**Verified on device against torch references (final env, pipelined time per call including ~90 us dispatch):**

| kernel | shape | max rel err vs fp32 ref | time/call |
|---|---|---|---:|
| nkilib `attention_cte` direct, causal | Hq=8, Hkv=2, **D=256**, S=512 | 2.2e-3 | 147 us (torch SDPA on device: 133 us) |
| same | S=2048 | 2.4e-3 | 369 us (torch SDPA: 357 us) |
| `NF.flash_attention` (default layouts q[B,S,D], k[B,D,S]) | Hq=8, Hkv=2, D=128, S=1024 | 2.9e-3 | 165 us |
| `gdn_conv1d_prefill` (grid 2 / no brackets) | C=6144, K=4, T=512 / 2048 | 2.7e-3 (state exact) | 265 / 497 us |
| `gdn_cte` (schur, mask_pack) | B=1, H=16, D=128, S=512 | out 5.3e-3, state 3.5e-3 | 421 us |
| `gdn_cte` | S=2048 | out 6.4e-3, state 5.2e-3 | 1203 us (device 1079 us) |
| `gdn_cte` + `recurrent_state` (2nd half of S, state from 1st half) | S=512 / 2048 | out 5.2e-3 / 7.1e-3; input buffer updated in place | |

**How vllm_neuron's `NF` ops dispatch** (`vllm_neuron/functional/...`). Each op checks a Python
`_can_use_*` gate at trace time and otherwise **silently** emits a torch fallback:

* `NF.flash_attention(q,k,v, scale, causal_mask, tp_q=True, tp_k=False, tp_out=False, ...)`:
  `wrap_nki(nki.jit()(attention_cte))[2](q=q*scale, scale=1.0, ..., softmax_dtype=fp32)`.
  Its gate requires B<=512, S<=131072 and **D<=128**. The nkilib kernel itself allows `d <= 512`, and
  D=256 works when called directly (table above).
* `NF.mlp(hidden[T,H], gate_w[H,I], up_w, down_w, ...)`:
  `wrap_nki(nkilib.core.mlp.mlp)[lnc](hidden.unsqueeze(0), ...)`. The CTE gate requires
  `ceil(I_eff/512) <= 8`, with I-sharding only when H >= 7168.
* `NF.qkv_proj(hidden[B,S,H], qkv_weights[H, fused], ...)`: `wrap_nki(_torch_compatible_qkv_kernel)`.
  The gate requires H <= 24576, H % 128 == 0 and `fused_qkv_dim <= 4096`.
* `NF.o_proj`: `wrap_nki(output_projection_cte)`.
* For Qwen3.5-2B shapes (`09_nf_gates.py`, meta tensors):

| TP | NF.mlp (I=6144/TP) | qkv_proj attn (5120/TP) | qkv_proj GDN in_proj (8192/TP) | flash_attention D=256 |
|---|---|---|---|---|
| 1 | **torch** | **torch** | **torch** | **torch** |
| 2, 4, 8 | NKI | NKI | NKI | **torch** |

  So at TP=1 none of these NF ops use NKI for this model. Call nkilib kernels directly via
  `wrap_nki`, or accept compiler-generated code (the compiler's SDPA was as fast as `attention_cte` here).

---------------------------------------------------------------------------------------------

## 3. Multi-core tensor parallel (`03_tp_collectives.py`, `03b_subgroups.py`)

**Launch:** use plain `torchrun`, one process per logical core. Each rank sets its env **before importing
torch/lite**:

```python
local_rank = int(os.environ["LOCAL_RANK"]); core_base = int(os.environ.get("CORE_BASE", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + local_rank)   # one core per process
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{61234 + core_base}")  # SAME on all ranks of the group
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register

dist.init_process_group("gloo")      # world PG is gloo. There is NO "neuron" PG backend: device collectives
BACKEND = register()                 # are traced into the NEFF and run by NRT/CCOM at execution time
dev = torch.device("neuron", 0)
G = dist.group.WORLD

f_ar  = lambda x: funcol.all_reduce(x, "sum", G)                 # string reduce op (picklable for precompile)
f_ag  = lambda x: funcol.all_gather_tensor(x, 0, G)              # gather_dim 0 or 1 both work
f_rs  = lambda x: funcol.reduce_scatter_tensor(x, "sum", 0, G)
f_a2a = lambda x: funcol.all_to_all_single(x, None, None, G)
y = torch.compile(f_ar, backend=BACKEND, fullgraph=True, dynamic=False)(x.to(dev))
```

```bash
torchrun --nproc_per_node=4 --master_port=29604 03_tp_collectives.py          # cores 0-3
CORE_BASE=4 torchrun --nproc_per_node=4 --master_port=29605 03_tp_collectives.py  # cores 4-7, concurrently
```

* The executor for a collective NEFF is built with `(dist.get_rank(), dist.get_world_size())`. Every rank
  traces the **same** graph, so all ranks share one NEFF (SPMD). Pass rank-dependent values as **device
  tensors**, not Python ints (vllm-neuron's `rank_in_group_tensor`). Otherwise each rank compiles its own
  graph.
* **`vllm_neuron.nn.ColumnParallelLinear` / `RowParallelLinear`** work standalone with no vLLM parallel
  state. They default to `dist.group.WORLD`, and `tp_group=` takes another group.
  * CPL holds `[out/tp, in]`. `gather_output=True` calls `all_gather_tensor(gather_dim=1)`, which is
    `allow_in_graph`'d for fullgraph.
  * RPL holds `[out, in/tp]` and calls `funcol.all_reduce(reduceOp="sum")` on the native route.
  * The weights are `torch.empty`+init: copy shards in yourself or use `load_state_dict` with the full tensor.
  * A Megatron MLP `RPL(silu(CPL(x)))` with H=2048, I=6144 matched the full CPU MLP at TP=2/4/8
    (maxrel 4.2e-3 / 5.4e-3 / 6.1e-3). CPL(gather_output=True) gave 3.2e-3.
* **Sub-groups (DP x TP in one job)** are verified at world=4 with TP groups [[0,1],[2,3]]. Every rank creates
  every group, inside `active_mesh(full_mesh)` so the replica groups are SPMD-consistent (this is what
  `NeuronGroupCoordinator` does):

```python
from libtorch_neuronx_lite.parallel.replica_mesh import active_mesh
mesh = [[0, 1], [2, 3]]
with active_mesh(mesh):
    for ranks in mesh:
        g = dist.new_group(ranks)
        if rank in ranks: my_group = g
y = torch.compile(lambda t: funcol.all_reduce(t, "sum", my_group), backend=BACKEND, fullgraph=True)(x)
```

* vllm-neuron's own launcher is `vllm_neuron.utils.executor.MPExecutor(world_size, model_load)`. It forks
  workers, sets the same env, uses a gloo PG and calls `initialize_neuron_parallel_state` to build vLLM
  groups. It needs vLLM imports. I read its source but did not run it.

**Collectives verified** (each rank's input differs; reference = gloo on CPU). Two other things ran
correctly: two independent TP=2 torchrun jobs at once (cores 0-1 and 2-3, each with its own master_port and
root-comm port), and the sub-group case.

| op | world 2 | world 4 | world 8 (2 chips) |
|---|---|---|---|
| all_reduce / all_gather(d0, d1) / reduce_scatter / all_to_all_single [1024,2048] bf16 | all OK | all OK | all OK |

**all_reduce [1024,2048] bf16 (4 MiB) latency:**

| world | 1 all_reduce per graph (us per call, incl. dispatch) | inside a graph (8 chained, per AR) | algbw |
|---|---:|---:|---:|
| 2 (cores 0-1) | 188 | 72 | 58 GB/s |
| 4 (one chip) | 203-238 | 60 | 70 GB/s |
| 8 (two chips) | 332 | 103 | 41 GB/s |

`NEURON_RT_DISABLE_EXECUTION_BARRIER=1` made no measurable difference (237 vs 203 us per call, 58 vs 60 us
in-graph). A collective NEFF call costs about 130-230 us more than a non-collective one, so keep
collectives inside the model graph.

---------------------------------------------------------------------------------------------

## 4. Data-parallel replicas (`04_dp_replica.py`, `04_dp_replicas.sh`)

Run one independent process per logical core. No process group is needed.

```bash
for c in 0 1 2 3 4 5 6 7; do
  NEURON_RT_VISIBLE_CORES=$c NEURON_RT_ROOT_COMM_ID=127.0.0.1:$((61234+c)) python replica.py &
done; wait
```

* In each process the device is still `neuron:0`.
* 8 concurrent replicas share the host fine (192 vCPU, 2 TB RAM). Each ran a 2-matmul MLP block
  (`silu(x@W0)@W1^T`, T=2048) at **108-128 TF/s per core** vs 127.5 TF/s solo, about 920 TF/s aggregate.
  Cores on the same chip lose 0-15% under contention.
* A shared `NEURON_RT_ROOT_COMM_ID` across independent processes also worked for non-collective NEFFs, but use
  distinct ports anyway. TP groups must use distinct ports.
* **Two processes on the same core fail:** the second gets
  `RuntimeError: The PyTorch Neuron Runtime could not be initialized`.
* Warm replicas start fast: about 0.2-1.2 s from import to first result when the caches are warm.

---------------------------------------------------------------------------------------------

## 5. Compilation caching, buckets, compile time (`05_caching.py`, `mock_hybrid.py`)

**Where artifacts live:**

| cache | location | key |
|---|---|---|
| NEFFs | `$NEURON_LIBTORCH_CACHE_ROOT/neuron/native_neff_cache/<k0:2>/<k2:4>/<k>.neff` + `cache_key_index/`. The root defaults to `~/.cache/neuron_libtorch`, or `/tmp/neuron_compile_wdir_$USER` when `~` is NFS. `TORCH_NEURONX_NEFF_CACHE_DIR` overrides it. | graph structure + I/O shapes (+ `NEURON_CC_FLAGS` in the persistent key) |
| Dynamo precompile packages + AOTAutograd cache | `$TORCHINDUCTOR_CACHE_DIR/{dynamo,aotautograd}` (default `/tmp/torchinductor_$USER`). Scoped by collective topology: `.../topo_<mesh digest>/`. | function identity + guards. **Not the compiler flags (G10)** |

* The precompile cache (`TORCH_CACHING_PRECOMPILE`) and the AOTAutograd cache are switched on **only when
  `VLLM_NEURON_BACKEND=neuron_native` (or `neuron`) is set before importing lite**. Check with
  `libtorch_neuronx_lite.COMPILE_CACHE_STATE`:
  * with the variable set: `{'route': 'native', 'precompile': True, 'autograd_cache': True, ...}`
  * unset: `{'route': 'xla', 'precompile': False, ...}`
* **Shape buckets:** with `dynamic=False` every distinct input shape is its own Dynamo graph and NEFF.
  Pad T up to a bucket (e.g. 512, 1024, 2048, 4096) and mask or ignore the padding.
* Dynamo's `recompile_limit` is 8: the 9th distinct shape on one compiled callable raises
  `FailOnRecompileLimitHit: Hard failure due to fullgraph=True`. Raise it as vllm-neuron does
  (`torch._dynamo.config.cache_size_limit = 2**62`), or keep one compiled callable per bucket.
* After warm-up, `torch.compiler.set_stance("fail_on_recompile")` turns any unexpected shape into an error
  instead of a silent recompile (verified).
* Warm up every bucket at start-up. The first call of each bucket traces, then compiles or loads.

**Measured with a mock Qwen3.5-2B hybrid slice** (`mock_hybrid.HybridStack(4)`): 3 GDN layers (in_proj, NKI
conv1d, l2norm, NKI gdn_cte, gated RMSNorm, out_proj, SwiGLU MLP) and 1 gated-attention layer (D=256, NKI
attention_cte, MLP), random weights, one graph per bucket, 1 core:

| run | T=512 first call | T=2048 first call | process wall |
|---|---:|---:|---:|
| cold (empty NEFF and inductor caches) | **16.1 s** | **20.9 s** | 52 s |
| warm (both caches) | 0.0 s (+0.95 s at `torch.compile()` time to load the package) | 0.0 s | 12.9 s |
| NEFF cache only (inductor cache deleted) | 1.9 s (retrace, NEFF hit) | 2.3 s | 17.9 s |

* Steady state: **3.82 ms per forward at T=512 and 13.47 ms at T=2048** for the 4 layers. A naive x6
  gives about 23 ms / 81 ms for the 24-layer torso on one core, before optimization (see 6).
* Other cold compile times: one matmul graph 8-12 s, gdn_cte S=2048 graph 7.7 s, a small collective
  graph about 1.5-2.5 s per rank (all ranks compile the same key at once and share the NEFF via the cache).
* Python imports take about 3 s.

---------------------------------------------------------------------------------------------

## 6. Profiling (`06_profile.py`, `06_profile.sh`, `06f_attribute.py`, `06d_overhead.py`)

**(a) Find the NEFF a compiled callable runs** using lite's execution hook:

```python
from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context, clear_execute_context
seen = {}
set_execute_context(ExecuteContext(pre_execute_hook=lambda inputs, md: seen.setdefault(md.neff_id, md.artifact_dir)))
compiled_fn(*args); clear_execute_context()
neff = glob.glob(os.path.join(next(iter(seen.values())), "*.neff"))[0]
```

**(b) In-process capture with NRT inspect.** This is the same call vllm-neuron's `NeuronProfiler` and
`MPExecutor.start_profile` make. It brackets any window of real executions:

```python
from libtorch_neuronx_lite import envs as lenvs
rt = torch.classes.neuron.Runtime()
rt.start_profiling(out_dir, ["device_profile", "system_profile"], None, None, lenvs.get_neff_cache_dir(BACKEND))
...run + fence...
rt.stop_profiling()
# -> out_dir/i-<inst>_pid_<pid>/<ts>/{profile_nc_0_session_0.ntff, ntrace.pb, trace_info.pb, cpu_util.pb, host_mem.pb}
#    and out_dir/neffs/<..>/<key>.neff. Copy the NEFFs next to the NTFF, then:
#    neuron-explorer view --output-format summary-json -d <that ts dir>
```

**(c) Offline per-NEFF capture and analysis.** This gives clean device time for one execution:

```bash
neuron-explorer capture -n $NEFF -s out/profile.ntff --profile-nth-exec=2      # writes out/profile_exec_2.ntff (!)
neuron-explorer view --output-format summary-json -n $NEFF -s out/profile_exec_2.ntff > summary.json
#   json: {"device_profile_profile_exec_2": {"summary": [{total_time, mfu_estimated_percent (fraction), ...}]}}
neuron-explorer view -n $NEFF -s out/profile_exec_2.ntff --output-format parquet --output-file out/pq --ingest-only
#   --ingest-only is required, otherwise view starts a UI server on :3001 and never exits
pip install --target /tmp/howto_pylib duckdb     # the image has no duckdb/pyarrow
PYTHONPATH=/tmp/howto_pylib python 06f_attribute.py out/pq
```

* Per-kernel time inside a fused graph comes from the `Instruction` parquet table (250k rows for the
  4-layer graph). It has columns `hlo_name`, `nki_source_location`, `engine`, `pcore_idx`, `start_ts`,
  `end_ts` and `duration_ns`.
* `06f_attribute.py` groups instructions by NKI kernel (via the source file), dot, transpose and other ops.
  For each group it reports the union of busy intervals on physical core 0.
* `FrameworkNode` names nodes by module path (`GDNLayer[0][0]_custom-call.203`). Its start/end spans
  interleave across the schedule, so do not use them as durations.

**Measured device times:**

| graph | device total_time | notes |
|---|---:|---|
| matmul [2048,2048]x[2048,6144] | 382 us | MFU 86%, PE active 93% |
| gdn_cte B=1 H=16 D=128 S=2048 | 1075-1079 us | MFU 17%, Scalar 72% / Vector 83% / PE 76-78% active |
| 4-layer hybrid T=2048 | 13.5 ms | MFU 49%, MBU 33% |

Breakdown of the 4-layer graph (busy time, categories overlap):

| category | busy time | share of wall |
|---|---:|---:|
| fp32 elementwise / reduce (norms, l2norm, softplus, gating) | 5.3 ms | 39% |
| NKI gdn_cte (3 calls) | 3.8 ms | 28% |
| dot (matmuls) | 3.2 ms | 24% |
| NKI gdn_conv1d_prefill (3 calls) | 1.26 ms | 9% |
| transpose | 0.84 ms | 6% |
| compiler custom-calls | 0.47 ms | 3.5% |
| NKI attention_cte | 0.42 ms | 3% |

The fp32 elementwise work and layout transposes are the first optimization targets.

**Host vs device time.** The gap between host-measured and profiled time is per-execution runtime overhead
(`06d_overhead.py`, gdn_cte S=2048):

| setting | 1 kernel/graph | 4 kernels in one graph, per kernel | blocking call |
|---|---:|---:|---:|
| default NRT | 3153 us/call | 1062 us | 1698 us |
| `NEURON_RT_IO_RING_CACHE_SIZE=32` | 1192 us/call | 1061 us | |
| + `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` | 1185 us/call | | |

---------------------------------------------------------------------------------------------

## 7. Gotchas hit (error -> fix)

| # | symptom | cause / fix |
|---|---|---|
| G1 | no precompile cache, XLA torch overrides installed (silent) | `libtorch_neuronx_lite` treats an **unset** `VLLM_NEURON_BACKEND` as the XLA route, even though vllm-neuron defaults to native. Set `VLLM_NEURON_BACKEND=neuron_native` before `import libtorch_neuronx_lite`. Check `COMPILE_CACHE_STATE['route']=='native'`. |
| G2 | `NRT:nrt_load_collectives NEURON_RT_ROOT_COMM_ID is not set; cannot load NEFF with collective operations!` then `RuntimeError: Could not load the model status=2 message=Invalid`, for a plain single-core matmul | Native LNC=2 NEFFs need a CCOM root. Always set `NEURON_RT_ROOT_COMM_ID=127.0.0.1:<port>`: identical within a TP group, distinct across groups. |
| G3 | timings far too small; `torch.neuron.synchronize()` returns in 20 us | It does not wait for executions. Block with `.cpu()` or the fence graph. |
| G4 | `Failed to schedule neff execution. status=7 message=Execution Queue Full` | Max in-flight = `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` (default 8). Raise it (vllm-neuron uses 32) and/or fence every <= N calls. |
| G5 | pipelined call 3x slower than device time (gdn_cte 3.15 ms vs 1.06 ms) | Set `NEURON_RT_IO_RING_CACHE_SIZE=32` (vllm-neuron worker default): 1.19 ms. |
| G6 | `dist.barrier()` -> `NotImplementedError: You should register PrivateUse1HooksInterface...` | Use a CPU gloo collective as the barrier: `dist.all_reduce(torch.zeros(1))`. |
| G7 | `NCC_EVRF066 NKI kernel '...' is configured with LNC=1, but neuronx-cc is using LNC=2` | The NKI grid must equal LNC. Use `k[2](...)` or no brackets, never `k[1]`. |
| G8 | NKI not used and no warning | `NF.mlp`/`NF.qkv_proj`/`NF.flash_attention` gates fall back to torch: at TP=1 for H=2048/I=6144/fused-qkv>4096, and for D=256 attention at any TP. Call nkilib via `wrap_nki` directly, or check the FX graph for `nki_kernel_wrapper`. |
| G9 | `FailOnRecompileLimitHit: Hard failure due to fullgraph=True` on the 9th input shape | Raise `torch._dynamo.config.cache_size_limit`/`recompile_limit`, bucket shapes, then `set_stance("fail_on_recompile")`. |
| G10 | changing `compiler_args`/flags has no effect (0.0 s "compile") | The Dynamo precompile/AOT package in `TORCHINDUCTOR_CACHE_DIR` restores the old NEFF and ignores flags. Delete that dir (or use a fresh one) when changing flags. With it deleted, the NEFF cache recompiled correctly. Also, in one process the same graph with different `compiler_args` raises `RuntimeError: cache_key ... was already compiled with compiler_args ...` (from source). |
| G11 | eager `x.to(dev, dtype=torch.float32)` -> `Expected self.dtype() == dst.dtype()`; eager `xd.t().contiguous()` -> `Expected self.is_contiguous()` | Convert dtype on the CPU before `.to(dev)` and do layout ops inside compiled graphs. Eager device ops work but are slow (21 ms for a 256^2 matmul). Put everything in graphs. |
| G12 | `UserWarning: Set seed for neuron device does not take effect` | Device RNG is not seeded. Generate random tensors on the CPU and `.to(dev)`. |
| G13 | `gdn_cte_torch_ref` AssertionError `algorithm='schur' requires mask_pack` | Pass `mask_pack=torch.from_numpy(build_mask_pack())` to the reference too. gdn_cte also needs L2-normalized q/k, fp32 g_log/beta/state and S % 128 == 0. |
| G14 | `neuron-explorer capture -s x.ntff` then `view` says file not found | capture writes `x_exec_<N>.ntff` (`profile_exec_2.ntff`). |
| G15 | `neuron-explorer view --output-format parquet` never returns | It starts a UI server on :3001 after writing. Add `--ingest-only`. |
| G16 | `view -d <session dir>`: `no .neff file found` | The in-process capture stores NEFFs in `<out>/neffs/..`. Copy them into the session dir. |
| G17 | second process on a busy core: `The PyTorch Neuron Runtime could not be initialized` | One process per logical core. |
| G18 | harmless log noise | `BUILTIN_MATCH guard ... dropped with caching_precompile=True`, `aliasing_analysis.py INFO`, `Defaulting to PJRT_DEVICE=CPU`. |

---------------------------------------------------------------------------------------------

## 8. What was not verified / not done

* `MPExecutor` and vLLM-parallel-state groups (`initialize_neuron_parallel_state`, `get_tp_group()`) were
  not run. The standalone path uses raw `torch.distributed` + funcol.
* No real Qwen3.5 weights or HF parity. `mock_hybrid.py` has the right shapes and ops, but RoPE is
  omitted and the weights are random. Only the individual kernels are numerically checked.
* The NKI MLP/QKV kernels at TP>=2 were only gate-checked, not run.
* Multi-node and EFA were not tested. TP across 2 chips (world 8) was tested.

## Script index (`howto_scripts/`, run on the trn2 host from `~/howto/scripts/`)

| script | question |
|---|---|
| `common_env.sh` | env |
| `01_matmul.py/.sh`, `01b_queue_diag.py/.sh` | Q1 placement, compile, async semantics, timing |
| `02_nki_kernels.py/.sh` (`02_rerun.sh` also prints route detection), `09_nf_gates.py/.sh` | Q2 NKI kernels, NF dispatch gates |
| `03_tp_collectives.py/.sh`, `03b_subgroups.py`, `03b_dp_x_tp.sh` | Q3 TP, collectives, CPL/RPL, sub-groups |
| `04_dp_replica.py`, `04_dp_replicas.sh` | Q4 DP replicas |
| `05_caching.py/.sh`, `mock_hybrid.py` | Q5 caching, compile times, recompile limit |
| `06_profile.py/.sh`, `06e_find_neff.py`, `06f_attribute.py`, `06d_overhead.py/.sh` | Q6 profiling, host vs device overhead |
| `07_eager_probe.py/.sh` | eager-op support, H2D/D2H, exec-barrier test |
| `08_autocast.py/.sh` | auto-cast default, stale-cache-with-flags gotcha |
