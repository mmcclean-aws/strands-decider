# Strands Decider 2B on AWS Trainium2 (Neuron SDK 2.33, PyTorch-native)

Port of `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` to trn2.48xlarge on the SDK 2.33
PyTorch-native backend (`libtorch_neuronx_lite`, `torch.compile(backend="neuron_native_lite")`), no NxDI.
Headline: **52.8% MFU at 67 ms** for a 2,048-token forward on one logical core; **46% MFU** with all 64
cores loaded (1.71M tokens/s per instance); accuracy within the bf16 floor of the CPU fp32 golden on 1,019
labeled eval rows; every request class in the repo's device-parity set under 100 ms (TP2 for long-state
multi-question requests). Full numbers: `results/summary.md`.

## Reports
- `report/architecture.html` — model architecture (Gated DeltaNet + gated attention torso, pointer readout).
- `report/cost_model.html` — sharding cost model (DP/TP/TP+SP/CP/Ulysses/hybrid/TSP/PP; EP n/a), calibrated.
- `report/optimization.html` — port, kernels, accuracy methodology, final performance, deployment.

## Layout
| path | what |
|---|---|
| `neuron/decider_model.py` | static-shape PyTorch re-implementation (merged LoRA, head-major layouts, TP sharding, shared-prefix prefix/suffix forwards); exact vs HF+PEFT on CPU fp32 (max dp 5e-7) |
| `neuron/kernels/gdn_fast.py` | WY-form chunked Gated DeltaNet NKI kernel (2.4x nkilib `gdn_cte`), initial/final state |
| `neuron/kernels/fused_ops.py` | fused GDN prologue (conv1d+SiLU+L2+beta/g), NKI RMSNorm / gated RMSNorm, pre-transposed GEMM layouts, fast prefix/suffix/attention layers, TP head counts |
| `neuron/nki_ops.py` | kernel bindings (nkilib `attention_cte` d=256 with prefix KV, nkilib `gdn_cte`, `FastGdn`) |
| `neuron/engine.py` | serving engine: strands-decider prompting/tokenization -> bucketed (T, B) device forwards, shared-prefix path for multi-question requests |
| `neuron/accuracy.py` | accuracy gate: golden tensors + labeled eval metrics (accuracy, ECE, NLL, flips) |
| `neuron/device_runner.py` | single-core: `layers` bring-up, `goldens` gate, `bench` |
| `neuron/tp_runner.py`, `neuron/tp_engine_test.py` | tensor parallel (torchrun, one process per logical core) |
| `neuron/test_engine.py` | request-level parity vs the reference `SystemOneEngine` |
| `neuron/bench_*.py`, `neuron/equiv_layer0.py` | component / attention / GDN / collective benchmarks; per-op equivalence |
| `tools/make_goldens.py`, `tools/make_request_goldens.py` | CPU fp32 reference generation (`goldens/`) |
| `tools/cost_model.py` | analytic cost model (`--measured` = calibrated to device data) |
| `collectives/` | trn2 LNC2 collective latencies (nightly nccom-test benchmarks) |
| `docs/native_stack_howto.md` | verified how-to for the SDK 2.33 native backend (API, TP, profiling, gotchas) |
| `results/` | raw benchmark / gate JSON and logs, `summary.md` |

## Reproduce (on a trn2.48xlarge host with the SDK 2.33 image; repo at `~/decider`, strands-decider cloned into `ext/strands-decider`)
```bash
export VLLM_PLUGINS=none VLLM_NEURON_BACKEND=neuron_native NEURON_LOGICAL_NC_CONFIG=2 \
       NEURON_RT_ROOT_COMM_ID=127.0.0.1:62100 NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=32 \
       NEURON_RT_IO_RING_CACHE_SIZE=32 NEURON_LIBTORCH_CACHE_ROOT=/tmp/decider_cache \
       HF_HOME=/tmp/hf DECIDER_FAST=1 NEURON_RT_VISIBLE_CORES=8
cd ~/decider/neuron
python3 device_runner.py bench --shapes 2048x1,512x4                       # latency / MFU
python3 device_runner.py goldens --gdir ~/decider/goldens/eval --buckets 512,1024,2048,4096   # accuracy gate
python3 test_engine.py ~/decider/goldens/requests.json --device neuron     # request-level parity
CORE_BASE=12 torchrun --nproc_per_node=2 tp_runner.py bench --shapes 4096x1  # TP2 (DECIDER_TP_CHUNKS=4)
```
