# Strands Decider 2B on trn2.48xlarge — results summary (2026-11-10)

Model: `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` (Qwen3.5-2B-Base torso: 18 Gated DeltaNet + 6 gated GQA
attention layers, hidden 2048, dense SwiGLU 6144, LoRA r=48 merged, pointer head dim 768 fp32). One prefill
forward per request; no decode.
Stack: Neuron SDK 2.33 PyTorch-native backend (`libtorch_neuronx_lite` 2.13.0.1.0, torch 2.13, neuronx-cc 2.28,
nki 0.7.0, image `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`)
on a trn2.48xlarge host, LNC=2 (64 logical cores, 4 per chip). No NxDI.
MFU = model FLOPs (2·N_nonembed·T + causal attention + GDN chunk FLOPs; 2.871 GFLOP/token at T=2048) /
(time × 166.75 TF/s per logical core [667 TF/s bf16 per chip / 4]). The profiler's own peak is 158 TF/s,
which would give ~5% higher MFU; we report the conservative figure.

## Optimization trajectory (one LNC2 core, B=1, full model, latency = blocking forward incl. read-back)

| step | T=512 | T=2048 | T=4096 | MFU @2048 |
|---|---|---|---|---|
| v0: torch ops + nkilib gdn_cte + attention_cte, token-major GDN prologue | 120 ms (est.) | — | — | — |
| v1: head-major projections (einsum straight to [B,H,T,D]) | 45.2 | 174.3 | 346.7 | 20.3% |
| v2: + fused NKI prologue/epilogue, NKI norms, pre-transposed GEMM weights | 23.0 | 78.1 | 154.7 | 45.4% |
| v3: + gdn_fast (WY-form GDN kernel, 2.4x nkilib) | **20.7** | **67.3** | **133.6** | **52.8%** |

Batched (v3): 256x4 35.3 ms (49.9%), 256x8 68.3 ms (51.2%), 512x4 67.1 ms (52.2%), 1024x2 67.8 ms (51.9%).
Intermediate buckets (v3): 256: 14.8 ms (30.3%), 768: 27.0 (49.2%), 1024: 35.2 (50.3%), 1536: 52.2 (50.9%),
2560: 85.7 (52.0%), 3072: 101.5 (52.8%), 3584: 119.4 (52.6%).

## Tensor parallel (v3 kernels, fused ops, intra-chip adjacent logical cores)

| config | T=2048 | T=3072 | T=4096 | MFU @4096 (over P cores) |
|---|---|---|---|---|
| TP2 | 47.8 ms | 72.1 (2 chunks) / 68.1 (4 chunks) | 93.2 / **87.9 (4 chunks)** | 38.5% / 40.8% |
| TP2, all-reduces removed (timing only) | | | 71.7 | 50.0% |
| TP4 | 27.3 ms | | 56.9 | 31.6% |
| TP2, v1 kernels | 82.2 | | 160.9 | 22.3% |

In-graph collectives (world=2, [T,2048] bf16, chain of 8 in one graph): all-reduce 4 MiB 72 us, 8 MiB 213 us,
16 MiB 402 us; RS+AG 72 / 123 / 344 us. They do not overlap with compute; 4-way token chunking of each layer's
tail recovers ~6 ms at T=4096.

## Full-instance data-parallel throughput (v3, one replica per logical core, pipelined)

| replicas | shape | per-core tok/s | ms/forward | aggregate tok/s | MFU per core |
|---|---|---|---|---|---|
| 1 | 2048x1 | 30,640 | 66.8 | — | 52.8% |
| 16 (4 chips) | 2048x1 | 26,392 | 77.6 | 422,275 | 45.4% |
| 64 (16 chips) | 2048x1 | 26,687 | 76.7 | **1,707,996** | **46.0%** |
| 16 | 512x4 | 26,741 | 76.6 | 427,856 | 45.4% |
| 64 | 512x4 | 26,985 | 75.9 | **1,727,018** | **45.9%** |

(v2 kernels, 16 replicas at 2048x1: 24,659 tok/s/core, 42.4%.)

## Accuracy (device bf16 vs CPU fp32 HF+PEFT golden; 1,019 labeled rows from the repo's eval sets)

Gate (based on the bf16 numerics floor measured by running the same code in bf16 on CPU, which drifts up to
0.016 on 24 rows, row-for-row comparable to the device): mean|dp| <= 0.005, p99 <= 0.02, max <= 0.05,
argmax flips only at golden top-2 margin < 0.05, accuracy within 0.5 pp, ECE within 0.01, NLL within 0.005.

| run | mean|dp| | p99 | max | flips (margins) | accuracy dev / golden | ECE dev / golden | NLL dev / golden | PASS |
|---|---|---|---|---|---|---|---|---|
| v1 (DP) | 0.0020 | 0.0107 | 0.024 | 2 (0.022, 0.0015) | 0.8037 / 0.8027 | 0.2079 / 0.2068 | 0.4826 / 0.4824 | yes |
| v2 (DP) | 0.0021 | 0.0112 | 0.035 | 2 (0.0012, 0.0015) | 0.8018 / 0.8027 | 0.2059 / 0.2068 | 0.4826 / 0.4824 | yes |
| **v3 (DP)** | 0.0021 | 0.0128 | 0.024 | 3 (0.014, 0.0012, 0.0087) | 0.8018 / 0.8027 | 0.2060 / 0.2068 | 0.4823 / 0.4824 | **yes** |
| v3 DP, fp16 GDN compute (`DECIDER_GDN_DTYPE=fp16`, +2% latency) | 0.0020 | 0.0104 | 0.029 | 2 (0.022, 0.0012) | 0.8027 / 0.8027 | 0.2069 / 0.2068 | 0.4826 / 0.4824 | yes |
| **v3 TP2 (4 chunks)** | 0.0022 | 0.0145 | 0.024 | 5 (max 0.034) | 0.8037 / 0.8027 | | | **yes** |

The CPU golden itself reproduces the model card's published per-task accuracies where the sets coincide
(gen:cross_artefact 0.710 n=31, gen:missing_fact 0.724 n=29). Per-op equivalence on real activations (layer 0
and 3, T=512): every device op matches CPU bf16 error (rms_norm 1.7e-3, einsum 2.1e-3, conv 1.3e-3,
transcendentals ~1e-7, attention 2.0e-3); only the bf16 GDN kernel is higher (2.1e-3 vs 0.9e-3), and an fp32 GDN
core does not change end-to-end drift.

Request-level parity (repo `evaluation/device_parity.py` request set: 18 requests, 54 answers, states 40-3,000
tokens, 1 and 5 questions, all question types) vs the reference `SystemOneEngine` on CPU fp32:
v1 batched 0/54 changed (max |dp| 0.0081); v1 shared-prefix 0/54 (0.0139); v3 shared-prefix 0/54 (0.0113);
v3 TP2 shared-prefix 0/36 (0.0097). The repo's own device parity: MPS 0.0051, MLX 0.0138.

## End-to-end request latency (engine incl. host tokenization; v3)

| request (device-parity set) | tokens | DP, one core | TP2 (2 cores) |
|---|---|---|---|
| 1 question, short state | ~100 | 15.1 ms | |
| 5 questions, short state (shared prefix) | ~390 | 43.3 ms | |
| 1 question, ~350-token state | ~440 | 21.0 ms | 21.8 ms |
| 5 questions, ~350-token state | ~720 | 48.1 ms | 40.4 ms |
| 1 question, ~2,500-token state | ~2,600 | **93.8 ms** (2816 bucket) | 68.6 ms |
| 5 questions, ~2,500-token state | ~2,880 | 113.3 ms | **81.6 ms** |

All 18 requests: 0 changed answers on either configuration (DP max |dp| 0.0113, TP2 0.0097).
Host overhead (render + tokenize + fit) 0.2-5 ms. Before the shared-prefix port the 5-question long-state request
took 1,712 ms (5x the state re-encoded).

## Kernel-level results
- gdn_fast (WY form, 4-head batched vector/scalar ops, HW-generated DMAs): 0.55 ms vs nkilib gdn_cte 1.30 ms
  (with NEURON_RT_IO_RING_CACHE_SIZE=32; 3.01 ms without) at B=1,H=16,T=2048; engines 85-90% busy; bf16 rel
  error 3.6e-3 (nkilib 3.3e-3); supports initial state + final state; fp16 mode 4.6e-4 error at the same speed.
- Fused GDN prologue (conv1d K=4 + SiLU + L2 + beta/g as tensor-engine diagonal-tap matmuls): 0.18 ms @2048;
  GDN layer without core 3.98 -> 1.06 ms; MLP 1.25 -> 1.13 ms (82% of peak) via pre-transposed [K,N] weights.
- Key compiler finding: neuronx-cc fusing an RMSNorm into a head-major einsum makes that GEMM 2-3x slower; NKI
  norms as custom-call boundaries avoid it.
- attention_cte supports head_dim 256 directly (0.40 ms @2048, 0.96 ms @4096, tp_k=False fastest).

## Recommended deployment (trn2.48xlarge)
- Single-core (LNC2) data-parallel replicas; dynamic batching per bucket (B*T <= ~2048 tokens) for short
  requests; buckets 256/512/768/1024/1536/2048/2560/2816 (+3072/4096 on TP2).
- Requests > ~2.8K tokens, or multi-question requests with a > ~2K-token state, route to a pool of TP2 pairs
  (adjacent logical cores in one chip), 4-way chunked all-reduce, prefix buckets incl. 2560, suffix batch {2,4,6,8}.
- Next step for long prompts: the v3-calibrated cost model (fits 7 DP points within 3%, predicts the 4 held-out
  TP measurements within 2.5-12.6%) ranks CP2 with the two-pass GDN state correction best at T=4096 (81.5 ms, 44% MFU
  vs TP2 91 ms model / 87.9 ms measured), because its traffic is only K/V + state all-gathers while in-graph
  all-reduce costs ~3.1x the nccom-test numbers and does not overlap compute.
- JevBench's length distribution (p50 178, p90 2,471, p99 3,072 tokens) puts 96.6% of requests on the DP pool (72% are <= 512 tokens, 86% <= 2048).
- Shared-prefix path for every multi-question request.
- Runtime env: NEURON_RT_IO_RING_CACHE_SIZE=32, NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=32,
  TORCH_CACHING_PRECOMPILE=0 when iterating on kernels.
