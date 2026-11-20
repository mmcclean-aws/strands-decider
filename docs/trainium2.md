# The Trainium2 backend

This document describes how `--device neuron` runs the 2B Qwen3.5 model on AWS Trainium2. It
covers the compiled model, the kernels, the shared prefix, tensor parallel, and what was
measured. [docs/inference.md, Serving on AWS Trainium](inference.md#serving-on-aws-trainium)
covers how to use it, [deploy/trainium2/](../deploy/trainium2/README.md) how to host it, and
[evaluation/results.md](../evaluation/results.md#serving-on-aws-trainium2-accuracy-and-latency)
the measurements. Module paths are relative to `src/strands_decider/`.

It uses the Neuron SDK's PyTorch-native backend (`libtorch_neuronx_lite`, SDK 2.33): a model is
`torch.compile`d with `backend="neuron_native_lite"`, and each input shape compiles to one
graph that runs on a logical NeuronCore. No NxDI, no vLLM, no torch-xla tracing.

## One graph per shape

A request is one forward per chunk of questions: there is no decode loop. So the whole model,
from the embedding to the calibrated probabilities, is one graph: the 24 layers (18 Gated
DeltaNet, 6 gated attention, an MLP after each), the final norm, and the pointer readout in
fp32 (gather `<answer>` and the options' last tokens, LayerNorm, the q and k projections, the
scaled dot product, temperature, masked softmax). The host keeps what `SystemOneEngine` does
around the forward: rendering, fitting the window, option positions and `to_answer` are its
code, because `NeuronEngine` subclasses it (`neuron_engine.py`).

A graph has fixed shapes, so the engine pads. Rows of state plus question go to the smallest
token bucket that holds them (256 to 4,096) and are run in as few forwards as the compiled batch
sizes allow. Options are padded to 32. Right padding needs no mask: attention is causal and the
Gated DeltaNet recurrence runs forward, so no real position reads a padded one.

transformers' Qwen3.5 does not compile this way, so `neuron_model.py` re-implements it for
static shapes. The LoRA adapter is merged into the base weights at load, in fp32 on the bf16
base weights, then rounded once. Each Gated DeltaNet layer's q, k, v and z projections are one
head-major weight, so one einsum gives `[B, heads, T, 128]` without a transpose; likewise the
attention layer's query, gate, key and value. On the CPU in fp32 it gives the same
probabilities as `StrandsDeciderModel` to within 2.1e-6 on the 54 answers of
`evaluation/device_parity.py` (`tests/test_neuron_model.py` checks it on a tiny random torso).

## Kernels

The port took a 2,048-token forward on one core from 174.3 to 67.3 ms (20.3% to 52.8% MFU) in
three steps:

1. Head-major projection layouts, with nkilib's Gated DeltaNet (`gdn_cte`) and attention
   (`attention_cte`) kernels: 174.3 ms.
2. Fused kernels around the Gated DeltaNet core (`neuron_kernels/fused_ops.py`), NKI norms and
   GEMM weights pre-transposed to `[K, N]` at load (`neuron_kernels/layouts.py`): 78.1 ms.
3. A new Gated DeltaNet kernel in WY form (`neuron_kernels/gdn_fast.py`): 67.3 ms.

In this package the fused path measures 68.3 ms (52.4% MFU) and nkilib's kernels alone 153.6 ms
(`NeuronConfig(fast_kernels=False)`), on the same SDK.

**gdn_fast.** Profiled, nkilib's `gdn_cte` is limited by instruction overhead, not arithmetic:
28 matmuls per 128-token chunk and about 40 small vector and scalar instructions, each with a
fixed cost of 65 to 290 ns, emitted one lane at a time. `gdn_fast` applies the chunk's
triangular inverse to both right-hand sides once, off the recurrence, so the sequential path per
chunk is four matmuls and two vector instructions. It processes four heads per instruction, and
interleaves one batch of chunks' state updates with the next batch's inverse, so every engine
has independent work. 0.55 ms against 1.30 ms per call at 2,048 tokens, with the same bf16
error (3.6e-3 against 3.3e-3); an fp16 mode is 8x more accurate at the same speed
(`NeuronConfig.gdn_dtype`).

**The fused prologue.** The depthwise convolution, SiLU, the L2 norm of q and k, and the gates
are one kernel. The convolution and the transpose to token-major are one tensor-engine
instruction per block and tap, with diagonal tap matrices accumulated in fp32. The rest of a
Gated DeltaNet layer went from 3.98 to 1.06 ms at 2,048 tokens.

**A compiler finding.** When neuronx-cc fuses an RMSNorm into the head-major einsum after it,
that GEMM runs 2 to 3x slower (1.13 against 0.58 ms). An NKI RMSNorm puts a custom-call boundary
in front of the GEMM, which prevents the fusion. In front of the token-major MLP GEMMs the fusion
is fine, so the MLP keeps its torch norm.

**Attention.** nkilib's `attention_cte` takes head dim 256 and a cached prefix's K and V directly,
but vLLM Neuron's attention op gates on head dim 128 and silently falls back to torch, so the
model calls the kernel through `wrap_nki` itself. `attention_cte` with K laid out `[B, D, S]` is
the fastest variant.

## Shared prefix

Several questions about one state encode the state once, as on other devices. `prefix_forward`
runs the state, padded to a bucket, and returns its caches: per Gated DeltaNet layer the final
recurrent state and the last three pre-convolution tokens, per attention layer K and V. The
padding is made inert: its gates are zeroed for the recurrence and masked out of attention. Then
`suffix_forward` runs every question, padded to 128 tokens, from those caches: the recurrence
starts from the state, the convolution from the three tokens. A state over the largest bucket or
a question over 128 tokens takes whole prompts instead.

## Tensor parallel

A long state is latency-bound on one core. `neuron_model.shard` splits each layer's Gated
DeltaNet and query heads and the MLP over 2 or 4 ranks (key and value heads split, or replicated
once there are more ranks than heads), and each layer's two output projections are all-reduced
inside the graph, in four token chunks so that one chunk's all-reduce runs while the next
computes. At 4,096 tokens TP2 measured 87.9 ms against 133.6 ms on one core, at 40.8% MFU: the
in-graph all-reduces (402 us per 16 MiB, 48 per forward) do not overlap with compute. The
calibrated cost model ranks context parallel over two cores best for long prompts (81.5 ms
modelled), which is not implemented.

The ranks must run the same graphs in the same order, so they run the same engine on the same
requests: rank 0 broadcasts each request, and the others follow (`neuron_tp.py`).

## Accuracy

Device bf16 is compared with the CPU in fp32. On `evaluation/device_accuracy.py`'s 1,019 labeled
rows, the gate is the bf16 floor measured by running the same code in bf16 on the CPU: mean |dp|
at most 0.005, p99 0.02, max 0.05, changed answers only at near-ties, accuracy, ECE and NLL
within 0.005, 0.01 and 0.005.

## Limits

- Only the 2B Qwen3.5 torso's layout: as many Gated DeltaNet key heads as value heads, equal head
  dims, default RoPE. The kernels are written for its dimensions (hidden 2,048, 16 heads of 128).
- The pointer readout only; token-readout checkpoints are refused, as by the other v1 engines.
- At most 32 options per question. No images.
- `serve` compiles every configured shape before it answers `/health`: 67 by default, 42 minutes
  cold, 16 with the Neuron SDK's cache warm, where torch still traces each shape again. Fewer
  shapes, or loading torch's compiled packages instead of tracing, would shorten it.
