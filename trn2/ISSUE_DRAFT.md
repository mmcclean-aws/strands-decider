**Title:** feat: serve on AWS Trainium2 (`--device neuron`)

## Proposal

Add AWS Trainium2 as a serving device, next to `cuda`, `mps`, `cpu` and `mlx`: `strands-decider serve CKPT --device neuron`. I have a working port of `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` on a trn2.48xlarge (Neuron SDK 2.33, PyTorch-native backend `libtorch_neuronx_lite`, no NxDI or vLLM) and would like to upstream it as a series of small PRs. Before I start, I'd like to agree on the shape below.

Measured in the integrated package on one trn2.48xlarge (Neuron SDK 2.33), bf16, LNC=2:

- A 2,048-token forward takes 68 ms on one of the 64 logical cores, 52.4% MFU (the port measured 1.71M tokens/s with all 64 serving).
- On `evaluation/device_parity.py`'s 54 answers, 0 change against `SystemOneEngine` on CPU fp32; max |dp| 0.0113 (MPS 0.0051, MLX 0.0138).
- On 1,019 labelled eval rows, accuracy 0.8027 vs 0.8027, ECE 0.2069 vs 0.2068, NLL 0.4823 vs 0.4824; 4 answers change, all near-ties.
- Five questions on a 2,500-token state: 121 ms on one core, 88 ms on TP2, 61 ms on TP4.

## Design

- **Engine.** `NeuronEngine(SystemOneEngine)`, as `MLXEngine` does. It reuses rendering, `_fit`, `_chunks`, option positions, temperatures and `to_answer`, and overrides only the two forward paths, `_slot_probs_batched` and `_slot_probs_shared_prefix`. Those pad rows into graphs compiled ahead of time for fixed (tokens, rows) buckets.
- **Model.** `neuron_model.py` is the Qwen3.5 torso and pointer head as one static-shape module. The LoRA adapter is merged in fp32 at load, and the readout and masked softmax run on the device. It loads through `StrandsDeciderModel.load`, so Hub ids, revisions and both head files work as on other devices. It is pure torch, so CI can test it on CPU against `StrandsDeciderModel` with a tiny random Qwen3.5. It refuses token-readout checkpoints, like the other v1 engines.
- **Kernels.** NKI kernels for the Gated DeltaNet chunk rule and fused layer prologue/epilogue in `neuron_kernels/`, with bindings to nkilib's `attention_cte`.
- **Install.** The Neuron packages are not on PyPI; PyPI has 0.0.1 placeholders. I propose a `neuron` extra whose minimum versions exist only on the Neuron repository, so an install without the extra index fails instead of installing placeholders:
  `pip install "strands-decider[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com`.
  `nkilib` ships only in the Neuron DLC image, so the DLC is the supported environment, and `neuron_available()` checks for nkilib and a device. All Neuron imports are lazy. Like `mlx`, `--device neuron` is never selected automatically.
- **Tests.**
  - CI covers the reference model and the engine logic on CPU.
  - Kernel and device tests get a `neuron` pytest marker. They skip without hardware; I run them on trn2 and quote the output in each PR.
  - The tensor-parallel sharding is tested on CPU under the existing `distributed` marker.
- **Typing.** The torch code is fully typed for `mypy ./src`. The two NKI kernel modules (`neuron_kernels/gdn_fast.py`, `fused_ops.py`) are exempted from the untyped-def checks, since nki's tracing API has no types.

The work is done and verified on trn2, as the stacked branches `neuron/01-model` ... `neuron/07-results` of my fork.

## PR series

1. `feat(neuron)`: the static-shape model (pure torch), with CPU parity tests.
2. `feat(neuron)`: `--device neuron` (engine, plumbing, extra, docs) on nkilib's kernels.
3. `feat(evaluation)`: a device accuracy gate for any device (CPU fp32 goldens, |dp|, flips, ECE, NLL) and a Neuron MFU benchmark.
4. `perf(neuron)`: the fused NKI kernels.
5. `feat(neuron)`: tensor parallel, experimental, as `python -m strands_decider.neuron_tp serve` (rank 0 serves HTTP and broadcasts each request); not wired into `strands-decider serve` yet.
6. `docs(deploy)`: `deploy/trainium2` for EC2: one `serve` per logical core, the environment, the buckets, a cost model for the sharding choices, Neuron how-to notes.
7. `docs`: a Trainium2 section in `evaluation/results.md`, `docs/trainium2.md`, the data in `research/data/`.

Two small changes outside the new modules: `StrandsDeciderModel.load` passes `torch_device="cpu"` to `PeftModel.from_pretrained` (with `libtorch_neuronx_lite` imported, PEFT otherwise stages the adapter on the `neuron` device and safetensors fails), and `training.evaluate` refuses `--device neuron` as it refuses `mlx`.

## Questions for maintainers

1. Are the module names and the layout above OK? The alternative is one `strands_decider/neuron/` subpackage.
2. Is an extra that needs `--extra-index-url` acceptable?
3. Is the mypy exception for NKI kernel bodies acceptable?
4. For tensor parallel inside `serve`, should it be in scope, or engine-level and experimental first?
5. Do you want the HTML design reports, or only the Markdown summary (docs/trainium2.md)?
6. The Neuron how-to scripts are kept as they ran and excluded from ruff (`extend-exclude`). Acceptable, or should they be fork-only?
7. `serve` compiles 67 shapes at start: 42 min cold, 16 min warm. Is a smaller default set preferable?
