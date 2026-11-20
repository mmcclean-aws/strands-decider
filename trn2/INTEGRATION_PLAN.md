# Plan: Trainium2 (`--device neuron`) into strands-decider

Status (2026-11-20): steps 1-7 done and verified, step 8 in this branch; see Outcome at the end. The upstream issue (step 0) is drafted in ISSUE_DRAFT.md and not posted. This file lives in the `trn2/` staging area of the
fork's `trn2-port` branch and is deleted with it at the end (step 8).

## Decisions

| Topic | Decision |
|---|---|
| Upstream scope | Everything goes upstream: the single-core engine and NKI kernels, tensor parallel, `deploy/trainium2`, results and the how-to docs. |
| Engine shape | `NeuronEngine(SystemOneEngine)`, as `MLXEngine` does. It reuses `evaluate`, `_fit`, `_chunks`, `_option_idx` and `to_answer`, and overrides only `_slot_probs_batched` and `_slot_probs_shared_prefix`, which pad into compiled (T, B) buckets and call the device graph. |
| Device name | `--device neuron`, opt-in like `mlx`. `_auto_device` never picks it. |
| Dependency | A `neuron` extra that lists the real Neuron packages with floors that exist only on the Neuron repo, so a PyPI-only install fails at resolution and does not install the 0.0.1 placeholders. Install with `pip install "strands-decider[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com`. `nkilib` is not on that index (checked 2026-11-19) and ships in the Neuron DLC image, so a runtime check covers it and the device. All Neuron imports are lazy. |
| Tests | CI runs on CPU: the torch reference torso and the engine on a tiny random Qwen3.5, plus device selection. Kernel and device tests carry a new `neuron` pytest marker. They skip without hardware and are run by hand on a trn2 host, with the output quoted in each PR. |
| Process | Open an upstream issue first, then send a series of small stacked PRs. Each is green on CPU CI and quotes its trn2 numbers. |
| Deploy | `deploy/trainium2/`: a README and scripts for EC2 (trn2.48xlarge running the Neuron DLC). No CDK stack for now. |

Checked on the Neuron index (2026-11-19): `libtorch-neuronx-lite` 2.13.0.1.0.4089 (cp311 to cp314,
linux_x86_64 only), `nki` 0.7.0, `neuronx-cc` 2.28.7631. The port was measured on lite 2.13.0.1.0.3112,
so re-measure on the current build in step 2.

Proposed extra (environment markers keep it a no-op off Linux x86):

```toml
# `--device neuron`: AWS Trainium2 through the Neuron SDK's PyTorch-native backend
# (src/strands_decider/neuron_engine.py). The packages are on the Neuron repository, not PyPI
# (PyPI holds 0.0.1 placeholders; the floors make a PyPI-only install fail instead):
#   pip install "strands-decider[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
# nkilib ships in the Neuron DLC image only; neuron_available() checks for it.
neuron = [
    "libtorch-neuronx-lite>=2.13; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "nki>=0.7; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "neuronx-cc>=2.28; sys_platform == 'linux' and platform_machine == 'x86_64'",
]
```

## Where the trn2/ files go

| trn2/ | Destination | Step |
|---|---|---|
| `neuron/decider_model.py`: config, loading, torch math, prefix and suffix forwards, readout | `src/strands_decider/neuron_model.py` | 1 |
| `neuron/decider_model.py`: `use_fast` path | `neuron_model.py` | 4 |
| `neuron/decider_model.py`: `shard_state`, TP reduce | `neuron_model.py` | 5 |
| `neuron/engine.py`, the build and runner parts of `device_runner.py` | `src/strands_decider/neuron_engine.py` | 2 |
| `neuron/nki_ops.py` (nkilib `gdn_cte`, `attention_cte`) | `src/strands_decider/neuron_kernels/__init__.py` | 2 |
| `neuron/nki_ops.py` `FastGdn`, `kernels/gdn_fast.py`, `kernels/fused_ops.py` | `src/strands_decider/neuron_kernels/{gdn_fast,fused_ops}.py` | 4 |
| `kernels/test_gdn_fast.py`, `kernels/test_fused_ops.py` (correctness) | `tests/test_neuron_kernels.py` (`-m neuron`) | 4 |
| `kernels/test_*` (timing), `bench_*.py`, `equiv_layer0.py`, `device_runner.py bench` | `evaluation/bench_neuron.py` (MFU per bucket); the rest stays fork-only | 3, 4 |
| `test_cpu_parity.py`, `test_prefix_cpu.py`, `test_cpu_bf16_budget.py` | Replaced by `tests/test_neuron_model.py` (tiny model) and the step-1 real-weights check | 1 |
| `test_engine.py`, `tools/make_request_goldens.py` | Replaced by `evaluation/device_parity.py --device cpu --device neuron` | 2 |
| `accuracy.py`, `tools/make_goldens.py` | `evaluation/device_accuracy.py`: CPU fp32 goldens and a gate for any device | 3 |
| `tp_runner.py`, `tp_engine_test.py` | TP in `neuron_engine.py` and `serve`; tests under `-m distributed` and `-m neuron` | 5 |
| `tools/cost_model.py`, `collectives/` | `deploy/trainium2/cost_model/` | 6 |
| `docs/native_stack_howto.md`, `howto_scripts/` | `deploy/trainium2/howto/` | 6 |
| `results/summary.md` | `evaluation/results.md`, new section "Serving on Trainium2: accuracy and latency" | 7 |
| `results/*.json` | `research/data/trn2_*.csv` (bench and gate summaries; per-row gate data as CSV if maintainers want it) | 7 |
| `report/*.html` | Condensed into `docs/trainium2.md` (design: static shapes, merged LoRA, layouts, kernels, compiler findings). The HTML itself goes only if maintainers ask; upstream docs are Markdown. | 7 |
| `README.md`, `.gitignore` | Content folded into the documents above | 7 |

The layout above (flat `neuron_engine.py` and `neuron_model.py`, like `mlx_engine.py`, plus a
`neuron_kernels/` package for the large kernel files) is the proposal to put in the issue. If
maintainers prefer one `strands_decider/neuron/` subpackage, the moves change, not the steps.

## Branches

- PR branches are stacked off `upstream/main`: `neuron/01-model`, `neuron/02-engine` and so on, pushed to
  `origin` (the fork). The `trn2/` import commit never goes upstream: each branch adds files at their
  destination and reads the originals with `git show trn2-port:trn2/...`.
- `trn2-port` stays the fork's integration branch. After each step, merge that step's branch into it and
  delete the `trn2/` files it replaced, so `trn2/` shrinks to what is still unported.
- Commits follow CONTRIBUTING: Conventional Commits (`feat(neuron): ...`); a move or a refactor in its own
  commit, separate from functional change; ruff with a py310 target, mypy strict on `src/`; nothing
  downloaded in tests; every Markdown reference to a moved path fixed. Each PR body gets the
  "Human Overview".
- Nothing is pushed or posted without the owner's go-ahead.

## Cross-cutting rules for every step

- Python 3.11 compatible. CI runs 3.11 and 3.12; the DLC runs 3.14, and lite has cp311 wheels.
- No `sys.path` edits and no `DECIDER_*` environment variables. Configuration goes in a `NeuronConfig`
  dataclass (buckets, batch sizes per bucket, `k_pad`, `suffix_len`, `suffix_batch`, GDN compute dtype,
  fast kernels on or off, TP chunks).
- Required Neuron environment variables (`VLLM_NEURON_BACKEND=neuron_native` before importing lite,
  `NEURON_RT_ROOT_COMM_ID` even on one core) are set with `os.environ.setdefault` in the lazy loader,
  and documented. The performance ones (`NEURON_RT_IO_RING_CACHE_SIZE`,
  `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS`) are documented and given defaults the same way, but the
  user can override them.
- mypy strict on `src/`: annotate the torch code fully. Add overrides for `libtorch_neuronx_lite.*`,
  `nki.*` and `nkilib.*`. NKI kernel bodies may need `disallow_untyped_defs = false` for
  `strands_decider.neuron_kernels.*`. Raise that in the issue and keep it as narrow as possible.
- Licensing: the kernels are original and call nkilib without copying it. Compare
  `gdn_chunk_ref`/`attn_ref` with transformers' `torch_chunk_gated_delta_rule`; if either is adapted,
  add a header and a `THIRD_PARTY_NOTICES.md` entry as `mps_kernels.py` has.

## Steps

### 0. Upstream issue (no code)

Propose `--device neuron`: the design (subclass engine, static buckets, on-device readout, merged LoRA),
the module layout, the extra with the external index, the `neuron` marker with manual trn2 runs, the
mypy exception for kernel bodies, TP inside `serve`, the PR series, and the headline numbers
(52.8% MFU at 67 ms for 2,048 tokens on one logical core; 0 of 54 parity answers changed).

- **Verify:** maintainers agree on layout, extra and testing before step 1, or the plan is updated.
  Draft the issue text and get the owner's OK before posting.

### 1. `feat(neuron): a static-shape Qwen3.5 torso for ahead-of-time compilation`

`neuron_model.py`, pure torch with no Neuron import: `DeciderConfig` read from the base model's
`text_config` and `StrandsDeciderConfig`, with no hard-coded 2B shapes (`reshape(48, 128, 2048)` and the
like become head counts and dims from the config). Loading goes through `modeling.checkpoint_dir`,
`base_revision` and `load_head_state`, so Hub ids, local checkpoints, `slot_head.pt` and
`head.safetensors` all load as on other devices. The LoRA merge refuses what it cannot represent, with
the same checks as `mlx_engine.merge_lora` (share a helper if it stays small). Also the reference
`gdn_chunk_ref` and `attn_ref`, whole, prefix and suffix forwards, and the readout. No fast path and
no TP yet.

A first commit moves the tiny-Qwen3.5 checkpoint builder from `tests/test_mlx_engine.py` into a shared
test helper (refactor only).

- **Verify (CPU, CI):** `tests/test_neuron_model.py` on the tiny model in fp32. Probabilities match
  `StrandsDeciderModel` to about 1e-5 for noul, choice and score; prefix plus suffix equals the whole
  forward; right-padded rows equal unpadded ones; both head formats load; unsupported adapters are
  refused. `pytest -q`, `ruff check .` and `mypy ./src` pass.
- **Verify (this machine, real weights, not CI):** against HF+PEFT in fp32 on the published
  `strands-decider-2B-qwen3.5-v1-2610`, max |dp| at most 5e-7, as trn2 measured. Run it with
  `uv run` in a `uv venv`.

### 2. `feat(neuron): --device neuron`

- `neuron_engine.py`: `NeuronEngine(SystemOneEngine)` and `load_neuron_engine`. Each (T, B) bucket is
  compiled lazily with `torch.compile(backend=register(), fullgraph=True, dynamic=False)` and cached;
  rows are padded into buckets; the shared-prefix path runs prefix and suffix graphs and falls back to
  batched when a suffix is longer than `suffix_len` or the state is longer than the largest prefix
  bucket. Readout, temperature and masked softmax run on the device. Token usage is counted as
  `SystemOneEngine` counts it. Includes the nkilib mixers (`gdn_cte`, `attention_cte`), which match
  the trn2 "v1" level.
- More than `k_pad` options: other devices accept up to the schema's 255. Measure a second `k_pad`
  bucket (256). If it is cheap, compile both; if not, refuse with a 422, as the window is refused,
  and document it.
- Plumbing:
  - `infer.neuron_available()` and a `device == "neuron"` branch in `load_engine`.
  - `server.create_app`: engine choice; refuse `--vision` and `--device-map`; `_needs_serialisation`
    (serialise unless trn2 shows concurrent submission is safe); a warm-up that compiles every
    configured bucket, not only every question count.
  - `cli.py` help text.
  - `training.evaluate` refuses `--device neuron`, as it refuses mlx.
  - The `pyproject` extra and mypy overrides, and the `neuron` pytest marker.
  - `docs/inference.md`: a device-table row and a short "Serving on Trainium2" section with the
    `--extra-index-url` install and the DLC image.
- **Verify (CPU, CI):**
  - The engine with an injected eager CPU runner over the step-1 reference torso matches
    `SystemOneEngine` on the tiny model, covering:
    - batched and prefix paths;
    - bucket and batch-size selection;
    - the suffix-length fallback, chunking at `max_batch`;
    - window refusal, answer order and usage.
  - `tests/test_devices.py` additions:
    - neuron is never auto-selected;
    - `neuron_available()` is true only with lite, nki and nkilib importable;
    - a missing install names `strands-decider[neuron]` and the index before loading anything;
    - the server builds the neuron engine for `--device neuron` (stub).
- **Verify (trn2, manual):**
  - `pip install -e ".[neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com` inside the
    DLC. Check that lite's torch pin agrees with `torch>=2.7` and that the image's transformers is at
    least 5.15.
  - `strands-decider ask` and `strands-decider serve --device neuron` work.
  - `python evaluation/device_parity.py StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device cpu --device neuron`
    gives 0 of 54 changed and max |dp| of about 0.014 or less.
  - `pytest -m neuron` passes.
  - Latency at 512 and 2,048 tokens is about 45 and 174 ms (the v1 row of `summary.md`).

### 3. `feat(evaluation): a device accuracy gate`

`evaluation/device_accuracy.py`, from `accuracy.py` and `tools/make_goldens.py`, for any device, not
only neuron:
- CPU fp32 goldens on the committed labeled eval sets, through `training.evaluate`'s collator;
- per-device mean, p99 and max |dp|, argmax flips with their golden margins, and accuracy, ECE and NLL
  deltas;
- the bf16-floor gate thresholds from `summary.md`.

Also `evaluation/bench_neuron.py` (latency, throughput and MFU per bucket, from `device_runner.py bench`)
and `bench_local.py --device neuron`, which reports no peak memory for neuron.

- **Verify (CPU, CI):** a small unit test of the metric and gate functions (ECE, flips, margins) on
  synthetic rows.
- **Verify (this machine):** the gate passes cpu-bf16 against cpu-fp32 and reproduces the bf16 floor
  (about 0.016 max).
- **Verify (trn2):** the step-2 engine passes the gate on 1,019 rows (the v1 row of `summary.md`).

### 4. `perf(neuron): fused NKI kernels for the Gated DeltaNet and attention layers`

`neuron_kernels/gdn_fast.py` and `fused_ops.py`, the `use_fast` path in `neuron_model.py`, and
`NeuronConfig` switches in place of `DECIDER_FAST` and `DECIDER_GDN_DTYPE`. The torch references in
`fused_ops` (`gdn_prologue_ref` and the rest) must import without nki; split them out if needed. Kernel
correctness tests move to `tests/test_neuron_kernels.py` (`-m neuron`) and timing moves to
`bench_neuron.py`.

- **Verify (CPU, CI):** the torch references match `neuron_model.py`'s math on random inputs; the kernel
  module imports without nki (or is skipped cleanly).
- **Verify (trn2):**
  - `pytest -m neuron` (kernel correctness within the test tolerances);
  - `bench_neuron.py` at 2048x1 is about 67 ms (52.8% MFU) and at 512 tokens about 21 ms;
  - `device_accuracy.py` passes (the v3 row);
  - `device_parity.py` gives 0 of 54 changed and max |dp| of about 0.011.

### 5. `feat(neuron): tensor parallel across logical cores`

`shard_state` and the in-graph all-reduce with token chunking. `serve --device neuron --tensor-parallel N`
launches N ranks, one logical core each, on a gloo group. Rank 0 runs the HTTP server and broadcasts
each padded forward's inputs; the other ranks loop on forwards. Settle this design in the issue first.
If it is too much for one PR, ship it as an engine-level API, documented as experimental, before
wiring it into `serve`.

- **Verify (CPU, CI, `-m distributed`, the existing marker):** sharded forwards with world 2 and 4 on
  the tiny model under the reference torso equal the unsharded forward.
- **Verify (trn2):** TP2 at 4096x1 is about 88 ms with 4 chunks; `device_parity` with TP2 changes 0
  answers; `device_accuracy` passes (the TP2 row).

### 6. `docs(deploy): Trainium2 on EC2`

- `deploy/trainium2/README.md`:
  - trn2.48xlarge running the Neuron DLC; the install with the extra index;
  - one `serve` per logical core (`NEURON_RT_VISIBLE_CORES`, a distinct `NEURON_RT_ROOT_COMM_ID` port
    per process); a TP2 pool for prompts over about 2.8K tokens;
  - the bucket configuration; the NEFF cache and warm-up time;
  - the JevBench length distribution, which motivates the DP/TP split.
- `deploy/trainium2/scripts/` (a replica launcher), `cost_model/` (`cost_model.py` and the collectives
  JSON, path-relative), `howto/` (`native_stack_howto.md`, `howto_scripts/`), and an entry in
  `deploy/README.md`.
- Exposure: servers keep upstream's `127.0.0.1` default with no auth. Before writing anything about
  reaching the replicas from other hosts (load balancer, ports, security groups), follow the
  `builder-security` policy and the upstream Security section. The guide must not suggest binding to
  0.0.0.0 unauthenticated.
- **Verify (CPU):** `ruff check .`; `python deploy/trainium2/cost_model/cost_model.py --measured`
  reproduces `summary.md`'s numbers (7 DP points within 3%, TP within 2.5–12.6%).
- **Verify (trn2):** the launcher brings up 64 replicas; a load test reproduces about 1.7M tokens/s per
  instance (`summary.md`'s 64-replica row); the how-to scripts still run on the current DLC.

### 7. `docs: Trainium2 results and design notes`

- `evaluation/results.md`: "Serving on Trainium2: accuracy and latency" from `summary.md` (optimisation
  steps, buckets, DP throughput, TP, accuracy gate, request latency), in the style of the Mac and MLX
  sections.
- `research/data/trn2_*.csv` from `results/*.json`.
- `docs/trainium2.md`, condensed from `report/*.html`.
- References from `docs/architecture.md` (layout list), `docs/inference.md` and `README.md`.
- **Verify:** every link resolves (grep for the moved `trn2/` paths across `*.md` and comments); every
  number matches `results/*.json`; the docs preview workflow builds (`docs-deploy-preview.yml`, and
  `docs/site-manifest.json` if a new page needs listing).

### 8. Clean up the fork

When steps 1–7 are merged into `trn2-port`, `trn2/` should hold only fork-only leftovers: the HTML
reports if declined, the ad hoc probes. Delete `trn2/` (this plan included) from `trn2-port`, and
leave `mmcclean-aws/strands-decider-trn2` as the archive.

- **Verify:** `git ls-files trn2` is empty; `ruff check .` passes over the whole tree (today it does
  not, because of `trn2/`); `pytest -q` passes.

## What needs a trn2 host

This machine has no Trainium. Steps 2–6 each need a trn2.48xlarge session running the SDK 2.33 DLC
(`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`
or newer) for the "Verify (trn2)" items: `pytest -m neuron`, `device_parity`, `device_accuracy`,
`bench_neuron` and the replica load test. Batch them: one session per PR, with the outputs saved
under `research/data/` or quoted in the PR.

## Risks and open questions

- `libtorch-neuronx-lite` may pin an exact torch (2.13). That is fine for the extra, but check that it
  doesn't fight CI's floor pins; CI never installs the extra.
- `nkilib` exists only in the DLC, so the DLC is effectively the supported environment. Say so plainly.
- The trn2 numbers were taken on lite build 3112; the index now has 4089. Re-measure in step 2 and
  quote the current numbers.
- Compile time per bucket at startup: measure the cold start and the NEFF-cache hit, and document
  health-check timeouts as the SageMaker guide does.
- `_needs_serialisation` for neuron is unverified.
- More than 32 options (`k_pad`): step 2 decides.
- Maintainers may not want the mypy relaxation for kernel bodies, the external-index extra or the
  HTML reports. Each has a fallback above.


## Outcome (2026-11-20)

Done on stacked local branches off `upstream/main` (7e8380c, which had gained the torch-free
`core` library and the token readout since the plan; the neuron engine refuses token-readout
checkpoints like the other v1 engines). Nothing is pushed.

| Step | Branch | Verified |
|---|---|---|
| 1 model | `neuron/01-model` | 16 CPU tests; real 2B weights vs StrandsDeciderModel fp32: max dp 2.1e-6 over 54 answers |
| 2 engine | `neuron/02-engine` | CPU engine tests; trn2: install with the extra, `ask`, device_parity 0/54 changed (0.0139, nkilib path) |
| 3 accuracy gate, MFU bench | `neuron/03-evaluation` | trn2: gate PASS on 1,019 rows (both kernel paths) |
| 4 fused kernels | `neuron/04-kernels` | trn2: 14 `-m neuron` tests; 2,048 tokens 68.3 ms, 52.4% MFU; parity 0/54 (0.0113) |
| 5 tensor parallel (experimental launcher) | `neuron/05-tp` | CPU gloo 2/4 ranks; trn2: TP2/TP4 0/54 changed, 5q on 2,500 tokens 88 / 61 ms |
| 6 deploy/trainium2 | `neuron/06-deploy` | cost model reproduces its calibration; trn2: 4 replicas on loopback |
| 7 results and docs | `neuron/07-results` | links and anchors checked; numbers from the trn2 runs |

Deviations from the plan, and why:

- Weights load through `StrandsDeciderModel.load` and a PEFT merge in fp32 instead of the port's
  safetensors reader; it needed `torch_device="cpu"` in `modeling.load`'s PEFT call (with
  libtorch_neuronx_lite imported, PEFT staged the adapter on `neuron`).
- Tensor parallel ships as `python -m strands_decider.neuron_tp serve`, not as a `serve` flag,
  pending the maintainers' view (the plan's fallback).
- The how-to scripts are kept verbatim and excluded from ruff: autofix removed their
  side-effect imports.
- k_pad stays 32 with a 422 beyond it; a 256 bucket was not measured.

Bugs found on trn2 and fixed: PEFT staging the adapter on `neuron`; `mlp_fast` missing from
`fused_ops` after the layouts split; the shared-prefix forwards not on the fused path (194 ms
instead of 84 ms for a 2,560-token prefix); torch's compile cache shared across hosts failing
with NCC_EVRF059. Each has a CPU regression test where one is possible.

Open: `serve` warm start is 16 minutes with warm caches (67 shapes); the upstream issue is not
posted; nothing is pushed.
