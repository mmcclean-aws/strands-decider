"""Serving on AWS Trainium (`--device neuron`): the decider as graphs compiled ahead of time.

The Neuron SDK's PyTorch-native backend (`libtorch_neuronx_lite`,
`torch.compile(backend="neuron_native_lite")`) compiles one graph per input shape. So the engine
pads each forward to a fixed bucket: rows of (state + question) to a token bucket T and a batch
size B, a shared state to a prefix bucket, question suffixes to `suffix_len`. The graph is
`neuron_model.NeuronDeciderModel`, which returns the calibrated probabilities [B, K] with the
readout on the device.

Only the forwards move. `NeuronEngine` subclasses `SystemOneEngine`, as `MLXEngine` does, so
rendering, tokenisation, the context-window check, option positions, chunking at `max_batch`,
temperatures and `to_answer` are the torch engine's own code. Both of `evaluate`'s paths are
kept: one question forwards the whole prompt; several encode the state once and continue every
question from its caches. A state longer than the largest prefix bucket, or a question longer
than `suffix_len`, takes the whole-prompt path.

The device work runs through a `Runtime`: `NeuronRuntime` compiles and runs the graphs on a
NeuronCore; `EagerRuntime` runs the same module eagerly, which is how the tests exercise the
engine on a CPU.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import torch

from .infer import EngineConfig, SystemOneEngine
from .modeling import StrandsDeciderConfig, StrandsDeciderModel, refuse_token_readout
from .neuron_model import NeuronDeciderModel, from_decider_model
from .prompting import RenderedQuestion
from .schema import SystemOneRequest, SystemOneResponse

# What `pip install "strands-decider[neuron]"` needs to find the Neuron packages.
INSTALL_HINT = (
    "device 'neuron' needs AWS Trainium and the Neuron SDK: run inside the Neuron Deep Learning "
    'Container, or pip install "strands-decider[neuron]" '
    "--extra-index-url https://pip.repos.neuron.amazonaws.com (nkilib ships in the container)"
)
# The kernels need every token count to be a multiple of this.
TILE = 128


def neuron_available() -> bool:
    """True on Linux with a Neuron device and the Neuron SDK's Python packages installed."""
    if platform.system() != "Linux" or not os.path.exists("/dev/neuron0"):
        return False
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in ("libtorch_neuronx_lite", "nki", "nkilib")
        )
    except (ImportError, ValueError):
        return False


@dataclass
class NeuronConfig:
    """The compiled shapes. Every shape is compiled on first use (or by `warm_up`), then cached
    on disk by the Neuron SDK, so the set should stay small."""

    # Token buckets for whole prompts, and the batch sizes compiled for each. The engine pads
    # a chunk's rows to the smallest bucket that holds them and runs them in as few forwards as
    # it can, taking the smallest batch size that holds what is left.
    buckets: list[int] = field(
        default_factory=lambda: [256, 512, 768, 1024, 1536, 2048, 2560, 2816, 3072, 4096]
    )
    batch: dict[int, list[int]] = field(
        default_factory=lambda: {256: [1, 2, 4, 8], 512: [1, 2, 4], 768: [1, 2], 1024: [1, 2]}
    )
    # Options per question, padded. More is refused with a ValueError (HTTP 422).
    k_pad: int = 32
    # Shared prefix: state buckets, the question suffix length, and suffix batch sizes.
    prefix_buckets: list[int] = field(
        default_factory=lambda: [256, 512, 768, 1024, 1536, 2048, 2560, 2816, 3072, 4096]
    )
    suffix_len: int = TILE
    suffix_batch: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in ("buckets", "prefix_buckets"):
            values = getattr(self, name)
            if not values or values != sorted(values) or any(b % TILE for b in values):
                raise ValueError(f"{name} must be ascending multiples of {TILE}: {values}")
        if self.suffix_len % TILE:
            raise ValueError(f"suffix_len must be a multiple of {TILE}: {self.suffix_len}")

    def batch_sizes(self, bucket: int) -> list[int]:
        return sorted(self.batch.get(bucket, [1]))

    def shapes(self) -> list[tuple[str, int, int]]:
        """Every compiled shape: ("forward", T, B), ("prefix", S, 1), ("suffix", S, B)."""
        out = [("forward", t, b) for t in self.buckets for b in self.batch_sizes(t)]
        out += [("prefix", s, 1) for s in self.prefix_buckets]
        out += [("suffix", s, b) for s in self.prefix_buckets for b in sorted(self.suffix_batch)]
        return out


class Runtime(Protocol):
    """Runs `NeuronDeciderModel`'s three forwards on padded CPU tensors; results come back on
    the CPU, except the prefix caches, which stay where the suffix forward reads them."""

    def forward(
        self,
        ids: torch.Tensor,
        ans: torch.Tensor,
        opt: torch.Tensor,
        n_slots: torch.Tensor,
        inv_temp: torch.Tensor,
    ) -> torch.Tensor: ...

    def prefix(self, ids: torch.Tensor, s_len: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    def suffix(
        self,
        ids: torch.Tensor,
        s_len: torch.Tensor,
        caches: Sequence[torch.Tensor],
        ans: torch.Tensor,
        opt: torch.Tensor,
        n_slots: torch.Tensor,
        inv_temp: torch.Tensor,
    ) -> torch.Tensor: ...


class EagerRuntime:
    """The model run eagerly where its weights are (the CPU, in the tests)."""

    def __init__(self, model: NeuronDeciderModel) -> None:
        self.model = model.eval()

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.model(*inputs)
        return out

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def prefix(self, ids: torch.Tensor, s_len: torch.Tensor) -> tuple[torch.Tensor, ...]:
        caches: tuple[torch.Tensor, ...] = self.model.prefix_forward(ids, s_len)
        return caches

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def suffix(
        self,
        ids: torch.Tensor,
        s_len: torch.Tensor,
        caches: Sequence[torch.Tensor],
        *rest: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.suffix_forward(ids, s_len, caches, *rest)


def configure_environment() -> None:
    """Set what the Neuron runtime needs before libtorch_neuronx_lite is imported, unless set.

    - VLLM_NEURON_BACKEND=neuron_native: unset, libtorch_neuronx_lite takes its XLA route.
    - NEURON_RT_ROOT_COMM_ID: needed even on one core, and distinct for every process that is
      not part of the same tensor-parallel group, so the port follows the first visible core.
    - NEURON_RT_IO_RING_CACHE_SIZE and NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS: the reuse of a
      graph's DMA rings and the depth of the execution queue, the vLLM Neuron worker's values.
    """
    os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "0")
    first = int(cores.split(",")[0].split("-")[0] or 0)
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{61234 + first}")
    os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "32")


class NeuronRuntime:
    """The model compiled for one NeuronCore: one compiled callable per forward, which compiles
    each new input shape once (the Neuron SDK caches the compiled graphs on disk)."""

    def __init__(self, model: NeuronDeciderModel, device: torch.device, n_shapes: int) -> None:
        from libtorch_neuronx_lite.compile.native_backend import register

        # Dynamo recompiles a compiled callable for each new shape, up to a limit (8 by
        # default) that a bucket set passes.
        dynamo = torch._dynamo.config  # noqa: SLF001 -- no public setting for these limits
        dynamo.cache_size_limit = max(dynamo.cache_size_limit, n_shapes)
        dynamo.recompile_limit = max(dynamo.recompile_limit, n_shapes)
        backend = register()
        self.device = device
        self.model = model.to(device).eval()

        def compile_(fn: Any) -> Any:
            return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)

        self._forward = compile_(self.model.forward)
        self._prefix = compile_(self.model.prefix_forward)
        self._suffix = compile_(self.model.suffix_forward)

    def _to(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        # Dtypes are set on the CPU: a conversion during the copy fails on this device.
        return [x.to(self.device) for x in inputs]

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self._forward(*self._to(inputs)).cpu()
        return out

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def prefix(self, ids: torch.Tensor, s_len: torch.Tensor) -> tuple[torch.Tensor, ...]:
        caches: tuple[torch.Tensor, ...] = self._prefix(*self._to((ids, s_len)))
        return caches

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def suffix(
        self,
        ids: torch.Tensor,
        s_len: torch.Tensor,
        caches: Sequence[torch.Tensor],
        *rest: torch.Tensor,
    ) -> torch.Tensor:
        ids_d, s_len_d, *rest_d = self._to((ids, s_len, *rest))
        out: torch.Tensor = self._suffix(ids_d, s_len_d, caches, *rest_d).cpu()
        return out


class _Readout:
    """What the inherited engine code reads from a `StrandsDeciderModel`: its config and
    tokenizer. The weights live in the runtime."""

    def __init__(self, config: StrandsDeciderConfig, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer


def _smallest(sizes: Sequence[int], n: int) -> int:
    """The smallest size that holds n, else the largest."""
    return next((s for s in sizes if s >= n), sizes[-1])


class NeuronEngine(SystemOneEngine):
    """`SystemOneEngine` with every forward padded to a compiled shape and run by a `Runtime`.
    Build one with `load_neuron_engine`."""

    def __init__(
        self,
        runtime: Runtime,
        config: StrandsDeciderConfig,
        tokenizer: Any,
        engine_config: EngineConfig,
        neuron_config: NeuronConfig | None = None,
    ) -> None:
        # Not super().__init__(): that moves a torch model to a torch device. These are the
        # attributes the inherited helpers read; `device` is where their small tensors live.
        self.cfg = engine_config
        self.neuron = neuron_config or NeuronConfig()
        if config.max_length > self.neuron.buckets[-1]:
            raise ValueError(
                f"the checkpoint's window is {config.max_length} tokens; the largest neuron "
                f"bucket is {self.neuron.buckets[-1]}"
            )
        self.model = _Readout(config, tokenizer)
        self.tok = tokenizer
        self.device = "cpu"
        self.runtime = runtime
        self._pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._prefix_cache_unforkable = False
        self._tok_lock = threading.Lock()
        # One evaluation on the NeuronCore at a time: the server runs requests on a thread pool.
        self._device_lock = threading.Lock()

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.images:
            raise ValueError("device 'neuron' has no vision tower")
        with self._device_lock:
            return super().evaluate(request)

    # ---- padding

    def _bucket(self, n: int, buckets: Sequence[int]) -> int:
        for b in buckets:
            if n <= b:
                return b
        raise ValueError(f"{n} tokens exceed the largest neuron bucket, {buckets[-1]}")

    def _readout_inputs(
        self,
        B: int,
        rendered: list[RenderedQuestion],
        last: list[int],
        options: torch.Tensor,
        kinds: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ans [B], opt [B, k_pad], n_slots [B], inv_temp [B] for len(rendered) <= B rows; the
        padding rows read position 0 and have no options."""
        k_pad, n = self.neuron.k_pad, len(rendered)
        if options.shape[1] > k_pad:
            raise ValueError(
                f"a question has {options.shape[1]} options; device 'neuron' takes at most {k_pad}"
            )
        ans = torch.zeros(B, dtype=torch.int32)
        ans[:n] = torch.tensor(last, dtype=torch.int32)
        opt = torch.zeros(B, k_pad, dtype=torch.int32)
        opt[:n, : options.shape[1]] = options.clamp_min(0).to(torch.int32)
        n_slots = torch.zeros(B, dtype=torch.int32)
        n_slots[:n] = torch.tensor([rq.n_slots for rq in rendered], dtype=torch.int32)
        inv_temp = torch.ones(B, dtype=torch.float32)
        inv_temp[:n] = 1.0 / self._temperatures(kinds).cpu()
        return ans, opt, n_slots, inv_temp

    def _ids(self, rows: list[list[int]], B: int, T: int) -> torch.Tensor:
        ids = torch.full((B, T), self._pad_id, dtype=torch.int32)
        for j, row in enumerate(rows):
            ids[j, : len(row)] = torch.tensor(row, dtype=torch.int32)
        return ids

    # ---- the two paths

    def _slot_probs_batched(
        self,
        s: list[int],
        q: list[list[int]],
        n_slots: list[int],
        kinds: list[str],
        offsets: list[list[tuple[int, int]]],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        assert rendered is not None
        rows = [s + qi for qi in q]
        options = self._option_idx(rendered, len(s), offsets)
        T = self._bucket(max(len(r) for r in rows), self.neuron.buckets)
        sizes = self.neuron.batch_sizes(T)
        out = torch.zeros(len(rows), self.neuron.k_pad)
        start = 0
        while start < len(rows):
            B = _smallest(sizes, len(rows) - start)
            part = slice(start, min(start + B, len(rows)))
            inputs = self._readout_inputs(
                B,
                rendered[part],
                [len(r) - 1 for r in rows[part]],
                options[part],
                kinds[part],
            )
            probs = self.runtime.forward(self._ids(rows[part], B, T), *inputs)
            out[part] = probs[: part.stop - part.start].float()
            start = part.stop
        return out, sum(len(r) for r in rows)

    def _slot_probs_shared_prefix(
        self,
        s: list[int],
        q: list[list[int]],
        n_slots: list[int],
        kinds: list[str],
        offsets: list[list[tuple[int, int]]],
        rendered: list[RenderedQuestion] | None = None,
    ) -> tuple[torch.Tensor, int]:
        assert rendered is not None
        L = self.neuron.suffix_len
        if len(s) > self.neuron.prefix_buckets[-1] or max(len(qi) for qi in q) > L:
            return self._slot_probs_batched(s, q, n_slots, kinds, offsets, rendered)
        S = self._bucket(len(s), self.neuron.prefix_buckets)
        s_len = torch.tensor([len(s)], dtype=torch.int32)
        caches = self.runtime.prefix(self._ids([s], 1, S), s_len)
        options = self._option_idx(rendered, 0, offsets)
        sizes = sorted(self.neuron.suffix_batch)
        out = torch.zeros(len(q), self.neuron.k_pad)
        start = 0
        while start < len(q):
            B = _smallest(sizes, len(q) - start)
            part = slice(start, min(start + B, len(q)))
            inputs = self._readout_inputs(
                B, rendered[part], [len(qi) - 1 for qi in q[part]], options[part], kinds[part]
            )
            probs = self.runtime.suffix(self._ids(q[part], B, L), s_len, caches, *inputs)
            out[part] = probs[: part.stop - part.start].float()
            start = part.stop
        return out, len(s) + sum(len(qi) for qi in q)

    # ---- compilation

    def warm_up(self) -> None:
        """Run every compiled shape once, so no request waits for a compile."""
        k = self.neuron.k_pad

        def readout(B: int, T: int) -> list[torch.Tensor]:
            return [
                torch.full((B,), T - 1, dtype=torch.int32),
                torch.zeros(B, k, dtype=torch.int32),
                torch.full((B,), 2, dtype=torch.int32),
                torch.ones(B, dtype=torch.float32),
            ]

        caches: dict[int, tuple[torch.Tensor, ...]] = {}
        for kind, T, B in self.neuron.shapes():
            ids = torch.full((B, T), self._pad_id, dtype=torch.int32)
            if kind == "forward":
                self.runtime.forward(ids, *readout(B, T))
            elif kind == "prefix":
                caches[T] = self.runtime.prefix(ids, torch.tensor([T], dtype=torch.int32))
            else:
                L = self.neuron.suffix_len
                ids = torch.full((B, L), self._pad_id, dtype=torch.int32)
                s_len = torch.tensor([T], dtype=torch.int32)
                self.runtime.suffix(ids, s_len, caches[T], *readout(B, L))


def load_neuron_engine(
    checkpoint: str,
    config: EngineConfig | None = None,
    neuron_config: NeuronConfig | None = None,
) -> NeuronEngine:
    """The neuron counterpart of `infer.load_engine`: a local checkpoint directory or a Hub id,
    on the first visible NeuronCore (NEURON_RT_VISIBLE_CORES)."""
    if not neuron_available():
        raise RuntimeError(INSTALL_HINT)
    neuron_config = neuron_config or NeuronConfig()
    configure_environment()
    import libtorch_neuronx_lite  # noqa: F401  (registers the "neuron" device)

    from .neuron_kernels import NkiMixers

    decider = StrandsDeciderModel.load(checkpoint)
    refuse_token_readout(decider.config)
    config_, tokenizer = decider.config, decider.tokenizer
    shape, sd = from_decider_model(decider, dtype=getattr(torch, neuron_config.dtype))
    del decider
    device = torch.device("neuron", 0)
    mixers = NkiMixers(device)
    model = NeuronDeciderModel(
        shape,
        sd,
        compute_dtype=getattr(torch, neuron_config.dtype),
        gdn_impl=mixers.gdn,
        attn_impl=mixers.attn,
    )
    runtime = NeuronRuntime(model, device, n_shapes=len(neuron_config.shapes()))
    engine_config = replace(config or EngineConfig(), device="neuron")
    return NeuronEngine(runtime, config_, tokenizer, engine_config, neuron_config)
