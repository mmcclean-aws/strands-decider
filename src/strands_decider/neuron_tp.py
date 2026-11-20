"""Tensor parallel on AWS Trainium (experimental): one model replica over N logical NeuronCores.

A request whose state is long is latency-bound on one core: at 4,096 tokens the forward takes
134 ms there, and 88 ms over two cores of one chip (TP2). `neuron_model.shard` splits each
layer's heads and MLP over N ranks, one process per core, and every layer's output projection
is all-reduced inside the compiled graph (`NeuronDeciderModel._reduce`).

Every rank must run the same graphs in the same order. So the ranks run the same engine on the
same requests: rank 0 takes each request, broadcasts it to the other ranks over a CPU (gloo)
process group, and answers it with them; the other ranks follow (`TensorParallelEngine.follow`).
Rank 0's answer is the response; every rank computes the same one, since the readout is whole
on every rank.

    python -m strands_decider.neuron_tp serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \\
        --tensor-parallel 2 --core-base 12 --port 8099

`--core-base` is the first logical core; the group takes the next N, which should be adjacent
cores of one chip (four logical cores per Trainium2 chip at LNC=2): collectives within a chip
are the fast ones. Pick N in {2, 4}.
"""

from __future__ import annotations

import argparse
import os
import socket
from dataclasses import replace
from typing import Any

import torch

from .infer import EngineConfig
from .neuron_engine import NeuronConfig, NeuronEngine, build_runtime
from .schema import SystemOneRequest, SystemOneResponse

_STOP, _EVALUATE, _WARM_UP = "stop", "evaluate", "warm_up"


class TensorParallelEngine(NeuronEngine):
    """A `NeuronEngine` on one rank of a tensor-parallel group. On rank 0, `evaluate` and
    `warm_up` run on every rank; the other ranks run `follow` until rank 0 closes."""

    def __init__(self, *args: Any, rank: int, group: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.group = group

    def _broadcast(self, message: list[Any]) -> list[Any]:
        import torch.distributed as dist

        dist.broadcast_object_list(message, src=0, group=self.group)
        return message

    def _evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if self.rank == 0:  # under the device lock: the ranks see requests in rank 0's order
            self._broadcast([_EVALUATE, request.model_dump_json()])
        return super()._evaluate(request)

    def warm_up(self) -> None:
        if self.rank == 0:
            self._broadcast([_WARM_UP, None])
        super().warm_up()

    def close(self) -> None:
        """On rank 0: release the other ranks' `follow`."""
        if self.rank == 0:
            self._broadcast([_STOP, None])

    def follow(self) -> None:
        """On ranks other than 0: run what rank 0 runs, until it closes."""
        while True:
            op, payload = self._broadcast([None, None])
            if op == _STOP:
                return
            if op == _WARM_UP:
                super().warm_up()
            else:
                self.evaluate(SystemOneRequest.model_validate_json(payload))


def load_tensor_parallel_engine(
    checkpoint: str,
    rank: int,
    world: int,
    group: Any,
    config: EngineConfig | None = None,
    neuron_config: NeuronConfig | None = None,
) -> TensorParallelEngine:
    """Rank `rank`'s engine, in a process whose NEURON_RT_VISIBLE_CORES is its own core and whose
    NEURON_RT_ROOT_COMM_ID is the group's (see `launch`)."""
    neuron_config = neuron_config or NeuronConfig()
    runtime, decider_config, tokenizer = build_runtime(
        checkpoint, neuron_config, rank=rank, world=world, group=group
    )
    engine_config = replace(config or EngineConfig(), device="neuron")
    return TensorParallelEngine(
        runtime, decider_config, tokenizer, engine_config, neuron_config, rank=rank, group=group
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def _rank_main(rank: int, args: argparse.Namespace, master_port: int) -> None:
    import torch.distributed as dist

    # Before the Neuron runtime starts: this rank's core, and the group's collectives root.
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.core_base + rank)
    os.environ["NEURON_RT_NUM_CORES"] = "1"
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"127.0.0.1:{62300 + args.core_base}"
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(master_port))
    world = args.tensor_parallel
    dist.init_process_group("gloo", rank=rank, world_size=world)
    group = dist.group.WORLD
    neuron_config = NeuronConfig(tp_chunks=args.tp_chunks)
    engine = load_tensor_parallel_engine(
        args.checkpoint,
        rank,
        world,
        group,
        EngineConfig(
            max_batch=args.max_batch,
            model_name=args.model_name or os.path.basename(args.checkpoint.rstrip("/")),
        ),
        neuron_config,
    )
    try:
        if rank != 0:
            engine.follow()
            return
        import uvicorn

        from .server import create_app

        app = create_app(
            args.checkpoint,
            device="neuron",
            model_name=args.model_name,
            max_batch=args.max_batch,
            warmup=not args.no_warmup,
            build_engine=lambda: engine,
        )
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m strands_decider.neuron_tp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", help="serve POST /v1/systemone from a tensor-parallel group")
    s.add_argument("checkpoint")
    s.add_argument("--tensor-parallel", type=int, default=2, choices=[2, 4])
    s.add_argument("--core-base", type=int, default=0, help="the group's first logical core")
    s.add_argument("--tp-chunks", type=int, default=NeuronConfig().tp_chunks)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model-name", default=None)
    s.add_argument("--max-batch", type=int, default=32)
    s.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)
    torch.multiprocessing.spawn(
        _rank_main, args=(args, _free_port()), nprocs=args.tensor_parallel, join=True
    )


if __name__ == "__main__":  # pragma: no cover
    main()
