"""Tensor parallel for --device neuron: sharded ranks must answer as the whole model does.

`neuron_model.shard`'s weights, with each layer's partial sums all-reduced over a CPU (gloo)
process group, and `TensorParallelEngine`'s lockstep (rank 0 broadcasts each request, the
others follow), on test_neuron_model's tiny random Qwen3.5 with the torch reference mixers.
The sharding itself is checked in-process: the partial sums of every shard add up to the
whole model's layer. The engine test launches worker processes (`-m distributed`).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from test_neuron_engine import QUESTIONS, TICKET, _words
from test_neuron_model import build_checkpoint

from strands_decider.modeling import StrandsDeciderModel
from strands_decider.neuron_model import (
    NeuronDeciderModel,
    from_decider_model,
    rope_cos_sin,
    shard,
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return build_checkpoint(tmp_path_factory.mktemp("neuron_tp"), _words())


class _Sum:
    """Stands in for a layer's all-reduce: records each rank's partial sum."""

    def __init__(self):
        self.parts = []

    def __call__(self, y):
        self.parts.append(y)
        return y


@pytest.mark.parametrize("world", [2, 4])
def test_the_shards_partial_sums_add_up_to_each_layer(checkpoint, world):
    shape, sd = from_decider_model(StrandsDeciderModel.load(checkpoint), dtype=torch.float32)
    whole = NeuronDeciderModel(shape, sd, compute_dtype=torch.float32)
    x = torch.randn(2, 40, shape.hidden, generator=torch.Generator().manual_seed(0))
    cos_sin = rope_cos_sin(40, shape.rope_dim, shape.rope_theta, x.device, x.dtype)
    layers = {
        "gdn": lambda m: m.gdn_layer(0, x),
        "attention": lambda m: m.attn_layer(shape.layer_types.index("full_attention"), x, *cos_sin),
        "mlp": lambda m: m.mlp(0, x),
    }
    for name, layer in layers.items():
        total = torch.zeros_like(x)
        for rank in range(world):
            local, lsd = shard(shape, sd, rank, world)
            part = NeuronDeciderModel(local, lsd, compute_dtype=torch.float32)
            part._reduce = _Sum()
            with torch.inference_mode():
                layer(part)
            [y] = part._reduce.parts
            total += y
        with torch.inference_mode():
            expected = layer(whole) - x
        torch.testing.assert_close(total, expected, atol=1e-5, rtol=1e-5, msg=name)


def test_a_layout_the_ranks_cannot_divide_is_refused(checkpoint):
    shape, sd = from_decider_model(StrandsDeciderModel.load(checkpoint), dtype=torch.float32)
    with pytest.raises(ValueError, match="tensor parallel over 3"):
        shard(shape, sd, 0, 3)


def _rank(rank, world, port, checkpoint, out):
    import os

    import torch.distributed as dist

    from strands_decider.infer import EngineConfig
    from strands_decider.neuron_engine import EagerRuntime, NeuronConfig
    from strands_decider.neuron_tp import TensorParallelEngine
    from strands_decider.schema import SystemOneRequest

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    decider = StrandsDeciderModel.load(checkpoint)
    config, tok = decider.config, decider.tokenizer
    shape, sd = from_decider_model(decider, dtype=torch.float32)
    local, lsd = shard(shape, sd, rank, world)
    model = NeuronDeciderModel(local, lsd, compute_dtype=torch.float32)
    model.tp_group = dist.group.WORLD
    neuron = NeuronConfig(buckets=[128, 256, 512], prefix_buckets=[128, 256], dtype="float32")
    engine = TensorParallelEngine(
        EagerRuntime(model),
        config,
        tok,
        EngineConfig(device="neuron"),
        neuron,
        rank=rank,
        group=dist.group.WORLD,
    )
    if rank:
        engine.follow()
    else:
        requests = [
            SystemOneRequest(state=TICKET, questions={"team": QUESTIONS["team"]}),
            SystemOneRequest(state=TICKET * 2, questions=QUESTIONS),
        ]
        out.put([engine.evaluate(r).model_dump_json() for r in requests])
        engine.close()
    dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("world", [2, 4])
def test_the_ranks_answer_as_the_whole_model_does(checkpoint, world):
    import torch.multiprocessing as mp
    from test_neuron_engine import _neuron, _probabilities

    from strands_decider.neuron_tp import _free_port
    from strands_decider.schema import SystemOneRequest, SystemOneResponse

    ctx = mp.get_context("spawn")
    out = ctx.SimpleQueue()
    mp.start_processes(
        _rank, args=(world, _free_port(), checkpoint, out), nprocs=world, start_method="spawn"
    )
    got = [SystemOneResponse.model_validate_json(r) for r in out.get()]
    whole = _neuron(checkpoint)
    requests = [
        SystemOneRequest(state=TICKET, questions={"team": QUESTIONS["team"]}),
        SystemOneRequest(state=TICKET * 2, questions=QUESTIONS),
    ]
    for response, request in zip(got, requests, strict=True):
        expected = _probabilities(whole.evaluate(request))
        for name, p in _probabilities(response).items():
            assert p == pytest.approx(expected[name], abs=1e-4), name
