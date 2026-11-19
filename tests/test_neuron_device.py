"""On a NeuronCore: the compiled engine, with nkilib's kernels, answers as the eager model does.

Runs only on AWS Trainium with the Neuron SDK (`pytest -m neuron`; it skips elsewhere). A random
four-layer torso with the 2B's layer dimensions (the kernels' tile sizes) is saved as a
checkpoint and loaded with `load_neuron_engine`, so the test covers the loader, compilation, both
kernels and both forward paths on the device. The reference is the same weights run eagerly in
fp32 on the CPU; the device runs bf16.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.neuron

transformers = pytest.importorskip("transformers")

from strands_decider import neuron_engine  # noqa: E402

if not neuron_engine.neuron_available():
    pytest.skip("needs AWS Trainium and the Neuron SDK", allow_module_level=True)

from test_neuron_engine import QUESTIONS, TICKET, _words  # noqa: E402
from test_neuron_model import LORA_TARGETS, _tokenizer  # noqa: E402

from strands_decider.infer import EngineConfig  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.neuron_engine import (  # noqa: E402
    EagerRuntime,
    NeuronConfig,
    NeuronEngine,
    load_neuron_engine,
)
from strands_decider.neuron_model import NeuronDeciderModel, from_decider_model  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402

SHAPES = NeuronConfig(
    buckets=[256, 512],
    batch={256: [1, 2]},
    prefix_buckets=[256],
    suffix_len=128,
    suffix_batch=[4],
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("neuron_device")
    tok = _tokenizer(_words())
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    base = tmp / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        pointer_dim=64,
        max_length=512,
        torch_dtype="bfloat16",
        lora_r=4,
        lora_alpha=8,
        lora_targets=LORA_TARGETS,
    )
    model = StrandsDeciderModel(
        cfg, transformers.Qwen3_5ForCausalLM.from_pretrained(base).model, tok
    )
    model.attach_lora()
    path = tmp / "ckpt"
    model.save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def engines(checkpoint):
    device = load_neuron_engine(checkpoint, EngineConfig(), SHAPES)
    decider = StrandsDeciderModel.load(checkpoint)
    config, tok = decider.config, decider.tokenizer
    shape, sd = from_decider_model(decider, dtype=torch.float32)
    eager = EagerRuntime(NeuronDeciderModel(shape, sd, compute_dtype=torch.float32))
    return device, NeuronEngine(eager, config, tok, EngineConfig(device="neuron"), SHAPES)


def _probabilities(response):
    return {
        n: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for n, a in response.answers.items()
    }


@pytest.mark.parametrize(
    "questions", [["team"], ["urgent", "team", "mood"]], ids=["whole-prompt", "shared-prefix"]
)
def test_the_device_answers_as_the_eager_model_does(engines, questions):
    device, eager = engines
    request = SystemOneRequest(state=TICKET * 3, questions={n: QUESTIONS[n] for n in questions})
    got, expected = (
        _probabilities(device.evaluate(request)),
        _probabilities(eager.evaluate(request)),
    )
    for name in expected:
        assert got[name] == pytest.approx(expected[name], abs=0.03), name


def test_every_shape_compiles_and_runs(engines):
    device, _ = engines
    device.warm_up()
