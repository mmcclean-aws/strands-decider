"""The neuron engine must answer as the torch engine does, from the same checkpoint.

`NeuronEngine` pads every forward to a compiled shape; here its runtime is `EagerRuntime`, the
same static-shape model run eagerly on the CPU, so the test covers everything the engine adds
to `SystemOneEngine` (buckets, batch sizes, the shared-prefix path and its fallbacks, the
readout inputs, the token count) without Trainium. The checkpoint is test_neuron_model's tiny
random Qwen3.5; nothing is downloaded.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from test_neuron_model import build_checkpoint

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.neuron_engine import (
    EagerRuntime,
    NeuronConfig,
    NeuronEngine,
)
from strands_decider.neuron_model import NeuronDeciderModel, from_decider_model
from strands_decider.prompting import render_question, render_state
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

QUESTIONS = {
    "urgent": NoulQuestion(instructions="Does this convey urgency?"),
    "team": ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={"billing": "payments and invoices", "technical": "bugs", "sales": "pricing"},
    ),
    "mood": ScoreQuestion(
        instructions="How frustrated is the writer?", criteria=["calm", "annoyed", "furious"]
    ),
}
LONG_QUESTION = ChoiceQuestion(
    instructions="Which of these " + "very " * 120 + "specific teams fits?",
    criteria={"a": "first", "b": "second"},
)
TICKET = "Help! My payouts have been failing for three days and nobody answers my emails."
POLICY = "Refunds are issued within five business days of an approved request. "


def _request(repeats: int, names: list[str], extra: dict | None = None) -> SystemOneRequest:
    questions = {n: QUESTIONS[n] for n in names} | (extra or {})
    return SystemOneRequest(state=POLICY * repeats + TICKET, questions=questions)


def _words():
    from tokenizers import pre_tokenizers

    texts = [render_state(POLICY + TICKET)]
    texts += [render_question(q).text for q in [*QUESTIONS.values(), LONG_QUESTION]]
    texts += [
        render_question(
            ChoiceQuestion(instructions="Pick.", criteria={c: None for c in "abcde"})
        ).text
    ]
    return {w for t in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(t)}


# Small buckets so that short requests cross them: prompts up to 512 tokens, states up to 256
# through the shared prefix, at most 4 rows per forward.
SMALL = NeuronConfig(
    buckets=[128, 256, 384, 512],
    batch={128: [1, 2], 256: [1, 2]},
    k_pad=4,
    prefix_buckets=[128, 256],
    suffix_len=128,
    suffix_batch=[2, 3],
    dtype="float32",
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return build_checkpoint(tmp_path_factory.mktemp("neuron_engine"), _words())


class _Counting(EagerRuntime):
    """Records the padded shape of every forward."""

    def __init__(self, model):
        super().__init__(model)
        self.calls = []

    def forward(self, ids, *rest):
        self.calls.append(("forward", *ids.shape))
        return super().forward(ids, *rest)

    def prefix(self, ids, s_len):
        self.calls.append(("prefix", *ids.shape))
        return super().prefix(ids, s_len)

    def suffix(self, ids, *rest):
        self.calls.append(("suffix", *ids.shape))
        return super().suffix(ids, *rest)


def _neuron(checkpoint, neuron_config=SMALL, **engine):
    decider = StrandsDeciderModel.load(checkpoint)
    config, tok = decider.config, decider.tokenizer
    shape, sd = from_decider_model(decider, dtype=torch.float32)
    runtime = _Counting(NeuronDeciderModel(shape, sd, compute_dtype=torch.float32))
    return NeuronEngine(
        runtime, config, tok, EngineConfig(device="neuron", **engine), neuron_config
    )


@pytest.fixture(scope="module")
def engines(checkpoint):
    model = StrandsDeciderModel.load(checkpoint)
    return {
        "torch": SystemOneEngine(model, EngineConfig(device="cpu")),
        "torch_batched": SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False)),
        "neuron": _neuron(checkpoint),
        "neuron_batched": _neuron(checkpoint, use_prefix_cache=False),
    }


def _probabilities(response):
    out = {}
    for name, a in response.answers.items():
        out[name] = [a.noul] if a.type == "noul" else list(a.probabilities.values())
    return out


def _assert_same(got, expected):
    assert got.answers.keys() == expected.answers.keys()
    for name, p in _probabilities(got).items():
        assert p == pytest.approx(_probabilities(expected)[name], abs=2e-4), name


REQUESTS = {
    "one-question": (_request(1, ["team"]), [("forward", 1, 128)]),
    "three-questions": (
        _request(1, ["urgent", "team", "mood"]),
        [("prefix", 1, 128), ("suffix", 3, 128)],
    ),
    # two suffix forwards: 3 then 2 (the smallest suffix batch that holds the rest)
    "five-questions": (
        _request(
            1, ["urgent", "team", "mood"], {"u2": QUESTIONS["urgent"], "m2": QUESTIONS["mood"]}
        ),
        [("prefix", 1, 128), ("suffix", 3, 128), ("suffix", 2, 128)],
    ),
    # a state past the largest prefix bucket (256): whole prompts, in the 384 bucket, which
    # has one batch size
    "long-state": (_request(20, ["urgent", "team"]), [("forward", 1, 384), ("forward", 1, 384)]),
    # a question past suffix_len: whole prompts
    "long-question": (
        _request(1, ["team"], {"long": LONG_QUESTION}),
        [("forward", 2, 256)],
    ),
}


@pytest.mark.parametrize("name", list(REQUESTS))
def test_neuron_answers_as_torch_does(engines, name):
    request, shapes = REQUESTS[name]
    neuron = engines["neuron"]
    neuron.runtime.calls.clear()
    got = neuron.evaluate(request)
    _assert_same(got, engines["torch"].evaluate(request))
    assert neuron.runtime.calls == shapes
    # The token count follows the path, as torch's does: the state once through the shared
    # prefix, once per question for whole prompts (which the fallbacks take).
    shared = any(kind == "prefix" for kind, *_ in shapes)
    assert got.usage == engines["torch" if shared else "torch_batched"].evaluate(request).usage


@pytest.mark.parametrize("name", list(REQUESTS))
def test_without_the_prefix_cache_every_prompt_is_whole(engines, name):
    request, _ = REQUESTS[name]
    neuron = engines["neuron_batched"]
    neuron.runtime.calls.clear()
    got = neuron.evaluate(request)
    expected = engines["torch_batched"].evaluate(request)
    _assert_same(got, expected)
    assert got.usage == expected.usage
    assert {kind for kind, *_ in neuron.runtime.calls} == {"forward"}


def test_max_batch_chunks_questions_as_torch_does(checkpoint, engines):
    neuron = _neuron(checkpoint, max_batch=2)
    request = _request(1, ["urgent", "team", "mood"])
    _assert_same(neuron.evaluate(request), engines["torch"].evaluate(request))
    # chunks of 2 then 1: a prefix and its suffixes, then one whole prompt
    assert neuron.runtime.calls == [("prefix", 1, 128), ("suffix", 2, 128), ("forward", 1, 128)]


def test_more_options_than_k_pad_are_refused(engines):
    question = ChoiceQuestion(instructions="Pick.", criteria={c: None for c in "abcde"})
    with pytest.raises(ValueError, match="at most 4"):
        engines["neuron"].evaluate(SystemOneRequest(state=TICKET, questions={"q": question}))


def test_images_are_refused(engines):
    request = _request(0, ["urgent"]).model_copy(update={"images": ["aGVsbG8="]})
    with pytest.raises(ValueError, match="vision"):
        engines["neuron"].evaluate(request)


def test_a_window_past_the_largest_bucket_is_refused_at_load(checkpoint):
    small = NeuronConfig(buckets=[128, 256], prefix_buckets=[128], dtype="float32")
    with pytest.raises(ValueError, match="largest neuron bucket is 256"):
        _neuron(checkpoint, neuron_config=small)


def test_warm_up_runs_every_compiled_shape_once(checkpoint):
    neuron = _neuron(checkpoint)
    neuron.warm_up()
    expected = [(k, B, T if k != "suffix" else SMALL.suffix_len) for k, T, B in SMALL.shapes()]
    assert neuron.runtime.calls == expected


@pytest.mark.parametrize(
    "bad",
    [{"buckets": [100]}, {"buckets": [256, 128]}, {"prefix_buckets": []}, {"suffix_len": 100}],
)
def test_shapes_off_the_kernel_tile_are_refused(bad):
    with pytest.raises(ValueError, match="multiple"):
        NeuronConfig(**bad)
