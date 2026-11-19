"""The static-shape model behind `--device neuron` must compute what StrandsDeciderModel computes.

A tiny random-weight Qwen3.5 text model (Gated DeltaNet and attention layers) is saved as the
base checkpoint, and a decider checkpoint is built on it as training saves one, with a LoRA
adapter over the projections v19's adapter targets. Both models load it from disk, so the test
covers loading through StrandsDeciderModel.load, the adapter merge, the weight layouts, the
readout and the shared-prefix forwards. The torch reference mixers run on the CPU; nothing
needs Trainium, and nothing is downloaded.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.neuron_model import (  # noqa: E402
    NeuronDeciderModel,
    TorsoShape,
    from_decider_model,
)

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
]
VOCAB = 96


def _tokenizer(words=()):
    """A word-level tokenizer: the given words, then filler words up to VOCAB."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    names = ["<pad>", "<eos>", "<unk>", *sorted(set(words))]
    names += [f"w{i}" for i in range(max(0, VOCAB - len(names)))]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(names)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


def _base_config(**overrides):
    kw = dict(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    kw.update(overrides)
    return transformers.Qwen3_5TextConfig(**kw)


def build_checkpoint(tmp, words=(), max_length=512):
    """A decider checkpoint on a tiny random Qwen3.5, saved as training saves one, with a
    LoRA adapter and readout that change the answers. Returns its path."""
    tok = _tokenizer(words)
    torch.manual_seed(0)
    base = tmp / "base"
    lm = transformers.Qwen3_5ForCausalLM(_base_config(vocab_size=len(tok)))
    with torch.no_grad():  # the norms start at zero (1 + w = 1): give the (1 + w) fold work
        for name, p in lm.named_parameters():
            if "norm" in name:
                p.normal_(std=0.2)
    lm.save_pretrained(base)
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        pointer_dim=16,
        max_length=max_length,
        torch_dtype="float32",
        lora_r=4,
        lora_alpha=8,
        lora_targets=LORA_TARGETS,
        temperature_by_kind={"choice": 1.7, "noul": 0.8},
    )
    torso = transformers.Qwen3_5ForCausalLM.from_pretrained(base).model
    model = StrandsDeciderModel(cfg, torso, tok)
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for name, p in model.torso.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
        for p in model.head.parameters():
            p.normal_(std=0.3)
    path = tmp / "ckpt"
    model.save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return build_checkpoint(tmp_path_factory.mktemp("neuron_model"))


@pytest.fixture(scope="module")
def models(checkpoint):
    reference = StrandsDeciderModel.load(checkpoint).eval()
    shape, sd = from_decider_model(StrandsDeciderModel.load(checkpoint), dtype=torch.float32)
    return reference, NeuronDeciderModel(shape, sd, compute_dtype=torch.float32).eval()


def _rows(lengths, k_pad=6, seed=1):
    """Right-padded random rows: ids [B, T], the mask, answer and option positions, option counts."""
    g = torch.Generator().manual_seed(seed)
    T = max(lengths)
    ids = torch.zeros(len(lengths), T, dtype=torch.long)
    mask = torch.zeros(len(lengths), T, dtype=torch.long)
    opt = torch.full((len(lengths), k_pad), -1, dtype=torch.long)
    n_slots = torch.zeros(len(lengths), dtype=torch.long)
    for j, L in enumerate(lengths):
        ids[j, :L] = torch.randint(3, VOCAB, (L,), generator=g)
        mask[j, :L] = 1
        n = 2 + j % (k_pad - 1)
        opt[j, :n] = torch.sort(torch.randperm(L - 1, generator=g)[:n]).values
        n_slots[j] = n
    ans = mask.sum(1) - 1
    return ids, mask, ans, opt, n_slots


def _reference_probs(reference, ids, mask, opt, n_slots, temperature):
    with torch.inference_mode():
        out = reference(
            input_ids=ids,
            attention_mask=mask,
            n_slots=n_slots,
            temperature=temperature,
            opt_idx=opt,
        )
    return out["log_probs"].exp()


def test_the_layout_is_read_from_the_torso_config(models):
    _, neuron = models
    c = neuron.cfg
    assert (c.hidden, c.gh, c.gd, c.nq, c.nkv, c.hd, c.rope_dim) == (64, 4, 16, 4, 2, 16, 4)
    assert c.temperature_by_kind == {"choice": 1.7, "noul": 0.8}
    assert c.max_length == 512


# Lengths that cross the Gated DeltaNet chunk (64) and leave right padding in every batch.
@pytest.mark.parametrize(
    "lengths", [[9], [40, 17, 33], [130, 70, 64]], ids=["one", "short", "chunks"]
)
def test_the_static_model_answers_as_the_torch_model_does(models, lengths):
    reference, neuron = models
    ids, mask, ans, opt, n_slots = _rows(lengths)
    temperature = torch.tensor([1.7, 0.8, 1.0][: len(lengths)])
    expected = _reference_probs(reference, ids, mask, opt, n_slots, temperature)
    with torch.inference_mode():
        got = neuron(ids, ans, opt, n_slots, 1.0 / temperature)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)
    for j, n in enumerate(n_slots.tolist()):  # zero past each row's options, and a distribution
        assert got[j, n:].abs().max() == 0
        assert abs(float(got[j].sum()) - 1) < 1e-5


def test_padding_does_not_reach_real_positions(models):
    _, neuron = models
    ids, _, ans, opt, n_slots = _rows([20])
    inv = torch.ones(1)
    padded = torch.cat([ids, torch.randint(3, VOCAB, (1, 50))], 1)
    with torch.inference_mode():
        torch.testing.assert_close(
            neuron(padded, ans, opt, n_slots, inv),
            neuron(ids, ans, opt, n_slots, inv),
            atol=1e-6,
            rtol=0,
        )


@pytest.mark.parametrize("s_len", [1, 3, 47, 64, 100])
def test_a_shared_prefix_answers_as_the_whole_prompt(models, s_len):
    _, neuron = models
    bucket = 128
    suffix_lengths = [12, 7, 20]
    L = 24
    ids, _, ans, opt, n_slots = _rows([s_len + n for n in suffix_lengths], seed=s_len)
    state = ids[:1, :s_len]
    for j in range(1, len(suffix_lengths)):  # every row shares the first row's state
        ids[j, :s_len] = state[0]
    opt = torch.where(opt >= s_len, opt, torch.full_like(opt, s_len))  # options in the question
    inv = torch.tensor([1.0, 0.5, 2.0])
    with torch.inference_mode():
        whole = neuron(ids, ans, opt, n_slots, inv)
        prefix = torch.zeros(1, bucket, dtype=torch.long)
        prefix[0, :s_len] = state[0]
        prefix[0, s_len:] = 5  # bucket padding: must not change a thing
        caches = neuron.prefix_forward(prefix, torch.tensor([s_len]))
        suffix = torch.zeros(len(suffix_lengths), L, dtype=torch.long)
        for j, n in enumerate(suffix_lengths):
            suffix[j, :n] = ids[j, s_len : s_len + n]
        got = neuron.suffix_forward(
            suffix, torch.tensor([s_len]), caches, ans - s_len, opt - s_len, n_slots, inv
        )
    torch.testing.assert_close(got, whole, atol=1e-5, rtol=0)


def test_bf16_stays_near_fp32(checkpoint, models):
    _, fp32 = models
    shape, sd = from_decider_model(StrandsDeciderModel.load(checkpoint), dtype=torch.bfloat16)
    bf16 = NeuronDeciderModel(shape, sd, compute_dtype=torch.bfloat16).eval()
    ids, _, ans, opt, n_slots = _rows([40, 17, 33])
    inv = torch.ones(3)
    with torch.inference_mode():
        diff = (bf16(ids, ans, opt, n_slots, inv) - fp32(ids, ans, opt, n_slots, inv)).abs().max()
    assert 0 < float(diff) < 0.05


@pytest.mark.parametrize(
    ("overrides", "named"),
    [
        ({"linear_num_key_heads": 2}, "key heads"),
        ({"linear_key_head_dim": 32}, "key dim"),
        (
            {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 1e4}},
            "rope_type",
        ),
    ],
)
def test_other_torso_layouts_are_refused(overrides, named):
    with pytest.raises(ValueError, match=named):
        TorsoShape.from_text_config(_base_config(**overrides), pointer_dim=16)


def test_a_token_readout_checkpoint_is_refused(checkpoint):
    model = StrandsDeciderModel.load(checkpoint)
    model.config = dataclasses.replace(model.config, head_type="token_readout")
    with pytest.raises(ValueError, match="token_readout"):
        from_decider_model(model)


def test_the_adapter_loads_on_the_cpu_whatever_accelerator_torch_reports(checkpoint, monkeypatch):
    # With libtorch_neuronx_lite imported, torch.accelerator reports "neuron"; PEFT would stage
    # the adapter there, and safetensors cannot address it.
    class _Neuron:
        type = "neuron"

    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *a, **k: _Neuron())
    model = StrandsDeciderModel.load(checkpoint)
    assert {p.device.type for p in model.torso.parameters()} == {"cpu"}
