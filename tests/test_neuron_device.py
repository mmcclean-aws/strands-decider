"""On a NeuronCore: the compiled engine, with nkilib's kernels, answers as the eager model does.

Runs only on AWS Trainium with the Neuron SDK (`pytest -m neuron`; it skips elsewhere). A random
four-layer torso with the 2B's layer dimensions (the kernels' tile sizes) is saved as a
checkpoint and loaded with `load_neuron_engine`, so the test covers the loader, compilation, both
kernels and both forward paths on the device. The reference is the same weights run eagerly in
fp32 on the CPU; the device runs bf16.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

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


@pytest.fixture(scope="module", params=[False, True], ids=["nkilib", "fast"])
def engines(checkpoint, request):
    shapes = dataclasses.replace(SHAPES, fast_kernels=request.param)
    device = load_neuron_engine(checkpoint, EngineConfig(), shapes)
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


# ---- the kernels alone, against the torch references (fp32 on the CPU)

DEV = torch.device("neuron", 0)


def _compile(fn):
    from libtorch_neuronx_lite.compile.native_backend import register

    return torch.compile(fn, backend=register(), fullgraph=True, dynamic=False)


def _rel(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _gdn_inputs(B, T, H=16, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, D, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, H, T, D, generator=g), dim=-1)
    v = torch.randn(B, H, T, D, generator=g) * 0.5
    gate = -0.5 * torch.rand(B, H, T, generator=g)
    beta = torch.sigmoid(torch.randn(B, H, T, generator=g) * 2)
    return q, k, v, gate, beta


@pytest.mark.parametrize("state", [False, True], ids=["zero-state", "initial-state"])
@pytest.mark.parametrize(("B", "T"), [(1, 128), (2, 512), (1, 2048)])
def test_gdn_fast_matches_the_reference(B, T, state):
    from strands_decider.neuron_kernels import FastGdn
    from strands_decider.neuron_model import gdn_chunk_ref

    q, k, v, g, beta = _gdn_inputs(B, T, seed=B * 7 + T)
    S0 = torch.randn(B, 16, 128, 128) * 0.05 if state else None
    core = FastGdn(DEV)
    fn = _compile(lambda *a: core(*a, return_state=True))
    args = [x.to(torch.bfloat16).to(DEV) for x in (q, k, v)] + [g.to(DEV), beta.to(DEV)]
    if state:
        args.append(S0.to(DEV))
    with torch.inference_mode():
        out, final = fn(*args)
    qq, kk, vv = (x.to(torch.bfloat16).float() for x in (q, k, v))
    ref, ref_final = gdn_chunk_ref(qq, kk, vv, g, beta, S0=S0, return_state=True)
    assert _rel(out, ref) <= 1e-2 and _rel(final, ref_final) <= 1e-2


@pytest.mark.parametrize("T", [128, 1024])
def test_the_fused_prologue_and_norms_match_their_references(T):
    from strands_decider.neuron_kernels import fused_ops, layouts
    from strands_decider.neuron_model import gated_rms_norm, rms_norm

    B, g = 2, torch.Generator().manual_seed(1)
    bf = torch.bfloat16
    proj_hm = (torch.randn(B, 48, T, 128, generator=g) * 0.7).to(bf)
    conv_w = (torch.randn(48, 128, 4, generator=g) * 0.4).to(bf)
    zba = (torch.randn(B, T, 2080, generator=g) * 1.5).to(bf)
    A, dtb = -(torch.rand(16, generator=g) * 4 + 0.1), torch.randn(16, generator=g) * 0.5
    hist = (torch.randn(1, 48, 3, 128, generator=g) * 0.7).to(bf)
    ref = layouts.gdn_prologue_ref(proj_hm, conv_w, zba[..., 2048:], A, dtb, conv_hist=hist)
    fn = _compile(lambda p, w, z, ab, h: fused_ops.gdn_prologue(p, w, z, ab, conv_hist=h))
    with torch.inference_mode():
        out = fn(
            proj_hm.permute(0, 1, 3, 2).contiguous().to(DEV),
            layouts.conv_diag_from_conv_w(conv_w).to(DEV),
            zba.to(DEV),
            torch.stack([A, dtb], 1).contiguous().to(DEV),
            hist.to(DEV),
        )
    for name, got, want in zip(["q", "k", "v", "g", "beta"], out, ref, strict=True):
        assert _rel(got, want) <= 5e-3, name

    x = (torch.randn(B, T, 2048, generator=g) * 0.5).to(bf)
    ln = 1 + 0.1 * torch.randn(2048, generator=g)
    with torch.inference_mode():
        got = _compile(fused_ops.rms_norm_nki)(x.to(DEV), ln.to(DEV))
    assert _rel(got, rms_norm(x.float(), ln, 1e-6)) <= 5e-3

    o = (torch.randn(B, 16, T, 128, generator=g) * 0.3).to(bf)
    gn = (1 + 0.1 * torch.randn(128, generator=g)).to(bf)
    z = zba[..., :2048].reshape(B, T, 16, 128).transpose(1, 2)
    want = (
        gated_rms_norm(o.float(), z.float(), gn.float(), 1e-6).transpose(1, 2).reshape(B, T, 2048)
    )
    with torch.inference_mode():
        got = _compile(fused_ops.gated_rms_norm_nki)(o.to(DEV), zba.to(DEV), gn.to(DEV))
    assert _rel(got, want) <= 5e-3
