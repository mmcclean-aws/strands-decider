"""The fused Neuron kernels' weight layouts (neuron_kernels/layouts.py) and torch references must
compute what NeuronDeciderModel's reference layers compute.

The kernels themselves run only on Trainium (tests/test_neuron_device.py); this checks, on the
CPU and on test_neuron_model's tiny random Qwen3.5, that the re-laid-out weights hold the same
numbers in the order the kernels read them, and that the torch reference of the fused
prologue is the model's math.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from test_neuron_model import checkpoint  # noqa: F401  (the fixture)

from strands_decider.modeling import StrandsDeciderModel
from strands_decider.neuron_kernels import layouts
from strands_decider.neuron_model import (
    NeuronDeciderModel,
    from_decider_model,
    rms_norm,
)


@pytest.fixture(scope="module")
def model_and_sd(checkpoint):  # noqa: F811
    shape, sd = from_decider_model(StrandsDeciderModel.load(checkpoint), dtype=torch.float32)
    return NeuronDeciderModel(shape, sd, compute_dtype=torch.float32).eval(), sd


def _x(model, T=24, B=2):
    g = torch.Generator().manual_seed(3)
    return torch.randn(B, T, model.cfg.hidden, generator=g)


def test_every_layer_gets_its_kind_of_layout(model_and_sd):
    model, sd = model_and_sd
    preps = layouts.prepare_layers(model.cfg.layer_types, sd, torch.float32)
    for lt, prep in zip(model.cfg.layer_types, preps, strict=True):
        assert {"ln2", "w_gate_up_t", "w_down_t"} <= set(prep)
        if lt == "linear_attention":
            assert {"w_qkv_t", "w_zba_t", "conv_diag", "gdn_ab", "w_out_t"} <= set(prep)
        else:
            assert {"attn_w_qgkv_t", "attn_w_o_t", "attn_q_norm", "attn_ln1"} <= set(prep)


def test_the_mlp_layout_computes_the_mlp(model_and_sd):
    model, sd = model_and_sd
    x = _x(model)
    with torch.inference_mode():
        got = layouts.mlp_fast(x, layouts.prepare_mlp_layer(sd, 0, torch.float32), model.cfg.eps)
        torch.testing.assert_close(got, model.mlp(0, x), atol=1e-5, rtol=1e-5)


def test_the_gdn_layouts_hold_the_model_projections(model_and_sd):
    model, sd = model_and_sd
    c, x = model.cfg, _x(model)
    prep = layouts.prepare_gdn_layer(sd, 0, torch.float32)
    with torch.inference_mode():
        qkvz, beta, g = model._gdn_proj(0, x)
        h = rms_norm(x, model.w("layers.0.ln1"), c.eps)
        proj_fm = torch.einsum("btc,chd->bhdt", h, prep["w_qkv_t"])
        zba = h @ prep["w_zba_t"]
    gh, gd = c.gh, c.gd
    torch.testing.assert_close(proj_fm, qkvz[:, : 3 * gh].transpose(2, 3), atol=1e-5, rtol=1e-5)
    z = qkvz[:, 3 * gh :].transpose(1, 2).reshape(x.shape[0], x.shape[1], gh * gd)
    torch.testing.assert_close(zba[..., : gh * gd], z, atol=1e-5, rtol=1e-5)
    # the prologue reads b and a from the last 2 gh columns, and (A, dt_bias) packed per head
    ref = layouts.gdn_prologue_ref(
        qkvz[:, : 3 * gh], sd["layers.0.conv_w"], zba[..., gh * gd :], *prep["gdn_ab"].unbind(1)
    )
    torch.testing.assert_close(ref[3], g, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ref[4], beta, atol=1e-5, rtol=1e-5)
    # the out-projection, token-major [(h, d), hidden]
    o = torch.randn(x.shape[0], gh, x.shape[1], gd)
    expected = torch.einsum("bhtd,chd->btc", o, model.w("layers.0.w_out"))
    got = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ prep["w_out_t"]
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_the_prologue_reference_is_the_model_math(model_and_sd):
    """q, k, v after the convolution, SiLU and L2 norm, as the model's layer computes them
    (rounded to bf16, as the kernel returns them), with and without a prefix's conv history."""
    model, sd = model_and_sd
    c, x = model.cfg, _x(model)
    gh = c.gh
    with torch.inference_mode():
        qkvz, _, _ = model._gdn_proj(0, x)
        pre = qkvz[:, : 3 * gh]
        from strands_decider.neuron_model import causal_conv1d_silu

        conv = causal_conv1d_silu(pre, sd["layers.0.conv_w"])
        expected = model._qkv(conv)
        zeros = torch.zeros(x.shape[0], x.shape[1], 2 * gh)
        ref = layouts.gdn_prologue_ref(
            pre, sd["layers.0.conv_w"], zeros, torch.ones(gh), torch.zeros(gh)
        )
        for got, want in zip(ref[:3], expected, strict=True):
            torch.testing.assert_close(
                got.float(), want.to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2
            )
        # history: the last K-1 pre-conv tokens of a prefix continue the convolution exactly
        s_len = 10
        tail = layouts.conv_tail_fm(pre[:1, :, :s_len].transpose(2, 3), torch.tensor([s_len]))
        torch.testing.assert_close(
            tail, model.conv_tail(pre[:1, :, :s_len], torch.tensor([s_len]), c.conv_k)
        )
        cont = layouts.gdn_prologue_ref(
            pre[:1, :, s_len:],
            sd["layers.0.conv_w"],
            zeros[:1, s_len:],
            torch.ones(gh),
            torch.zeros(gh),
            conv_hist=tail,
        )
        whole = layouts.gdn_prologue_ref(
            pre[:1], sd["layers.0.conv_w"], zeros[:1], torch.ones(gh), torch.zeros(gh)
        )
        for a, b in zip(cont[:3], whole[:3], strict=True):
            torch.testing.assert_close(a.float(), b[:, :, s_len:].float())


def test_the_attention_layouts_hold_the_model_projections(model_and_sd):
    model, sd = model_and_sd
    c, x = model.cfg, _x(model)
    i = c.layer_types.index("full_attention")
    prep = layouts.prepare_attn_layer(sd, i, torch.float32)
    with torch.inference_mode():
        h = rms_norm(x, model.w(f"layers.{i}.ln1"), c.eps)
        expected = torch.einsum("btc,hdc->bhtd", h, model.w(f"layers.{i}.w_qgkv"))
        got = torch.einsum("btc,chd->bhtd", h, prep["w_qgkv_t"])
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    o = torch.randn(x.shape[0], c.nq, x.shape[1], c.hd)
    expected = torch.einsum("bhtd,chd->btc", o, model.w(f"layers.{i}.w_o"))
    got = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1) @ prep["w_o_t"]
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)
