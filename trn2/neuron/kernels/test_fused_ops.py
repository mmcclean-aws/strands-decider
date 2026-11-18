"""Correctness + benchmark for neuron/kernels/fused_ops.py on ONE LNC=2 NeuronCore (trn2).

Run on the trn2 host (one process per core):
    export VLLM_PLUGINS=none VLLM_NEURON_BACKEND=neuron_native NEURON_LOGICAL_NC_CONFIG=2 \
           NEURON_RT_VISIBLE_CORES=40 NEURON_RT_ROOT_COMM_ID=127.0.0.1:63108 TORCH_CACHING_PRECOMPILE=0 \
           NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS=32 NEURON_RT_IO_RING_CACHE_SIZE=4
    python neuron/kernels/test_fused_ops.py --T 512,2048 --only correctness,prologue,epilogue,layer,mlp,gemm,nkilib

Sections
    correctness  NKI kernels vs the torch math of decider_model.py (rel L2 error, CPU fp32 reference):
                 gdn_prologue (+ feature-major variant), rms_norm_nki, gated_rms_norm_nki, and the
                 whole gdn_layer_fast / mlp_fast vs DeciderNeuron.gdn_layer / .mlp (bf16, same weights,
                 the GDN core replaced by a cheap stand-in so only the non-core path is compared).
    prologue     current decider prologue (RMSNorm + einsum + conv + SiLU + L2 + beta/g) vs
                 gdn_in_proj + gdn_prologue; the NKI kernel alone.
    epilogue     current gated_rms_norm + einsum out-proj + residual vs gdn_epilogue.
    layer        GDN layer minus the core: DeciderNeuron.gdn_layer vs gdn_layer_fast.
    mlp          DeciderNeuron.mlp vs mlp_fast.
    tp           TP shards (H = 16 / 8 / 4 heads per rank, shard_state order): gdn_prologue (incl. conv history,
                 Bh = 1 / B, suffix B = 2 / 8, T = 128 and T = 1024), gdn_layer_fast, gdn_layer_suffix_fast and
                 gdn_layer_prefix_fast vs the DeciderNeuron methods on the same shard (CPU fp32 reference).
    gemm         compiler GEMM TF/s, decider layouts (x @ W.t() / head-major einsum) vs pre-transposed [K, N].
    nkilib       nkilib qkv_cte / output_projection_cte / mlp (CTE) at the decider shapes.

Timing: torch.compile(backend=neuron_native_lite), pipelined calls, fence on a tiny graph; each call is one
graph, so per-call dispatch (~20-80 us) is included -- compare rows at equal T.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
os.environ.setdefault("TORCH_CACHING_PRECOMPILE", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import libtorch_neuronx_lite  # noqa: E402,F401
from libtorch_neuronx_lite.compile.native_backend import register  # noqa: E402

import fused_ops as fo  # noqa: E402
from decider_model import (DeciderConfig, DeciderNeuron, causal_conv1d_silu_hm, gated_rms_norm, l2norm,  # noqa: E402
                           rms_norm, shard_state)

BACKEND = register()
torch._dynamo.config.recompile_limit = 256   # many shapes per lambda code object in this script
DEV = torch.device("neuron", 0)
PEAK = 667e12 / 4            # LNC2 bf16 dense peak, 166.75 TF/s
DT = torch.bfloat16
EPS = 1e-6
ROWS: list[tuple] = []       # (section, name, T, ms, tflops or None, extra)

_ft = torch.zeros(1, device=DEV)
_ff = torch.compile(lambda t: t + 1, backend=BACKEND, fullgraph=True, dynamic=False)


def fence():
    _ff(_ft).cpu()


def bench(section, name, T, fn, args, flops=0.0, iters=20, extra=""):
    f = torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        y = f(*args)
        fence()
        for _ in range(3):
            y = f(*args)
        fence()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = f(*args)
        fence()
        d = (time.perf_counter() - t0) / iters
    tf = flops / d / 1e12 if flops else None
    ROWS.append((section, name, T, d * 1e3, tf, extra))
    tfs = f"{tf:6.1f} TF/s {tf * 1e12 / PEAK:5.1%}" if tf else ""
    print(f"[{section}] {name:58s} T={T:5d} {d * 1e3:8.3f} ms {tfs} {extra}", flush=True)
    return d, y


def rel(a, b):
    a, b = a.detach().cpu().float(), b.detach().cpu().float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def record_err(name, T, err, tol=5e-3):
    ok = "PASS" if err <= tol else "FAIL"
    ROWS.append(("correctness", name, T, None, None, f"rel L2 {err:.2e} {ok}"))
    print(f"[correctness] {name:58s} T={T:5d} rel L2 {err:.2e}  {ok}", flush=True)


# ----------------------------------------------------------------------------- random layer weights
def random_layer_sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc  # noqa: E731
    p = "layers.0."
    return {
        p + "ln1": 1 + 0.1 * r(2048), p + "ln2": 1 + 0.1 * r(2048),
        p + "w_qkvz": r(64, 128, 2048, sc=0.02).to(DT), p + "w_ba": r(32, 2048, sc=0.02).to(DT),
        p + "conv_w": r(48, 128, 4, sc=0.4).to(DT), p + "A": -(r(16).abs() * 2 + 0.1), p + "dt_bias": 0.5 * r(16),
        p + "gnorm": (1 + 0.1 * r(128)).to(DT), p + "w_out": r(2048, 16, 128, sc=0.02).to(DT),
        p + "w_gate_up": r(12288, 2048, sc=0.02).to(DT), p + "w_down": r(2048, 6144, sc=0.02).to(DT),
    }


def core_stub(q, k, v, g, beta):
    """Cheap stand-in for the GDN core (keeps every prologue output live)."""
    return (v.float() * beta[..., None] + q.float() * g[..., None] + 0.1 * k.float()).to(v.dtype)


def decider_module(sd, dev, compute_dtype=DT):
    cfg = DeciderConfig(n_layers=1, layer_types=["linear_attention"])
    return DeciderNeuron(cfg, {k: v.to(dev) for k, v in sd.items()}, gdn_impl=core_stub,
                         compute_dtype=compute_dtype).to(dev)


def to_dev(d):
    return {k: v.to(DEV) for k, v in d.items()}


# ----------------------------------------------------------------------------- correctness
def correctness(T, B):
    g = torch.Generator().manual_seed(1)
    proj_hm = (torch.randn(B, 48, T, 128, generator=g) * 0.7).to(DT)
    conv_w = (torch.randn(48, 128, 4, generator=g) * 0.4).to(DT)
    zba = (torch.randn(B, T, 2080, generator=g) * 1.5).to(DT)
    A = -(torch.rand(16, generator=g) * 4 + 0.1)
    dtb = torch.randn(16, generator=g) * 0.5
    gab = torch.stack([A, dtb], 1).contiguous()
    ref = fo.gdn_prologue_ref(proj_hm, conv_w, zba[..., 2048:], A, dtb)
    proj_fm = proj_hm.permute(0, 1, 3, 2).contiguous()
    fm = T <= 1024
    f = torch.compile(lambda p, w, z, ab: fo.gdn_prologue(p, w, z, ab, emit_qk_fm=fm), backend=BACKEND, fullgraph=True)
    with torch.inference_mode():
        out = f(proj_fm.to(DEV), fo.conv_diag_from_conv_w(conv_w).to(DEV), zba.to(DEV), gab.to(DEV))
    for n, o, r in zip(["q", "k", "v", "g", "beta"], out, ref):
        record_err(f"gdn_prologue {n} (B={B})", T, rel(o, r))
    if fm:
        record_err(f"gdn_prologue q_fm (B={B})", T, rel(out[5], ref[0].transpose(2, 3)))
        record_err(f"gdn_prologue k_fm (B={B})", T, rel(out[6], ref[1].transpose(2, 3)))

    x = (torch.randn(B, T, 2048, generator=g) * 0.5).to(DT)
    ln = 1 + 0.1 * torch.randn(2048, generator=g)
    f = torch.compile(lambda x, ln: fo.rms_norm_nki(x, ln), backend=BACKEND, fullgraph=True)
    with torch.inference_mode():
        record_err("rms_norm_nki", T, rel(f(x.to(DEV), ln.to(DEV)), rms_norm(x.float(), ln, EPS)))

    o = (torch.randn(B, 16, T, 128, generator=g) * 0.3).to(DT)
    gn = (1 + 0.1 * torch.randn(128, generator=g)).to(DT)
    z_hm = zba[..., :2048].reshape(B, T, 16, 128).transpose(1, 2)
    refg = gated_rms_norm(o.float(), z_hm.float(), gn.float(), EPS).transpose(1, 2).reshape(B, T, 2048)
    f = torch.compile(lambda o, z, w: fo.gated_rms_norm_nki(o, z, w), backend=BACKEND, fullgraph=True)
    with torch.inference_mode():
        record_err("gated_rms_norm_nki", T, rel(f(o.to(DEV), zba.to(DEV), gn.to(DEV)), refg))

    # whole layers vs DeciderNeuron (same bf16 weights; CPU fp32 copy of the decider math as the reference)
    sd = random_layer_sd()
    xs = (torch.randn(B, T, 2048, generator=g) * 0.5).to(DT)
    m_ref = decider_module({k: v.float() for k, v in sd.items()}, "cpu", compute_dtype=torch.float32)
    with torch.inference_mode():
        y_ref = m_ref.gdn_layer(0, xs.float())
        y_mlp_ref = m_ref.mlp(0, xs.float())
    m_dev = decider_module(sd, DEV)
    prep = to_dev(fo.prepare_gdn_layer(sd, 0))
    mprep = to_dev(fo.prepare_mlp_layer(sd, 0))
    xd = xs.to(DEV)
    fc = torch.compile(lambda x: m_dev.gdn_layer(0, x), backend=BACKEND, fullgraph=True)
    fn = torch.compile(lambda x: fo.gdn_layer_fast(x, prep, core_stub), backend=BACKEND, fullgraph=True)
    mc = torch.compile(lambda x: m_dev.mlp(0, x), backend=BACKEND, fullgraph=True)
    mn = torch.compile(lambda x: fo.mlp_fast(x, mprep), backend=BACKEND, fullgraph=True)
    with torch.inference_mode():
        # compare the layer DELTA (y - x): the residual would otherwise hide errors
        record_err("decider gdn_layer (bf16, device) delta", T, rel(fc(xd).cpu().float() - xs.float(), y_ref - xs.float()), 2e-2)
        record_err("gdn_layer_fast delta", T, rel(fn(xd).cpu().float() - xs.float(), y_ref - xs.float()), 2e-2)
        record_err("decider mlp (bf16, device) delta", T, rel(mc(xd).cpu().float() - xs.float(), y_mlp_ref - xs.float()), 2e-2)
        record_err("mlp_fast delta", T, rel(mn(xd).cpu().float() - xs.float(), y_mlp_ref - xs.float()), 2e-2)


# ----------------------------------------------------------------------------- perf sections
def perf_prologue(T, B, sd):
    m = decider_module(sd, DEV)
    prep = to_dev(fo.prepare_gdn_layer(sd, 0))
    x = (torch.randn(B, T, 2048) * 0.5).to(DT).to(DEV)
    fl = 2 * B * T * 2048 * 8224

    def cur(x):
        c, p = m.cfg, "layers.0."
        h = rms_norm(x, m.w(p + "ln1"), c.eps)
        qkvz = torch.einsum("btc,hdc->bhtd", h, m.w(p + "w_qkvz"))
        conv = causal_conv1d_silu_hm(qkvz[:, :48], m.w(p + "conv_w"))
        q, k, v, z = l2norm(conv[:, :16]), l2norm(conv[:, 16:32]), conv[:, 32:48], qkvz[:, 48:]
        ba = (h @ m.w(p + "w_ba").t()).float().transpose(1, 2)
        beta = torch.sigmoid(ba[:, :16])
        g = m.w(p + "A")[None, :, None] * F.softplus(ba[:, 16:] + m.w(p + "dt_bias")[None, :, None])
        return q.to(DT), k.to(DT), v.to(DT), g, beta, z

    def new(x):
        proj_fm, zba = fo.gdn_in_proj(x, prep["ln1"], prep)
        return fo.gdn_prologue(proj_fm, prep["conv_diag"], zba, prep["gdn_ab"]) + (zba,)

    bench("prologue", "current: rms+einsum+conv+silu+l2+beta/g (torch)", T, cur, (x,), fl)
    bench("prologue", "new: NKI rms + 2 GEMMs + NKI prologue", T, new, (x,), fl)
    proj_fm = (torch.randn(B, 48, 128, T) * 0.7).to(DT).to(DEV)
    zba = torch.randn(B, T, 2080).to(DT).to(DEV)
    bench("prologue", "NKI gdn_prologue kernel alone", T, lambda p, z: fo.gdn_prologue(p, prep["conv_diag"], z, prep["gdn_ab"]),
          (proj_fm, zba))
    bench("prologue", "gdn_in_proj alone (NKI rms + 2 GEMMs)", T, lambda x: fo.gdn_in_proj(x, prep["ln1"], prep), (x,), fl)
    bench("prologue", "gdn_in_proj with torch rms (fused by compiler)", T,
          lambda x: fo.gdn_in_proj(x, prep["ln1"], prep, nki_norm=False), (x,), fl)


def perf_epilogue(T, B, sd):
    prep = to_dev(fo.prepare_gdn_layer(sd, 0))
    w_out = sd["layers.0.w_out"].to(DEV)
    x = (torch.randn(B, T, 2048) * 0.5).to(DT).to(DEV)
    o = (torch.randn(B, 16, T, 128) * 0.3).to(DT).to(DEV)
    zba = torch.randn(B, T, 2080).to(DT).to(DEV)
    fl = 2 * B * T * 2048 * 2048

    def cur(x, o, zba):
        z = zba[..., :2048].reshape(B, T, 16, 128).transpose(1, 2)
        return x + torch.einsum("bhtd,chd->btc", gated_rms_norm(o, z, prep["gnorm"], EPS), w_out)

    bench("epilogue", "current: gated_rms_norm + einsum out-proj + resid", T, cur, (x, o, zba), fl)
    bench("epilogue", "new: NKI gated_rms_norm + x@Wt + resid", T,
          lambda x, o, zba: fo.gdn_epilogue(x, o, zba, prep["gnorm"], prep["w_out_t"]), (x, o, zba), fl)
    bench("epilogue", "NKI gated_rms_norm alone", T, lambda o, zba: fo.gated_rms_norm_nki(o, zba, prep["gnorm"]), (o, zba))


def perf_layer(T, B, sd):
    m = decider_module(sd, DEV)
    prep = to_dev(fo.prepare_gdn_layer(sd, 0))
    x = (torch.randn(B, T, 2048) * 0.5).to(DT).to(DEV)
    fl = 2 * B * T * 2048 * (8224 + 2048)
    bench("layer", "GDN layer w/o core: DeciderNeuron.gdn_layer", T, lambda x: m.gdn_layer(0, x), (x,), fl)
    bench("layer", "GDN layer w/o core: gdn_layer_fast", T, lambda x: fo.gdn_layer_fast(x, prep, core_stub), (x,), fl)


def perf_mlp(T, B, sd):
    m = decider_module(sd, DEV)
    mprep = to_dev(fo.prepare_mlp_layer(sd, 0))
    x = (torch.randn(B, T, 2048) * 0.5).to(DT).to(DEV)
    fl = 2 * B * T * 3 * 2048 * 6144
    bench("mlp", "MLP block: DeciderNeuron.mlp", T, lambda x: m.mlp(0, x), (x,), fl)
    bench("mlp", "MLP block: mlp_fast (pre-transposed weights)", T, lambda x: fo.mlp_fast(x, mprep), (x,), fl)


def perf_gemm(T, B):
    shapes = {"GDN in-proj 2048->8192": (2048, 8192), "out-proj 2048->2048": (2048, 2048),
              "MLP gate_up 2048->12288": (2048, 12288), "MLP down 6144->2048": (6144, 2048)}
    R = 2
    for name, (K, N) in shapes.items():
        x = (torch.randn(B, T, K) * 0.5).to(DT).to(DEV)
        Wc = [(torch.randn(N, K) * 0.02).to(DT) for _ in range(R)]
        fl = 2 * B * T * K * N * R
        Ws = [w.to(DEV) for w in Wc]
        bench("gemm", f"{name}: x @ W.t()  (W [N,K], decider layout)", T, lambda x, *ws: tuple(x @ w.t() for w in ws),
              (x, *Ws), fl)
        if N == 8192:
            Wh = [w.reshape(64, 128, K) for w in Ws]
            bench("gemm", f"{name}: einsum btc,hdc->bhtd (decider)", T,
                  lambda x, *ws: tuple(torch.einsum("btc,hdc->bhtd", x, w) for w in ws), (x, *Wh), fl)
        del Ws
        Wt = [w.t().contiguous().to(DEV) for w in Wc]
        bench("gemm", f"{name}: x @ Wt     (Wt [K,N], pre-transposed)", T, lambda x, *ws: tuple(x @ w for w in ws),
              (x, *Wt), fl)
        if N == 8192:
            Wth = [w.reshape(K, 64, 128) for w in Wt]
            bench("gemm", f"{name}: einsum btc,chd->bhdt (pre-T, feature-major)", T,
                  lambda x, *ws: tuple(torch.einsum("btc,chd->bhdt", x, w) for w in ws), (x, *Wth), fl)
        del Wt, x


def perf_nkilib(T, B):
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.mlp.mlp import mlp as nk_mlp
    from nkilib.core.output_projection.output_projection_cte.output_projection_cte import output_projection_cte
    from nkilib.core.qkv.qkv import qkv as nk_qkv
    from nkilib.core.utils.common_types import ActFnType, NormType, OProjAttentionLayout, QKVOutputLayout

    x = (torch.randn(B, T, 2048) * 0.5).to(DT).to(DEV)
    lnb = (1 + 0.1 * torch.randn(1, 2048)).to(DT).to(DEV)
    kq = wrap_nki(nk_qkv)
    w = (torch.randn(2048, 4096) * 0.02).to(DT).to(DEV)          # qkv_cte max I = 4096 (GDN in-proj = 2 calls)
    bench("nkilib", "nkilib qkv_cte 2048->4096 + fused RMSNorm (BSD)", T, lambda x, w, g: kq[2](
        input=x, fused_qkv_weights=w, output_layout=QKVOutputLayout.BSD, fused_norm_type=NormType.RMS_NORM,
        gamma_norm_weights=g, norm_eps=EPS), (x, w, lnb), 2 * B * T * 2048 * 4096)
    bench("nkilib", "torch rms + x @ Wt 2048->4096 (reference for qkv_cte)", T,
          lambda x, w, g: rms_norm(x, g[0].float(), EPS) @ w, (x, w, lnb), 2 * B * T * 2048 * 4096)
    ko = wrap_nki(output_projection_cte)
    a = (torch.randn(B, 16, T, 128) * 0.5).to(DT).to(DEV)
    wo = (torch.randn(2048, 2048) * 0.02).to(DT).to(DEV)
    bench("nkilib", "nkilib output_projection_cte BNSd 2048->2048", T,
          lambda a, w: ko[2](attention=a, weight=w, attention_layout=OProjAttentionLayout.BNSd), (a, wo), 2 * B * T * 2048 ** 2)
    km = wrap_nki(nk_mlp)
    wg, wu = [(torch.randn(2048, 6144) * 0.02).to(DT).to(DEV) for _ in range(2)]
    wd = (torch.randn(6144, 2048) * 0.02).to(DT).to(DEV)
    bench("nkilib", "nkilib mlp (CTE) + fused RMSNorm", T, lambda x, g, a, b, c: km[2](
        hidden_tensor=x, gate_proj_weights_tensor=a, up_proj_weights_tensor=b, down_proj_weights_tensor=c,
        normalization_weights_tensor=g, normalization_type=NormType.RMS_NORM, activation_fn=ActFnType.SiLU, eps=EPS),
        (x, lnb, wg, wu, wd), 2 * B * T * 3 * 2048 * 6144)


# ----------------------------------------------------------------------------- tensor parallel / prefix / suffix
def core_stub_kw(q, k, v, g, beta, S0=None, return_state=False):
    o = core_stub(q, k, v, g, beta)
    if S0 is not None:
        o = (o.float() + 0.01 * S0[:, :, :1, :].float()).to(v.dtype)
    if return_state:
        return o, torch.einsum("bhtd,bhte->bhde", k.float(), v.float())
    return o


def check_tp(T_long=1024):
    full = random_layer_sd()
    gdn_keys = [k for k in full if k.rsplit(".", 1)[-1] in ("ln1", "w_qkvz", "w_ba", "conv_w", "A", "dt_bias", "gnorm", "w_out")]
    full = {k: full[k] for k in gdn_keys}
    for world in (1, 2, 4):
        cfg = DeciderConfig(n_layers=1, layer_types=["linear_attention"])
        c, sd = shard_state(cfg, full, world - 1, world)        # last rank of each world size
        H = c.gv
        m_ref = DeciderNeuron(c, {k: v.float() for k, v in sd.items()}, gdn_impl=core_stub_kw, compute_dtype=torch.float32)
        prep = to_dev(fo.prepare_gdn_layer(sd, 0))
        gab = prep["gdn_ab"].cpu()
        g = torch.Generator().manual_seed(world)
        for B, T, Bh in ((1, T_long, 1), (2, 128, 1), (8, 128, 8)):
            ph = (torch.randn(B, 3 * H, T, 128, generator=g) * 0.7).to(DT)
            zba = (torch.randn(B, T, 130 * H, generator=g) * 1.5).to(DT)
            hist = (torch.randn(Bh, 3 * H, 3, 128, generator=g) * 0.7).to(DT)
            ref = fo.gdn_prologue_ref(ph, sd["layers.0.conv_w"], zba[..., 128 * H:], gab[:, 0], gab[:, 1], conv_hist=hist)
            f = torch.compile(lambda p, z, hh: fo.gdn_prologue(p, prep["conv_diag"], z, prep["gdn_ab"], conv_hist=hh),
                              backend=BACKEND, fullgraph=True)
            with torch.inference_mode():
                out = f(ph.permute(0, 1, 3, 2).contiguous().to(DEV), zba.to(DEV), hist.to(DEV))
            err = max(rel(o, r) for o, r in zip(out, ref))
            record_err(f"TP{world} H={H} prologue+hist B={B} Bh={Bh} (max over q,k,v,g,beta)", T, err)
        # layers vs the DeciderNeuron methods on this shard
        x = (torch.randn(1, 512, 2048, generator=g) * 0.5).to(DT)
        xs = (torch.randn(2, 128, 2048, generator=g) * 0.5).to(DT)
        tail = (torch.randn(1, 3 * H, 3, 128, generator=g) * 0.7).to(DT)
        S0 = torch.randn(1, H, 128, 128, generator=g) * 0.1
        valid = torch.arange(512)[None, :] < 300
        s_len = torch.tensor([300])
        with torch.inference_mode():
            r_full = m_ref.gdn_layer(0, x.float()) - x.float()
            r_suf = m_ref.gdn_layer_suffix(0, xs.float(), S0, tail.float()) - xs.float()
            r_pre, r_S, r_tail = m_ref.gdn_layer_prefix(0, x.float(), valid, s_len)
            fl = torch.compile(lambda x: fo.gdn_layer_fast(x, prep, core_stub_kw), backend=BACKEND, fullgraph=True)
            fs = torch.compile(lambda x, t, s: fo.gdn_layer_suffix_fast(x, prep, core_stub_kw, s, t), backend=BACKEND, fullgraph=True)
            fp = torch.compile(lambda x, v, s: fo.gdn_layer_prefix_fast(x, prep, core_stub_kw, v, s), backend=BACKEND, fullgraph=True)
            y = fl(x.to(DEV)).cpu().float() - x.float()
            ys = fs(xs.to(DEV), tail.to(DEV), S0.to(DEV)).cpu().float() - xs.float()
            yp, Sp, tp = fp(x.to(DEV), valid.to(DEV), s_len.to(DEV))
        record_err(f"TP{world} H={H} gdn_layer_fast delta", 512, rel(y, r_full), 2e-2)
        record_err(f"TP{world} H={H} gdn_layer_suffix_fast delta (B=2)", 128, rel(ys, r_suf), 2e-2)
        record_err(f"TP{world} H={H} gdn_layer_prefix_fast delta", 512, rel(yp.cpu().float() - x.float(), r_pre - x.float()), 2e-2)
        record_err(f"TP{world} H={H} gdn_layer_prefix_fast state", 512, rel(Sp, r_S), 2e-2)
        record_err(f"TP{world} H={H} gdn_layer_prefix_fast conv tail", 512, rel(tp, r_tail), 2e-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", default="512,2048")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--only", default="correctness,tp,prologue,epilogue,layer,mlp,gemm,nkilib")
    a = ap.parse_args()
    secs = a.only.split(",")
    sd = random_layer_sd()
    if "tp" in secs:
        check_tp()
    for T in [int(t) for t in a.T.split(",")]:
        if "correctness" in secs:
            correctness(T, a.B)
            if T <= 1024 and a.B == 1:
                correctness(T, 2)
        if "prologue" in secs:
            perf_prologue(T, a.B, sd)
        if "epilogue" in secs:
            perf_epilogue(T, a.B, sd)
        if "layer" in secs:
            perf_layer(T, a.B, sd)
        if "mlp" in secs:
            perf_mlp(T, a.B, sd)
        if "gemm" in secs:
            perf_gemm(T, a.B)
        if "nkilib" in secs:
            perf_nkilib(T, a.B)
    print("\n==== summary ====")
    for sec, name, T, ms, tf, extra in ROWS:
        tfs = f"{tf:6.1f} TF/s ({tf * 1e12 / PEAK:5.1%})" if tf else ""
        mss = f"{ms:8.3f} ms" if ms is not None else " " * 11
        print(f"{sec:11s} T={T:5d} {mss} {tfs:22s} {name} {extra}")


if __name__ == "__main__":
    main()
