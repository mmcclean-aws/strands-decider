"""Correctness + timing for neuron/kernels/gdn_fast.py vs the fp32 reference and nkilib gdn_cte.

Run on a trn2 host (env as in scripts/env.sh: LNC=2, VLLM_NEURON_BACKEND=neuron_native, ...):

    python test_gdn_fast.py --check            # correctness (out + final state, zero / nonzero S0)
    python test_gdn_fast.py --acc              # bf16 / fp16 / fp32 accuracy, gdn_fast vs gdn_cte, T=2048
    python test_gdn_fast.py --timing           # timing table vs nkilib gdn_cte, B in {1,4}, T in {512..4096}
    python test_gdn_fast.py --timing --fm      # same with feature-major q/k inputs (qt/kt)
    python test_gdn_fast.py --suffix           # shared-prefix suffix calls (B<=8, T in {128,256}, nonzero S0)

Correctness gate: rel L2 <= 1e-2 and max abs <= 3e-2 vs the fp32 reference (bf16-rounded inputs).

Profile notes (neuron-explorer, B=1 H=16 T=2048, LNC=2, per physical core):
  nkilib gdn_cte : device total_time 1071 us; PE/DVE/ACT active 660/712/578 us; 28 matmuls per chunk
                   (2 fp32, ~184 ns avg) + ~40 unbatched [128,128] DVE/ACT ops (283/280 ns each, 65/290 ns
                   of it fixed cost); lanes emitted one after another. Host wall 2.9 ms/call: ~1.8 ms is the
                   runtime rewriting its static (dge_mode=none) DMA descriptors every execution
                   (1.12 ms/call with NEURON_RT_IO_RING_CACHE_SIZE=32).
  gdn_fast       : device total_time ~497 us; PE/DVE/ACT active 336/394/312 us (all engines ~85-90% busy).
Timing: host loop of `iters` back-to-back calls with one read-back at the end (async execution), i.e.
per-call wall time of a compiled graph containing only the kernel. Set NEURON_RT_IO_RING_CACHE_SIZE=32
(the vllm-neuron worker default) to remove the per-execution static-DMA-descriptor rewrite that dominates
nkilib gdn_cte's standalone wall time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import libtorch_neuronx_lite  # noqa: E402,F401  (registers the "neuron" device)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # neuron/ (decider_model.py)

import gdn_fast as gf  # noqa: E402

try:  # the model's reference (supports S0 / return_state in recent versions)
    from decider_model import gdn_chunk_ref as _ref_model

    def gdn_ref(q, k, v, g, beta, S0=None):
        try:
            return _ref_model(q, k, v, g, beta, S0=S0, return_state=True)
        except TypeError:
            from dev.ref_state import gdn_ref_state
            return gdn_ref_state(q, k, v, g, beta, S0)
except Exception:  # pragma: no cover
    from dev.ref_state import gdn_ref_state as gdn_ref


def make_inputs(B, H, T, D=128, seed=0, gscale=0.5):
    """Realistic GDN inputs: q,k L2-normalised, v~N(0,0.5), g in [-gscale,0], beta in (0,1)."""
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, T, D, generator=gen), dim=-1)
    k = F.normalize(torch.randn(B, H, T, D, generator=gen), dim=-1)
    v = torch.randn(B, H, T, D, generator=gen) * 0.5
    g = -gscale * torch.rand(B, H, T, generator=gen)
    beta = torch.sigmoid(torch.randn(B, H, T, generator=gen) * 2)
    return q, k, v, g, beta


def err(o, r):
    o, r = o.float(), r.float()
    return ((o - r).norm() / r.norm()).item(), (o - r).abs().max().item()


_BACKEND = None
for _k in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, 256)


def compile_(fn):
    global _BACKEND
    if _BACKEND is None:
        import libtorch_neuronx_lite  # noqa: F401
        from libtorch_neuronx_lite.compile.native_backend import register
        _BACKEND = register()
    return torch.compile(fn, backend=_BACKEND, fullgraph=True, dynamic=False)


def first(y):
    while isinstance(y, (tuple, list)):
        y = y[0]
    return y


def bench(f, args, iters=20, warm=3):
    with torch.inference_mode():
        first(f(*args)).cpu()
        for _ in range(warm):
            y = f(*args)
        first(y).cpu()
        t0 = time.perf_counter()
        for _ in range(iters):
            y = f(*args)
        first(y).cpu()
    return (time.perf_counter() - t0) / iters


DEV = torch.device("neuron", 0)
H = 16


def fast_fn(cdt, fm=False, with_state=False, odt=None):
    consts = gf.gdn_fast_consts(DEV)

    def core(q, k, v, g, b, *rest):
        rest = list(rest)
        qt = kt = s0 = None
        if fm:
            qt, kt = rest[0], rest[1]
            rest = rest[2:]
        if with_state:
            s0 = rest[0]
        return gf.gdn_fast(q, k, v, g, b, consts, initial_state=s0, compute_dtype=cdt, return_state=True,
                           qt=qt, kt=kt, out_dtype=odt)
    return compile_(core)


def fast_args(q, k, v, g, beta, cdt, fm=False, S0=None):
    a = [x.to(cdt).to(DEV) for x in (q, k, v)] + [g.to(DEV), beta.to(DEV)]
    if fm:
        a += [q.transpose(-1, -2).contiguous().to(cdt).to(DEV), k.transpose(-1, -2).contiguous().to(cdt).to(DEV)]
    if S0 is not None:
        a.append(S0.to(DEV))
    return a


def base_fn():
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.gdn import build_mask_pack, gdn_cte
    mp = torch.as_tensor(build_mask_pack(), dtype=torch.float32).to(DEV)
    kern = wrap_nki(gdn_cte)

    def core(q, k, v, g, b):
        return kern(q, k, v, g, b, scale=128 ** -0.5, mask_pack=mp)
    return compile_(core)


def check(shapes, cdt=torch.bfloat16, fm=False, state=False, tol=(1e-2, 3e-2), gscale=0.5):
    ok = True
    for (B, T) in shapes:
        q, k, v, g, beta = make_inputs(B, H, T, seed=B * 7 + T, gscale=gscale)
        S0 = (torch.randn(B, H, 128, 128) * 0.05) if state else None
        f = fast_fn(cdt, fm=fm, with_state=state)
        with torch.inference_mode():
            o, st = f(*fast_args(q, k, v, g, beta, cdt, fm, S0))
            o, st = o.cpu().float(), st.cpu().float()
        qq, kk, vv = (x.to(cdt).float() for x in (q, k, v))
        ref, ref_st = gdn_ref(qq, kk, vv, g, beta, S0)
        rel, mx = err(o, ref)
        srel, smx = err(st, ref_st)
        good = rel <= tol[0] and mx <= tol[1] and srel <= tol[0] and not torch.isnan(o).any()
        ok &= good
        print(f"check B={B} T={T:4d} {str(cdt)[6:]:8s} fm={int(fm)} S0={int(state)} gmax={gscale}: out rel {rel:.2e} max {mx:.2e} | "
              f"S_final rel {srel:.2e} max {smx:.2e}  {'OK' if good else 'FAIL'}", flush=True)
    return ok


def accuracy(B=1, T=2048):
    q, k, v, g, beta = make_inputs(B, H, T, seed=1)
    rows = []
    for cdt, odt in ((torch.bfloat16, None), (torch.float16, None), (torch.float16, torch.float32),
                     (torch.float32, None)):
        f = fast_fn(cdt, odt=odt)
        with torch.inference_mode():
            o, _ = f(*fast_args(q, k, v, g, beta, cdt))
        qq, kk, vv = (x.to(cdt).float() for x in (q, k, v))
        ref, _ = gdn_ref(qq, kk, vv, g, beta)
        tag = str(cdt)[6:] + ("/out fp32" if odt is not None else "")
        rows.append(("gdn_fast", tag, *err(o.cpu(), ref)))
        if odt is None and cdt != torch.float32:
            # output rounding share: the same reference rounded to the output dtype
            rows.append(("  (ref rounded to out dtype)", str(cdt)[6:], *err(ref.to(cdt), ref)))
    bf = base_fn()
    for cdt in (torch.bfloat16, torch.float32):
        with torch.inference_mode():
            o, _ = bf(*[x.to(cdt).to(DEV) for x in (q, k, v)], g.to(DEV), beta.to(DEV))
        qq, kk, vv = (x.to(cdt).float() for x in (q, k, v))
        ref, _ = gdn_ref(qq, kk, vv, g, beta)
        rows.append(("nkilib gdn_cte", str(cdt)[6:], *err(o.cpu(), ref)))
    print(f"\naccuracy B={B} T={T} H={H} vs fp32 reference (inputs rounded to the compute dtype)")
    for r in rows:
        print(f"  {r[0]:30s} {r[1]:18s} rel {r[2]:.3e}  max {r[3]:.3e}")


def timing(Bs=(1, 4), Ts=(512, 1024, 2048, 4096), fm=False, cdts=(torch.bfloat16,)):
    print(f"\ntiming (ms/call, host loop) IO_RING_CACHE={os.environ.get('NEURON_RT_IO_RING_CACHE_SIZE', '-')} fm={int(fm)}")
    print(f"{'B':>3s} {'T':>5s} {'nkilib':>9s} " + " ".join(f"{'fast_' + str(c)[6:]:>14s}" for c in cdts) + "  speedup")
    bf = base_fn()
    for B in Bs:
        for T in Ts:
            q, k, v, g, beta = make_inputs(B, H, T)
            tb = bench(bf, [x.to(torch.bfloat16).to(DEV) for x in (q, k, v)] + [g.to(DEV), beta.to(DEV)])
            tf = []
            for cdt in cdts:
                tf.append(bench(fast_fn(cdt, fm=fm), fast_args(q, k, v, g, beta, cdt, fm)))
            print(f"{B:3d} {T:5d} {tb * 1e3:9.3f} " + " ".join(f"{t * 1e3:14.3f}" for t in tf)
                  + f"  {tb / tf[0]:6.2f}x", flush=True)


def timing_suffix(Bs=(1, 4, 8), Ts=(128, 256)):
    """Shared-prefix suffix calls: nonzero S0, final state returned (nkilib gdn_cte with recurrent_state)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.experimental.gdn import build_mask_pack, gdn_cte
    mp = torch.as_tensor(build_mask_pack(), dtype=torch.float32).to(DEV)
    kern = wrap_nki(gdn_cte)

    def base(q, k, v, g, b, s0):
        return kern(q, k, v, g, b, scale=128 ** -0.5, mask_pack=mp, recurrent_state=s0)
    bf = compile_(base)
    print(f"\nsuffix timing with S0 (ms/call) IO_RING_CACHE={os.environ.get('NEURON_RT_IO_RING_CACHE_SIZE', '-')}")
    print(f"{'B':>3s} {'T':>5s} {'nkilib':>9s} {'fast_bf16':>10s}")
    for B in Bs:
        for T in Ts:
            q, k, v, g, beta = make_inputs(B, H, T)
            S0 = torch.randn(B, H, 128, 128) * 0.05
            tb = bench(bf, [x.to(torch.bfloat16).to(DEV) for x in (q, k, v)] + [g.to(DEV), beta.to(DEV), S0.to(DEV)])
            tf = bench(fast_fn(torch.bfloat16, with_state=True), fast_args(q, k, v, g, beta, torch.bfloat16, S0=S0))
            print(f"{B:3d} {T:5d} {tb * 1e3:9.3f} {tf * 1e3:10.3f}  {tb / tf:5.2f}x", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--acc", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--fm", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--suffix", action="store_true")
    a = ap.parse_args()
    if a.check:
        sh = [(1, 256), (2, 512)] if a.quick else [(1, 128), (1, 512), (1, 1024), (1, 2048), (1, 4096), (2, 256),
                                                    (4, 512), (4, 2048), (8, 256)]
        ok = check(sh, fm=a.fm)
        ok &= check([(1, 128), (4, 256), (8, 128)] if not a.quick else [(2, 256)], state=True, fm=a.fm)
        ok &= check([(1, 2048)], cdt=torch.float16, fm=a.fm)
        ok &= check([(1, 512)], gscale=8.0, fm=a.fm)  # strong decay: within-chunk |G| up to ~1000
        print("ALL OK" if ok else "SOME FAILED")
    if a.acc:
        accuracy()
    if a.timing:
        timing(fm=a.fm, cdts=(torch.bfloat16, torch.float16))
    if a.suffix:
        timing_suffix()
