"""Q5: compile caching, cold/warm compile times, shape buckets, recompile limits."""
import os, sys, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
t_import = time.perf_counter()
import torch
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite import envs as lenvs
sys.path.insert(0, os.path.dirname(__file__))
from mock_hybrid import HybridStack
t_import = time.perf_counter() - t_import
mode = sys.argv[1]
BACKEND = register(); dev = torch.device("neuron", 0)
compile_ = lambda fn: torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
print(f"[{mode}] imports {t_import:.1f}s; NEFF cache dir = {lenvs.get_neff_cache_dir(BACKEND)}; "
      f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}", flush=True)
_ft = torch.zeros(1, device=dev); _ff = compile_(lambda t: t + 1)
def fence(): _ff(_ft).cpu()
fence()

if mode in ("cold", "warm", "neff_only"):
    torch.manual_seed(0)
    model = HybridStack(n_layers=4).to(dev).eval()
    t0 = time.perf_counter(); cm = compile_(model); t_wrap = time.perf_counter() - t0
    print(f"[{mode}] torch.compile(model) wrapper: {t_wrap:.2f}s", flush=True)
    for T in (512, 2048):                      # shape buckets: one graph/NEFF per T
        x = (torch.randn(T, 2048, dtype=torch.bfloat16)).to(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            y = cm(x)
        y.cpu(); t_first = time.perf_counter() - t0
        with torch.no_grad():
            for _ in range(3): cm(x)
            fence(); t0 = time.perf_counter()
            for _ in range(16): cm(x)
            fence(); t_run = (time.perf_counter() - t0) / 16
        print(f"[{mode}] 4-layer hybrid (3 GDN + 1 attn) T={T}: first call {t_first:6.1f}s  steady {t_run*1e3:.2f} ms/forward  finite={bool(torch.isfinite(y.cpu().float()).all())}", flush=True)

if mode == "recompile":
    import torch._dynamo
    print("recompile_limit =", torch._dynamo.config.recompile_limit, flush=True)
    w = torch.randn(256, 256, dtype=torch.bfloat16).to(dev)
    cf = compile_(lambda x, w: x @ w)
    for i, T in enumerate(range(128, 128 * 12, 128)):
        try:
            t0 = time.perf_counter(); cf(torch.randn(T, 256, dtype=torch.bfloat16).to(dev), w).cpu()
            print(f"  shape #{i+1} T={T}: ok ({time.perf_counter()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  shape #{i+1} T={T}: {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
            break
    # serving: freeze the set of compiled shapes
    torch.compiler.set_stance("fail_on_recompile")
    try:
        cf(torch.randn(100, 256, dtype=torch.bfloat16).to(dev), w).cpu()
        print("  unseen shape under fail_on_recompile: ran (unexpected)")
    except Exception as e:
        print(f"  unseen shape under fail_on_recompile: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    cf(torch.randn(128, 256, dtype=torch.bfloat16).to(dev), w).cpu(); print("  seen shape under fail_on_recompile: ok")
