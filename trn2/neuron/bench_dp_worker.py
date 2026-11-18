"""One DP replica: pipelined forwards of a fixed (T, B) for DURATION seconds; prints tokens/s."""
import os, sys, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_runner as dr
T, B, dur = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
_, _, m = dr.build(None, True, True)
fn = dr.compiled(m)
ids = torch.randint(0, 248000, (B, T), dtype=torch.int32)
args = [x.to(dr.DEV) for x in (ids, torch.full((B,), T - 1, dtype=torch.int32),
        (torch.arange(32, dtype=torch.int32)[None].repeat(B, 1) * 7 + 10).clamp_max(T - 2),
        torch.full((B,), 4, dtype=torch.int32), torch.ones(B))]
with torch.inference_mode():
    for _ in range(3):
        fn(*args).cpu()
    open(sys.argv[4], "w").write("ready\n")
    while not os.path.exists(sys.argv[5]):  # start barrier file
        time.sleep(0.01)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < dur:
        for _ in range(8):
            y = fn(*args)
        y.cpu(); n += 8
    dt = time.perf_counter() - t0
print(f"RESULT core={os.environ['NEURON_RT_VISIBLE_CORES']} forwards={n} {n * T * B / dt:.0f} tok/s {1e3 * dt / n:.2f} ms/forward", flush=True)
