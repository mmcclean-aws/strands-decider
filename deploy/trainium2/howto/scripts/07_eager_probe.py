"""Which tensor ops work EAGERLY (outside torch.compile) on the lite 'neuron' device?"""
import os, time
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch, libtorch_neuronx_lite
dev = torch.device("neuron", 0)
x = torch.randn(256, 256, dtype=torch.bfloat16)
xd = x.to(dev)
tests = {
    "torch.empty(device=neuron)": lambda: torch.empty(4, 4, device=dev),
    "torch.zeros(device=neuron)": lambda: torch.zeros(4, 4, device=dev).cpu(),
    "torch.randn(device=neuron)": lambda: torch.randn(4, 4, device=dev).cpu(),
    "x.to(neuron) / .cpu() roundtrip exact": lambda: bool(torch.equal(xd.cpu(), x)),
    "x.to(neuron, dtype=float32)": lambda: xd.to(torch.float32).cpu().dtype,
    "copy_ cpu->neuron in place": lambda: torch.empty(256, 256, dtype=torch.bfloat16, device=dev).copy_(x).cpu().equal(x),
    "eager add (xd + 1)": lambda: (xd + 1).cpu().float().sub(x.float() + 1).abs().max().item(),
    "eager matmul (xd @ xd)": lambda: (xd @ xd).cpu().shape,
    "eager slice view xd[:16]": lambda: xd[:16].cpu().equal(x[:16]),
    "eager .contiguous() of transpose": lambda: xd.t().contiguous().cpu().equal(x.t().contiguous()),
    "nn.Module.to(neuron)": lambda: next(torch.nn.Linear(4, 4).to(dev).parameters()).device,
    "torch.neuron.is_available()": lambda: torch.neuron.is_available(),
}
for name, fn in tests.items():
    try:
        t0 = time.perf_counter(); r = fn(); dt = time.perf_counter() - t0
        print(f"OK    {name:42s} -> {r}  ({dt*1e3:.1f} ms)", flush=True)
    except Exception as e:
        print(f"FAIL  {name:42s} -> {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
# H2D / D2H bandwidth
big = torch.randn(64 * 1024 * 1024 // 2, dtype=torch.bfloat16)   # 64 MiB
big.to(dev)
t0 = time.perf_counter(); bd = big.to(dev); bd.cpu()[:1]; 
t0 = time.perf_counter(); bd = big.to(dev); t_h2d = time.perf_counter() - t0
t0 = time.perf_counter(); bd.cpu(); t_d2h = time.perf_counter() - t0
print(f"H2D 64MiB {64/1024/t_h2d:.1f} GB/s ({t_h2d*1e3:.1f} ms), D2H 64MiB {64/1024/t_d2h:.1f} GB/s ({t_d2h*1e3:.1f} ms)")
