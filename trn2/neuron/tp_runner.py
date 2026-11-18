"""Tensor-parallel decider on P logical NeuronCores (torchrun, one process per core).

  CORE_BASE=12 torchrun --nproc_per_node=2 tp_runner.py bench --shapes 4096x1,2048x1
  CORE_BASE=12 torchrun --nproc_per_node=2 tp_runner.py goldens --gdir ~/decider/goldens/eval --limit 64
"""

import os
import sys

local_rank = int(os.environ["LOCAL_RANK"])
core_base = int(os.environ.get("CORE_BASE", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + local_rank)
os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{62300 + core_base}")
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import libtorch_neuronx_lite  # noqa: E402,F401
from libtorch_neuronx_lite.compile.native_backend import register  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from decider_model import DeciderNeuron, _snapshot, load_state, shard_state  # noqa: E402

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
BACKEND = register()
DEV = torch.device("neuron", 0)


def log(*a):
    if rank == 0:
        print(*a, flush=True)


def build(gdn_impl_name: str):
    cfg, sd = load_state(_snapshot("Qwen/Qwen3.5-2B-Base"), _snapshot("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"))
    lcfg, lsd = shard_state(cfg, sd, rank, world)
    from nki_ops import NkiMixers

    mix = NkiMixers(DEV)
    gdn = mix.gdn
    if gdn_impl_name == "fast":
        from nki_ops import FastGdn

        gdn = FastGdn(DEV)
    m = DeciderNeuron(lcfg, lsd, gdn_impl=gdn, attn_impl=mix.attn, use_fast=bool(int(os.environ.get("DECIDER_FAST", "1")))).eval()
    m.tp_group = dist.group.WORLD
    m._noreduce = bool(int(os.environ.get("DECIDER_TP_NOREDUCE", "0")))  # timing-only experiment
    m.tp_chunks = int(os.environ.get("DECIDER_TP_CHUNKS", "1"))
    return cfg, m.to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "goldens"])
    ap.add_argument("--shapes", default="2048x1,4096x1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--gdir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gdn", default="fast")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg, m = build(a.gdn)
    fn = torch.compile(m.forward, backend=BACKEND, fullgraph=True, dynamic=False)

    def run(ids, ans, opt, ns, it):
        with torch.inference_mode():
            return fn(ids.to(DEV), ans.to(DEV), opt.to(DEV), ns.to(DEV), it.to(DEV)).cpu()

    if a.mode == "bench":
        from cost_model import Model

        res = {}
        for spec in a.shapes.split(","):
            T, B = (int(x) for x in spec.split("x"))
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(0, 248000, (B, T), dtype=torch.int32, generator=g)
            ans = torch.full((B,), T - 1, dtype=torch.int32)
            opt = (torch.arange(32, dtype=torch.int32)[None].repeat(B, 1) * 7 + 10).clamp_max(T - 2)
            ns = torch.full((B,), 4, dtype=torch.int32)
            it = torch.ones(B)
            t0 = time.time()
            run(ids, ans, opt, ns, it)
            tc = time.time() - t0
            for _ in range(3):
                run(ids, ans, opt, ns, it)
            dist.all_reduce(torch.zeros(1))
            lat = []
            for _ in range(a.iters):
                t1 = time.perf_counter()
                run(ids, ans, opt, ns, it)
                lat.append(time.perf_counter() - t1)
            lat.sort()
            p50 = lat[len(lat) // 2]
            fl = Model().flops(T)["total"] * B
            mfu = fl / (p50 * world * 667e12 / 4)
            res[spec] = {"P": world, "compile_s": tc, "p50_ms": 1e3 * p50, "p99_ms": 1e3 * lat[-1], "mfu": mfu}
            log(f"TP{world} T={T} B={B} compile {tc:.0f}s p50 {1e3 * p50:.2f} ms MFU(over {world} cores) {mfu:.1%}")
        if rank == 0 and a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    else:
        from accuracy import evaluate

        r = evaluate(run, a.gdir, [512, 1024, 2048, 4096], {}, limit=a.limit, log=log)
        if rank == 0:
            print(json.dumps({k: v for k, v in r.items() if k != "per_task"}, indent=1))
            if a.out:
                json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
