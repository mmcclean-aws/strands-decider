"""Q2: calling nkilib NKI kernels inside a torch.compile'd function (native backend).

Pattern:   K = wrap_nki(nki_jit_kernel);  out = K[lnc](**kwargs)   inside the compiled fn.
Shapes follow Qwen3.5-2B: full-attn 8 q heads / 2 kv heads, head_dim 256;
GDN 16 k heads / 16 v heads, Dk=Dv=128, conv_dim 6144, conv kernel 4.
"""
import os, time, traceback
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
import torch.nn.functional as F
import nki
import libtorch_neuronx_lite
from libtorch_neuronx_lite.compile.native_backend import register
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.experimental.gdn import gdn_cte, gdn_conv1d_prefill, build_mask_pack
from nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_ref
from nkilib.experimental.gdn.gdn_conv1d_torch import gdn_conv1d_prefill_torch_ref

BACKEND = register()
dev = torch.device("neuron", 0)
LNC = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
def compile_(fn):
    return torch.compile(fn, backend=BACKEND, fullgraph=True, dynamic=False)
_ft = torch.zeros(1, device=dev); _ff = compile_(lambda t: t + 1)
def fence(): _ff(_ft).cpu()
fence()
def bench(fn, *args, iters=32, chunk=8):
    for _ in range(3): fn(*args)
    fence(); t0 = time.perf_counter()
    for _ in range(iters // chunk):
        for _ in range(chunk): fn(*args)
        fence()
    return (time.perf_counter() - t0) / iters
def rel(a, b):
    a, b = a.float(), b.float(); return ((a - b).abs().max() / b.abs().max()).item()

# ---------------------------------------------------------------- attention_cte (GQA, head_dim 256)
attn_k = wrap_nki(nki.jit()(attention_cte))      # same as vllm_neuron NF.flash_attention
def attn_nki(q, k, v):
    # q [Hq,S,D] (tp_q=True), k [Hkv,S,D] (tp_k=True), v [Hkv,S,D]; native GQA (Hq % Hkv == 0)
    scale = q.shape[-1] ** -0.5
    return attn_k[LNC](q=q * scale, k=k, v=v, scale=1.0, causal_mask=True,
                       tp_q=True, tp_k=True, tp_out=False)
def attn_torch(q, k, v):                         # what NF.flash_attention falls back to for D>128
    rep = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(rep, 0); v = v.repeat_interleave(rep, 0)
    return F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True).squeeze(0)

for S in (512, 2048):
    try:
        torch.manual_seed(0)
        q = torch.randn(8, S, 256, dtype=torch.bfloat16); k = torch.randn(2, S, 256, dtype=torch.bfloat16)
        v = torch.randn(2, S, 256, dtype=torch.bfloat16)
        ref = attn_torch(q.float(), k.float(), v.float())
        qd, kd, vd = q.to(dev), k.to(dev), v.to(dev)
        c_nki, c_t = compile_(attn_nki), compile_(attn_torch)
        t0 = time.perf_counter(); o = c_nki(qd, kd, vd).cpu(); tc = time.perf_counter() - t0
        o_t = c_t(qd, kd, vd).cpu()
        print(f"attention_cte  Hq=8 Hkv=2 D=256 S={S}: nki maxrel={rel(o, ref):.2e} (first call {tc:.1f}s) "
              f"torch-on-device maxrel={rel(o_t, ref):.2e} | nki {bench(c_nki, qd, kd, vd)*1e6:.0f} us  "
              f"torch {bench(c_t, qd, kd, vd)*1e6:.0f} us", flush=True)
    except Exception:
        print(f"attention_cte S={S} FAILED"); traceback.print_exc()

# ---------------------------------------------------------------- vllm_neuron NF.flash_attention (D=128 -> kernel)
try:
    import vllm_neuron.functional as NF
    S = 1024
    torch.manual_seed(0)
    q = torch.randn(8, S, 128, dtype=torch.bfloat16); k = torch.randn(2, 128, S, dtype=torch.bfloat16)
    v = torch.randn(2, S, 128, dtype=torch.bfloat16)
    ref = attn_torch(q.float(), k.float().transpose(1, 2), v.float())
    c = compile_(lambda q, k, v: NF.flash_attention(q, k, v))   # tp_q=True, tp_k=False default layouts
    o = c(q.to(dev), k.to(dev), v.to(dev)).cpu()
    print(f"NF.flash_attention D=128 S={S} GQA 8/2: maxrel={rel(o, ref):.2e} "
          f"{bench(c, q.to(dev), k.to(dev), v.to(dev))*1e6:.0f} us", flush=True)
except Exception:
    print("NF.flash_attention FAILED"); traceback.print_exc()

# ---------------------------------------------------------------- gdn_conv1d_prefill
conv_k = wrap_nki(gdn_conv1d_prefill)            # already @nki.jit in nkilib
C, KW = 6144, 4
for grid in (None, LNC):   # None = no brackets -> grid taken from NEURON_LOGICAL_NC_CONFIG. grid [1] FAILS (NCC_EVRF066)
    for T in (512, 2048):
        try:
            torch.manual_seed(0)
            mixed = torch.randn(1, C, T, dtype=torch.bfloat16); w = torch.randn(C, KW, dtype=torch.bfloat16) * 0.5
            ref_o, ref_s = gdn_conv1d_prefill_torch_ref(mixed.float(), w.float())
            kern = conv_k if grid is None else conv_k[grid]
            fn = compile_(lambda m, w: kern(mixed=m, conv_weight=w))
            o, s = fn(mixed.to(dev), w.to(dev))
            print(f"gdn_conv1d_prefill grid=[{grid}] C={C} T={T}: out maxrel={rel(o.cpu(), ref_o):.2e} "
                  f"state maxrel={rel(s.cpu(), ref_s):.2e} | {bench(fn, mixed.to(dev), w.to(dev))*1e6:.0f} us", flush=True)
        except Exception:
            print(f"gdn_conv1d_prefill grid={grid} T={T} FAILED"); traceback.print_exc()

# ---------------------------------------------------------------- gdn_cte (chunked gated delta rule)
cte_k = wrap_nki(gdn_cte)
mask_pack_cpu = torch.from_numpy(build_mask_pack())
mask_pack_dev = mask_pack_cpu.to(dev)    # host-built constant, upload once
def gdn_nki(q, k, v, g, b, mp):
    return cte_k[LNC](query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp)
def gdn_nki_state(q, k, v, g, b, mp, st):
    # recurrent_state supplied -> aliased to output 1 and updated IN PLACE
    return cte_k[LNC](query=q, key=k, value=v, g_log=g, beta=b, scale=128 ** -0.5, mask_pack=mp,
                      recurrent_state=st)
for S in (512, 2048):
    try:
        torch.manual_seed(0)
        B, Hk, Hv, D = 1, 16, 16, 128
        q = F.normalize(torch.randn(B, Hk, S, D), dim=-1).bfloat16()
        k = F.normalize(torch.randn(B, Hk, S, D), dim=-1).bfloat16()
        v = torch.randn(B, Hv, S, D).bfloat16()
        g = -F.softplus(torch.randn(B, Hv, S)) * 0.1          # negative log decay, fp32
        b = torch.sigmoid(torch.randn(B, Hv, S))               # (0,1), fp32
        ref_o, ref_s = gdn_cte_torch_ref(q.float(), k.float(), v.float(), g, b, scale=128 ** -0.5, mask_pack=mask_pack_cpu)
        args = [t.to(dev) for t in (q, k, v, g, b)] + [mask_pack_dev]
        fn = compile_(gdn_nki)
        t0 = time.perf_counter(); o, st = fn(*args); o = o.cpu(); st = st.cpu(); tc = time.perf_counter() - t0
        msg = (f"gdn_cte grid=[{LNC}] B=1 H=16 D=128 S={S}: out maxrel={rel(o, ref_o):.2e} "
               f"state maxrel={rel(st, ref_s):.2e} (first call {tc:.1f}s) | {bench(fn, *args)*1e6:.0f} us")
        # continuation with an initial state (in-place aliasing): run on the 2nd half given state of 1st half
        h = S // 2
        r1_o, r1_s = gdn_cte_torch_ref(q[:, :, :h].float(), k[:, :, :h].float(), v[:, :, :h].float(), g[..., :h], b[..., :h], scale=128 ** -0.5, mask_pack=mask_pack_cpu)
        r2_o, r2_s = gdn_cte_torch_ref(q[:, :, h:].float(), k[:, :, h:].float(), v[:, :, h:].float(), g[..., h:], b[..., h:], scale=128 ** -0.5, mask_pack=mask_pack_cpu, recurrent_state=r1_s.clone())
        st_dev = r1_s.clone().float().to(dev)
        fn2 = compile_(gdn_nki_state)
        a2 = [t.to(dev) for t in (q[:, :, h:].contiguous(), k[:, :, h:].contiguous(), v[:, :, h:].contiguous(),
                                  g[..., h:].contiguous(), b[..., h:].contiguous())] + [mask_pack_dev, st_dev]
        o2, s2 = fn2(*a2)
        msg += (f" || with recurrent_state: out maxrel={rel(o2.cpu(), r2_o):.2e} state maxrel={rel(s2.cpu(), r2_s):.2e} "
                f"input buffer updated in place: {rel(st_dev.cpu(), r2_s) < 2e-2}")
        print(msg, flush=True)
    except Exception:
        print(f"gdn_cte S={S} FAILED"); traceback.print_exc()
