"""Mock Qwen3.5-2B-style hybrid layers (random weights) using nkilib kernels, for compile/caching/perf tests.
NOT numerically validated against HF -- shapes/ops only (kernels are validated individually in 02_nki_kernels.py)."""
import torch, torch.nn as nn, torch.nn.functional as F, nki
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from nkilib.core.attention.attention_cte import attention_cte
from nkilib.experimental.gdn import gdn_cte, gdn_conv1d_prefill, build_mask_pack

H, I = 2048, 6144
NK, NV, DK, DV = 16, 16, 128, 128          # GDN heads
NQ, NKV, DH = 8, 2, 256                     # full attention heads
conv_k = wrap_nki(gdn_conv1d_prefill); cte_k = wrap_nki(gdn_cte); attn_k = wrap_nki(nki.jit()(attention_cte))

def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)

def P(*shape, s=0.02):
    return nn.Parameter(torch.randn(*shape, dtype=torch.bfloat16) * s, requires_grad=False)

class MLP(nn.Module):
    def __init__(s):
        super().__init__(); s.norm = nn.Parameter(torch.ones(H), requires_grad=False)
        s.wg, s.wu, s.wd = P(H, I), P(H, I), P(I, H)
    def forward(s, x):
        h = rms(x, s.norm)
        return x + (F.silu(h @ s.wg) * (h @ s.wu)) @ s.wd

class GDNLayer(nn.Module):
    def __init__(s, mask_pack):
        super().__init__()
        s.norm = nn.Parameter(torch.ones(H), requires_grad=False)
        s.w_qkvz = P(H, 2 * NK * DK + 2 * NV * DV); s.w_ba = P(H, 2 * NV)
        s.conv_w = P(2 * NK * DK + NV * DV, 4, s=0.3)
        s.A_log = nn.Parameter(torch.zeros(NV), requires_grad=False); s.dt_bias = nn.Parameter(torch.zeros(NV), requires_grad=False)
        s.onorm = nn.Parameter(torch.ones(DV), requires_grad=False); s.w_out = P(NV * DV, H)
        s.register_buffer("mask_pack", mask_pack, persistent=False); s.mlp = MLP()
    def forward(s, x):
        T = x.shape[0]
        h = rms(x, s.norm)
        qkvz = h @ s.w_qkvz
        qkv, z = qkvz[:, :2 * NK * DK + NV * DV], qkvz[:, 2 * NK * DK + NV * DV:]
        ba = h @ s.w_ba; b, a = ba[:, :NV], ba[:, NV:]
        conv, _state = conv_k(mixed=qkv.t().unsqueeze(0).contiguous(), conv_weight=s.conv_w)   # [1, C, T]
        conv = conv[0].t()                                                                     # [T, C]
        q, k, v = conv[:, :NK * DK], conv[:, NK * DK:2 * NK * DK], conv[:, 2 * NK * DK:]
        q = F.normalize(q.reshape(T, NK, DK).float(), dim=-1).to(x.dtype).transpose(0, 1).unsqueeze(0).contiguous()
        k = F.normalize(k.reshape(T, NK, DK).float(), dim=-1).to(x.dtype).transpose(0, 1).unsqueeze(0).contiguous()
        v = v.reshape(T, NV, DV).transpose(0, 1).unsqueeze(0).contiguous()
        g = (-torch.exp(s.A_log.float()) * F.softplus(a.float() + s.dt_bias.float())).t().unsqueeze(0).contiguous()
        beta = torch.sigmoid(b.float()).t().unsqueeze(0).contiguous()
        o, _st = cte_k(query=q, key=k, value=v, g_log=g, beta=beta, scale=DK ** -0.5, mask_pack=s.mask_pack)
        o = o[0].transpose(0, 1)                                                               # [T, NV, DV]
        o = rms(o, s.onorm) * F.silu(z.reshape(T, NV, DV))
        x = x + o.reshape(T, NV * DV) @ s.w_out
        return s.mlp(x)

class AttnLayer(nn.Module):
    def __init__(s):
        super().__init__()
        s.norm = nn.Parameter(torch.ones(H), requires_grad=False)
        s.w_qkv = P(H, 2 * NQ * DH + 2 * NKV * DH)     # q + output gate, k, v
        s.qn = nn.Parameter(torch.ones(DH), requires_grad=False); s.kn = nn.Parameter(torch.ones(DH), requires_grad=False)
        s.w_o = P(NQ * DH, H); s.mlp = MLP()
    def forward(s, x):
        T = x.shape[0]
        h = rms(x, s.norm)
        qkv = h @ s.w_qkv
        qg = qkv[:, :2 * NQ * DH].reshape(T, NQ, 2 * DH)
        q, gate = qg[..., :DH], qg[..., DH:]
        k = qkv[:, 2 * NQ * DH:2 * NQ * DH + NKV * DH].reshape(T, NKV, DH)
        v = qkv[:, 2 * NQ * DH + NKV * DH:].reshape(T, NKV, DH)
        q = (rms(q, s.qn) * DH ** -0.5).transpose(0, 1).contiguous()     # [NQ, T, DH]   (RoPE omitted in mock)
        k = rms(k, s.kn).transpose(0, 1).contiguous()                     # [NKV, T, DH]
        v = v.transpose(0, 1).contiguous()
        o = attn_k(q=q, k=k, v=v, scale=1.0, causal_mask=True, tp_q=True, tp_k=True, tp_out=False)   # [NQ, T, DH]
        o = o.transpose(0, 1) * torch.sigmoid(gate)
        x = x + o.reshape(T, NQ * DH) @ s.w_o
        return s.mlp(x)

class HybridStack(nn.Module):
    """layer pattern of Qwen3.5: [lin, lin, lin, full] * (n_layers // 4)"""
    def __init__(s, n_layers=4):
        super().__init__()
        mp = torch.from_numpy(build_mask_pack())
        s.layers = nn.ModuleList([AttnLayer() if (i % 4 == 3) else GDNLayer(mp) for i in range(n_layers)])
    def forward(s, x):
        for l in s.layers:
            x = l(x)
        return x
