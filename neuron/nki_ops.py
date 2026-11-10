"""NKI kernel bindings (nkilib shipped in the SDK 2.33 image) for the decider's token mixers.

Both are called from inside a torch.compile(backend="neuron_native_lite") graph through
libtorch_neuronx_lite's `wrap_nki`, launched on the LNC=2 SPMD grid `[2]`.

  gdn_nki(q, k, v, g, beta)   -> nkilib.experimental.gdn.gdn_cte  (schur, chunk 128)
  attn_nki(q, k, v, scale)    -> nkilib.core.attention.attention_cte (causal, GQA native, d<=512)
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def _wrapped():
    import nki
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.attention.attention_cte import attention_cte
    from nkilib.experimental.gdn import gdn_cte

    return {
        "attn": wrap_nki(nki.jit()(attention_cte)),
        "gdn": wrap_nki(gdn_cte),  # already @nki.jit in nkilib
    }


@functools.lru_cache(maxsize=None)
def mask_pack_cpu() -> torch.Tensor:
    from nkilib.experimental.gdn import build_mask_pack

    mp = build_mask_pack()
    return torch.as_tensor(mp, dtype=torch.float32)


class NkiMixers:
    """Holds device-resident constants (the GDN schur mask pack) and the kernel callables."""

    def __init__(self, device, lnc: int = 2, gdn_fp32: bool | None = None):
        import os

        self.lnc = lnc
        self.gdn_fp32 = bool(int(os.environ.get("DECIDER_GDN_FP32", "0"))) if gdn_fp32 is None else gdn_fp32
        self.mask_pack = mask_pack_cpu().to(device)

    def gdn(self, q, k, v, g, beta, S0=None, return_state=False):
        """q,k [B,H,T,Dk] (L2-normalised, compute dtype), v [B,H,T,Dv], g/beta [B,H,T] fp32.
        S0 [B,H,Dk,Dv] fp32 optional initial state; return_state -> (out, final_state)."""
        dk = q.shape[-1]
        if self.gdn_fp32:  # the kernel's compute dtype follows q
            q, k, v = q.float(), k.float(), v.float()
        kw = {} if S0 is None else {"recurrent_state": S0.float().contiguous()}
        out, state = _wrapped()["gdn"][self.lnc](
            q.contiguous(), k.contiguous(), v.contiguous(), g.float().contiguous(), beta.float().contiguous(),
            scale=dk ** -0.5, mask_pack=self.mask_pack, **kw)
        return (out, state) if return_state else out

    def attn(self, q, k, v, scale, k_prior=None, v_prior=None, prior_len=None):
        """q [B,Hq,T,D], k/v [B,Hkv,T,D] -> [B,Hq,T,D]; causal; GQA handled by the kernel.
        Optional cached prefix k_prior/v_prior [B,Hkv,S,D] with prior_len [1] valid positions."""
        B, Hq, T, D = q.shape
        Hkv = k.shape[1]
        qq = (q * scale).reshape(B * Hq, T, D).contiguous()          # tp_q=True: [B, S, D]
        kk = k.reshape(B * Hkv, T, D).transpose(1, 2).contiguous()   # tp_k=False: [B, D, S] (fastest)
        vv = v.reshape(B * Hkv, T, D).contiguous()
        kw = {}
        if k_prior is not None:
            S = k_prior.shape[2]
            kw = dict(k_prior=k_prior.reshape(B * Hkv, S, D).transpose(1, 2).contiguous(),
                      v_prior=v_prior.reshape(B * Hkv, S, D).contiguous(),
                      prior_used_len=prior_len.reshape(1).to(torch.float32))
        o = _wrapped()["attn"][self.lnc](
            q=qq, k=kk, v=vv, scale=1.0, causal_mask=True, tp_q=True, tp_k=False, tp_out=False,
            cache_softmax=False, softmax_dtype=torch.float32, **kw)
        return o.reshape(B, Hq, T, D)


class FastGdn:
    """kernels/gdn_fast.py (WY-form chunked GDN, ~2.4x nkilib gdn_cte) with the decider's mixer API.
    compute dtype from DECIDER_GDN_DTYPE in {bf16, fp16, fp32} (default bf16)."""

    def __init__(self, device, compute_dtype=None):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels"))
        import gdn_fast as gf

        self._gf = gf
        self.consts = gf.gdn_fast_consts(device)
        name = os.environ.get("DECIDER_GDN_DTYPE", "bf16")
        self.cdt = compute_dtype or {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]

    def __call__(self, q, k, v, g, beta, S0=None, return_state=False):
        return self._gf.gdn_fast(q, k, v, g, beta, self.consts, initial_state=S0, return_state=return_state,
                                 compute_dtype=self.cdt, out_dtype=torch.bfloat16)
