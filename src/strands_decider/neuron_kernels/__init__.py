"""NKI kernels for the decider's token mixers on AWS Trainium (`--device neuron`).

`NeuronDeciderModel`'s two pluggable token mixers:

  NkiMixers.gdn   the Gated DeltaNet chunk rule: nkilib.experimental.gdn.gdn_cte (schur form)
  NkiMixers.attn  causal grouped-query attention: nkilib.core.attention.attention_cte, which
                  takes head_dim 256 and a cached prefix's K and V directly
  FastGdn         the Gated DeltaNet chunk rule in WY form (gdn_fast.py), 2.4x gdn_cte

nkilib ships in the Neuron Deep Learning Container image. With `fast`, the model also runs the
rest of each layer through fused_ops.py: the Gated DeltaNet prologue (convolution, SiLU, L2
norm, beta and g in one kernel) and epilogue (the gated norm), NKI RMSNorms, and GEMM weights
pre-transposed once at load (layouts.py).

Both are called from inside a `torch.compile(backend="neuron_native_lite")` graph through
libtorch_neuronx_lite's `wrap_nki`, on the LNC=2 grid (two physical NeuronCores per logical
core). Nothing here is imported off Trainium: neuron_engine.py imports this module only after
`neuron_available()`.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

# NKI's launch grid must equal the logical-core configuration (LNC=2 on trn2).
LNC = 2


@functools.cache
def _kernels() -> dict[str, Any]:
    import nki
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.attention.attention_cte import attention_cte
    from nkilib.experimental.gdn import gdn_cte

    return {"attn": wrap_nki(nki.jit()(attention_cte)), "gdn": wrap_nki(gdn_cte)}


def _mask_pack() -> torch.Tensor:
    from nkilib.experimental.gdn import build_mask_pack

    return torch.as_tensor(build_mask_pack(), dtype=torch.float32)


class NkiMixers:
    """The kernels, and the device constant gdn_cte needs (its Schur mask pack), made once
    outside any compiled graph."""

    def __init__(self, device: torch.device) -> None:
        self.mask_pack = _mask_pack().to(device)

    def gdn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S0: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """gdn_chunk_ref's contract: q, k [B, H, T, D] L2-normalised, v [B, H, T, D], g and beta
        [B, H, T]; T a multiple of 128. S0 [B, H, D, D] is an optional initial state."""
        kw = {} if S0 is None else {"recurrent_state": S0.float().contiguous()}
        out, state = _kernels()["gdn"][LNC](
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.float().contiguous(),
            beta.float().contiguous(),
            scale=q.shape[-1] ** -0.5,
            mask_pack=self.mask_pack,
            **kw,
        )
        return (out, state) if return_state else out

    def attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        k_prior: torch.Tensor | None = None,
        v_prior: torch.Tensor | None = None,
        prior_len: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """attn_ref's contract: q [B, Hq, T, D], k and v [B, Hkv, T, D], causal; an optional
        cached prefix k_prior and v_prior [B, Hkv, S, D] with prior_len [1] real positions."""
        B, Hq, T, D = q.shape
        Hkv = k.shape[1]
        kw: dict[str, Any] = {}
        if k_prior is not None:
            assert v_prior is not None and prior_len is not None
            S = k_prior.shape[2]
            kw = {
                "k_prior": k_prior.reshape(B * Hkv, S, D).transpose(1, 2).contiguous(),
                "v_prior": v_prior.reshape(B * Hkv, S, D).contiguous(),
                "prior_used_len": prior_len.reshape(1).to(torch.float32),
            }
        o: torch.Tensor = _kernels()["attn"][LNC](
            q=(q * scale).reshape(B * Hq, T, D).contiguous(),  # tp_q=True: [B, S, D]
            k=k.reshape(B * Hkv, T, D).transpose(1, 2).contiguous(),  # tp_k=False: [B, D, S]
            v=v.reshape(B * Hkv, T, D).contiguous(),
            scale=1.0,
            causal_mask=True,
            tp_q=True,
            tp_k=False,
            tp_out=False,
            cache_softmax=False,
            softmax_dtype=torch.float32,
            **kw,
        )
        return o.reshape(B, Hq, T, D)


class FastGdn:
    """gdn_fast.py's kernel with `NkiMixers.gdn`'s contract. `compute_dtype` is its matmul
    operand dtype: bf16, or fp16 at the same speed with about 8x less rounding error."""

    def __init__(self, device: torch.device, compute_dtype: torch.dtype = torch.bfloat16) -> None:
        from .gdn_fast import gdn_fast_consts

        self.consts = gdn_fast_consts(device)
        self.compute_dtype = compute_dtype

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        S0: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from .gdn_fast import gdn_fast

        out: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = gdn_fast(
            q,
            k,
            v,
            g,
            beta,
            self.consts,
            initial_state=S0,
            return_state=return_state,
            compute_dtype=self.compute_dtype,
            out_dtype=torch.bfloat16,
        )
        return out
