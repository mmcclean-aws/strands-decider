"""Which vllm_neuron NF ops take the NKI path at Qwen3.5-2B shapes? (meta tensors; no device needed)"""
import os
os.environ.setdefault("VLLM_NEURON_BACKEND", "neuron_native")
import torch
from nkilib.core.utils.common_types import NormType
from vllm_neuron.functional.mlp import _can_use_kernel as mlp_ok
from vllm_neuron.functional.attention.qkv import _can_use_qkv_kernel as qkv_ok
from vllm_neuron.functional.attention.attention_cte import _can_use_flash_attention_kernel as fa_ok
from nkilib.core.utils.common_types import QKVOutputLayout
m = lambda *s: torch.empty(*s, dtype=torch.bfloat16, device="meta")
H, I = 2048, 6144
for tp in (1, 2, 4, 8):
    for T in (512, 2048):
        r = dict(
            mlp=mlp_ok(m(T, H), m(H, I // tp)),
            attn_qkv=qkv_ok(m(1, T, H), m(H, (8 * 256 * 2 + 2 * 2 * 256) // tp), None, None, 256, QKVOutputLayout.BSD, NormType.NO_NORM, None),
            gdn_in_proj=qkv_ok(m(1, T, H), m(H, 8192 // tp), None, None, 128, QKVOutputLayout.BSD, NormType.NO_NORM, None),
            flash_attn_d256=fa_ok(m(max(1, 8 // tp), T, 256)),
            flash_attn_d128=fa_ok(m(8, T, 128)),
        )
        print(f"TP={tp} T={T}: " + "  ".join(f"{k}={'NKI' if v else 'torch'}" for k, v in r.items()), flush=True)
