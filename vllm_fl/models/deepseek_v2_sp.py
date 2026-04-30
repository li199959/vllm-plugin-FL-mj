# Copyright (c) 2025 BAAI. All rights reserved.

"""Sequence-parallel residual patch for vLLM native DeepSeek V2/V3 models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sp_active() -> bool:
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_fl.ops.register_custom_ops import is_sp_enabled

    return bool(is_sp_enabled() and get_tp_group().world_size > 1)


def _sp_slice_for_rank(x: torch.Tensor) -> torch.Tensor:
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_fl.ops.register_custom_ops import get_sp_pad_size

    tp_group = get_tp_group()
    tp_size = tp_group.world_size
    if tp_size == 1:
        return x

    pad_size = get_sp_pad_size()
    if pad_size > 0:
        x = F.pad(x, (0, 0, 0, pad_size))

    assert x.shape[0] % tp_size == 0, (
        f"Sequence-parallel residual length {x.shape[0]} must be divisible "
        f"by tensor parallel size {tp_size}."
    )
    chunk = x.shape[0] // tp_size
    start = tp_group.rank_in_group * chunk
    return torch.narrow(x, 0, start, chunk).contiguous()


def _sp_gather_and_unpad(x: torch.Tensor) -> torch.Tensor:
    from vllm_fl.ops.register_custom_ops import maybe_all_gather_and_unpad

    return maybe_all_gather_and_unpad(x)


def patch_deepseek_v2_decoder_layer_sp() -> None:
    """Patch native vLLM DeepSeek decoder layers for manual Linear SP.

    Linear SP changes row-parallel projections from all_reduce to
    reduce_scatter over the token dimension. vLLM's native DeepSeek layer keeps
    the residual full-sized, so slice it before the post-attention RMSNorm and
    gather it again before returning to the next native layer.
    """

    import vllm.model_executor.models.deepseek_v2 as deepseek_v2

    layer_cls = deepseek_v2.DeepseekV2DecoderLayer
    if getattr(layer_cls, "_vllm_fl_sp_patched", False):
        return

    dense_mlp_cls = deepseek_v2.DeepseekV2MLP
    mha_attn_cls = getattr(deepseek_v2, "DeepseekAttention", ())

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ):
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )

        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not getattr(self, "use_mha", False):
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, mha_attn_cls)
            and hidden_states.dtype == torch.float16
        ):
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        sp_active = _sp_active()
        if sp_active:
            residual = _sp_slice_for_rank(residual)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )

        if isinstance(self.mlp, dense_mlp_cls):
            hidden_states = self.mlp(hidden_states)
            if hidden_states.dtype == torch.float16:
                hidden_states *= 1.0 / self.routed_scaling_factor
            if sp_active:
                hidden_states = _sp_gather_and_unpad(hidden_states)
                residual = _sp_gather_and_unpad(residual)
        else:
            if sp_active:
                hidden_states = _sp_gather_and_unpad(hidden_states)
                residual = _sp_gather_and_unpad(residual)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    layer_cls.forward = forward
    layer_cls._vllm_fl_sp_patched = True
