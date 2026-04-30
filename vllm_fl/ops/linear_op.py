"""Custom linear op dispatch for DSA-CP and Sequence Parallelism.

Mirrors vllm-ascend's linear_op.py. Each op overrides how a linear layer's
forward pass handles TP communication:

- ShardedCPColumnParallelOp: fake TP=1, each rank holds full weight
- ShardedCPRowParallelOp: fake TP=1, no reduce on output
- SequenceColumnParallelOp: all_gather input before matmul
- SequenceRowParallelOp: reduce_scatter output instead of all_reduce
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tp_group,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers.linear import split_tensor_along_last_dim


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class CustomLinearOp:
    def __init__(self, layer):
        self.layer = layer
        self.bias = None
        self.skip_bias_add = None
        self.return_bias = None
        self.quant_method = None
        self.prefix = None

    @property
    def comm_group(self):
        return get_tp_group()

    @property
    def tp_rank(self):
        return self.comm_group.rank_in_group

    @property
    def tp_size(self):
        return self.comm_group.world_size

    def update_attrs(self):
        self.bias = self.layer.bias
        self.skip_bias_add = self.layer.skip_bias_add
        self.return_bias = self.layer.return_bias
        self.quant_method = self.layer.quant_method
        self.prefix = self.layer.prefix

    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)
        if not self.return_bias:
            return output
        return output, output_bias

    def apply_impl(self, input_):
        raise NotImplementedError


class CustomColumnParallelOp(CustomLinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.gather_output = None

    def update_attrs(self):
        super().update_attrs()
        self.gather_output = self.layer.gather_output


class CustomRowParallelOp(CustomLinearOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.reduce_results = None
        self.input_is_parallel = None
        self.input_size_per_partition = None

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results
        self.input_size_per_partition = self.layer.input_size_per_partition


# ---------------------------------------------------------------------------
# ShardedCP ops — fake TP=1, each rank holds full weight
# ---------------------------------------------------------------------------

class ShardedCPColumnParallelOp(CustomColumnParallelOp):
    @property
    def comm_group(self):
        return SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)

    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self.layer, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class ShardedCPRowParallelOp(CustomRowParallelOp):
    @property
    def comm_group(self):
        return SimpleNamespace(world_size=1, rank_in_group=0, device_group=None)

    def apply_impl(self, input_):
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output = self.quant_method.apply(self.layer, input_, bias_)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def update_attrs(self):
        super().update_attrs()
        self.layer.reduce_results = False


# ---------------------------------------------------------------------------
# Sequence Parallelism ops — all_gather / reduce_scatter
# ---------------------------------------------------------------------------

class SequenceColumnParallelOp(CustomColumnParallelOp):
    def apply_impl(self, input_):
        from vllm_fl.ops.register_custom_ops import maybe_all_gather_and_unpad

        bias = self.bias if not self.skip_bias_add else None
        input_ = maybe_all_gather_and_unpad(input_)
        output = self.quant_method.apply(self.layer, input_, bias)
        if self.gather_output:
            output = self.comm_group.all_gather(output)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class SequenceRowParallelOp(CustomRowParallelOp):
    def apply_impl(self, input_):
        from vllm_fl.ops.register_custom_ops import is_sp_enabled, get_sp_pad_size

        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = splitted[self.tp_rank].contiguous()

        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        if self.tp_size == 1 or not self.reduce_results:
            output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        elif is_sp_enabled():
            x = input_parallel
            pad_size = get_sp_pad_size()
            if pad_size > 0:
                x = F.pad(x, (0, 0, 0, pad_size))
            output_parallel = self.quant_method.apply(self.layer, x, bias=bias_)
            output = tensor_model_parallel_reduce_scatter(output_parallel, 0)
        else:
            from vllm.distributed import tensor_model_parallel_all_reduce
            output_parallel = self.quant_method.apply(
                self.layer, input_parallel, bias=bias_
            )
            output = tensor_model_parallel_all_reduce(output_parallel)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

def _get_column_parallel_op(prefix, layer):
    from vllm_fl.utils import enable_dsa_cp, enable_mla_oot, enable_sp

    if enable_dsa_cp() and ("q_b_proj" in prefix or "kv_b_proj" in prefix):
        # FL SFA expects these projections in its fake-TP layout, while native
        # vLLM FLASHMLA(_SPARSE) expects the normal TP-local layout.
        return ShardedCPColumnParallelOp(layer) if enable_mla_oot() else None
    if enable_sp():
        if "shared_expert" in prefix:
            return None
        sp_prefixes = ["gate_up_proj", "qkv_proj"]
        for p in sp_prefixes:
            if p in prefix:
                return SequenceColumnParallelOp(layer)
    return None


def _get_row_parallel_op(prefix, layer):
    from vllm_fl.utils import enable_dsa_cp_with_layer_shard, enable_sp

    if enable_dsa_cp_with_layer_shard() and "o_proj" in prefix:
        return ShardedCPRowParallelOp(layer)
    if enable_sp():
        if "shared_expert" in prefix:
            return None
        sp_prefixes = ["o_proj", "down_proj"]
        for p in sp_prefixes:
            if p in prefix:
                return SequenceRowParallelOp(layer)
    return None


def get_parallel_op(disable_tp, prefix, layer, direct):
    """Get the custom linear op for a given layer prefix.

    Returns:
        (custom_op, tp_rank, tp_size) tuple.
    """
    if disable_tp:
        return None, 0, 1

    if direct == "row":
        custom_op = _get_row_parallel_op(prefix, layer)
    else:
        custom_op = _get_column_parallel_op(prefix, layer)

    if custom_op is not None:
        return custom_op, custom_op.tp_rank, custom_op.tp_size
    return None, get_tp_group().rank_in_group, get_tp_group().world_size
