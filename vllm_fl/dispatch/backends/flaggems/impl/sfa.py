# Copyright (c) 2025 BAAI. All rights reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

import torch
from tqdm import tqdm

from vllm import _custom_ops as ops
import vllm.envs as envs_vllm
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
try:
    # vLLM >= 0.13 moved MLA attention backends under vllm.v1.
    from vllm.v1.attention.backends.mla.common import MLACommonMetadataBuilder
except ModuleNotFoundError:  # pragma: no cover - compatibility with older vLLM
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.distributed.parallel_state import is_global_first_rank

from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec
try:
    from vllm.v1.attention.backends.mla.common import (  # type: ignore
        dynamic_per_batched_tensor_quant,
    )
except Exception:  # pragma: no cover
    from vllm.v1.attention.backends.mla import dynamic_per_batched_tensor_quant  # type: ignore
from vllm._aiter_ops import rocm_aiter_ops

from vllm_fl import envs
from vllm_fl.vllmfl_config import get_vllm_fl_config
from vllm_fl.dispatch.backends.flaggems.impl.attention import AttentionFLState
from vllm_fl.dispatch.backends.flaggems.impl.utils import (
    FLCommonAttentionMetadata,
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_fl.distributed.utils import all_gather_async
from vllm_fl.ops.layer_shard_linear import (
    is_hidden_layer,
    post_process_after_loading_for_shard_weight_series,
    reach_layer_for_shard_weight_series,
    register_all_layers_to_shard_weight_series,
)
from vllm_fl.utils import (
    _round_up,
    dispose_layer,
    enable_cp,
    enable_dsa_cp,
    enable_dsa_cp_with_layer_shard,
    enable_dsa_cp_with_o_proj_tp,
)
from vllm_fl.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_fl.ops.triton.rope import rope_forward_triton_siso

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024


class FLSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "FL_SFA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        if enable_cp():
            from vllm_fl.dispatch.backends.flaggems.impl.sfa_cp import FLSFACPMetadataBuilder
            return FLSFACPMetadataBuilder
        return FLSFAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["FLSFAImpl"]:
        if enable_cp():
            from vllm_fl.dispatch.backends.flaggems.impl.sfa_cp import FLSFACPImpl
            return FLSFACPImpl
        return FLSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # return [128]
        return [32, 64, 128]


@dataclass
class DSACPContext:
    num_tokens: int
    num_tokens_pad: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    pad_size: int
    local_pad_size: int
    slot_mapping_cp: torch.Tensor
    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor


@dataclass
class FLSFAMetadata:
    """Metadata for SFA.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    # The dimension of the attention heads
    head_dim: int | None = None
    # chunked prefill by default if no attn_states passed
    attn_state: AttentionFLState = AttentionFLState.ChunkedPrefill
    dsa_cp_context: DSACPContext | None = None
    reshape_cache_event: torch.Event = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0


M = TypeVar("M", bound=FLSFAMetadata)

class FLSFAMetadataBuilder(MLACommonMetadataBuilder[FLSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[FLSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else FLSFAMetadata,
            supports_dcp_with_varlen,
        )

        self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
        self.reorder_batch_threshold = self.decode_threshold

        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.enable_dsa_cp = enable_dsa_cp()

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.actual_seq_lengths_query = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.actual_seq_lengths_key = torch.empty_like(self.actual_seq_lengths_query)

    @classmethod
    def get_cudagraph_support(
        cls: type["FLSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: FLCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FLSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens

        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]

        cos, sin = get_cos_and_sin_mla(input_positions, True)

        dsa_cp_context = None
        if self.enable_dsa_cp:
            global_tp_size = get_tp_group().world_size
            num_tokens = num_input_tokens
            num_tokens_pad = _round_up(num_tokens, global_tp_size)
            num_tokens_per_device = num_tokens_pad // global_tp_size
            local_start = get_tp_group().rank_in_group * num_tokens_per_device
            local_end_with_pad = local_start + num_tokens_per_device
            local_end = min(local_end_with_pad, num_actual_tokens)

            pad_size = num_tokens_pad - cos.shape[0]
            assert cos.shape == sin.shape, f"cos.shape must be equal to sin.shape, got {cos.shape} and {sin.shape}"

            if pad_size > 0:
                cos = nn.functional.pad(cos, (0, 0, 0, 0, 0, 0, 0, pad_size))
                sin = nn.functional.pad(sin, (0, 0, 0, 0, 0, 0, 0, pad_size))

            pad_size_slot = num_tokens_pad - slot_mapping.shape[0]
            if pad_size_slot > 0:
                slot_mapping = nn.functional.pad(slot_mapping, (0, pad_size_slot), value=-1)
            else:
                slot_mapping = slot_mapping[:num_tokens_pad]
            slot_mapping_cp = slot_mapping[local_start:local_end_with_pad]

            cos = cos[local_start:local_end_with_pad]
            sin = sin[local_start:local_end_with_pad]

            assert cos.shape[0] == num_tokens_per_device, (
                f"cos.shape[0] must be equal to num_tokens_per_device, \
                    got {cos.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping_cp.shape[0] == num_tokens_per_device, (
                f"slot_mapping_cp.shape[0] must be equal to num_tokens_per_device, \
                    got {slot_mapping_cp.shape[0]} and {num_tokens_per_device}"
            )
            assert slot_mapping.shape[0] == num_tokens_pad, (
                f"slot_mapping.shape[0] must be equal to num_tokens_pad, \
                    got {slot_mapping.shape[0]} and {num_tokens_pad}"
            )

            actual_seq_lengths_query = self.actual_seq_lengths_query
            actual_seq_lengths_key = self.actual_seq_lengths_key

            num_segs = cum_query_lens.shape[0]
            last_token = 0
            cum = 0
            for i in range(0, num_segs):
                global_start = last_token
                global_end = cum_query_lens[i].item()
                last_token = global_end

                req_local_start = max(global_start, local_start)
                req_local_end = min(global_end, local_end_with_pad)
                num_local_tokens = req_local_end - req_local_start

                if num_local_tokens > 0:
                    cum += num_local_tokens
                    actual_seq_lengths_query[i] = cum

                    offset = global_end - req_local_end
                    actual_seq_lengths_key[i] = seq_lens[i].item() - offset
                else:
                    actual_seq_lengths_query[i] = cum
                    actual_seq_lengths_key[i] = 0

            actual_seq_lengths_query = actual_seq_lengths_query[:num_reqs]
            actual_seq_lengths_key = actual_seq_lengths_key[:num_reqs]

            dsa_cp_context = DSACPContext(
                num_tokens=num_tokens,
                num_tokens_pad=num_tokens_pad,
                local_start=local_start,
                local_end=local_end,
                local_end_with_pad=local_end_with_pad,
                pad_size=pad_size,
                local_pad_size=local_end_with_pad - local_end,
                slot_mapping_cp=slot_mapping_cp,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            dsa_cp_context=dsa_cp_context,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: FLCommonAttentionMetadata,
        attn_state: AttentionFLState = AttentionFLState.DecodeOnly,
    ):
        if attn_state in {AttentionFLState.DecodeOnly, AttentionFLState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata

class FLSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Supports forward using the all-gather o_proj weight for decode requests when Sharded CP is enabled.
    o_proj_full_pool: torch.Tensor | None = None

    # qk_hadamard tensor shared when dsa c8 enabled
    qk_hadamard: torch.Tensor | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # vLLM's MLA common impl uses this flag to guard ROCm Aiter fp8-bmm
        # specific weight processing. On CUDA platforms this should be False.
        try:
            self.is_aiter_triton_fp8_bmm_enabled = bool(
                getattr(rocm_aiter_ops, "is_fp8bmm_enabled", lambda: False)()
            )
        except Exception:
            self.is_aiter_triton_fp8_bmm_enabled = False

        # MLA Args
        required_kwargs = (
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "rotary_emb",
            "kv_b_proj",
            "o_proj",
            "indexer",
            "q_b_proj",
        )
        missing_kwargs = [name for name in required_kwargs if name not in kwargs]
        if missing_kwargs:
            raise RuntimeError(
                "FLSFABackend requires the FL MultiHeadLatentAttentionWrapper "
                "kwargs, but the current MLA wrapper did not provide: "
                f"{missing_kwargs}. Enable the FL MLA OOT wrapper with "
                "VLLM_FL_ENABLE_MLA_OOT=1 and do not blacklist "
                "MultiHeadLatentAttentionWrapper, or use the native "
                "FLASHMLA_SPARSE backend."
            )
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.rotary_emb = kwargs["rotary_emb"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        self.q_b_proj = kwargs["q_b_proj"]

        fl_config = get_vllm_fl_config()

        assert self.indexer is not None, "Indexer is required for DSA."

        self.local_num_heads = self.num_heads
        self.vllm_config = get_current_vllm_config()
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )

        # indexer param
        self.n_head: int = self.indexer.n_head  # 64
        self.head_dim: int = self.indexer.head_dim  # 128
        self.wq_b = self.indexer.wq_b
        self.wk = self.indexer.wk
        self.weights_proj = self.indexer.weights_proj
        self.k_norm = self.indexer.k_norm
        self.cp_size = 1
        self.is_rope_neox_style = True
        self.is_glm_moe_dsa = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.is_glm_moe_dsa = True

        # Effective in SFA when FlashComm is enabled.
        self.enable_dsa_cp = enable_dsa_cp()

        # Enable layer sharding via DSA-CP on the P node in the PD-disaggregated setup.
        # TODO: now not distinct PD-mixed and PD-disaggregated
        self.enable_dsa_cp_with_layer_shard = enable_dsa_cp_with_layer_shard()

        # use original TP o_proj weight in PD mix stage, and full gather
        # for o_proj weight for prefill stage.
        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp()

        # TODO: need to distinct PD-mixed and PD-disaggregated
        if self.enable_dsa_cp:
            self.local_num_heads = self.num_heads * self.tp_size
            self.layer_sharding_kwargs = []
            for layer_name in get_vllm_fl_config().layer_sharding or []:
                if layer_name in kwargs:
                    self.layer_sharding_kwargs.append(kwargs[layer_name])
                else:
                    logger.warning_once(
                        f"[SFAImpl init] Layer '{layer_name}' not found in kwargs for layer sharding, "
                        "skipping sharding configuration"
                    )
            register_all_layers_to_shard_weight_series(self.layer_sharding_kwargs)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # NOTE: Ascend currently do not support quant kv_b_proj.
        # assert isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod)
        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute: {WEIGHT_NAMES}."
            )

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(
                    layer.input_size_per_partition,
                    dtype=act_dtype,
                    device=get_layer_weight(layer).device,
                )
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        if self.is_aiter_triton_fp8_bmm_enabled:
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=current_platform.fp8_dtype()
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=current_platform.fp8_dtype()
            )

            # The kernel operates on non-padded inputs. Hence, pre-compiling
            # triton kernel to avoid runtime compilation for unseen batch sizes
            # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
            # On DS-R1, this step adds roughly 50s to the model loading time.
            max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            pre_compilation_list = list(range(1, max_batch_size + 1))
            if is_global_first_rank():
                pre_compilation_list = tqdm(
                    pre_compilation_list,
                    desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                    total=max_batch_size,
                )

            for m in pre_compilation_list:
                x = torch.empty(
                    (self.W_K.shape[0], m, self.W_K.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_K.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
        # TODO: need to distinct PD-mixed and PD-disaggregated
        dispose_layer(self.kv_b_proj)
        if self.enable_dsa_cp:
            for layer in self.layer_sharding_kwargs or []:
                if is_hidden_layer(layer):
                    post_process_after_loading_for_shard_weight_series(layer)

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single_query(
        self,
        q_pe: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.rotary_emb is not None:
            assert (
                self.rotary_emb.rope_single_query is not None
            ), "Rotary embedding rope_single_query must be specified for SFA."
            q_pe = self.rotary_emb.rope_single_query(q_pe, cos, sin)
        return q_pe

    def _init_o_proj_tp_full_params(self):
        """
        Initialize TP-mode and Full-mode parameters for o_proj weight,
        preparing for weight switching in PD mix stage.

        For PD mix stage:
        - Use original TP o_proj weight for decode phase
        - Need full-gather o_proj weight from all TP ranks for prefill phase
        """
        if FLSFAImpl.o_proj_full_pool is None:
            sample = self.o_proj.weight
            FLSFAImpl.o_proj_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, sample.shape[1]), dtype=sample.dtype, device=sample.device
            )

        # Save TP-mode parameters (original sharded weights)
        self.o_proj_tp_weight = self.o_proj.weight.clone().detach()
        self.o_proj_tp_aclnn_input_scale = self.o_proj.aclnn_input_scale.clone().detach()
        self.o_proj_tp_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.clone().detach()
        self.o_proj_tp_aclnn_input_offset = self.o_proj.aclnn_input_offset.clone().detach()

        # Initially switch to TP mode for graph capture
        self.o_proj.weight.set_(self.o_proj_tp_weight)
        self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
        self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
        self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

        # Precompute Full-mode quantization parameters by repeating TP parameters across all TP ranks
        self.o_proj_full_aclnn_input_scale = self.o_proj.aclnn_input_scale.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_scale_reciprocal = self.o_proj.aclnn_input_scale_reciprocal.repeat(self.tp_size)
        self.o_proj_full_aclnn_input_offset = self.o_proj.aclnn_input_offset.repeat(self.tp_size)

    def _handle_o_proj_weight_switch_and_forward(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        o_proj_full_handle: torch.distributed.Work | None,
        should_shard_weight: bool,
    ) -> tuple[torch.Tensor, bool]:
        """
        Handle o_proj weight switching between TP-mode and Full-mode, and execute forward computation.
        """
        # Gather o_proj weight from all TP ranks for Full-mode computation
        if should_shard_weight:
            # Wait for the completion of o_proj weight all-gather operation
            if o_proj_full_handle is not None:
                o_proj_full_handle.wait()

            # Switch o_proj to Full-mode (gathered weight from all TP ranks)
            self.o_proj.weight.set_(FLSFAImpl.o_proj_full_pool)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_full_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_full_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_full_aclnn_input_offset)

            # Apply quantization method and execute forward computation
            output[...] = self.o_proj.quant_method.quant_method.apply(self.o_proj, attn_output)

            # Switch o_proj back to TP-mode for subsequent decode operations
            self.o_proj.weight.set_(self.o_proj_tp_weight)
            self.o_proj.aclnn_input_scale.set_(self.o_proj_tp_aclnn_input_scale)
            self.o_proj.aclnn_input_scale_reciprocal.set_(self.o_proj_tp_aclnn_input_scale_reciprocal)
            self.o_proj.aclnn_input_offset.set_(self.o_proj_tp_aclnn_input_offset)

            return output, False
        else:
            # For decode scenario: perform all-to-all communication on o_proj input activations
            # Reshape for all-to-all: [batch * seq, tp_size, head_dim] -> [tp_size, batch * seq, head_dim]
            send = (
                attn_output.view(-1, self.tp_size, self.num_heads * self.v_head_dim)
                .permute(1, 0, 2)
                .reshape(-1, self.num_heads * self.v_head_dim)
            )

            attn_output = torch.empty_like(send)
            torch.distributed.all_to_all_single(attn_output, send, group=get_tp_group().device_group)

            return attn_output, True

    def _get_full_kv(self, k, attn_metadata):
        return k

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
    ):
        # RMSNorm + RoPE + KVCache
        kv_c, k_pe = kv_no_split.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        # Add RoPE to k_pe
        if self.rotary_emb is not None:
            assert (
                self.rotary_emb.rope_single_key is not None
            ), "Rotary embedding rope_single_key must be specified for SFA."
            k_pe = self.rotary_emb.rope_single_key(k_pe, cos, sin)

        k_pe = k_pe.squeeze(1)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
            )

        return k_pe, kv_c_normed

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _v_up_proj(self, x: torch.Tensor):
        out: torch.Tensor = None
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)

        if self.is_aiter_triton_fp8_bmm_enabled:
            out = out.view(-1, self.local_num_heads, self.v_head_dim)
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            x = rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
            )
        else:
            # Convert from (B, N * V) to (N, B, V)
            out = out.view(-1, self.local_num_heads, self.v_head_dim).transpose(0, 1)

            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            torch.bmm(x, self.W_UV, out=out)  # Reuse "out" to make it "hot"

            # Convert from (N, B, V) to (B, N * V)
            out = out.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return out

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        k_li, _ = self.wk(x)  # [b,s,7168] @ [7168,128] = [b,s,128]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            raise RuntimeError(
                "Only support triton in FLSFA now."
            )

        k_li_scale = None

        return k_li, k_li_scale

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        weights, _ = self.weights_proj(x)

        q_li, _ = self.wq_b(q_c)  # [b,s,1536] @ [1536,64*128] = [b,s,64*128]
        q_li = q_li.view(-1, self.n_head, self.head_dim)  # [n_toks,64,128]
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            raise RuntimeError("Only support triton in FLSFA now.")

        from vllm_fl.ops.triton.sparse_attn import triton_paged_indexer_k_tiled_interface
        from flag_gems.fused.DSA.bin_topk import bucket_sort_topk
        Q = q_li.shape[0]
        cu_q = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=q_li.device),
            actual_seq_lengths_query.to(torch.int64),
        ])
        logits, kv_ends = triton_paged_indexer_k_tiled_interface(
            q=q_li,
            kv_cache=kv_cache[2],
            block_table=attn_metadata.block_table,
            weights=weights,
            cu_q_seqlens=cu_q,
            seq_kv_lens=actual_seq_lengths_key.to(torch.int32),
        )
        starts = torch.zeros(Q, dtype=torch.int32, device=logits.device)
        topk_indices = bucket_sort_topk(logits, starts, kv_ends, topk=2048)
        return topk_indices

    def _execute_sparse_flash_attention_process(
        self, ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
    ):
        from vllm_fl.ops.triton.sparse_attn import triton_sparse_mla_fwd_paged_interface
        block_table = attn_metadata.block_table
        cu_q = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=ql_nope.device),
            actual_seq_lengths_query.to(torch.int64),
        ])
        attn_output, _ = triton_sparse_mla_fwd_paged_interface(
            q_nope=ql_nope,
            q_rope=q_pe,
            kv_nope=kv_cache[0],
            kv_rope=kv_cache[1],
            block_table=block_table,
            indices=topk_indices,
            sm_scale=self.scale,
            cu_q_seqlens=cu_q,
            d_v=self.kv_lora_rank,
        )
        return attn_output

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attn_metadata: M,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            if self.enable_dsa_cp:
                for layer in self.layer_sharding_kwargs or []:
                    if is_hidden_layer(layer):
                        reach_layer_for_shard_weight_series(layer)
            return output.fill_(0)

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping = attn_metadata.slot_mapping
        slot_mapping_cp = None
        if self.enable_dsa_cp:
            assert attn_metadata.dsa_cp_context is not None
            slot_mapping_cp = attn_metadata.dsa_cp_context.slot_mapping_cp
            actual_seq_lengths_query = attn_metadata.dsa_cp_context.actual_seq_lengths_query
            actual_seq_lengths_key = attn_metadata.dsa_cp_context.actual_seq_lengths_key
        else:
            actual_seq_lengths_query = attn_metadata.cum_query_lens
            actual_seq_lengths_key = attn_metadata.seq_lens

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        output_padded = output

        # all-gather o_proj weight for prefill stage of PD mix node
        # TODO: need to distinct PD-mixed and PD-disaggregated
        # o_proj_full_handle = None
        # # if is PD mix stage, using original TP o_proj weight, and also need to full gather for o_proj
        # # weight for prefill stage.
        # full_gather_o_proj_enabled = self.enable_dsa_cp_with_o_proj_tp and attn_metadata.attn_state not in {
        #     AttentionFLState.DecodeOnly,
        #     AttentionFLState.SpecDecoding,
        # }

        assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
        q_c = self.q_a_layernorm(q_c)
        k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
        wait_for_kv_layer_from_connector(layer_name)
        if self.enable_dsa_cp:
            assert slot_mapping_cp is not None
            k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping_cp, attn_metadata, k_scale)
        else:
            k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata, k_scale)
        if self.enable_dsa_cp:
            assert k_pe is not None
            assert k_nope is not None
            assert k_li is not None
            # TODO: need to distinct PD-mixed and PD-disaggregated
            # support all_gather kv async for communication calculation overlap
            fused_kv_no_split, kv_ag_handle = all_gather_async(
                torch.cat(
                    [
                        k_pe.view(-1, k_pe.shape[-1]),
                        k_nope.view(-1, k_nope.shape[-1]),
                        k_li.view(-1, k_li.shape[-1]),
                    ],
                    dim=1,
                ),
                get_tp_group(),
            )
        ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
        q_pe = self.rope_single_query(q_pe, cos, sin)

        if self.enable_dsa_cp:
            if kv_ag_handle is not None:
                kv_ag_handle.wait()

            # TODO: need to distinct PD-mixed and PD-disaggregated
            for layer in self.layer_sharding_kwargs or []:
                if is_hidden_layer(layer):
                    reach_layer_for_shard_weight_series(layer)

            if kv_cache is not None:
                assert fused_kv_no_split is not None
                k_pe, k_nope, k_li = fused_kv_no_split.split(
                    [self.qk_rope_head_dim, self.kv_lora_rank, self.head_dim], dim=-1
                )
                k_nope = k_nope.view(k_nope.shape[0], 1, -1)
                k_pe = k_pe.view(k_pe.shape[0], 1, -1)
                DeviceOperator.reshape_and_cache(
                    key=k_nope[: attn_metadata.num_actual_tokens],
                    value=k_pe[: attn_metadata.num_actual_tokens],
                    key_cache=kv_cache[0],
                    value_cache=kv_cache[1],
                    slot_mapping=slot_mapping[: attn_metadata.num_actual_tokens],
                )
        k_li = self._get_full_kv(k_li, attn_metadata)

        topk_indices = self.indexer_select_post_process(
            x=hidden_states,
            q_c=q_c,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            cos=cos,
            sin=sin,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
        )

        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope, q_pe, kv_cache, topk_indices, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key
        )

        attn_output = self._v_up_proj(attn_output)

        # TODO: need to distinct PD-mixed and PD-disaggregated
        output[...] = self.o_proj(attn_output)[0]

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output_padded
