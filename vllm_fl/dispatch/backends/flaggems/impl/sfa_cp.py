# Copyright (c) 2025 BAAI. All rights reserved.

"""Minimal SFA context-parallel entrypoints for FL DSA-CP.

The first FL DSA-CP milestone supports the DCP/SP path used by
DeepSeek V3.2 serving.  Full PCP request splitting is intentionally kept out
of this module for now, but the classes mirror vllm-ascend's SFA-CP boundary
so the richer PCP metadata can be added without changing backend selection.
"""

from __future__ import annotations

from vllm.logger import logger

from vllm_fl.dispatch.backends.flaggems.impl.sfa import (
    FLSFAImpl,
    FLSFAMetadata,
    FLSFAMetadataBuilder,
)


class FLSFACPMetadataBuilder(FLSFAMetadataBuilder):
    """SFA metadata builder for the first DSA-CP serving milestone.

    DCP reuses the TP communication domain in vLLM, so the base SFA metadata
    builder's DSA-CP token slicing and slot mapping are sufficient for the
    initial multi-card path.  PCP-specific head/tail request splitting will be
    layered here later.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        self.vllm_config = vllm_config
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        parallel_config = vllm_config.parallel_config
        self.pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        if self.pcp_size > 1:
            logger.warning_once(
                "FLSFACPMetadataBuilder is using the DCP/SP-only DSA-CP path. "
                "Full PCP request splitting is not implemented in vllm_fl yet; "
                "set prefill_context_parallel_size=1 for the supported first "
                "milestone."
            )

    def build(self, *args, **kwargs) -> FLSFAMetadata:
        return super().build(*args, **kwargs)


class FLSFACPImpl(FLSFAImpl):
    """Implementation alias for the DCP/SP SFA path.

    The base FLSFA implementation already performs DSA-CP local token handling,
    KV all-gather, and sparse attention execution.  This subclass exists as the
    explicit context-parallel backend hook selected by ``FLSFABackend``.
    """

    pass
