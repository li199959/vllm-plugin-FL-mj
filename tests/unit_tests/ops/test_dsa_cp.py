# Copyright (c) 2025 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _clear_dsa_cp_caches():
    from vllm_fl import utils

    for fn_name in (
        "enable_cp",
        "_is_dsa_cp_model",
        "enable_sp",
        "enable_dsa_cp",
        "enable_dsa_cp_with_layer_shard",
        "enable_dsa_cp_with_o_proj_tp",
    ):
        getattr(utils, fn_name).cache_clear()


def _fake_vllm_config(
    *,
    has_index_topk=True,
    tp_size=1,
    dcp_size=1,
    pcp_size=1,
    kv_transfer_config=None,
):
    hf_text_config = SimpleNamespace()
    if has_index_topk:
        hf_text_config.index_topk = 16
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
        kv_transfer_config=kv_transfer_config,
    )


class TestDSACPFlags:

    def teardown_method(self):
        _clear_dsa_cp_caches()

    def test_dsa_cp_requires_fl_mla_oot(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.delenv("VLLM_FL_ENABLE_MLA_OOT", raising=False)
        config = _fake_vllm_config(has_index_topk=True, tp_size=4)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_sp() is False
            assert utils.enable_dsa_cp() is False

    def test_dsa_cp_enabled_for_dsa_model_with_tp_when_fl_mla_oot(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.setenv("VLLM_FL_ENABLE_MLA_OOT", "1")
        config = _fake_vllm_config(has_index_topk=True, tp_size=4)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_sp() is True
            assert utils.enable_dsa_cp() is True

    def test_dsa_cp_requires_tp_parallelism_even_with_dcp(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.setenv("VLLM_FL_ENABLE_MLA_OOT", "1")
        config = _fake_vllm_config(has_index_topk=True, tp_size=1, dcp_size=2)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_sp() is False
            assert utils.enable_dsa_cp() is False

    def test_dsa_cp_disabled_for_single_card(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.setenv("VLLM_FL_ENABLE_MLA_OOT", "1")
        config = _fake_vllm_config(has_index_topk=True, tp_size=1, dcp_size=1)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_sp() is False
            assert utils.enable_dsa_cp() is False

    def test_dsa_cp_disabled_for_non_dsa_model(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.setenv("VLLM_FL_ENABLE_MLA_OOT", "1")
        config = _fake_vllm_config(has_index_topk=False, tp_size=4)
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_sp() is False
            assert utils.enable_dsa_cp() is False

    def test_o_proj_tp_enabled_for_pd_mixed_role(self, monkeypatch):
        from vllm_fl import utils

        monkeypatch.setenv("VLLM_FL_ENABLE_MLA_OOT", "1")
        config = _fake_vllm_config(
            has_index_topk=True,
            tp_size=4,
            kv_transfer_config=SimpleNamespace(kv_role="kv_both"),
        )
        with patch("vllm.config.get_current_vllm_config", return_value=config):
            assert utils.enable_dsa_cp_with_o_proj_tp() is True


class TestDSACPLinearDispatch:

    @pytest.fixture
    def mock_tp_group(self):
        return SimpleNamespace(world_size=4, rank_in_group=1, device_group=None)

    def test_q_b_proj_uses_sharded_cp_only_for_fl_mla_oot(self, mock_tp_group):
        from vllm_fl.ops import linear_op

        layer = SimpleNamespace(prefix="model.layers.0.self_attn.q_b_proj")
        with (
            patch("vllm_fl.utils.enable_dsa_cp", return_value=True),
            patch("vllm_fl.utils.enable_mla_oot", return_value=True),
            patch("vllm_fl.utils.enable_sp", return_value=True),
            patch("vllm_fl.ops.linear_op.get_tp_group", return_value=mock_tp_group),
        ):
            op, rank, size = linear_op.get_parallel_op(
                False, layer.prefix, layer, "column"
            )
        assert isinstance(op, linear_op.ShardedCPColumnParallelOp)
        assert rank == 0
        assert size == 1

    def test_q_b_proj_keeps_native_tp_layout_without_fl_mla_oot(self, mock_tp_group):
        from vllm_fl.ops import linear_op

        prefix = "model.layers.0.self_attn.q_b_proj"
        layer = SimpleNamespace(prefix=prefix)
        with (
            patch("vllm_fl.utils.enable_dsa_cp", return_value=True),
            patch("vllm_fl.utils.enable_mla_oot", return_value=False),
            patch("vllm_fl.utils.enable_sp", return_value=True),
            patch("vllm_fl.ops.linear_op.get_tp_group", return_value=mock_tp_group),
        ):
            op, rank, size = linear_op.get_parallel_op(False, prefix, layer, "column")
        assert op is None
        assert rank == 1
        assert size == 4

    @pytest.mark.parametrize(
        "prefix,direct,expected_type",
        [
            ("model.layers.0.self_attn.query_key_value", "column", "SequenceColumnParallelOp"),
            ("model.layers.0.self_attn.out_proj", "row", "SequenceRowParallelOp"),
            ("model.layers.0.attention.dense", "row", "SequenceRowParallelOp"),
        ],
    )
    def test_sequence_parallel_prefixes(self, mock_tp_group, prefix, direct, expected_type):
        from vllm_fl.ops import linear_op

        layer = SimpleNamespace(prefix=prefix)
        with (
            patch("vllm_fl.utils.enable_dsa_cp", return_value=False),
            patch("vllm_fl.utils.enable_dsa_cp_with_layer_shard", return_value=False),
            patch("vllm_fl.utils.enable_sp", return_value=True),
            patch("vllm_fl.ops.linear_op.get_tp_group", return_value=mock_tp_group),
        ):
            op, _, _ = linear_op.get_parallel_op(False, prefix, layer, direct)
        assert type(op).__name__ == expected_type


def test_sfa_cp_module_exports_backend_hooks():
    from vllm_fl.dispatch.backends.flaggems.impl.sfa_cp import (
        FLSFACPImpl,
        FLSFACPMetadataBuilder,
    )
    from vllm_fl.dispatch.backends.flaggems.impl.sfa import (
        FLSFAImpl,
        FLSFAMetadataBuilder,
    )

    assert issubclass(FLSFACPMetadataBuilder, FLSFAMetadataBuilder)
    assert issubclass(FLSFACPImpl, FLSFAImpl)


def test_sfa_backend_selects_cp_hooks_when_cp_enabled():
    from vllm_fl.dispatch.backends.flaggems.impl.sfa import FLSFABackend
    from vllm_fl.dispatch.backends.flaggems.impl.sfa_cp import (
        FLSFACPImpl,
        FLSFACPMetadataBuilder,
    )

    with patch("vllm_fl.dispatch.backends.flaggems.impl.sfa.enable_cp", return_value=True):
        assert FLSFABackend.get_builder_cls() is FLSFACPMetadataBuilder
        assert FLSFABackend.get_impl_cls() is FLSFACPImpl
