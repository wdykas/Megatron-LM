# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for ``megatron.core.inference.attention_bounded_segments``.

These tests cover the pure-Python segment computation: parsing a layer type
list (e.g., ``"M*M*"``) into ``Segment`` objects with the right boundaries,
MoE layer indices, and stateful (Mamba/GDN) layer indices. No CUDA or
distributed setup is required.
"""

import pytest

from megatron.core.inference.attention_bounded_segments import (
    SegmentRuntime,
    build_layer_to_segment_map,
    compute_segments,
    summarize_segments,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols


class TestComputeSegments:
    def test_empty_layer_list_returns_no_segments(self):
        assert compute_segments([]) == []

    def test_all_attention_layers_returns_no_segments(self):
        # Pure attention stack — nothing to optimize.
        segs = compute_segments([Symbols.ATTENTION] * 4)
        assert segs == []

    def test_simple_two_segments(self):
        # A * M E * M E pattern -> two segments around the middle attention.
        # idx:  0  1  2  3  4  5
        layer_types = [
            Symbols.ATTENTION,
            Symbols.MAMBA,
            Symbols.MOE,
            Symbols.ATTENTION,
            Symbols.MAMBA,
            Symbols.MOE,
        ]
        segs = compute_segments(layer_types)
        assert len(segs) == 2

        s0, s1 = segs
        assert (s0.start_layer, s0.end_layer) == (1, 2)
        assert s0.layer_indices == [1, 2]
        assert s0.moe_layer_indices == [2]
        assert s0.stateful_layer_indices == [1]
        assert s0.has_attention_boundary_before is True
        assert s0.has_attention_boundary_after is True

        assert (s1.start_layer, s1.end_layer) == (4, 5)
        assert s1.layer_indices == [4, 5]
        assert s1.moe_layer_indices == [5]
        assert s1.stateful_layer_indices == [4]
        assert s1.has_attention_boundary_before is True
        # No attention layer follows -> end-of-model boundary.
        assert s1.has_attention_boundary_after is False

    def test_no_attention_at_edges(self):
        # Pure Mamba+MoE -> one segment covering everything, no attention
        # boundaries at either edge.
        layer_types = [Symbols.MAMBA, Symbols.MOE, Symbols.MAMBA, Symbols.MOE]
        segs = compute_segments(layer_types)
        assert len(segs) == 1
        seg = segs[0]
        assert seg.start_layer == 0
        assert seg.end_layer == 3
        assert seg.has_attention_boundary_before is False
        assert seg.has_attention_boundary_after is False
        assert seg.moe_layer_indices == [1, 3]
        assert seg.stateful_layer_indices == [0, 2]

    def test_nemotron_like_pattern(self):
        # Nemotron-style: many Mamba+MoE between sparse attention layers.
        # Pattern: M E M E * M E M E M E * M E
        layer_types = list("MEME*MEMEME*ME")
        segs = compute_segments(layer_types)
        assert len(segs) == 3
        assert [s.num_layers for s in segs] == [4, 6, 2]
        assert [s.num_moe_layers for s in segs] == [2, 3, 1]

    def test_gdn_counts_as_stateful_not_moe(self):
        # GDN layers also carry recurrent state -> stateful, not MoE.
        layer_types = [Symbols.GDN, Symbols.MOE]
        segs = compute_segments(layer_types)
        assert len(segs) == 1
        assert segs[0].stateful_layer_indices == [0]
        assert segs[0].moe_layer_indices == [1]

    def test_dsa_attention_is_a_boundary(self):
        # Deepseek-style attention should also act as a boundary.
        layer_types = [Symbols.MAMBA, Symbols.DS_ATTENTION, Symbols.MOE]
        segs = compute_segments(layer_types)
        assert len(segs) == 2
        assert segs[0].layer_indices == [0]
        assert segs[1].layer_indices == [2]

    def test_segment_ids_are_contiguous(self):
        # Segment IDs should run 0..N-1 in execution order.
        layer_types = list("M*ME*EM")
        segs = compute_segments(layer_types)
        assert [s.segment_id for s in segs] == list(range(len(segs)))


class TestLayerToSegmentMap:
    def test_attention_layers_map_to_none(self):
        layer_types = [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MOE, Symbols.ATTENTION]
        segs = compute_segments(layer_types)
        mapping = build_layer_to_segment_map(segs, len(layer_types))
        assert mapping == [None, 0, 0, None]

    def test_all_layers_in_one_segment(self):
        layer_types = [Symbols.MAMBA, Symbols.MOE]
        segs = compute_segments(layer_types)
        mapping = build_layer_to_segment_map(segs, len(layer_types))
        assert mapping == [0, 0]


class TestSegmentRuntime:
    def test_disabled_runtime_returns_baseline_combine_destination(self):
        # Even if a non-baseline policy is configured, the runtime must
        # report "original_owner" while disabled — this is what guarantees
        # baseline equivalence with the feature flag off.
        rt = SegmentRuntime.from_layer_type_list(
            list("ME*ME"),
            enabled=False,
            moe_combine_destination_policy="current_segment_owner",
        )
        assert rt.combine_destination_for_layer(0) == "original_owner"
        assert rt.combine_destination_for_layer(4) == "original_owner"

    def test_enabled_runtime_reports_configured_policy(self):
        rt = SegmentRuntime.from_layer_type_list(
            list("ME*ME"),
            enabled=True,
            moe_combine_destination_policy="original_owner",
        )
        assert rt.combine_destination_for_layer(0) == "original_owner"

    def test_segment_for_layer_lookup(self):
        # Pattern: M E * M E  -> seg0=[0,1], seg1=[3,4], idx 2 is attention.
        rt = SegmentRuntime.from_layer_type_list(list("ME*ME"))
        s0 = rt.segment_for_layer(0)
        assert s0 is not None and s0.segment_id == 0
        # Attention layer maps to no segment.
        assert rt.segment_for_layer(2) is None
        s1 = rt.segment_for_layer(4)
        assert s1 is not None and s1.segment_id == 1

    def test_is_attention_boundary(self):
        rt = SegmentRuntime.from_layer_type_list(list("ME*ME"))
        assert rt.is_attention_boundary(2) is True
        assert rt.is_attention_boundary(0) is False
        # Out-of-range indices are not boundaries (they're not anything).
        assert rt.is_attention_boundary(99) is False

    def test_enabled_runtime_reports_variant_b_policy(self):
        rt = SegmentRuntime.from_layer_type_list(
            list("ME*ME"),
            enabled=True,
            moe_combine_destination_policy="current_segment_owner",
        )
        assert rt.combine_destination_for_layer(0) == "current_segment_owner"
        assert rt.combine_destination_for_layer(4) == "current_segment_owner"


class TestSummarize:
    def test_empty_summary(self):
        assert summarize_segments([]) == "<no attention-bounded segments>"

    def test_summary_contains_layer_pattern(self):
        segs = compute_segments(list("ME*ME"))
        summary = summarize_segments(segs)
        assert "seg0" in summary
        assert "seg1" in summary
        assert "ME" in summary  # the layer types should appear


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
