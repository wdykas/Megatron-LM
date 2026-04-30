# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention-bounded segment execution for hybrid Mamba+MoE inference.

This module identifies "attention-bounded segments" — maximal runs of
non-attention layers (Mamba/MoE/MLP/GDN) delimited by attention layers — and
exposes a small runtime that callers (the hybrid block, the MoE dispatcher)
read to decide whether to take the Variant B fast path.

The Variant B fast path skips the AllGather before each MoE layer and replaces
the ReduceScatter at combine with an AllReduce that returns the full global
view. It is enabled by setting
``TransformerConfig.enable_attention_bounded_segments=True`` together with
``moe_combine_destination_policy="current_segment_owner"``. With either
condition off, every hook is a no-op and behavior matches baseline exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols

# Layer type symbols that act as attention boundaries.
_ATTENTION_SYMBOLS = frozenset({LayerSymbols.ATTENTION, LayerSymbols.DS_ATTENTION})

# Layer type symbols that participate in MoE-style routing.
_MOE_SYMBOLS = frozenset({LayerSymbols.MOE})

# Layer type symbols whose recurrent state must be migrated when ownership
# changes. MLP and dense layers have no per-request state.
_STATEFUL_SYMBOLS = frozenset({LayerSymbols.MAMBA, LayerSymbols.GDN})


CombineDestinationPolicy = Literal["original_owner", "current_segment_owner"]


@dataclass
class Segment:
    """An attention-bounded segment of layers.

    A segment is a maximal run of non-attention layers (Mamba/MoE/MLP/GDN)
    delimited by attention layers (or the beginning/end of the model).

    Attributes:
        segment_id: Index of this segment within the layer sequence.
        start_layer: Local index (within the pipeline stage's layer list) of
            the first layer in this segment.
        end_layer: Local index of the last layer in this segment (inclusive).
        layer_indices: The list of local layer indices belonging to this
            segment.
        layer_types: The list of layer type characters for this segment.
        moe_layer_indices: Local indices of MoE layers in this segment.
        stateful_layer_indices: Local indices of layers with recurrent state
            (Mamba/GDN) in this segment.
        has_attention_boundary_before: True if there is an attention layer
            immediately before this segment.
        has_attention_boundary_after: True if there is an attention layer
            immediately after this segment.
    """

    segment_id: int
    start_layer: int
    end_layer: int
    layer_indices: List[int]
    layer_types: List[str]
    moe_layer_indices: List[int] = field(default_factory=list)
    stateful_layer_indices: List[int] = field(default_factory=list)
    has_attention_boundary_before: bool = False
    has_attention_boundary_after: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    @property
    def num_stateful_layers(self) -> int:
        return len(self.stateful_layer_indices)

    def contains(self, local_layer_idx: int) -> bool:
        return self.start_layer <= local_layer_idx <= self.end_layer


def compute_segments(layer_type_list: List[str]) -> List[Segment]:
    """Identify attention-bounded segments in a layer sequence.

    A segment is the set of consecutive non-attention layers between two
    attention layers (or model start/end). Attention layers themselves are
    not part of any segment.

    Args:
        layer_type_list: Layer type characters for this pipeline stage,
            using the symbols in ``LayerSymbols`` (e.g., ``'M'`` for Mamba,
            ``'*'`` for attention, ``'E'`` for MoE).

    Returns:
        A list of ``Segment`` objects in execution order. May be empty if
        ``layer_type_list`` is empty or consists entirely of attention layers.

    Examples:
        >>> from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
        >>> # Pattern: Attention, Mamba, MoE, Mamba, MoE, Attention, Mamba, MoE
        >>> ltl = ['*', 'M', 'E', 'M', 'E', '*', 'M', 'E']
        >>> segs = compute_segments(ltl)
        >>> [(s.start_layer, s.end_layer) for s in segs]
        [(1, 4), (6, 7)]
    """
    segments: List[Segment] = []
    n = len(layer_type_list)
    i = 0
    seg_id = 0
    while i < n:
        if layer_type_list[i] in _ATTENTION_SYMBOLS:
            i += 1
            continue
        start = i
        while i < n and layer_type_list[i] not in _ATTENTION_SYMBOLS:
            i += 1
        end = i - 1
        layer_indices = list(range(start, end + 1))
        layer_types = layer_type_list[start : end + 1]
        moe_indices = [
            idx for idx in layer_indices if layer_type_list[idx] in _MOE_SYMBOLS
        ]
        stateful_indices = [
            idx for idx in layer_indices if layer_type_list[idx] in _STATEFUL_SYMBOLS
        ]
        has_boundary_before = start > 0 and layer_type_list[start - 1] in _ATTENTION_SYMBOLS
        has_boundary_after = end < n - 1 and layer_type_list[end + 1] in _ATTENTION_SYMBOLS
        segments.append(
            Segment(
                segment_id=seg_id,
                start_layer=start,
                end_layer=end,
                layer_indices=layer_indices,
                layer_types=layer_types,
                moe_layer_indices=moe_indices,
                stateful_layer_indices=stateful_indices,
                has_attention_boundary_before=has_boundary_before,
                has_attention_boundary_after=has_boundary_after,
            )
        )
        seg_id += 1
    return segments


def build_layer_to_segment_map(
    segments: List[Segment], num_layers: int
) -> List[Optional[int]]:
    """Build a lookup table from local layer index to segment id.

    Layers that are attention boundaries map to ``None``.
    """
    mapping: List[Optional[int]] = [None] * num_layers
    for seg in segments:
        for idx in seg.layer_indices:
            mapping[idx] = seg.segment_id
    return mapping


def summarize_segments(segments: List[Segment]) -> str:
    """Return a short human-readable summary of segment topology."""
    if not segments:
        return "<no attention-bounded segments>"
    parts = []
    for seg in segments:
        parts.append(
            f"seg{seg.segment_id}=[{seg.start_layer}..{seg.end_layer}] "
            f"types={''.join(seg.layer_types)} "
            f"({seg.num_moe_layers} MoE, {seg.num_stateful_layers} stateful)"
        )
    return " | ".join(parts)


@dataclass
class SegmentRuntime:
    """Per-pipeline-stage runtime state for attention-bounded segment execution.

    Holds the static segment topology and the configured combine-destination
    policy. With ``enabled=False`` (default) every
    ``combine_destination_for_layer`` call returns ``"original_owner"`` so
    output matches baseline exactly.

    Attributes:
        layer_type_list: Layer types for this pipeline stage.
        segments: Attention-bounded segments computed from
            ``layer_type_list``.
        layer_to_segment: ``layer_to_segment[i]`` is the segment id of layer
            ``i``, or ``None`` if layer ``i`` is an attention boundary.
        enabled: Master switch.
        moe_combine_destination_policy: Either ``"original_owner"`` (baseline)
            or ``"current_segment_owner"`` (Variant B AR return).
    """

    layer_type_list: List[str]
    segments: List[Segment]
    layer_to_segment: List[Optional[int]]
    enabled: bool = False
    moe_combine_destination_policy: CombineDestinationPolicy = "original_owner"

    @classmethod
    def from_layer_type_list(
        cls,
        layer_type_list: List[str],
        *,
        enabled: bool = False,
        moe_combine_destination_policy: CombineDestinationPolicy = "original_owner",
    ) -> "SegmentRuntime":
        """Build a runtime from a list of layer type characters."""
        segments = compute_segments(layer_type_list)
        layer_to_segment = build_layer_to_segment_map(segments, len(layer_type_list))
        return cls(
            layer_type_list=list(layer_type_list),
            segments=segments,
            layer_to_segment=layer_to_segment,
            enabled=enabled,
            moe_combine_destination_policy=moe_combine_destination_policy,
        )

    def segment_for_layer(self, local_layer_idx: int) -> Optional[Segment]:
        """Return the segment containing this local layer, or None."""
        if not (0 <= local_layer_idx < len(self.layer_to_segment)):
            return None
        seg_id = self.layer_to_segment[local_layer_idx]
        if seg_id is None:
            return None
        return self.segments[seg_id]

    def is_attention_boundary(self, local_layer_idx: int) -> bool:
        if not (0 <= local_layer_idx < len(self.layer_type_list)):
            return False
        return self.layer_type_list[local_layer_idx] in _ATTENTION_SYMBOLS

    def combine_destination_for_layer(self, local_layer_idx: int) -> str:
        """Return the policy name for where MoE combine should send tokens.

        Returns ``"original_owner"`` when the runtime is disabled (baseline
        behavior). When enabled, returns the configured
        ``moe_combine_destination_policy``.
        """
        if not self.enabled:
            return "original_owner"
        return self.moe_combine_destination_policy
