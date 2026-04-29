# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention-bounded segment execution for hybrid Mamba+MoE inference.

This module provides infrastructure for an exact (non-approximate) inference
runtime optimization for hybrid Mamba+MoE models with sparse attention layers.

The core idea is to canonicalize hidden state ownership only at attention layers,
and between attention layers run Mamba+MoE blocks within a chosen execution
"island" (segment owner rank). This avoids paying the MoE combine-back cost to
the original KV owner after every MoE layer.

For the MVP, the runtime tracks segments and ownership metadata but defaults to
the baseline behavior (combine destination = original owner). The feature is
guarded by ``TransformerConfig.enable_attention_bounded_segments``.

References:
    See ``attention_bounded_segment_execution_plan.md`` for the design doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols

# Layer type symbols that act as attention boundaries. Hidden state must be
# canonicalized to the KV owner before these layers run.
_ATTENTION_SYMBOLS = frozenset({LayerSymbols.ATTENTION, LayerSymbols.DS_ATTENTION})

# Layer type symbols that participate in MoE-style routing.
_MOE_SYMBOLS = frozenset({LayerSymbols.MOE})

# Layer type symbols whose recurrent state must be migrated when ownership
# changes. MLP and dense layers have no per-request state.
_STATEFUL_SYMBOLS = frozenset({LayerSymbols.MAMBA, LayerSymbols.GDN})


SegmentOwnerPolicy = Literal[
    "original_owner",
    "fixed_rank",
    "same_node_as_attention_owner",
    "hottest_expert_affinity",
    "measured_cost_model",
]

CombineDestinationPolicy = Literal[
    "original_owner", "current_segment_owner", "next_mamba_owner", "cost_model"
]


@dataclass
class Segment:
    """An attention-bounded segment of layers.

    A segment is a maximal run of non-attention layers (Mamba/MoE/MLP/GDN)
    delimited by attention layers (or the beginning/end of the model). Within
    a segment, hidden state ownership may move freely; outside (i.e., at
    attention layers), ownership is canonicalized to the KV owner.

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
        # Collect a run of non-attention layers.
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

    Args:
        segments: Segments produced by ``compute_segments``.
        num_layers: Total number of layers on this pipeline stage.

    Returns:
        A list of length ``num_layers`` where entry ``i`` is the segment id
        containing layer ``i`` (or ``None`` for attention layers).
    """
    mapping: List[Optional[int]] = [None] * num_layers
    for seg in segments:
        for idx in seg.layer_indices:
            mapping[idx] = seg.segment_id
    return mapping


def summarize_segments(segments: List[Segment]) -> str:
    """Return a short human-readable summary of segment topology.

    Useful to log once at startup so users can verify what the runtime sees.
    """
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

    This object holds the static segment topology and a small amount of
    mutable per-request ownership state. It is constructed once per
    ``HybridStack`` (one per pipeline stage) and lives for the lifetime of
    the model.

    The MVP wires this into the model's forward path purely as metadata: with
    ``enable_attention_bounded_segments=False`` (the default) every
    ``combine_destination_for_layer`` call returns ``"original_owner"`` and
    the canonicalize / Mamba-state migration hooks are no-ops, so output
    matches baseline exactly.

    Future stages will use this state to actually redirect MoE combine, move
    Mamba state between ranks, and choose segment owners based on routing.

    Attributes:
        layer_type_list: Layer types for this pipeline stage.
        segments: Attention-bounded segments computed from
            ``layer_type_list``.
        layer_to_segment: ``layer_to_segment[i]`` is the segment id of layer
            ``i``, or ``None`` if layer ``i`` is an attention boundary.
        enabled: Master switch. When False, all hooks are no-ops and
            ``combine_destination_for_layer`` returns ``"original_owner"``.
        segment_owner_policy: Policy for choosing the segment owner rank.
            Only ``"original_owner"`` is implemented in the MVP.
        moe_combine_destination_policy: Policy for the MoE combine
            destination. Only ``"original_owner"`` is implemented in the MVP.
    """

    layer_type_list: List[str]
    segments: List[Segment]
    layer_to_segment: List[Optional[int]]
    enabled: bool = False
    segment_owner_policy: SegmentOwnerPolicy = "original_owner"
    moe_combine_destination_policy: CombineDestinationPolicy = "original_owner"

    # Per-request bookkeeping. Populated lazily as requests appear. The
    # initial implementation does not migrate state, so these are all
    # initialized to and remain at the request's KV owner.
    _attention_owner: Dict[int, int] = field(default_factory=dict)
    _current_owner: Dict[int, int] = field(default_factory=dict)
    _segment_owner: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @classmethod
    def from_layer_type_list(
        cls,
        layer_type_list: List[str],
        *,
        enabled: bool = False,
        segment_owner_policy: SegmentOwnerPolicy = "original_owner",
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
            segment_owner_policy=segment_owner_policy,
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

    def combine_destination_for_layer(
        self, local_layer_idx: int, request_id: Optional[int] = None
    ) -> str:
        """Return the policy name for where MoE combine should send tokens.

        For the MVP this always returns ``"original_owner"`` so behavior
        matches baseline. The signature accepts a ``request_id`` so that
        future cost-model policies can vary per-request without changing
        callers.
        """
        if not self.enabled:
            return "original_owner"
        # Only the baseline policy is implemented in the MVP. Future stages
        # will branch on self.moe_combine_destination_policy and self.segments.
        return self.moe_combine_destination_policy

    # ------------------------------------------------------------------
    # Attention-boundary canonicalization. The MVP keeps current_owner ==
    # attention_owner at all times, so this is a no-op. The hook exists so
    # that later stages can drop in real send/recv code without touching
    # the model forward path again.
    # ------------------------------------------------------------------
    def canonicalize_to_attention_owner(self, request_ids: Optional[List[int]] = None) -> None:
        if not self.enabled:
            return
        if request_ids is None:
            request_ids = list(self._current_owner.keys())
        for req_id in request_ids:
            owner = self._attention_owner.get(req_id)
            if owner is not None:
                self._current_owner[req_id] = owner

    def set_attention_owner(self, request_id: int, rank: int) -> None:
        """Record the (stable) KV owner rank for a request."""
        self._attention_owner[request_id] = rank
        # Until we actually redirect, current_owner tracks attention_owner.
        self._current_owner.setdefault(request_id, rank)

    def get_current_owner(self, request_id: int) -> Optional[int]:
        return self._current_owner.get(request_id)

    def forget_request(self, request_id: int) -> None:
        """Drop bookkeeping for a finished request."""
        self._attention_owner.pop(request_id, None)
        self._current_owner.pop(request_id, None)
        # Also drop segment_owner entries for this request.
        stale = [k for k in self._segment_owner if k[0] == request_id]
        for k in stale:
            self._segment_owner.pop(k, None)
