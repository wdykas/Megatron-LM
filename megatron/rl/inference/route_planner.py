# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-request route planner for layer-kind-disaggregated inference.

Given a list of :class:`InferenceShard` whose ``layer_indices`` cover
the model's layer space, produces a :class:`Route` — the ordered
sequence of shard hops a request walks during its forward pass. Each
hop carries the consecutive layer indices that shard runs before
handing the activation off to the next hop.

The planner is deliberately stateless and side-effect-free: same
inputs always produce the same route. Per-request dynamic routing is
the property of *which* layout the route is computed against; the
mapping from layout → route is deterministic.

See :doc:`megatron/core/inference/DISAGG_DESIGN.md` for how the route
is consumed at forward-pass time and what assumptions it implies for
the coord and engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.inference.shards import InferenceShard


@dataclass(frozen=True)
class RouteHop:
    """One shard's contribution to a request's forward pass.

    Attributes:
        shard_idx: Index of the shard in the global shards list.
        layer_indices: Consecutive global layer indices this shard runs
            during this hop. Always non-empty; always a strict ascending
            sub-range of the layer space.
        succs: Explicit list of successor hop positions this hop fans
            its outbound activation to. ``None`` (default) means "use
            the implicit linear chain" — i.e. ``(hop_pos + 1,)`` when
            not the last hop, else ``()``. Set explicitly when the
            route is a DAG (fan-out point, parallel branch tip, or
            exit hop with multiple inbound branches). ``()`` is valid
            and means "this hop terminates a branch" (only one tip
            may terminate without descendants — the route's exit).
        preds: Explicit list of predecessor hop positions whose
            outbound activations this hop ingests. ``None`` (default)
            means "use the implicit linear chain" — i.e.
            ``(hop_pos - 1,)`` when not the first hop, else ``()``.
            When ``len(preds) > 1`` the dispatcher reduces inbound
            activations per :attr:`reduce_op`.
        reduce_op: Reduction kernel applied at the merge point: one of
            ``"sum"`` or ``"mean"``. ``None`` when there's exactly one
            inbound activation (the linear case). Required when
            ``preds`` is set and has length > 1.

    The previous-hop shard (source of the inbound activation) is
    derivable from the hop's position in :attr:`Route.hops` — see
    :meth:`Route.src_of`. For DAG branches use :meth:`Route.predecessors`
    / :meth:`Route.successors`.
    """

    shard_idx: int
    layer_indices: Tuple[int, ...]
    succs: Optional[Tuple[int, ...]] = None
    preds: Optional[Tuple[int, ...]] = None
    reduce_op: Optional[str] = None


_VALID_REDUCE_OPS = ("sum", "mean")


@dataclass(frozen=True)
class Route:
    """Complete forward-pass plan for one request.

    Attributes:
        hops: Ordered tuple of :class:`RouteHop`. Length-1 routes
            correspond to "no disaggregation; this request runs entirely
            on one shard" — back-compat with collocated requests. For
            DAG routes the linear ``hop_pos → hop_pos + 1`` chain is
            still meaningful (it expresses the "spine"), but branches
            and merges are encoded via :attr:`RouteHop.parallel_succs`
            and :attr:`RouteHop.reduce_from`.

    Linear-hop fast paths (``predecessors`` / ``successors``) collapse
    to the historical ``hop_pos ± 1`` behavior when no DAG fields are
    set, so existing back-compat routes keep working.
    """

    hops: Tuple[RouteHop, ...]

    @property
    def entry_shard(self) -> int:
        """Shard the request is submitted to (== ``hops[0].shard_idx``)."""
        return self.hops[0].shard_idx

    @property
    def exit_shard(self) -> int:
        """Shard producing the final logits and next token; coord routes
        ``ENGINE_REPLY`` from here (== ``hops[-1].shard_idx``)."""
        return self.hops[-1].shard_idx

    def visits(self, shard_idx: int) -> bool:
        return any(h.shard_idx == shard_idx for h in self.hops)

    def src_of(self, hop_pos: int) -> Optional[int]:
        """Shard that produced the inbound activation for ``hops[hop_pos]``,
        or ``None`` for the entry hop (input is from the embedding)."""
        return self.hops[hop_pos - 1].shard_idx if hop_pos > 0 else None

    def predecessors(self, hop_pos: int) -> Tuple[int, ...]:
        """All predecessor hop positions for ``hops[hop_pos]``.

        Returns :attr:`RouteHop.preds` when set explicitly; otherwise
        falls back to the implicit linear chain (``hop_pos - 1`` when
        ``hop_pos > 0``, empty for the entry hop).
        """
        hop = self.hops[hop_pos]
        if hop.preds is not None:
            return hop.preds
        return (hop_pos - 1,) if hop_pos > 0 else ()

    def successors(self, hop_pos: int) -> Tuple[int, ...]:
        """All successor hop positions for ``hops[hop_pos]``.

        Returns :attr:`RouteHop.succs` when set explicitly; otherwise
        falls back to the implicit linear chain (``hop_pos + 1`` when
        ``hop_pos < len(hops) - 1``, empty for the exit hop).
        """
        hop = self.hops[hop_pos]
        if hop.succs is not None:
            return hop.succs
        return (hop_pos + 1,) if hop_pos < len(self.hops) - 1 else ()

    def is_dag(self) -> bool:
        """True iff any hop overrides the implicit linear chain via
        explicit :attr:`RouteHop.succs` / :attr:`RouteHop.preds`.
        Linear routes return False — the dispatcher keeps its
        single-input single-output fast path for them."""
        return any(h.succs is not None or h.preds is not None for h in self.hops)


def _build_ownership_map(
    shards: List[InferenceShard],
    layer_type_list: Tuple[str, ...],
) -> List[int]:
    """For each global layer index, return the owning shard index.

    Raises ``AssertionError`` if any layer is unowned or double-owned.
    For layouts where no shard declares ``kinds=``, falls back to the
    PP-style "shard 0 owns everything" — which means there's only one
    shard and the route degenerates to a single hop.
    """
    n_layers = len(layer_type_list)
    owner: List[Optional[int]] = [None] * n_layers
    any_kinds = any(s.layer_indices is not None for s in shards)

    if not any_kinds:
        # No disagg. Assign every layer to shard 0 (the request stays
        # on its entry shard for the whole forward pass — collocated
        # behavior).
        assert len(shards) >= 1, "plan_route needs at least one shard"
        return [0] * n_layers

    for s in shards:
        if s.layer_indices is None:
            raise AssertionError(
                f"Shard {s.index} has no layer_indices but other shards "
                f"do. Mixed layouts are rejected by "
                f"assert_kinds_partition_layers; this shouldn't happen "
                f"at route-planning time. Re-validate the layout."
            )
        for li in s.layer_indices:
            assert 0 <= li < n_layers, (
                f"Shard {s.index} owns layer {li}, out of range for "
                f"layer_type_list of length {n_layers}."
            )
            if owner[li] is not None:
                raise AssertionError(
                    f"Layer {li} owned by both shard {owner[li]} and "
                    f"shard {s.index}; layouts must be a partition."
                )
            owner[li] = s.index

    missing = [i for i, o in enumerate(owner) if o is None]
    if missing:
        raise AssertionError(
            f"{len(missing)} layer(s) have no owning shard "
            f"(first few: {missing[:5]}). Extend a shard's kinds= to "
            f"cover the missing kind(s) or re-check layer_type_list."
        )
    # owner is now List[int] (all filled) — type-narrow for clarity.
    return [o for o in owner]  # type: ignore[misc]


def plan_route(
    shards: List[InferenceShard],
    layer_type_list: Tuple[str, ...],
) -> Route:
    """Compute the route a request takes through a disaggregated layout.

    Walks layers ``0..L-1`` in model order. Each contiguous run of
    layers owned by the same shard collapses into a single
    :class:`RouteHop` (the engine on that shard then runs all of them
    locally with no inter-layer activation transport).

    Args:
        shards: List of shards in the layout. Order = shard index.
        layer_type_list: Per-block kind symbols (length = total layer
            count).

    Returns:
        :class:`Route` with at least one hop. For non-disagg layouts
        (no shard declares ``kinds=``), the route has exactly one hop
        covering every layer on shard 0.
    """
    assert layer_type_list, "layer_type_list must be non-empty."
    owner_by_layer = _build_ownership_map(shards, layer_type_list)

    hops: List[RouteHop] = []
    current_shard: Optional[int] = None
    current_layers: List[int] = []
    for li, shard_idx in enumerate(owner_by_layer):
        if shard_idx == current_shard:
            current_layers.append(li)
            continue
        if current_shard is not None:
            hops.append(RouteHop(current_shard, tuple(current_layers)))
        current_shard = shard_idx
        current_layers = [li]
    assert current_shard is not None  # layer_type_list non-empty
    hops.append(RouteHop(current_shard, tuple(current_layers)))
    return Route(hops=tuple(hops))


def make_moe_dag_route(
    backbone_shard: int,
    expert_shards: Tuple[int, ...],
    moe_layer: int,
    *,
    backbone_pre_layers: Tuple[int, ...] = (),
    backbone_post_layers: Tuple[int, ...] = (),
    reduce_op: str = "sum",
) -> Route:
    """Build a DAG route where a single backbone shard fans an MoE layer
    out to ``len(expert_shards)`` expert shards in parallel, then reduces
    the experts' outputs and continues on the backbone.

    Shape::

        [backbone_pre]
            ├─→ E0 (moe_layer)
            ├─→ E1 (moe_layer)
            └─→ ...
            ↓ reduce
        [backbone_post]

    The reduce hop runs back on the backbone (its first hop position is
    1 — the linear successor of the fan-out hop — and it merges the rest
    of the expert hops via ``reduce_from``). The backbone fan-out hop's
    ``parallel_succs`` lists the expert hop positions beyond the linear
    successor.

    Args:
        backbone_shard: Shard index running the non-MoE backbone (pre +
            post MoE layers).
        expert_shards: Shard indices running the MoE layer in parallel.
            Must contain at least one entry; routes with a single
            expert shard are valid (degenerates to a linear hop).
        moe_layer: Global layer index of the MoE block — each expert
            shard runs only this one layer.
        backbone_pre_layers: Backbone layers before the MoE block.
            Empty when the MoE block is the very first layer.
        backbone_post_layers: Backbone layers after the MoE block.
            Empty when the MoE block is the very last layer.
        reduce_op: ``"sum"`` (default) or ``"mean"``. The dispatcher's
            reduce kernel; pick ``"mean"`` if the expert outputs are
            already weighted and you want the average rather than the
            sum.
    """
    assert expert_shards, (
        "make_moe_dag_route requires at least one expert shard"
    )

    n_experts = len(expert_shards)
    if n_experts == 1:
        # Single-expert "DAG" is just a linear route — emit the simple
        # form so the dispatcher's linear fast path picks it up.
        hops: List[RouteHop] = []
        if backbone_pre_layers:
            hops.append(RouteHop(backbone_shard, tuple(backbone_pre_layers)))
        hops.append(RouteHop(expert_shards[0], (moe_layer,)))
        if backbone_post_layers:
            hops.append(RouteHop(backbone_shard, tuple(backbone_post_layers)))
        return Route(hops=tuple(hops))

    if not backbone_post_layers:
        # No post-hop: experts converge nowhere. Route is malformed.
        raise AssertionError(
            "make_moe_dag_route with multiple experts requires a "
            "non-empty backbone_post_layers to host the reduce hop."
        )

    # Multi-expert DAG. Positions:
    #
    #   0 .. K-1 : optional pre hop(s) — at most one in this builder
    #   K .. K+N-1 : expert hops (parallel branches)
    #   K+N : post hop (reduce target)
    has_pre = bool(backbone_pre_layers)
    pre_pos = 0 if has_pre else None
    expert_start = 1 if has_pre else 0
    expert_positions = tuple(range(expert_start, expert_start + n_experts))
    post_pos = expert_start + n_experts

    hops = []
    if has_pre:
        # Pre hop fans to ALL experts in parallel — explicit succs to
        # override the implicit "linear next == expert_positions[0]"
        # chain so the dispatcher knows about every branch.
        hops.append(
            RouteHop(
                shard_idx=backbone_shard,
                layer_indices=tuple(backbone_pre_layers),
                succs=expert_positions,
            )
        )
    for e_shard in expert_shards:
        # Each expert hop sends straight to the post (reduce) hop AND
        # receives straight from the pre hop. Both edges must be
        # explicit, otherwise the implicit linear chain would make
        # neighboring experts send / receive to each other instead.
        hops.append(
            RouteHop(
                shard_idx=e_shard,
                layer_indices=(moe_layer,),
                succs=(post_pos,),
                preds=(pre_pos,) if pre_pos is not None else (),
            )
        )
    # Post hop ingests one activation per expert; reduce_op combines them.
    hops.append(
        RouteHop(
            shard_idx=backbone_shard,
            layer_indices=tuple(backbone_post_layers),
            preds=expert_positions,
            reduce_op=reduce_op,
        )
    )

    return Route(hops=tuple(hops))


def serialize_route(route: Route) -> list:
    """Flatten a :class:`Route` to a msgpack-compatible list.

    Linear wire form (one entry per hop) — preserved as the default
    when no DAG extension is set on any hop:

        [[shard_idx, [layer_idx, ...]], ...]

    DAG wire form — used when at least one hop declares
    :attr:`RouteHop.parallel_succs` / :attr:`RouteHop.reduce_from`:

        [[shard_idx, [layer_idx, ...], [parallel_succs...],
          [reduce_from...], reduce_op_or_null], ...]

    The 5-element form is emitted for every hop in a DAG route so the
    deserializer can dispatch on length without ambiguity. Existing
    callers that only construct linear routes keep emitting the legacy
    2-element form and existing wire-format consumers keep working
    bitwise unchanged.

    The inbound linear src shard for each hop is still derivable from
    position (``hops[i-1].shard_idx``); ``reduce_from`` carries only
    the *extra* DAG predecessors.
    """
    if route.is_dag():
        return [
            [
                h.shard_idx,
                list(h.layer_indices),
                (None if h.succs is None else list(h.succs)),
                (None if h.preds is None else list(h.preds)),
                h.reduce_op,
            ]
            for h in route.hops
        ]
    return [[h.shard_idx, list(h.layer_indices)] for h in route.hops]


def deserialize_route(obj: list) -> Route:
    """Inverse of :func:`serialize_route`. Accepts both the legacy
    2-element linear form and the 5-element DAG form on a per-hop
    basis (a route may not mix forms within itself — the serializer
    never produces a mixed wire form). Validates each hop has at
    least one layer (a zero-layer hop is meaningless and would
    silently break the forward-pass router) and that every hop has a
    well-formed reduce_op when ``reduce_from`` is set."""
    assert obj, "deserialize_route: empty hop list."
    hops: List[RouteHop] = []
    for i, h in enumerate(obj):
        if len(h) == 2:
            shard_idx, layer_indices = h
            succs: Optional[Tuple[int, ...]] = None
            preds: Optional[Tuple[int, ...]] = None
            reduce_op: Optional[str] = None
        elif len(h) == 5:
            shard_idx, layer_indices, wire_succs, wire_preds, reduce_op = h
            succs = (
                None if wire_succs is None
                else tuple(int(x) for x in wire_succs)
            )
            preds = (
                None if wire_preds is None
                else tuple(int(x) for x in wire_preds)
            )
            reduce_op = None if reduce_op is None else str(reduce_op)
        else:
            raise AssertionError(
                f"deserialize_route: hop {i} must be 2-element [shard_idx, "
                f"layer_indices] or 5-element DAG form, got {h!r}."
            )
        assert layer_indices, (
            f"deserialize_route: hop {i} has empty layer_indices."
        )
        if preds is not None and len(preds) > 1:
            assert reduce_op in _VALID_REDUCE_OPS, (
                f"deserialize_route: hop {i} has preds={preds} (multi-input) "
                f"but reduce_op={reduce_op!r}; must be one of "
                f"{_VALID_REDUCE_OPS}."
            )
        hops.append(
            RouteHop(
                shard_idx=int(shard_idx),
                layer_indices=tuple(int(li) for li in layer_indices),
                succs=succs,
                preds=preds,
                reduce_op=reduce_op,
            )
        )
    return Route(hops=tuple(hops))


