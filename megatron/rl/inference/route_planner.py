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
        src_shard: Index of the shard that produced this hop's input
            activation, or ``None`` for the entry hop (input from the
            embedding layer / token ids, not from another shard).
    """

    shard_idx: int
    layer_indices: Tuple[int, ...]
    src_shard: Optional[int]


@dataclass(frozen=True)
class Route:
    """Complete forward-pass plan for one request.

    Attributes:
        hops: Ordered tuple of :class:`RouteHop`. Length-1 routes
            correspond to "no disaggregation; this request runs entirely
            on one shard" — back-compat with collocated requests.
        entry_shard: Shard the request is submitted to (== ``hops[0].shard_idx``).
        exit_shard: Shard that produces the final logits and the next
            token (== ``hops[-1].shard_idx``). The coord routes
            ``ENGINE_REPLY`` from this shard.
    """

    hops: Tuple[RouteHop, ...]
    entry_shard: int
    exit_shard: int

    def num_hops(self) -> int:
        return len(self.hops)

    def hops_through(self, shard_idx: int) -> Tuple[RouteHop, ...]:
        """All hops where ``shard_idx`` is the running shard. Typically
        length 0 or 1; length > 1 happens when a request bounces back
        through a shard (e.g., dense MLP after an MoE detour)."""
        return tuple(h for h in self.hops if h.shard_idx == shard_idx)

    def visits(self, shard_idx: int) -> bool:
        return any(h.shard_idx == shard_idx for h in self.hops)


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
    *,
    entry_shard: Optional[int] = None,
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
        entry_shard: Optional override for the request's entry shard.
            If given, the route MUST start with this shard — otherwise
            an ``AssertionError`` is raised. Used when the submitter is
            pinned to a specific shard (e.g., a sticky routing policy)
            and wants the planner to validate compatibility rather than
            silently pick another entry. ``None`` means "use whatever
            shard owns layer 0."

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
    prev_shard: Optional[int] = None
    for li, shard_idx in enumerate(owner_by_layer):
        if shard_idx == current_shard:
            current_layers.append(li)
            continue
        if current_shard is not None:
            hops.append(
                RouteHop(
                    shard_idx=current_shard,
                    layer_indices=tuple(current_layers),
                    src_shard=prev_shard,
                )
            )
            prev_shard = current_shard
        current_shard = shard_idx
        current_layers = [li]
    assert current_shard is not None  # layer_type_list non-empty
    hops.append(
        RouteHop(
            shard_idx=current_shard,
            layer_indices=tuple(current_layers),
            src_shard=prev_shard,
        )
    )

    entry = hops[0].shard_idx
    if entry_shard is not None:
        assert entry == entry_shard, (
            f"Caller pinned entry_shard={entry_shard} but layer 0 is "
            f"owned by shard {entry}. Either re-submit to shard {entry} "
            f"or extend shard {entry_shard}'s kinds to include the "
            f"first layer's kind ({layer_type_list[0]!r})."
        )
    exit_ = hops[-1].shard_idx
    return Route(hops=tuple(hops), entry_shard=entry, exit_shard=exit_)


def serialize_route(route: Route) -> list:
    """Flatten a :class:`Route` to a msgpack-compatible list.

    Wire form (one entry per hop):

        [[shard_idx, [layer_idx, ...], src_shard_or_None], ...]

    ``entry_shard`` and ``exit_shard`` are derivable from the hops
    themselves so they're not on the wire.
    """
    return [
        [h.shard_idx, list(h.layer_indices), h.src_shard]
        for h in route.hops
    ]


def deserialize_route(obj: list) -> Route:
    """Inverse of :func:`serialize_route`. Validates each hop has at
    least one layer (a zero-layer hop is meaningless and would silently
    break the forward-pass router)."""
    assert obj, "deserialize_route: empty hop list."
    hops: List[RouteHop] = []
    for i, h in enumerate(obj):
        assert len(h) == 3, (
            f"deserialize_route: hop {i} must be "
            f"[shard_idx, layer_indices, src_shard], got {h!r}."
        )
        shard_idx, layer_indices, src_shard = h
        assert layer_indices, (
            f"deserialize_route: hop {i} has empty layer_indices."
        )
        hops.append(
            RouteHop(
                shard_idx=int(shard_idx),
                layer_indices=tuple(int(li) for li in layer_indices),
                src_shard=None if src_shard is None else int(src_shard),
            )
        )
    return Route(
        hops=tuple(hops),
        entry_shard=hops[0].shard_idx,
        exit_shard=hops[-1].shard_idx,
    )


def explain_route(route: Route, layer_type_list: Tuple[str, ...]) -> str:
    """Human-readable rendering of a route. Useful in logs + tests.

    Example output::

        Route(entry=0, exit=2): 0[M:0,2,4] -> 1[*:1,3] -> 2[E:5,6,7]
    """
    parts: List[str] = []
    for h in route.hops:
        kinds = "".join(layer_type_list[i] for i in h.layer_indices)
        layers = ",".join(str(i) for i in h.layer_indices)
        parts.append(f"{h.shard_idx}[{kinds}:{layers}]")
    return (
        f"Route(entry={route.entry_shard}, exit={route.exit_shard}): "
        + " -> ".join(parts)
    )
