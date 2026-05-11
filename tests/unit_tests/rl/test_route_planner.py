# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for ``megatron.rl.inference.route_planner``."""

import pytest

from megatron.core.inference.shards import InferenceShard
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    deserialize_route,
    explain_route,
    plan_route,
    serialize_route,
)


def _shard(
    index: int,
    kinds=None,
    layer_indices=None,
    rank_offset=0,
    world_size=2,
) -> InferenceShard:
    spec = {"tp": world_size, "pp": 1, "ep": 1, "expt_tp": world_size, "dp": 1}
    if kinds is not None:
        spec["kinds"] = tuple(kinds)
    return InferenceShard(
        index=index,
        spec=spec,
        rank_offset=rank_offset,
        world_size=world_size,
        pg_collection=None,
        kinds=tuple(kinds) if kinds is not None else None,
        layer_indices=tuple(layer_indices) if layer_indices is not None else None,
    )


def test_route_no_disagg_single_hop():
    """When no shard declares kinds=, the route is one hop covering
    every layer on shard 0 (collocated back-compat)."""
    shards = [_shard(0)]
    route = plan_route(shards, layer_type_list=("*",) * 6)
    assert route.num_hops() == 1
    assert route.entry_shard == 0 and route.exit_shard == 0
    assert route.hops[0].layer_indices == (0, 1, 2, 3, 4, 5)
    assert route.hops[0].src_shard is None


def test_route_three_kind_disagg_alternating_pattern():
    """Mamba/attention/MoE shards on a M*E... pattern produce a hop per layer."""
    pattern = ("M", "*", "E", "M", "*", "E")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0, 3)),
        _shard(1, kinds=("*",), layer_indices=(1, 4)),
        _shard(2, kinds=("E",), layer_indices=(2, 5)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    assert route.num_hops() == 6
    # Each hop covers one layer because the pattern is fully alternating.
    expected_shards = [0, 1, 2, 0, 1, 2]
    actual_shards = [h.shard_idx for h in route.hops]
    assert actual_shards == expected_shards
    # Each hop has src_shard = previous hop's shard.
    assert route.hops[0].src_shard is None
    for i in range(1, len(route.hops)):
        assert route.hops[i].src_shard == route.hops[i - 1].shard_idx


def test_route_consecutive_same_shard_layers_collapse_to_one_hop():
    """A run of layers on the same shard becomes one hop (saving
    activation hops at the wire level)."""
    pattern = ("M", "M", "M", "*", "*", "E", "E", "M")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0, 1, 2, 7)),
        _shard(1, kinds=("*",), layer_indices=(3, 4)),
        _shard(2, kinds=("E",), layer_indices=(5, 6)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    # M,M,M -> 1 hop; *,* -> 1; E,E -> 1; M -> 1. Total = 4.
    assert route.num_hops() == 4
    assert route.hops[0].shard_idx == 0
    assert route.hops[0].layer_indices == (0, 1, 2)
    assert route.hops[1].shard_idx == 1
    assert route.hops[1].layer_indices == (3, 4)
    assert route.hops[2].shard_idx == 2
    assert route.hops[2].layer_indices == (5, 6)
    assert route.hops[3].shard_idx == 0
    assert route.hops[3].layer_indices == (7,)
    # The last hop's src_shard is shard 2 (the previous hop's shard).
    assert route.hops[3].src_shard == 2


def test_route_entry_and_exit_shards():
    """entry_shard / exit_shard reflect the first and last hops, not
    just the first and last in the shards list."""
    pattern = ("*", "M", "M", "E")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(1, 2)),
        _shard(1, kinds=("*",), layer_indices=(0,)),
        _shard(2, kinds=("E",), layer_indices=(3,)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    # Layer 0 is owned by shard 1; layer 3 by shard 2.
    assert route.entry_shard == 1
    assert route.exit_shard == 2


def test_route_planner_rejects_double_owned_layer():
    """Layouts where two shards claim the same layer fail at plan time."""
    pattern = ("M", "*")
    shards = [
        _shard(0, kinds=("M", "*"), layer_indices=(0, 1)),
        _shard(1, kinds=("*",), layer_indices=(1,)),
    ]
    with pytest.raises(AssertionError, match="owned by both"):
        plan_route(shards, layer_type_list=pattern)


def test_route_planner_rejects_unowned_layer():
    pattern = ("M", "E", "*")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0,)),
        _shard(1, kinds=("*",), layer_indices=(2,)),
        # Layer 1 (E) has no owner.
    ]
    with pytest.raises(AssertionError, match="have no owning shard"):
        plan_route(shards, layer_type_list=pattern)


def test_route_planner_rejects_pinned_entry_when_layer0_not_owned():
    """If a caller pins entry_shard=X but X doesn't own layer 0, we
    fail loudly rather than silently re-route."""
    pattern = ("M", "*")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0,)),
        _shard(1, kinds=("*",), layer_indices=(1,)),
    ]
    with pytest.raises(AssertionError, match="pinned entry_shard"):
        plan_route(shards, layer_type_list=pattern, entry_shard=1)


def test_route_planner_accepts_compatible_pinned_entry():
    pattern = ("M", "*")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0,)),
        _shard(1, kinds=("*",), layer_indices=(1,)),
    ]
    route = plan_route(shards, layer_type_list=pattern, entry_shard=0)
    assert route.entry_shard == 0


def test_route_visits_and_hops_through():
    """Helpers: ``visits(s)`` and ``hops_through(s)``."""
    pattern = ("M", "*", "E", "M", "*")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0, 3)),
        _shard(1, kinds=("*",), layer_indices=(1, 4)),
        _shard(2, kinds=("E",), layer_indices=(2,)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    assert route.visits(0) and route.visits(1) and route.visits(2)
    # Shard 0 appears in 2 hops (layers 0 and 3); shard 2 in 1 hop.
    assert len(route.hops_through(0)) == 2
    assert len(route.hops_through(2)) == 1


def test_route_roundtrip_serialization():
    """A Route survives serialize → deserialize unchanged."""
    pattern = ("M", "M", "*", "E", "M")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0, 1, 4)),
        _shard(1, kinds=("*",), layer_indices=(2,)),
        _shard(2, kinds=("E",), layer_indices=(3,)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    wire = serialize_route(route)
    # Wire form is msgpack-compatible: only list / int / None.
    for hop_wire in wire:
        assert isinstance(hop_wire, list) and len(hop_wire) == 3
        shard_idx, layer_indices, src_shard = hop_wire
        assert isinstance(shard_idx, int)
        assert isinstance(layer_indices, list)
        assert all(isinstance(li, int) for li in layer_indices)
        assert src_shard is None or isinstance(src_shard, int)
    restored = deserialize_route(wire)
    assert restored == route


def test_deserialize_route_rejects_empty_hop():
    """A hop with zero layer indices is meaningless; reject loudly."""
    with pytest.raises(AssertionError, match="empty layer_indices"):
        deserialize_route([[0, [], None]])


def test_deserialize_route_rejects_empty_list():
    """A zero-hop route would have no entry/exit; reject loudly."""
    with pytest.raises(AssertionError, match="empty hop list"):
        deserialize_route([])


def test_deserialize_route_normalizes_types():
    """Integers come back as ints regardless of wire encoding
    (msgpack sometimes produces numpy ints for example)."""
    # Simulate a "weird" wire form with non-int-typed values.
    class _IntLike:
        def __init__(self, v): self.v = v
        def __index__(self): return self.v

    route = deserialize_route([[_IntLike(0), [_IntLike(0), _IntLike(1)], None]])
    assert route.hops[0].shard_idx == 0
    assert route.hops[0].layer_indices == (0, 1)


def test_explain_route_human_readable():
    pattern = ("M", "*", "E")
    shards = [
        _shard(0, kinds=("M",), layer_indices=(0,)),
        _shard(1, kinds=("*",), layer_indices=(1,)),
        _shard(2, kinds=("E",), layer_indices=(2,)),
    ]
    route = plan_route(shards, layer_type_list=pattern)
    s = explain_route(route, pattern)
    assert "Route(entry=0, exit=2)" in s
    assert "0[M:0]" in s
    assert "1[*:1]" in s
    assert "2[E:2]" in s
    assert "->" in s
