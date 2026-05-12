# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end pipeline test for layer-kind disaggregation primitives.

Stitches together the parser, builder, and route planner to verify the
whole shape works without a GPU or distributed runtime. If this test
passes, the primitives compose correctly.
"""

from typing import List

from megatron.core.inference.shards import InferenceShard
from megatron.rl.inference.route_planner import (
    deserialize_route,
    plan_route,
    serialize_route,
)
from megatron.rl.inference.shards_spec import (
    assert_kinds_partition_layers,
    compute_layer_indices_for_kinds,
    parse_inference_shards_spec,
)


def _make_shards_directly(specs: List[dict], layer_types: tuple) -> List[InferenceShard]:
    """Skip the distributed builder (which needs torch.distributed init)
    and construct InferenceShard objects with the same layer_indices logic
    we'd get from build_inference_pg_collections_for_shards."""
    shards: List[InferenceShard] = []
    offset = 0
    for i, spec in enumerate(specs):
        tp = spec["tp"]
        pp = spec["pp"]
        dp = spec["dp"]
        world = tp * pp * dp
        kinds = spec.get("kinds")
        layer_indices = (
            compute_layer_indices_for_kinds(kinds, layer_types) if kinds else None
        )
        shards.append(
            InferenceShard(
                index=i,
                spec=spec,
                rank_offset=offset,
                world_size=world,
                pg_collection=None,
                kinds=tuple(kinds) if kinds is not None else None,
                layer_indices=layer_indices,
            )
        )
        offset += world
    return shards


def _layers_owned_by_shard(route, shard_idx: int) -> set:
    """Union of layer_indices across all hops that ``shard_idx`` runs."""
    owned = set()
    for hop in route.hops:
        if hop.shard_idx == shard_idx:
            owned.update(hop.layer_indices)
    return owned


def test_pipeline_disagg_three_shards_alternating_pattern():
    """End-to-end: CLI string → parse → validate → shards → route.
    On an alternating M/*/E pattern with 3 kind-disagg shards, the
    route's hops partition every layer to exactly one shard, and that
    shard agrees with the shards' layer_indices."""
    spec_str = "tp=4,kinds=M+tp=4,kinds=*+tp=4,kinds=E"
    layer_types = ("M", "*", "E", "M", "*", "E")
    world_size = 12

    parsed = parse_inference_shards_spec(spec_str, world_size=world_size)
    assert len(parsed) == 3
    assert_kinds_partition_layers(parsed, layer_types)

    shards = _make_shards_directly(parsed, layer_types)
    assert shards[0].layer_indices == (0, 3)
    assert shards[1].layer_indices == (1, 4)
    assert shards[2].layer_indices == (2, 5)

    route = plan_route(shards, layer_type_list=layer_types)
    assert route.entry_shard == 0
    assert route.exit_shard == 2
    assert len(route.hops) == 6  # full alternation

    # Wire round-trip preserves the route.
    assert deserialize_route(serialize_route(route)) == route

    # Each layer is owned by exactly one shard in the route's hops,
    # and that shard matches the shard's declared layer_indices.
    for li in range(len(layer_types)):
        owners = [s for s in range(3) if li in _layers_owned_by_shard(route, s)]
        assert owners == [s for s in range(3)
                          if shards[s].layer_indices is not None
                          and li in shards[s].layer_indices]
        assert len(owners) == 1


def test_pipeline_no_disagg_layout_degrades_gracefully():
    """A spec with no kinds= produces a 1-hop route covering every
    layer on shard 0. The disagg primitives don't perturb collocated
    layouts."""
    layer_types = ("*",) * 4
    specs = parse_inference_shards_spec("tp=4", world_size=4)
    shards = _make_shards_directly(specs, layer_types)

    route = plan_route(shards, layer_type_list=layer_types)
    assert len(route.hops) == 1
    assert route.entry_shard == route.exit_shard == 0
    assert route.hops[0].layer_indices == (0, 1, 2, 3)
