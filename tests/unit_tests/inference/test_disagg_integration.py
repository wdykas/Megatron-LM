# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end pipeline test for layer-kind disaggregation primitives.

Stitches together the parser, builder, route planner, and route walker
to verify the whole shape works without a GPU or distributed runtime.
If this test passes, the primitives compose correctly; what remains is
the engine + coord + model-construction integration that consumes them.
"""

from typing import List

import pytest

from megatron.core.inference.partial_model import (
    PartialModelOwnership,
    filter_layer_pattern,
    ownership_for_shard,
)
from megatron.core.inference.route_walker import LayerAction, RouteWalker
from megatron.core.inference.shards import (
    InferenceShard,
    build_inference_pg_collections_for_shards,
)
from megatron.rl.inference.route_planner import (
    deserialize_route,
    explain_route,
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


def test_pipeline_disagg_three_shards_alternating_pattern():
    """End-to-end: CLI string -> parse -> validate -> shards -> route ->
    walker. Verifies a 3-kind disagg (M / * / E) on an alternating
    pattern correctly drives the route walker across every shard."""
    # CLI flag.
    spec_str = "tp=4,kinds=M+tp=4,kinds=*+tp=4,kinds=E"
    layer_types = ("M", "*", "E", "M", "*", "E")
    world_size = 12

    # Parse + validate.
    parsed = parse_inference_shards_spec(spec_str, world_size=world_size)
    assert len(parsed) == 3
    assert_kinds_partition_layers(parsed, layer_types)

    # Build (skipping the torch.distributed bit — same layer_indices logic).
    shards = _make_shards_directly(parsed, layer_types)
    assert shards[0].layer_indices == (0, 3)
    assert shards[1].layer_indices == (1, 4)
    assert shards[2].layer_indices == (2, 5)

    # Plan the route.
    route = plan_route(shards, layer_type_list=layer_types)
    assert route.entry_shard == 0
    assert route.exit_shard == 2
    assert route.num_hops() == 6  # full alternation

    # Round-trip through the wire format.
    restored = deserialize_route(serialize_route(route))
    assert restored == route

    # Drive walkers on each shard. After all three walkers have stepped
    # through their layer space, the union of LOCAL actions covers every
    # layer index exactly once.
    walkers = [RouteWalker(route, my_shard_idx=i) for i in range(3)]
    local_by_shard: List[List[int]] = [[], [], []]
    for li in range(len(layer_types)):
        for shard_idx, walker in enumerate(walkers):
            dec = walker.before_layer(li)
            if dec.action is LayerAction.LOCAL:
                local_by_shard[shard_idx].append(li)
            elif dec.action is LayerAction.RECEIVE:
                walker.after_receive()
                dec2 = walker.before_layer(li)
                assert dec2.action is LayerAction.LOCAL, (
                    f"after RECEIVE on shard {shard_idx} layer {li}, "
                    f"expected LOCAL, got {dec2.action}"
                )
                local_by_shard[shard_idx].append(li)
            elif dec.action is LayerAction.SEND:
                walker.after_send()

    # Every layer ran on exactly one shard, and that shard is the one
    # whose layer_indices contains the layer.
    for li in range(len(layer_types)):
        owners = [s for s, locals_ in enumerate(local_by_shard) if li in locals_]
        assert len(owners) == 1, (
            f"layer {li} ran on shards {owners}; expected exactly one"
        )
        assert shards[owners[0]].layer_indices is not None
        assert li in shards[owners[0]].layer_indices


def test_pipeline_partial_model_ownership_consistent_with_route():
    """The partial-model ownership derived from a shard agrees with
    the route walker's LOCAL set for that shard."""
    layer_types = ("M", "M", "*", "E", "*", "M")
    specs = parse_inference_shards_spec(
        "tp=2,kinds=M+tp=2,kinds=*+tp=2,kinds=E", world_size=6
    )
    shards = _make_shards_directly(specs, layer_types)

    route = plan_route(shards, layer_type_list=layer_types)

    for shard_idx, shard in enumerate(shards):
        own = ownership_for_shard(shard, layer_types)
        walker = RouteWalker(route, my_shard_idx=shard_idx)
        ran_locally: List[int] = []
        for li in range(len(layer_types)):
            dec = walker.before_layer(li)
            if dec.action is LayerAction.RECEIVE:
                walker.after_receive()
                walker.before_layer(li)  # consume the LOCAL
                ran_locally.append(li)
            elif dec.action is LayerAction.LOCAL:
                ran_locally.append(li)
            elif dec.action is LayerAction.SEND:
                walker.after_send()
        # The set of layers the walker ran locally must equal the
        # ownership descriptor's layer_indices.
        assert tuple(sorted(ran_locally)) == own.layer_indices, (
            f"shard {shard_idx}: walker ran {sorted(ran_locally)} but "
            f"ownership says {own.layer_indices}"
        )


def test_pipeline_memory_savings_match_layer_distribution():
    """Per-shard memory savings reflect the kind partition."""
    layer_types = tuple("MMMM****EEEE")  # 4 of each kind, 12 total
    specs = parse_inference_shards_spec(
        "tp=4,kinds=M+tp=4,kinds=*+tp=4,kinds=E", world_size=12
    )
    shards = _make_shards_directly(specs, layer_types)
    for shard in shards:
        own = ownership_for_shard(shard, layer_types)
        # Each shard owns 4 of 12 layers -> ~67% savings vs full model.
        assert own.num_owned_layers() == 4
        assert own.memory_savings_ratio() == pytest.approx(2 / 3)


def test_pipeline_no_disagg_layout_degrades_gracefully():
    """A spec with no kinds= produces a 1-hop route and full-model
    ownership — the disagg primitives don't get in the way of
    collocated layouts."""
    layer_types = ("*",) * 4
    specs = parse_inference_shards_spec("tp=4", world_size=4)
    shards = _make_shards_directly(specs, layer_types)

    route = plan_route(shards, layer_type_list=layer_types)
    assert route.num_hops() == 1
    assert route.entry_shard == route.exit_shard == 0

    walker = RouteWalker(route, my_shard_idx=0)
    for li in range(4):
        assert walker.before_layer(li).action is LayerAction.LOCAL

    own = ownership_for_shard(shards[0], layer_types)
    assert own.layer_indices == (0, 1, 2, 3)
    assert own.memory_savings_ratio() == 0.0


def test_pipeline_filter_pattern_matches_route_walk():
    """The filtered layer pattern (real-block vs stub) should agree with
    the walker's LOCAL set for each shard."""
    layer_types = ("M", "*", "E", "M", "*", "E")
    specs = parse_inference_shards_spec(
        "tp=2,kinds=M+tp=2,kinds=*+tp=2,kinds=E", world_size=6
    )
    shards = _make_shards_directly(specs, layer_types)
    pattern = "".join(layer_types)

    for shard in shards:
        own = ownership_for_shard(shard, layer_types)
        filtered = filter_layer_pattern(pattern, own)
        # Every owned position keeps its kind; every other becomes "_".
        for i, c in enumerate(filtered):
            if i in own.layer_indices:
                assert c == pattern[i], f"shard {shard.index} layer {i}: kept-but-changed"
            else:
                assert c == "_", f"shard {shard.index} layer {i}: not stubbed"


def test_pipeline_explain_route_renders_full_chain():
    """Smoke check on the human-readable rendering."""
    layer_types = ("M", "*", "E")
    specs = parse_inference_shards_spec(
        "tp=1,kinds=M+tp=1,kinds=*+tp=1,kinds=E", world_size=3
    )
    shards = _make_shards_directly(specs, layer_types)
    route = plan_route(shards, layer_type_list=layer_types)
    text = explain_route(route, layer_types)
    assert "0[M:0]" in text and "1[*:1]" in text and "2[E:2]" in text
    assert "->" in text
