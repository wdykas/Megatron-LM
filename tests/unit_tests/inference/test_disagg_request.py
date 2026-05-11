# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the disagg request bundle wire format."""

import pytest

from megatron.core.inference.disagg_request import DisaggRequestBundle
from megatron.rl.inference.route_planner import Route, RouteHop


def _make_route() -> Route:
    hops = (
        RouteHop(shard_idx=0, layer_indices=(0, 1), src_shard=None),
        RouteHop(shard_idx=1, layer_indices=(2, 3), src_shard=0),
        RouteHop(shard_idx=2, layer_indices=(4,), src_shard=1),
    )
    return Route(hops=hops, entry_shard=0, exit_shard=2)


def test_bundle_to_wire_msgpack_compatible():
    bundle = DisaggRequestBundle(
        request_id=42,
        route=_make_route(),
        prompt_tokens=[1, 2, 3],
        sampling_params={"temperature": 0.7},
    )
    wire = bundle.to_wire()
    assert wire["request_id"] == 42
    # Route on the wire is a list of [shard, [layers], src_shard].
    assert isinstance(wire["route"], list) and len(wire["route"]) == 3
    assert wire["route"][0] == [0, [0, 1], None]
    assert wire["route"][1] == [1, [2, 3], 0]
    assert wire["route"][2] == [2, [4], 1]
    assert wire["prompt_tokens"] == [1, 2, 3]
    assert wire["sampling_params"] == {"temperature": 0.7}


def test_bundle_roundtrip():
    bundle = DisaggRequestBundle(
        request_id=7,
        route=_make_route(),
        prompt_tokens=[10, 11],
        sampling_params={"top_p": 0.95, "top_k": 40},
        generated_tokens=[100, 101, 102],
    )
    restored = DisaggRequestBundle.from_wire(bundle.to_wire())
    assert restored.request_id == bundle.request_id
    assert restored.route == bundle.route
    assert restored.prompt_tokens == bundle.prompt_tokens
    assert restored.sampling_params == bundle.sampling_params
    assert restored.generated_tokens == bundle.generated_tokens


def test_bundle_shards_participating_dedup():
    """When the route revisits a shard, ``shards_participating`` still
    lists it once. The coord's fan-out should send ROUTE_REQUEST to
    each shard at most once."""
    hops = (
        RouteHop(shard_idx=0, layer_indices=(0, 1), src_shard=None),
        RouteHop(shard_idx=1, layer_indices=(2,), src_shard=0),
        RouteHop(shard_idx=0, layer_indices=(3,), src_shard=1),  # revisit
    )
    route = Route(hops=hops, entry_shard=0, exit_shard=0)
    bundle = DisaggRequestBundle(request_id=1, route=route)
    parts = bundle.shards_participating()
    assert set(parts) == {0, 1}


def test_bundle_expects_reply_from_exit_shard():
    bundle = DisaggRequestBundle(request_id=1, route=_make_route())
    assert bundle.expects_reply_from() == 2  # exit_shard


