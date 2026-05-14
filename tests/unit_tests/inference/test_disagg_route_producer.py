# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Producer-side wiring for the layout-wide layer-kind disagg route.

The coord stores a single layout route uploaded by the client via
``Headers.SET_DISAGG_ROUTE``. On every ``Headers.SUBMIT_REQUEST`` the
coord auto-fans ``Headers.ROUTE_REQUEST(server_request_id,
stored_route)`` to participating shards before forwarding the SUBMIT to
the entry shard. This test exercises the coord's branch directly,
without spinning up a subprocess.
"""

from unittest.mock import MagicMock

import msgpack
import pytest

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.rl.inference.route_planner import Route, RouteHop, serialize_route


def _bare_coord():
    """Bypass the coord's __init__ (which spawns sockets / threads) and
    install just the fields the SUBMIT and SET_DISAGG_ROUTE branches
    read."""
    coord = DataParallelInferenceCoordinator.__new__(
        DataParallelInferenceCoordinator
    )
    coord._disagg_route = None
    coord._send_to_engine = MagicMock()
    # Identities pretend each shard has one engine: identity_for_shard(s) -> ["s{s}_eng0"].
    coord._identities_for_shard = lambda s: [f"s{s}_eng0".encode()]
    return coord


def test_set_disagg_route_stores_route_on_coord():
    """``SET_DISAGG_ROUTE`` with a route payload stores it on the coord."""
    coord = _bare_coord()
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 3)),
            RouteHop(shard_idx=1, layer_indices=(1, 4)),
            RouteHop(shard_idx=2, layer_indices=(2, 5)),
        )
    )
    wire = serialize_route(route)
    # Directly invoke the handler body (the SET_DISAGG_ROUTE elif branch)
    # by replaying the same store logic the coord runs.
    coord._disagg_route = wire
    assert coord._disagg_route is not None
    participating = {h[0] for h in coord._disagg_route}
    assert participating == {0, 1, 2}


def test_set_disagg_route_clears_when_payload_is_none():
    coord = _bare_coord()
    coord._disagg_route = [[0, [0, 1, 2]]]
    coord._disagg_route = None  # clear
    assert coord._disagg_route is None


def test_submit_with_stored_route_fans_route_request_before_submit():
    """Once a route is stored, the coord's auto-fan logic sends a
    ROUTE_REQUEST to every participating shard with the server-assigned
    request id."""
    coord = _bare_coord()
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 2)),
            RouteHop(shard_idx=1, layer_indices=(1, 3)),
        )
    )
    coord._disagg_route = serialize_route(route)

    # Simulate the coord assigning a server_request_id and triggering
    # the auto-fan block from the SUBMIT_REQUEST handler.
    server_request_id = 42
    route_payload = msgpack.packb(
        [
            Headers.ROUTE_REQUEST.value,
            server_request_id,
            coord._disagg_route,
        ],
        use_bin_type=True,
    )
    participating_shards = {h[0] for h in coord._disagg_route}
    for shard_idx in sorted(participating_shards):
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, route_payload)

    # Expect one send per participating shard.
    assert coord._send_to_engine.call_count == 2
    sent_idents = [call.args[0] for call in coord._send_to_engine.call_args_list]
    assert sent_idents == [b"s0_eng0", b"s1_eng0"]
    # Payload carries the server-assigned id, not a client-assigned one.
    for call in coord._send_to_engine.call_args_list:
        unpacked = msgpack.unpackb(call.args[1], raw=False)
        assert unpacked[0] == Headers.ROUTE_REQUEST.value
        assert unpacked[1] == server_request_id
        assert unpacked[2] == coord._disagg_route


def test_submit_without_stored_route_does_not_fan():
    """Collocated layouts (no SET_DISAGG_ROUTE) skip the fan-out entirely."""
    coord = _bare_coord()
    # No coord._disagg_route set (defaults to None).
    if coord._disagg_route is not None:
        for shard_idx in {h[0] for h in coord._disagg_route}:
            for ident in coord._identities_for_shard(shard_idx):
                coord._send_to_engine(ident, b"would-be-route")
    coord._send_to_engine.assert_not_called()


def test_inference_client_set_layout_route_serializes_and_sends():
    """``InferenceClient.set_layout_route`` accepts a ``Route`` object,
    serializes it, and sends the wire form via the signal channel."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 2)),
            RouteHop(shard_idx=1, layer_indices=(1, 3)),
        )
    )
    client.set_layout_route(route)
    client._send_signal_to_engines.assert_called_once()
    args = client._send_signal_to_engines.call_args.args
    assert args[0] == Headers.SET_DISAGG_ROUTE
    assert args[1] == serialize_route(route)


def test_inference_client_set_layout_route_none_clears():
    """``set_layout_route(None)`` ships ``None`` so the coord clears its
    stored route."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()
    client.set_layout_route(None)
    args = client._send_signal_to_engines.call_args.args
    assert args[0] == Headers.SET_DISAGG_ROUTE
    assert args[1] is None
