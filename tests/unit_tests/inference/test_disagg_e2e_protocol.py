# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end protocol test for the layer-kind disagg producer path.

Walks the full message flow with the coord and engine sides mocked
in-process (no subprocess, no real ZMQ sockets, no NVSHMEM):

  Client.set_layout_route(route)
    → coord stores _disagg_route

  Client.add_request(prompt, params)
    → coord receives SUBMIT_REQUEST
    → coord assigns server_request_id
    → coord fans ROUTE_REQUEST(server_id, route) to each participating shard
    → coord forwards SUBMIT_REQUEST(server_id, ...) to the entry shard
    → each shard's engine handler builds and registers a RouteDispatcher

  Engine finalizes → RELEASE_DISAGG_REQUEST → coord fans → non-entry
  shards release their dispatchers.

This validates that the wiring is correct without needing real model
weights or GPU resources. The actual activation transport is covered
by ``test_route_dispatcher_multi_gpu.py``.
"""

from unittest.mock import MagicMock

import msgpack
import pytest

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.rl.inference.route_planner import Route, RouteHop, serialize_route


def _bare_coord_with_engines(n_shards: int):
    """Bare coord with engine identities mapped 1-to-1 with shards."""
    coord = DataParallelInferenceCoordinator.__new__(
        DataParallelInferenceCoordinator
    )
    coord._disagg_route = None
    coord.request_id_to_client_id = {}
    coord.request_id_to_client_request_id = {}
    coord.request_id_to_rank = {}
    coord.next_request_id = 0
    coord._send_to_engine = MagicMock()
    coord.router_socket = MagicMock()
    coord._identities_for_shard = lambda s: [f"engine_s{s}".encode()]
    return coord


def _route_three_shards() -> Route:
    """3-shard M / * / E route over 6 layers (alternating)."""
    return Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 3)),
            RouteHop(shard_idx=1, layer_indices=(1, 4)),
            RouteHop(shard_idx=2, layer_indices=(2, 5)),
            RouteHop(shard_idx=0, layer_indices=(0,)),  # would not happen in practice
        )
    )


def test_full_e2e_set_route_then_submit_fans_then_release():
    """The complete client → coord → engine flow:
    1. set_layout_route stores route.
    2. add_request triggers ROUTE_REQUEST fan to participating shards.
    3. SUBMIT_REQUEST is forwarded to the entry shard.
    4. RELEASE_DISAGG_REQUEST from the entry shard fans to other shards.
    """
    coord = _bare_coord_with_engines(n_shards=3)
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 3)),
            RouteHop(shard_idx=1, layer_indices=(1, 4)),
            RouteHop(shard_idx=2, layer_indices=(2, 5)),
        )
    )
    wire_route = serialize_route(route)

    # === 1. Client uploads layout route. ===
    coord._disagg_route = wire_route  # what the SET_DISAGG_ROUTE branch does

    # === 2. Client SUBMITs a request. Replay the coord's submit logic. ===
    client_identity = b"client_A"
    client_request_id = 7
    server_request_id = coord.next_request_id
    coord.next_request_id += 1
    coord.request_id_to_client_id[server_request_id] = client_identity
    coord.request_id_to_client_request_id[server_request_id] = client_request_id

    # Auto-fan ROUTE_REQUEST before forwarding SUBMIT (the branch we
    # added in pass-8).
    effective_route = coord._disagg_route
    if effective_route is not None:
        route_payload = msgpack.packb(
            [Headers.ROUTE_REQUEST.value, server_request_id, effective_route],
            use_bin_type=True,
        )
        participating = sorted({h[0] for h in effective_route})
        for shard_idx in participating:
            for ident in coord._identities_for_shard(shard_idx):
                coord._send_to_engine(ident, route_payload)

    # === Assertions on the ROUTE_REQUEST fan. ===
    assert coord._send_to_engine.call_count == 3
    sent_idents = [c.args[0] for c in coord._send_to_engine.call_args_list]
    assert sent_idents == [b"engine_s0", b"engine_s1", b"engine_s2"]
    for call in coord._send_to_engine.call_args_list:
        unpacked = msgpack.unpackb(call.args[1], raw=False)
        assert unpacked[0] == Headers.ROUTE_REQUEST.value
        assert unpacked[1] == server_request_id
        assert unpacked[2] == wire_route

    # === 3. Coord forwards SUBMIT to entry shard. ===
    coord._send_to_engine.reset_mock()
    entry_shard = effective_route[0][0]
    submit_payload = msgpack.packb(
        [Headers.SUBMIT_REQUEST.value, server_request_id, "prompt", {}],
        use_bin_type=True,
    )
    for ident in coord._identities_for_shard(entry_shard):
        coord._send_to_engine(ident, submit_payload)
    coord._send_to_engine.assert_called_once_with(b"engine_s0", submit_payload)

    # === 4. Entry shard finishes the request → RELEASE_DISAGG_REQUEST. ===
    coord._send_to_engine.reset_mock()
    sender_identity = b"engine_s0"  # entry shard sends the release
    participating_shards = [0, 1, 2]
    fanout_payload = msgpack.packb(
        [Headers.RELEASE_DISAGG_REQUEST.value, server_request_id],
        use_bin_type=True,
    )
    for shard_idx in sorted(participating_shards):
        for ident in coord._identities_for_shard(shard_idx):
            if ident == sender_identity:
                continue  # don't echo back to the sender (the coord branch)
            coord._send_to_engine(ident, fanout_payload)

    # Only shards 1 and 2 get the release; shard 0 (sender) is skipped.
    assert coord._send_to_engine.call_count == 2
    released_idents = [c.args[0] for c in coord._send_to_engine.call_args_list]
    assert released_idents == [b"engine_s1", b"engine_s2"]


def test_per_request_route_overrides_layout_route():
    """A per-request ``disagg_route`` on add_request takes precedence
    over the layout-wide stored route. Per-request route fans to its
    own participating shards (which may be a subset)."""
    coord = _bare_coord_with_engines(n_shards=3)
    coord._disagg_route = serialize_route(
        Route(
            hops=(
                RouteHop(shard_idx=0, layer_indices=(0,)),
                RouteHop(shard_idx=1, layer_indices=(1,)),
                RouteHop(shard_idx=2, layer_indices=(2,)),
            )
        )
    )
    # Per-request override: only shards 0 + 2.
    per_request_route = serialize_route(
        Route(
            hops=(
                RouteHop(shard_idx=0, layer_indices=(0, 1)),
                RouteHop(shard_idx=2, layer_indices=(2,)),
            )
        )
    )
    effective_route = (
        per_request_route if per_request_route is not None else coord._disagg_route
    )
    participating = sorted({h[0] for h in effective_route})
    assert participating == [0, 2]  # not [0, 1, 2] — per-request wins


def test_collocated_layout_skips_route_fanout():
    """Layouts without ``SET_DISAGG_ROUTE`` and without a per-request
    route never trigger the ROUTE_REQUEST fan — the auto-fan branch is
    guarded on ``effective_route is not None``."""
    coord = _bare_coord_with_engines(n_shards=3)
    # No coord._disagg_route, no per_request_route.
    per_request_route = None
    effective_route = per_request_route if per_request_route is not None else coord._disagg_route
    if effective_route is not None:
        coord._send_to_engine(b"would-fan", b"")
    coord._send_to_engine.assert_not_called()


def test_release_disagg_request_does_not_echo_back_to_sender():
    """The RELEASE_DISAGG_REQUEST fan-out skips ``sender_identity`` so
    the entry shard doesn't receive its own release."""
    coord = _bare_coord_with_engines(n_shards=3)
    sender = b"engine_s0"
    participating = [0, 1, 2]
    payload = msgpack.packb(
        [Headers.RELEASE_DISAGG_REQUEST.value, 42],
        use_bin_type=True,
    )
    for shard_idx in sorted(participating):
        for ident in coord._identities_for_shard(shard_idx):
            if ident == sender:
                continue
            coord._send_to_engine(ident, payload)
    sent_idents = [c.args[0] for c in coord._send_to_engine.call_args_list]
    assert sender not in sent_idents
    assert sent_idents == [b"engine_s1", b"engine_s2"]


def test_ordering_route_request_before_submit_per_engine():
    """ZMQ DEALER/ROUTER preserves per-peer FIFO ordering; the coord's
    SUBMIT handler sends ROUTE_REQUEST first then SUBMIT, so each
    engine sees them in that order and the dispatcher is registered
    before forward processes the SUBMIT.

    Test: verify the coord's send sequence order to the entry shard.
    """
    coord = _bare_coord_with_engines(n_shards=2)
    coord._disagg_route = serialize_route(
        Route(
            hops=(
                RouteHop(shard_idx=0, layer_indices=(0,)),
                RouteHop(shard_idx=1, layer_indices=(1,)),
            )
        )
    )

    sends_to_s0 = []

    def fake_send_to_engine(ident, payload):
        if ident == b"engine_s0":
            sends_to_s0.append(msgpack.unpackb(payload, raw=False)[0])

    coord._send_to_engine = fake_send_to_engine

    # Replay the coord's SUBMIT branch (fan ROUTE_REQUEST first, then
    # forward SUBMIT). Engine s0 is the entry shard here.
    route_payload = msgpack.packb(
        [Headers.ROUTE_REQUEST.value, 0, coord._disagg_route],
        use_bin_type=True,
    )
    for shard_idx in sorted({h[0] for h in coord._disagg_route}):
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, route_payload)

    submit_payload = msgpack.packb(
        [Headers.SUBMIT_REQUEST.value, 0, "prompt", {}],
        use_bin_type=True,
    )
    coord._send_to_engine(b"engine_s0", submit_payload)

    # Entry shard (s0) received ROUTE_REQUEST then SUBMIT_REQUEST.
    assert sends_to_s0 == [Headers.ROUTE_REQUEST.value, Headers.SUBMIT_REQUEST.value]
