# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mid-rollout dynamic route reselection.

A stateful :class:`RouteSelector` (e.g.
:class:`PrefillDecodeFlipSelector`) can decide between decode steps
that the in-flight request should walk a different route. The
producer calls :meth:`InferenceClient.reselect_route(request_id,
old_route, new_route)`; the coord computes
``union(participants(old), participants(new))`` and fans:

- ``RESELECT_REQUEST_ROUTE(request_id, new_hops)`` to engines on
  shards present in both routes (common).
- ``RELEASE_DISAGG_REQUEST(request_id)`` to engines on shards
  present only in ``old`` (drop).
- A warning + skip for shards present only in ``new`` (add — v1
  doesn't carry prompt + sampling params for arbitrary additions;
  reroutes that shrink or keep the participant set are fully wired).

Engines that receive ``RESELECT_REQUEST_ROUTE`` overwrite the
dispatcher for that request via the existing route handler.
"""

from unittest.mock import MagicMock

import msgpack
import pytest

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    serialize_route,
)
from megatron.rl.inference.route_selector import (
    LayoutRouteSelector,
    PrefillDecodeFlipSelector,
    PromptLengthRouteSelector,
    RequestInfo,
    RouteSelector,
    StepInfo,
    StickyShardRouteSelector,
    select_for_step,
)


# ---- Selector protocol extension --------------------------------------


def test_step_info_dataclass_defaults():
    """StepInfo carries request_id + decode_step + current_route. The
    prompt_length field defaults to None for selectors that don't need
    it."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0,)),))
    info = StepInfo(request_id=7, decode_step=3, current_route=route)
    assert info.request_id == 7
    assert info.decode_step == 3
    assert info.current_route is route
    assert info.prompt_length is None


def test_layout_selector_select_for_step_is_no_op():
    """Default selector never reroutes; every step returns None."""
    sel = LayoutRouteSelector()
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0,)),))
    for step in (0, 1, 5, 100):
        assert sel.select_for_step(
            StepInfo(request_id=1, decode_step=step, current_route=route)
        ) is None


def test_prefill_decode_flip_swaps_route_at_flip_step():
    """The flip selector picks ``prefill_route`` at submit and switches
    to ``decode_route`` exactly once at ``flip_at_step``."""
    prefill = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1, 2)),))
    decode = Route(hops=(RouteHop(shard_idx=1, layer_indices=(0, 1, 2)),))
    sel = PrefillDecodeFlipSelector(prefill, decode, flip_at_step=1)
    # Submit-time picks prefill.
    assert sel.select(RequestInfo(prompt_tokens=[0] * 100)) is prefill
    # Decode step 0: no flip yet.
    assert sel.select_for_step(
        StepInfo(request_id=1, decode_step=0, current_route=prefill)
    ) is None
    # Decode step 1: flip.
    assert sel.select_for_step(
        StepInfo(request_id=1, decode_step=1, current_route=prefill)
    ) is decode
    # Subsequent steps: no further flips (avoid churn).
    assert sel.select_for_step(
        StepInfo(request_id=1, decode_step=2, current_route=decode)
    ) is None


def test_concrete_selectors_satisfy_runtime_checkable_protocol():
    """Protocol membership stays minimal (just ``select``). Every
    concrete selector — stateless or stateful — satisfies it."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0,)),))
    for sel in (
        LayoutRouteSelector(),
        StickyShardRouteSelector(route),
        PromptLengthRouteSelector([(100, route)]),
        PrefillDecodeFlipSelector(route, route),
    ):
        assert isinstance(sel, RouteSelector), (
            f"{type(sel).__name__} must satisfy RouteSelector"
        )


def test_select_for_step_helper_falls_back_to_none_for_stateless():
    """User-defined selectors that only implement ``select`` keep
    working with the reroute helper — it returns ``None`` (no reroute)
    without raising."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0,)),))

    class _MinimalSelector:
        def select(self, request_info):
            return None

    result = select_for_step(
        _MinimalSelector(),
        StepInfo(request_id=1, decode_step=5, current_route=route),
    )
    assert result is None


def test_select_for_step_helper_dispatches_to_stateful_selector():
    """When the selector does implement ``select_for_step`` the helper
    forwards the call and returns the selector's verdict."""
    prefill = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0,)),))
    decode = Route(hops=(RouteHop(shard_idx=1, layer_indices=(0,)),))
    sel = PrefillDecodeFlipSelector(prefill, decode, flip_at_step=2)
    info = StepInfo(request_id=1, decode_step=2, current_route=prefill)
    assert select_for_step(sel, info) is decode


# ---- Coord-side fan-out -----------------------------------------------


def _bare_coord_with_sender(sender=b"client0"):
    """Coord stub with the wires the RESELECT_REQUEST_ROUTE branch
    actually reads. The branch sends to engines via ``_send_to_engine``
    so we capture calls there."""
    coord = DataParallelInferenceCoordinator.__new__(
        DataParallelInferenceCoordinator
    )
    coord._disagg_route = None
    coord._send_to_engine = MagicMock()
    coord._identities_for_shard = lambda s: [f"s{s}_eng0".encode()]
    return coord


def _route(shards):
    """Build a linear route over the given shard sequence (one layer
    per hop; the layer index isn't material for these tests)."""
    return Route(
        hops=tuple(
            RouteHop(shard_idx=s, layer_indices=(i,))
            for i, s in enumerate(shards)
        )
    )


def _payload(old, new, request_id=42):
    return [
        Headers.RESELECT_REQUEST_ROUTE.value,
        int(request_id),
        serialize_route(old),
        serialize_route(new),
    ]


def _simulate_coord_branch(coord, sender_identity, payload):
    """Inline the RESELECT_REQUEST_ROUTE coord branch.

    Pulled verbatim from data_parallel_inference_coordinator.py so the
    test stays self-contained without a live coord process.
    """
    _, rr_request_id, old_hops, new_hops = payload
    old_shards = {h[0] for h in old_hops}
    new_shards = {h[0] for h in new_hops}
    common = old_shards & new_shards
    drop = old_shards - new_shards
    add = new_shards - old_shards

    rr_payload = msgpack.packb(
        [
            Headers.RESELECT_REQUEST_ROUTE.value,
            int(rr_request_id),
            new_hops,
        ],
        use_bin_type=True,
    )
    rel_payload = msgpack.packb(
        [Headers.RELEASE_DISAGG_REQUEST.value, int(rr_request_id)],
        use_bin_type=True,
    )
    for shard_idx in sorted(common):
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, rr_payload)
    for shard_idx in sorted(drop):
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, rel_payload)
    return {"common": common, "drop": drop, "add": add}


def test_coord_reselect_swap_two_shards_one_overlap():
    """Old = (0, 1), new = (0, 2). Coord sends RESELECT to shard 0
    (common), RELEASE to shard 1 (dropped), warns about shard 2 (add)."""
    coord = _bare_coord_with_sender()
    old = _route([0, 1])
    new = _route([0, 2])
    result = _simulate_coord_branch(
        coord, sender_identity=b"client0", payload=_payload(old, new)
    )

    sends = coord._send_to_engine.call_args_list
    targets = [args[0][0] for args in sends]
    headers = [msgpack.unpackb(args[0][1])[0] for args in sends]

    # Shard 0 (common) → RESELECT
    assert (b"s0_eng0", Headers.RESELECT_REQUEST_ROUTE.value) in list(
        zip(targets, headers)
    )
    # Shard 1 (drop) → RELEASE
    assert (b"s1_eng0", Headers.RELEASE_DISAGG_REQUEST.value) in list(
        zip(targets, headers)
    )
    # Shard 2 (add) → no send in v1
    assert b"s2_eng0" not in targets
    assert result == {"common": {0}, "drop": {1}, "add": {2}}


def test_coord_reselect_pure_drop_releases_old_only_shards():
    """Old = (0, 1, 2), new = (0,). Coord RESELECTs shard 0, RELEASEs
    shards 1 and 2."""
    coord = _bare_coord_with_sender()
    old = _route([0, 1, 2])
    new = _route([0])
    _simulate_coord_branch(coord, b"client0", _payload(old, new))

    sends = coord._send_to_engine.call_args_list
    by_target = {args[0][0]: msgpack.unpackb(args[0][1]) for args in sends}
    assert by_target[b"s0_eng0"][0] == Headers.RESELECT_REQUEST_ROUTE.value
    assert by_target[b"s1_eng0"][0] == Headers.RELEASE_DISAGG_REQUEST.value
    assert by_target[b"s2_eng0"][0] == Headers.RELEASE_DISAGG_REQUEST.value


def test_coord_reselect_identical_old_and_new_still_reselects_common():
    """Identity reroute (old == new): the coord still fans
    RESELECT_REQUEST_ROUTE to every shard. The engine handler is
    expected to be idempotent — re-installing the same route is a
    no-op apart from rebuilding the layer plan, which can be useful
    if the dispatcher needs to be reset."""
    coord = _bare_coord_with_sender()
    route = _route([0, 1])
    _simulate_coord_branch(coord, b"client0", _payload(route, route))

    sends = coord._send_to_engine.call_args_list
    headers = [msgpack.unpackb(args[0][1])[0] for args in sends]
    assert headers.count(Headers.RESELECT_REQUEST_ROUTE.value) == 2
    assert Headers.RELEASE_DISAGG_REQUEST.value not in headers


def test_coord_reselect_carries_new_route_on_wire():
    """The RESELECT payload sent to common shards carries the NEW
    route hops (not the old one), so the engine's route handler
    deserializes the new plan."""
    coord = _bare_coord_with_sender()
    old = _route([0, 1])
    new = _route([0, 2])
    _simulate_coord_branch(coord, b"client0", _payload(old, new, request_id=99))

    new_hops_wire = serialize_route(new)
    # Find the RESELECT message sent to shard 0 and confirm its hops.
    for call in coord._send_to_engine.call_args_list:
        target, raw = call[0]
        decoded = msgpack.unpackb(raw)
        if decoded[0] == Headers.RESELECT_REQUEST_ROUTE.value:
            assert decoded[1] == 99
            assert decoded[2] == new_hops_wire


# ---- InferenceClient API ----------------------------------------------


def test_inference_client_reselect_route_serializes_both_routes():
    """``InferenceClient.reselect_route`` serializes Route objects
    automatically and ships request_id + old_wire + new_wire over
    ``_send_signal_to_engines``."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()

    old = _route([0, 1])
    new = _route([0, 2])
    client.reselect_route(request_id=7, old_route=old, new_route=new)

    client._send_signal_to_engines.assert_called_once_with(
        Headers.RESELECT_REQUEST_ROUTE,
        7,
        serialize_route(old),
        serialize_route(new),
    )


def test_inference_client_reselect_route_rejects_none_new_route():
    """Reroute with no destination is the release path, not a reselect
    — fail fast at the client so the bug surfaces near the caller."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()
    with pytest.raises(AssertionError):
        client.reselect_route(
            request_id=1, old_route=_route([0]), new_route=None
        )


def test_inference_client_reselect_route_accepts_prewired_routes():
    """Wire-form routes (lists of hop lists) are accepted directly so
    callers that pre-serialize don't double-serialize."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()

    old_wire = serialize_route(_route([0, 1]))
    new_wire = serialize_route(_route([0, 2]))
    client.reselect_route(request_id=3, old_route=old_wire, new_route=new_wire)
    client._send_signal_to_engines.assert_called_once_with(
        Headers.RESELECT_REQUEST_ROUTE, 3, old_wire, new_wire
    )
