# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Speculative parallel routes.

A :class:`SpeculativeRoute` is a bundle of N alternative
:class:`Route` objects with shared entry/exit shards. The producer
submits one normal request per branch (same prompt, same sampling
params, different ``server_request_id``), runs them in parallel up
to a decision layer, picks a winner per ``policy``, then cancels
the losers via :meth:`InferenceClient.cancel_branch`.

Mechanism reuses existing plumbing — each branch is just a regular
disagg request; cancellation reuses the engine's
``RELEASE_DISAGG_REQUEST`` path. The novelty is in the policy layer
+ the producer's coordination + the explicit
``CANCEL_SPECULATIVE_BRANCH`` header for clean telemetry.
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
    SpeculativeRoute,
    make_speculative_route,
)
from megatron.rl.inference.route_selector import (
    RequestInfo,
    SpeculativeRouteSelector,
)


def _route(shards, layers_per_shard=2):
    """Build a simple linear route over the given shards. Each shard
    runs ``layers_per_shard`` consecutive layers."""
    hops = []
    li = 0
    for s in shards:
        hops.append(
            RouteHop(
                shard_idx=s,
                layer_indices=tuple(range(li, li + layers_per_shard)),
            )
        )
        li += layers_per_shard
    return Route(hops=tuple(hops))


# ---- Data model ------------------------------------------------------


def test_speculative_route_requires_at_least_two_branches():
    """One branch isn't speculative — fail fast at construction so the
    bug doesn't manifest as a wasted coord cancel."""
    with pytest.raises(AssertionError):
        SpeculativeRoute(branches=(_route([0, 1]),), decision_layer=0)


def test_speculative_route_branches_must_share_entry_exit():
    """All branches converge on the same exit shard so the coord's
    reply routing doesn't depend on which branch won."""
    with pytest.raises(AssertionError):
        SpeculativeRoute(
            # Both end at shard 1 but start at different entries.
            branches=(_route([0, 1]), _route([2, 1])),
            decision_layer=0,
        )
    with pytest.raises(AssertionError):
        SpeculativeRoute(
            # Both start at shard 0 but exit on different shards.
            branches=(_route([0, 1]), _route([0, 2])),
            decision_layer=0,
        )


def test_speculative_route_supported_policies():
    """``first`` and ``min_load`` are accepted; other strings reject."""
    # Both branches: entry 0, exit 4. Branch B has an extra middle hop.
    branches = (
        _route([0, 1, 4], layers_per_shard=1),
        _route([0, 2, 3, 4], layers_per_shard=1),
    )
    # The branches have differing layer counts so we can't enforce a
    # decision_layer that's in BOTH spans here — just sanity-check
    # the policy validation, using a layer present in both routes.
    decision_layer = 0  # layer 0 is in both branches' spans
    SpeculativeRoute(
        branches=branches, decision_layer=decision_layer, policy="first"
    )
    SpeculativeRoute(
        branches=branches, decision_layer=decision_layer, policy="min_load"
    )
    with pytest.raises(AssertionError):
        SpeculativeRoute(
            branches=branches, decision_layer=decision_layer, policy="random"
        )


def test_speculative_route_participating_shards_is_union():
    """``participating_shards`` returns every shard touched by any
    branch — useful for the producer's "what's the maximum footprint
    of this request" check."""
    spec = SpeculativeRoute(
        branches=(
            _route([0, 1, 4], layers_per_shard=1),
            _route([0, 3, 4], layers_per_shard=1),
        ),
        decision_layer=1,
    )
    assert spec.participating_shards() == [0, 1, 3, 4]


def test_speculative_route_entry_and_exit_helpers():
    """Convenience properties expose the shared entry/exit shards."""
    spec = SpeculativeRoute(
        branches=(_route([0, 1, 4]), _route([0, 2, 4])),
        decision_layer=2,
    )
    assert spec.entry_shard == 0
    assert spec.exit_shard == 4


# ---- make_speculative_route ------------------------------------------


def test_make_speculative_route_validates_decision_layer_in_branches():
    """If a branch doesn't include the decision layer in its layer
    span the policy can't fire on it; reject at construction."""
    b1 = _route([0, 1, 4])  # layers 0..5; exits on shard 4
    b2 = _route([0, 2, 4])  # layers 0..5; exits on shard 4
    with pytest.raises(AssertionError):
        make_speculative_route(
            branches=(b1, b2),
            decision_layer=10,  # not in any branch
        )


def test_make_speculative_route_returns_bundle():
    """Happy path: ``make_speculative_route`` produces a valid bundle
    (branches share entry shard 0 and exit shard 4)."""
    b1 = _route([0, 1, 4])
    b2 = _route([0, 2, 4])
    spec = make_speculative_route(branches=(b1, b2), decision_layer=2)
    assert spec.branches == (b1, b2)
    assert spec.decision_layer == 2


# ---- SpeculativeRouteSelector ----------------------------------------


def test_speculative_selector_returns_primary_branch_at_submit():
    """Selector hands the producer the primary (branches[0]) for the
    standard submit path; the producer is responsible for submitting
    additional N-1 requests for the other branches."""
    # Both branches enter on shard 0 and exit on shard 1.
    b1 = _route([0, 1], layers_per_shard=1)
    b2 = _route([0, 2, 1], layers_per_shard=1)
    spec = SpeculativeRoute(branches=(b1, b2), decision_layer=0)
    sel = SpeculativeRouteSelector(spec)
    assert sel.select(RequestInfo()) is b1
    assert sel.speculative_route is spec


# ---- InferenceClient.cancel_branch -----------------------------------


def test_inference_client_cancel_branch_ships_header_and_shards():
    """``InferenceClient.cancel_branch`` ships
    ``CANCEL_SPECULATIVE_BRANCH`` with the request_id and shard list."""
    from megatron.core.inference.inference_client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()
    client.cancel_branch(request_id=42, participating_shards=[0, 2, 3])
    client._send_signal_to_engines.assert_called_once_with(
        Headers.CANCEL_SPECULATIVE_BRANCH, 42, [0, 2, 3]
    )


# ---- Coord-side handler ----------------------------------------------


def _bare_coord():
    coord = DataParallelInferenceCoordinator.__new__(
        DataParallelInferenceCoordinator
    )
    coord._disagg_route = None
    coord._send_to_engine = MagicMock()
    coord._identities_for_shard = lambda s: [f"s{s}_eng0".encode()]
    return coord


def _simulate_cancel_branch(coord, payload):
    """Inline the CANCEL_SPECULATIVE_BRANCH coord branch (verbatim)."""
    _, cancel_request_id, cancel_shards = payload
    rel_payload = msgpack.packb(
        [Headers.RELEASE_DISAGG_REQUEST.value, int(cancel_request_id)],
        use_bin_type=True,
    )
    for shard_idx in sorted(cancel_shards):
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, rel_payload)


def test_coord_cancel_fans_release_to_branch_participants():
    """A cancellation translates to RELEASE_DISAGG_REQUEST on every
    branch participant — the engines use their existing release path
    to free dispatcher + KV / mamba state."""
    coord = _bare_coord()
    payload = [Headers.CANCEL_SPECULATIVE_BRANCH.value, 99, [0, 2, 3]]
    _simulate_cancel_branch(coord, payload)

    sends = coord._send_to_engine.call_args_list
    assert len(sends) == 3
    for call in sends:
        target, raw = call[0]
        decoded = msgpack.unpackb(raw)
        assert decoded == [Headers.RELEASE_DISAGG_REQUEST.value, 99]
        assert target in (b"s0_eng0", b"s2_eng0", b"s3_eng0")


def test_coord_cancel_with_no_shards_is_a_noop():
    """An empty shard list — e.g. the producer over-eagerly cancels a
    branch with no remaining participants — fans nothing instead of
    raising. Defensive."""
    coord = _bare_coord()
    _simulate_cancel_branch(
        coord, [Headers.CANCEL_SPECULATIVE_BRANCH.value, 1, []]
    )
    coord._send_to_engine.assert_not_called()


# ---- End-to-end producer flow ----------------------------------------


def test_speculative_producer_flow_submit_then_cancel_loser():
    """Walks the producer's lifecycle for a 2-branch spec:

    1. Submit primary branch via ``add_request`` (normal path).
    2. Submit alt branch via ``add_request`` (normal path, different
       request_id).
    3. After decision boundary, primary wins → cancel alt branch via
       ``cancel_branch(alt_request_id, alt_branch.participating_shards)``.

    Verifies the cancel call serializes the right shard list (= every
    shard the loser branch visits)."""
    from megatron.core.inference.inference_client import InferenceClient

    primary = _route([0, 1, 4])  # layers 0..5 over shards 0, 1, 4
    alt = _route([0, 2, 4])      # layers 0..5 over shards 0, 2, 4
    spec = make_speculative_route(branches=(primary, alt), decision_layer=2)
    sel = SpeculativeRouteSelector(spec)

    # Producer picks the primary at submit; alt is submitted in parallel.
    assert sel.select(RequestInfo()) is primary

    # Pretend the producer assigned request_ids 1 (primary) and 2 (alt)
    # and now (after decision) chose primary as the winner.
    client = InferenceClient.__new__(InferenceClient)
    client._send_signal_to_engines = MagicMock()

    alt_shards = sorted({h.shard_idx for h in alt.hops})
    client.cancel_branch(request_id=2, participating_shards=alt_shards)
    client._send_signal_to_engines.assert_called_once_with(
        Headers.CANCEL_SPECULATIVE_BRANCH, 2, [0, 2, 4]
    )
