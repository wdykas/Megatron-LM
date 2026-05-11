# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the per-request forward-pass route walker.

These tests simulate an engine iterating layer_idx=0..N-1 and verify
the walker returns the correct LayerAction at every step for various
disagg topologies.
"""

import pytest

from megatron.core.inference.route_walker import (
    LayerAction,
    LayerDecision,
    RouteWalker,
)
from megatron.rl.inference.route_planner import Route, RouteHop


def _route(*hops: tuple) -> Route:
    """hops are (shard_idx, layer_indices, src_shard) tuples."""
    rh = tuple(
        RouteHop(
            shard_idx=s,
            layer_indices=tuple(layers),
            src_shard=src,
        )
        for s, layers, src in hops
    )
    return Route(
        hops=rh,
        entry_shard=rh[0].shard_idx,
        exit_shard=rh[-1].shard_idx,
    )


def _walk(walker: RouteWalker, num_layers: int) -> list:
    """Drive the walker through every layer and collect actions.
    Auto-dispatches RECEIVE / SEND callbacks so we can see the full
    sequence in one call."""
    out: list = []
    for li in range(num_layers):
        dec = walker.before_layer(li)
        out.append((li, dec.action, dec.peer_shard))
        if dec.action is LayerAction.RECEIVE:
            walker.after_receive()
            # Re-issue: the next decision should be LOCAL on this same
            # layer.
            dec2 = walker.before_layer(li)
            out.append((li, dec2.action, dec2.peer_shard))
        elif dec.action is LayerAction.SEND:
            walker.after_send()
        elif dec.action is LayerAction.DONE:
            break
    return out


def test_walker_entry_shard_single_hop():
    """Shard 0 owns all layers (no disagg): every layer is LOCAL, then DONE."""
    route = _route((0, [0, 1, 2, 3], None))
    walker = RouteWalker(route, my_shard_idx=0)
    actions = _walk(walker, num_layers=4)
    # Layers 0..3 LOCAL, then on the next layer we'd see DONE — but the
    # walker is exhausted only after the engine asks for layer 4 (which
    # _walk doesn't because num_layers=4). Check final state instead.
    assert all(a is LayerAction.LOCAL for _, a, _ in actions)
    # Probe layer 4 -> DONE.
    dec = walker.before_layer(4)
    assert dec.action is LayerAction.DONE


def test_walker_three_kind_disagg_entry_shard():
    """3-kind alternating layout, viewed from the entry shard (shard 0).
    Shard 0 owns layers 0 and 3. Expected sequence from shard 0's POV:
        layer 0: LOCAL (hop 0)
        layer 1: SEND -> shard 1 (after hop 0 ends)
        layer 2: NOT_MY_REQUEST (it's shard 2's hop)
        layer 3: RECEIVE from shard 2, then LOCAL
        layer 4: SEND -> shard 1
        ..."""
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
        (0, [3], 2),
        (1, [4], 0),
        (2, [5], 1),
    )
    walker = RouteWalker(route, my_shard_idx=0)
    actions = _walk(walker, num_layers=6)

    # Layer 0: LOCAL (entry hop).
    assert actions[0] == (0, LayerAction.LOCAL, None)
    # Layer 1: hop 0 ended — SEND to shard 1.
    assert actions[1] == (1, LayerAction.SEND, 1)
    # Layer 2: NOT_MY_REQUEST.
    assert actions[2] == (2, LayerAction.NOT_MY_REQUEST, None)
    # Layer 3: RECEIVE from shard 2 (the hop that ran layer 2), then LOCAL.
    assert actions[3] == (3, LayerAction.RECEIVE, 2)
    assert actions[4] == (3, LayerAction.LOCAL, None)
    # Layer 4: hop 3 ended — SEND to shard 1.
    assert actions[5] == (4, LayerAction.SEND, 1)


def test_walker_middle_shard_perspective():
    """Same layout as above but viewed from shard 1.
    Shard 1 owns layers 1 and 4. Expected:
        layers 0: NOT_MY_REQUEST
        layer 1: RECEIVE from shard 0, then LOCAL
        layer 2: SEND -> shard 2
        layers 3: NOT_MY_REQUEST
        layer 4: RECEIVE from shard 0 (yes, layer 3 was shard 0), then LOCAL
        layer 5: SEND -> shard 2
    """
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
        (0, [3], 2),
        (1, [4], 0),
        (2, [5], 1),
    )
    walker = RouteWalker(route, my_shard_idx=1)
    actions = _walk(walker, num_layers=6)
    assert actions[0] == (0, LayerAction.NOT_MY_REQUEST, None)
    assert actions[1] == (1, LayerAction.RECEIVE, 0)
    assert actions[2] == (1, LayerAction.LOCAL, None)
    assert actions[3] == (2, LayerAction.SEND, 2)
    assert actions[4] == (3, LayerAction.NOT_MY_REQUEST, None)
    assert actions[5] == (4, LayerAction.RECEIVE, 0)
    assert actions[6] == (4, LayerAction.LOCAL, None)
    assert actions[7] == (5, LayerAction.SEND, 2)


def test_walker_exit_shard_done_after_last_layer():
    """The exit shard's last hop ends with DONE, not SEND."""
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
    )
    walker = RouteWalker(route, my_shard_idx=1)
    actions = _walk(walker, num_layers=4)
    # Layer 0,1 -> NOT_MY_REQUEST. Layer 2 -> RECEIVE from shard 0, then LOCAL.
    # Layer 3 -> LOCAL. After layer 3, walker.is_done() must be True.
    assert actions[-1] == (3, LayerAction.LOCAL, None)
    # Probe layer 4 -> DONE.
    assert walker.before_layer(4).action is LayerAction.DONE


def test_walker_not_my_request_when_shard_not_in_route():
    """A shard not visited by the route gets NOT_MY_REQUEST for every
    layer."""
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
    )
    walker = RouteWalker(route, my_shard_idx=5)  # not in route
    for li in range(4):
        dec = walker.before_layer(li)
        assert dec.action is LayerAction.NOT_MY_REQUEST


def test_walker_is_entry_is_exit():
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
    )
    assert RouteWalker(route, my_shard_idx=0).is_entry()
    assert not RouteWalker(route, my_shard_idx=1).is_entry()
    assert not RouteWalker(route, my_shard_idx=2).is_entry()
    assert RouteWalker(route, my_shard_idx=2).is_exit()
    assert not RouteWalker(route, my_shard_idx=0).is_exit()


def test_walker_revisit_same_shard_collapses_to_local_run():
    """Shard 0 owns layers 0,1 then comes back for 4,5 (after a detour
    through shard 1 for layers 2,3). From shard 0's POV, we expect
    two separate hops separated by NOT_MY_REQUEST."""
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
        (0, [4, 5], 1),
    )
    walker = RouteWalker(route, my_shard_idx=0)
    actions = _walk(walker, num_layers=6)
    # Layer 0: LOCAL; layer 1: LOCAL.
    assert actions[0] == (0, LayerAction.LOCAL, None)
    assert actions[1] == (1, LayerAction.LOCAL, None)
    # Layer 2 (out of hop 0): SEND to shard 1.
    assert actions[2] == (2, LayerAction.SEND, 1)
    # Layer 3: NOT_MY_REQUEST (we sent already, hop 2 is shard 1's).
    assert actions[3] == (3, LayerAction.NOT_MY_REQUEST, None)
    # Layer 4: RECEIVE from shard 1, then LOCAL.
    assert actions[4] == (4, LayerAction.RECEIVE, 1)
    assert actions[5] == (4, LayerAction.LOCAL, None)
    # Layer 5: LOCAL.
    assert actions[6] == (5, LayerAction.LOCAL, None)


def test_walker_remaining_hops_decreases_after_send():
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (0, [2], 1),
    )
    walker = RouteWalker(route, my_shard_idx=0)
    assert walker.remaining_hops() == 2
    walker.before_layer(0)  # LOCAL
    # Trigger SEND at layer 1.
    dec = walker.before_layer(1)
    assert dec.action is LayerAction.SEND
    walker.after_send()
    assert walker.remaining_hops() == 1
    # Layer 2: RECEIVE then LOCAL.
    walker.before_layer(2)
    walker.after_receive()
    walker.before_layer(2)
    # After the last LOCAL, asking for one more layer marks DONE.
    walker.before_layer(3)
    assert walker.remaining_hops() == 0


def test_walker_done_is_idempotent():
    """Probing past the route end repeatedly yields DONE without error."""
    route = _route((0, [0, 1], None))
    walker = RouteWalker(route, my_shard_idx=0)
    walker.before_layer(0)
    walker.before_layer(1)
    # Now exhausted.
    assert walker.before_layer(2).action is LayerAction.DONE
    assert walker.before_layer(3).action is LayerAction.DONE


def test_walker_current_hop_layers_during_local_run():
    route = _route(
        (0, [0, 1, 2], None),
        (1, [3, 4], 0),
    )
    walker = RouteWalker(route, my_shard_idx=0)
    walker.before_layer(0)  # LOCAL
    assert walker.current_hop_layers() == (0, 1, 2)
    walker.before_layer(3)  # SEND
    walker.after_send()
    assert walker.current_hop_layers() is None


def test_walker_after_send_outside_hop_raises():
    """Calling after_send when not inside a hop is a logic error."""
    route = _route((0, [0], None))
    walker = RouteWalker(route, my_shard_idx=1)  # shard 1 not in route
    with pytest.raises(AssertionError, match="after_send called outside"):
        walker.after_send()
