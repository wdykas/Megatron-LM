# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-distributed tests for the stateless route dispatcher.

The dispatcher's ``dispatch_layer`` is the per-layer hook the model
forward calls. Given ``(route, my_shard_idx, layer_idx)`` it returns
the action without any cursor state. Tests drive the dispatcher
through scripted scenarios with mocked _send / _receive to verify
correct layering of RECEIVE-then-LOCAL, SEND-on-hop-end, and DONE.
"""

import pytest
import torch

from megatron.core.inference.route_dispatcher import LayerAction, RouteDispatcher
from megatron.rl.inference.route_planner import Route, RouteHop


def _route(*hops: tuple) -> Route:
    """hops are (shard_idx, layer_indices) tuples. Legacy 3-tuples with
    a trailing src_shard are accepted and the src is ignored (now
    derived from hop position)."""
    rh = tuple(
        RouteHop(shard_idx=h[0], layer_indices=tuple(h[1]))
        for h in hops
    )
    return Route(hops=rh)


class _MockDispatcher(RouteDispatcher):
    """Dispatcher with _send / _receive stubbed to record calls and
    inject canned receives. Skips the parent __init__ since the real
    NVSHMEM bits aren't set up here."""

    def __init__(self, route, my_shard_idx, received_payload=None):
        self._route = route
        self._my_shard = my_shard_idx
        # Build the parent's precomputed plan with matched-TP convention
        # (peer_pe == peer_shard_idx in these tests — tests treat the
        # shard index AS the peer PE for inspection purposes).
        from megatron.core.inference.route_dispatcher import _LayerPlan
        self._plan = {}
        last_hop_pos = len(route.hops) - 1
        for hop_pos, hop in enumerate(route.hops):
            if hop.shard_idx != my_shard_idx:
                for li in hop.layer_indices:
                    self._plan[li] = None
                continue
            receive_from_pe = (
                route.hops[hop_pos - 1].shard_idx if hop_pos > 0 else None
            )
            send_to_pes = (
                (route.hops[hop_pos + 1].shard_idx,)
                if hop_pos < last_hop_pos
                else ()
            )
            for i, li in enumerate(hop.layer_indices):
                self._plan[li] = _LayerPlan(
                    receive_from_pe=(
                        receive_from_pe if i == 0 else None
                    ),
                    send_to_pes=(
                        send_to_pes if i == len(hop.layer_indices) - 1 else ()
                    ),
                )
        # Test plumbing.
        self.sends: list = []
        self.receives: list = []
        self._received_payload = (
            received_payload
            if received_payload is not None
            else torch.full((4,), 100.0)
        )

    def _send(self, dst_pes, hidden):
        # Mock unpacks the single-peer matched-TP case for test
        # readability; multi-peer (hetero-TP) tests can inspect
        # ``self.sends`` directly as a list of (dst_pes, hidden) tuples.
        if len(dst_pes) == 1:
            self.sends.append((dst_pes[0], None if hidden is None else hidden.clone()))
        else:
            self.sends.append((dst_pes, None if hidden is None else hidden.clone()))

    def _receive(self, src_pe):
        self.receives.append(src_pe)
        return self._received_payload


def _walk_layers(dispatcher, num_layers, initial_hidden, run_local, stop_on=()):
    """Drive ``dispatch_layer`` for layers 0..num_layers-1. By default
    runs every layer; pass ``stop_on=(LayerAction.SEND,)`` to break
    early for tests that want to inspect single-hop boundaries."""
    hidden = initial_hidden
    actions = []
    for li in range(num_layers):
        hidden, action = dispatcher.dispatch_layer(li, hidden, run_local)
        actions.append(action)
        if action in stop_on:
            break
    return hidden, actions


def test_entry_shard_single_hop_runs_every_layer_locally():
    """Shard 0 owns layers [0..3] (no disagg): LOCAL × 4, then DONE on
    the next probe."""
    route = _route((0, [0, 1, 2, 3], None))
    disp = _MockDispatcher(route, my_shard_idx=0)

    def run_local(h):
        return h + 1

    hidden, actions = _walk_layers(disp, 4, torch.zeros(4), run_local)
    assert actions == [LayerAction.LOCAL] * 4
    assert torch.allclose(hidden, torch.full((4,), 4.0))
    # Probe layer 4 → DONE.
    _, action = disp.dispatch_layer(4, hidden, run_local)
    assert action is LayerAction.DONE


def test_three_kind_disagg_entry_shard():
    """3-kind alternating layout. From shard 0's POV (owns layers 0
    and 3): layer 0 is the hop's only layer AND another hop follows,
    so dispatch_layer(0) does run_local then SENDs — one combined
    action. The forward loop breaks here; layer 3's hop is handled
    on a later forward call (after RECEIVE)."""
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
        (0, [3], 2),
        (1, [4], 0),
        (2, [5], 1),
    )
    disp = _MockDispatcher(route, my_shard_idx=0)

    def run_local(h):
        return h * 2

    h0 = torch.ones(4)
    hidden, actions = _walk_layers(disp, 6, h0, run_local, stop_on=(LayerAction.SEND,))
    # SEND at layer 0 — loop bailed early to inspect. run_local was
    # called (h=1 → 2) and that result was sent to shard 1.
    assert actions == [LayerAction.SEND]
    assert len(disp.sends) == 1
    assert disp.sends[0][0] == 1  # dst shard
    assert torch.allclose(disp.sends[0][1], torch.full((4,), 2.0))
    assert hidden is None


def test_middle_shard_receives_runs_sends_in_one_dispatch():
    """Shard 1 owns layer 1 (a single-layer hop). The dispatcher
    coalesces RECEIVE + run_local + SEND into one ``dispatch_layer``
    call when the hop is one layer wide AND has both a predecessor
    and a successor shard."""
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
    )
    disp = _MockDispatcher(route, my_shard_idx=1, received_payload=torch.full((4,), 99.0))

    def run_local(h):
        return h + 1  # detect run_local ran

    h, a0 = disp.dispatch_layer(0, torch.zeros(4), run_local)
    assert a0 is LayerAction.NOT_MY_REQUEST

    # Single call covers receive (99 from shard 0) + run_local (→ 100)
    # + send (100 to shard 2). Returns SEND.
    h, a1 = disp.dispatch_layer(1, h, run_local)
    assert a1 is LayerAction.SEND
    assert disp.receives == [0]
    assert len(disp.sends) == 1 and disp.sends[0][0] == 2
    assert torch.allclose(disp.sends[0][1], torch.full((4,), 100.0))
    assert h is None  # SEND returns None hidden


def test_middle_shard_multi_layer_hop_locals_then_send():
    """When the middle shard's hop has multiple layers, the first
    layer is RECEIVE+LOCAL (returns LOCAL — receive absorbed), the
    middle layers are pure LOCAL, the last layer is run+SEND."""
    route = _route(
        (0, [0], None),
        (1, [1, 2, 3], 0),
        (2, [4], 1),
    )
    disp = _MockDispatcher(route, my_shard_idx=1, received_payload=torch.full((4,), 10.0))

    def run_local(h):
        return h + 1

    _, a0 = disp.dispatch_layer(0, torch.zeros(4), run_local)
    assert a0 is LayerAction.NOT_MY_REQUEST

    h, a1 = disp.dispatch_layer(1, None, run_local)
    # RECEIVE+LOCAL: receives 10, run_local → 11. Action is LOCAL (not SEND;
    # this is the hop's start, not its end).
    assert a1 is LayerAction.LOCAL
    assert torch.allclose(h, torch.full((4,), 11.0))
    assert disp.receives == [0]

    h, a2 = disp.dispatch_layer(2, h, run_local)
    assert a2 is LayerAction.LOCAL
    assert torch.allclose(h, torch.full((4,), 12.0))

    h, a3 = disp.dispatch_layer(3, h, run_local)
    # Hop's last layer + another hop follows → run_local then SEND.
    assert a3 is LayerAction.SEND
    assert len(disp.sends) == 1 and disp.sends[0][0] == 2
    assert torch.allclose(disp.sends[0][1], torch.full((4,), 13.0))


def test_exit_shard_done_after_last_layer():
    """The exit shard's last hop ends with the final LOCAL; the next
    layer probe returns DONE."""
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
    )
    disp = _MockDispatcher(route, my_shard_idx=1)

    def run_local(h):
        return h

    h, a0 = disp.dispatch_layer(0, torch.zeros(4), run_local)
    h, a1 = disp.dispatch_layer(1, h, run_local)
    h, a2 = disp.dispatch_layer(2, h, run_local)  # RECEIVE-then-LOCAL
    h, a3 = disp.dispatch_layer(3, h, run_local)  # LOCAL
    assert (a0, a1, a2, a3) == (
        LayerAction.NOT_MY_REQUEST,
        LayerAction.NOT_MY_REQUEST,
        LayerAction.LOCAL,
        LayerAction.LOCAL,
    )
    _, a4 = disp.dispatch_layer(4, h, run_local)
    assert a4 is LayerAction.DONE


def test_not_my_request_when_shard_not_in_route():
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
    )
    disp = _MockDispatcher(route, my_shard_idx=5)
    for li in range(4):
        _, a = disp.dispatch_layer(li, torch.zeros(4), lambda h: h)
        assert a is LayerAction.NOT_MY_REQUEST


def test_is_entry_and_is_exit():
    route = _route(
        (0, [0], None),
        (1, [1], 0),
        (2, [2], 1),
    )
    assert _MockDispatcher(route, 0).is_entry_shard()
    assert not _MockDispatcher(route, 1).is_entry_shard()
    assert _MockDispatcher(route, 2).is_exit_shard()
    assert not _MockDispatcher(route, 0).is_exit_shard()


def test_revisit_same_shard_first_hop_sends_at_hop_end():
    """Shard 0 owns layers 0,1 then comes back for 4,5 after a detour
    through shard 1. The first forward pass covers hop 0 only —
    layers 0,1 LOCAL, with the SEND coalesced into layer 1's
    dispatch (since layer 1 is hop 0's end). The second hop (layers
    4,5) is handled on a subsequent forward after the request returns
    via an activation."""
    route = _route(
        (0, [0, 1], None),
        (1, [2, 3], 0),
        (0, [4, 5], 1),
    )
    disp = _MockDispatcher(route, my_shard_idx=0)

    def run_local(h):
        return h + 1

    h = torch.zeros(4)
    h, a0 = disp.dispatch_layer(0, h, run_local)
    h, a1 = disp.dispatch_layer(1, h, run_local)
    assert (a0, a1) == (LayerAction.LOCAL, LayerAction.SEND)
    # SEND at layer 1 carries the post-layer-1 hidden state.
    assert len(disp.sends) == 1 and disp.sends[0][0] == 1
    assert torch.allclose(disp.sends[0][1], torch.full((4,), 2.0))


def test_done_is_idempotent():
    route = _route((0, [0, 1], None))
    disp = _MockDispatcher(route, my_shard_idx=0)
    for li in (0, 1):
        disp.dispatch_layer(li, torch.zeros(4), lambda h: h)
    # Probing past the route end returns DONE every time.
    _, a2 = disp.dispatch_layer(2, torch.zeros(4), lambda h: h)
    _, a3 = disp.dispatch_layer(3, torch.zeros(4), lambda h: h)
    assert a2 is LayerAction.DONE
    assert a3 is LayerAction.DONE


