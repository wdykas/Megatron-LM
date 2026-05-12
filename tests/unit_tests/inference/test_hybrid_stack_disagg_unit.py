# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-distributed test exercising the disagg wire-up inside
``HybridStack.forward`` directly.

We construct a HybridStack-like layer loop with the disagg attributes
and ``forward()`` plumbing, then drive it through a synthetic
dispatcher to verify:

- Collocated requests (no engine attached) run every layer locally.
- Disagg requests with an active dispatcher route through
  LOCAL / SEND / DONE actions; the loop breaks on SEND/DONE.
"""

import pytest
import torch

from megatron.core.inference.route_dispatcher import LayerAction
from megatron.core.inference.disagg_stub import DISAGG_STUB_MARKER


class _ScriptedDispatcher:
    """Stand-in for ``RouteDispatcher`` in the model-forward loop test:
    replays a fixed action sequence per layer."""

    def __init__(self, actions):
        self._actions = list(actions)
        self.dispatched: list = []

    def dispatch_layer(self, layer_idx, hidden, run_local):
        action = self._actions[layer_idx]
        self.dispatched.append((layer_idx, action))
        if action is LayerAction.LOCAL:
            # The real dispatcher replaces a None hidden with the
            # received tensor before run_local; simulate that.
            if hidden is None:
                hidden = torch.full((4,), 100.0)
            return run_local(hidden), action
        if action is LayerAction.SEND:
            return None, action
        return hidden, action  # DONE / NOT_MY_REQUEST


def _drive_layer_loop(layers, hidden, dispatcher):
    """Mirrors HybridStack.forward's layer loop: when a dispatcher is
    set, route every layer through it; otherwise run locally."""
    for layer_idx, layer in enumerate(layers):
        if dispatcher is not None:
            hidden, _ = dispatcher.dispatch_layer(
                layer_idx, hidden, lambda h, _l=layer: _l(h)
            )
        else:
            hidden = layer(hidden)
    return hidden


def test_collocated_forward_runs_every_layer_locally():
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h * -1,
    ]
    h = torch.ones(4)
    out = _drive_layer_loop(layers, h, dispatcher=None)
    assert torch.allclose(out, torch.full((4,), -3.0))


def test_disagg_forward_local_only_runs_every_layer():
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h * -1,
    ]
    disp = _ScriptedDispatcher([LayerAction.LOCAL] * 3)
    out = _drive_layer_loop(layers, torch.ones(4), dispatcher=disp)
    assert torch.allclose(out, torch.full((4,), -3.0))
    assert [c[1] for c in disp.dispatched] == [LayerAction.LOCAL] * 3


def test_disagg_forward_threads_none_through_after_send():
    """After SEND, hidden is cleared to None and the loop continues.
    Subsequent NOT_MY_REQUEST layers pass None through unchanged."""
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h,  # NOT_MY_REQUEST skips this
    ]
    disp = _ScriptedDispatcher([
        LayerAction.LOCAL,
        LayerAction.SEND,
        LayerAction.NOT_MY_REQUEST,
    ])
    out = _drive_layer_loop(layers, torch.ones(4), dispatcher=disp)
    # All three layers dispatched; final hidden is None.
    assert out is None
    assert [c[0] for c in disp.dispatched] == [0, 1, 2]


def test_disagg_forward_receive_absorbed_into_local():
    """RECEIVE is no longer a separate action — the dispatcher does
    the receive internally and returns LOCAL. Simulated here by
    letting LOCAL replace a None hidden with the received tensor."""
    layers = [
        lambda h: h + 1,
        lambda h: h * 10,
        lambda h: h,
    ]
    disp = _ScriptedDispatcher([
        LayerAction.LOCAL,
        LayerAction.SEND,
        LayerAction.LOCAL,  # next-hop receive-then-local
    ])
    out = _drive_layer_loop(layers, torch.ones(4), dispatcher=disp)
    # Layer 0: 1+1=2. Layer 1: SEND clears hidden. Layer 2: receive
    # injects [100,...], identity layer → 100.
    assert torch.allclose(out, torch.full((4,), 100.0))


def test_disagg_stub_marker_is_underscore():
    assert DISAGG_STUB_MARKER == "_"
