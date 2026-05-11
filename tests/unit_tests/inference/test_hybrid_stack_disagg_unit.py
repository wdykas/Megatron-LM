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

from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.disagg_forward import (
    maybe_dispatch_layer,
    should_stop_layer_loop,
)
from megatron.core.inference.route_walker import LayerAction
from megatron.core.models.hybrid.hybrid_block import DISAGG_STUB_MARKER


class _ScriptedDispatcher:
    def __init__(self, actions):
        self._actions = list(actions)
        self.dispatched: list = []

    def dispatch_layer(self, layer_idx, hidden, run_local):
        action = self._actions[layer_idx]
        self.dispatched.append((layer_idx, action))
        if action is LayerAction.LOCAL:
            return run_local(hidden), action
        if action is LayerAction.RECEIVE:
            inbound = torch.full_like(hidden, 100.0) if hidden is not None else torch.full((4,), 100.0)
            return run_local(inbound), action
        if action is LayerAction.SEND:
            return None, action
        if action is LayerAction.DONE:
            return hidden, action
        return hidden, action


def _drive_layer_loop(layers, hidden, engine, request_id):
    """Mirrors HybridStack.forward's layer loop in miniature."""
    for layer_idx, layer in enumerate(layers):
        hidden, action = maybe_dispatch_layer(
            engine=engine,
            request_id=request_id,
            layer_idx=layer_idx,
            hidden=hidden,
            run_local=lambda h, _layer=layer: _layer(h),
        )
        if should_stop_layer_loop(action):
            break
    return hidden


def test_collocated_forward_runs_every_layer_locally():
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h * -1,
    ]
    h = torch.ones(4)
    out = _drive_layer_loop(layers, h, engine=None, request_id=None)
    assert torch.allclose(out, torch.full((4,), -3.0))


def test_disagg_forward_local_only_runs_every_layer():
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h * -1,
    ]
    h = torch.ones(4)
    disp = _ScriptedDispatcher([LayerAction.LOCAL] * 3)
    engine = MagicMock()
    engine.get_route_dispatcher.return_value = disp
    out = _drive_layer_loop(layers, h, engine=engine, request_id=42)
    assert torch.allclose(out, torch.full((4,), -3.0))
    assert [c[1] for c in disp.dispatched] == [LayerAction.LOCAL] * 3


def test_disagg_forward_breaks_on_send():
    layers = [
        lambda h: h * 2,
        lambda h: h + 1,
        lambda h: h * -1,  # never runs
    ]
    h = torch.ones(4)
    disp = _ScriptedDispatcher([
        LayerAction.LOCAL,
        LayerAction.SEND,
        LayerAction.LOCAL,
    ])
    engine = MagicMock()
    engine.get_route_dispatcher.return_value = disp

    out = _drive_layer_loop(layers, h, engine=engine, request_id=1)
    assert out is None
    assert [c[0] for c in disp.dispatched] == [0, 1]


def test_disagg_forward_receive_then_local():
    layers = [
        lambda h: h + 1,
        lambda h: h * 10,  # local on RECEIVED tensor
        lambda h: h,
    ]
    disp = _ScriptedDispatcher([
        LayerAction.LOCAL,
        LayerAction.RECEIVE,
        LayerAction.DONE,
    ])
    engine = MagicMock()
    engine.get_route_dispatcher.return_value = disp

    out = _drive_layer_loop(layers, torch.ones(4), engine=engine, request_id=1)
    # Layer 0 LOCAL: 1+1=2. Layer 1 RECEIVE: gets [100,...] → 100*10=1000.
    # Layer 2 DONE: break.
    assert torch.allclose(out, torch.full((4,), 1000.0))


def test_disagg_stub_marker_is_underscore():
    assert DISAGG_STUB_MARKER == "_"
