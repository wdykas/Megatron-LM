# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the per-layer disagg forward-pass hook."""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.disagg_forward import (
    finalize_dispatch,
    maybe_dispatch_layer,
    should_stop_layer_loop,
)
from megatron.core.inference.route_walker import LayerAction


class _FakeDispatcher:
    """Records dispatch calls and returns scripted actions/values."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []
        self._done = False

    def dispatch_layer(self, layer_idx, hidden, run_local):
        if not self._script:
            raise RuntimeError(f"dispatcher script exhausted at layer {layer_idx}")
        action, hidden_factory = self._script.pop(0)
        self.calls.append((layer_idx, action, hidden is not None))
        if hidden_factory is None:
            return hidden, action
        return hidden_factory(hidden), action

    def is_done(self):
        return self._done

    def maybe_send_final(self, hidden, final_layer_idx):
        return False


class _FakeEngine:
    def __init__(self, dispatcher_by_id):
        self._dispatchers = dispatcher_by_id

    def get_route_dispatcher(self, request_id):
        return self._dispatchers.get(request_id)


def test_passthrough_when_no_request_id():
    eng = _FakeEngine({})
    h = torch.zeros(3)
    out, action = maybe_dispatch_layer(
        eng, request_id=None, layer_idx=0, hidden=h,
        run_local=lambda x: x + 1,
    )
    assert torch.allclose(out, h + 1)
    assert action is LayerAction.LOCAL


def test_passthrough_when_no_dispatcher_registered():
    eng = _FakeEngine({})
    h = torch.zeros(3)
    out, action = maybe_dispatch_layer(
        eng, request_id=7, layer_idx=0, hidden=h,
        run_local=lambda x: x + 2,
    )
    assert torch.allclose(out, h + 2)
    assert action is LayerAction.LOCAL


def test_local_action_passes_through_dispatcher_result():
    h = torch.zeros(3)
    disp = _FakeDispatcher(script=[(LayerAction.LOCAL, lambda h: h + 5)])
    eng = _FakeEngine({1: disp})
    out, action = maybe_dispatch_layer(
        eng, request_id=1, layer_idx=0, hidden=h,
        run_local=lambda x: x,
    )
    assert torch.allclose(out, h + 5)
    assert action is LayerAction.LOCAL


def test_receive_action_passes_through_dispatcher_result():
    h = torch.zeros(3)
    disp = _FakeDispatcher(script=[(LayerAction.RECEIVE, lambda h: torch.full((3,), 9.0))])
    eng = _FakeEngine({1: disp})
    out, action = maybe_dispatch_layer(
        eng, request_id=1, layer_idx=2, hidden=None,
        run_local=lambda x: x,
    )
    assert torch.allclose(out, torch.full((3,), 9.0))
    assert action is LayerAction.RECEIVE


def test_not_my_request_returns_hidden_unchanged():
    h = torch.tensor([1.0, 2.0, 3.0])
    disp = _FakeDispatcher(script=[(LayerAction.NOT_MY_REQUEST, None)])
    eng = _FakeEngine({1: disp})
    out, action = maybe_dispatch_layer(
        eng, request_id=1, layer_idx=0, hidden=h,
        run_local=lambda x: x * 100,
    )
    assert torch.allclose(out, h)
    assert action is LayerAction.NOT_MY_REQUEST


def test_send_returns_send_action_for_caller_break():
    h = torch.zeros(3)
    disp = _FakeDispatcher(script=[(LayerAction.SEND, lambda h: None)])
    eng = _FakeEngine({1: disp})
    out, action = maybe_dispatch_layer(
        eng, request_id=1, layer_idx=1, hidden=h,
        run_local=lambda x: x,
    )
    assert action is LayerAction.SEND
    assert out is None
    assert should_stop_layer_loop(action)


def test_done_returns_done_action_for_caller_break():
    h = torch.zeros(3)
    disp = _FakeDispatcher(script=[(LayerAction.DONE, lambda h: h)])
    eng = _FakeEngine({1: disp})
    _, action = maybe_dispatch_layer(
        eng, request_id=1, layer_idx=3, hidden=h,
        run_local=lambda x: x,
    )
    assert action is LayerAction.DONE
    assert should_stop_layer_loop(action)


def test_should_stop_layer_loop_for_local_actions():
    assert not should_stop_layer_loop(LayerAction.LOCAL)
    assert not should_stop_layer_loop(LayerAction.RECEIVE)
    assert not should_stop_layer_loop(LayerAction.NOT_MY_REQUEST)


def test_finalize_dispatch_no_op_when_no_request_id():
    eng = _FakeEngine({})
    assert finalize_dispatch(eng, None, 4, torch.zeros(3)) is False


def test_finalize_dispatch_no_op_when_no_dispatcher():
    eng = _FakeEngine({})
    assert finalize_dispatch(eng, 7, 4, torch.zeros(3)) is False


def test_finalize_dispatch_delegates_to_dispatcher():
    eng = _FakeEngine({})
    disp = MagicMock()
    disp.is_done.return_value = False
    disp.maybe_send_final.return_value = True
    eng._dispatchers[1] = disp

    assert finalize_dispatch(eng, 1, 5, torch.zeros(3)) is True
    disp.maybe_send_final.assert_called_once()


def test_finalize_dispatch_skips_done_dispatcher():
    eng = _FakeEngine({})
    disp = MagicMock()
    disp.is_done.return_value = True
    eng._dispatchers[1] = disp

    assert finalize_dispatch(eng, 1, 5, torch.zeros(3)) is False
    disp.maybe_send_final.assert_not_called()


def test_full_layer_loop_using_helper():
    """Simulate a 4-layer model.forward driven by maybe_dispatch_layer.
    Verify the loop breaks cleanly on SEND."""
    h = torch.full((4,), 1.0)
    disp = _FakeDispatcher(script=[
        (LayerAction.LOCAL, lambda h: h * 2),  # → [2,2,2,2]
        (LayerAction.LOCAL, lambda h: h * 3),  # → [6,6,6,6]
        (LayerAction.SEND, lambda h: None),    # caller breaks
    ])
    eng = _FakeEngine({1: disp})

    hidden = h
    actions_seen = []
    for li in range(4):
        hidden, action = maybe_dispatch_layer(
            eng, request_id=1, layer_idx=li, hidden=hidden,
            run_local=lambda x: x,
        )
        actions_seen.append(action)
        if should_stop_layer_loop(action):
            break
    assert actions_seen == [LayerAction.LOCAL, LayerAction.LOCAL, LayerAction.SEND]
    assert [c[0] for c in disp.calls] == [0, 1, 2]
