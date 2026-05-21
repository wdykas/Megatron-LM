# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pluggable activation-transport backend.

The dispatcher used to call ``activation_transport.send_hidden`` /
``receive_hidden`` directly, hardcoding NVSHMEM. After the backend
refactor it goes through ``get_activation_transport_backend()``, so
alternative transports (NCCL, Gloo, mock) plug in without touching
the dispatcher.

These tests verify the protocol shape + swap semantics without
requiring real NVSHMEM init.
"""

from typing import Tuple
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    NvshmemActivationTransportBackend,
    get_activation_transport_backend,
    set_activation_transport_backend,
)


class _MockBackend(ActivationTransportBackend):
    """Records calls; doesn't actually move bytes. Useful for
    asserting the dispatcher routes through the backend interface."""

    def __init__(self) -> None:
        self.sends: list = []
        self.receives: list = []
        self._inited = False

    def is_initialized(self) -> bool:
        return self._inited

    def init(self, **kwargs) -> None:
        self._inited = True

    def stream(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream()

    def send_hidden(self, my_pe, dst_pe, hidden, payload_nbytes, *, stream=None):
        self.sends.append((my_pe, dst_pe, hidden.shape, payload_nbytes))

    def receive_hidden(self, my_pe, src_pe, hidden_shape, hidden_dtype,
                       payload_nbytes, *, stream=None):
        self.receives.append((my_pe, src_pe, hidden_shape, payload_nbytes))
        return torch.zeros(hidden_shape, dtype=hidden_dtype)


def test_default_backend_is_nvshmem():
    """Without an explicit set, the singleton constructs the NVSHMEM
    backend on first access."""
    set_activation_transport_backend(None)  # reset
    backend = get_activation_transport_backend()
    assert isinstance(backend, NvshmemActivationTransportBackend)


def test_set_backend_swaps_singleton():
    """``set_activation_transport_backend(mock)`` redirects all future
    transport calls through the mock — the dispatcher's _send/_receive
    transparently route via the new backend."""
    mock = _MockBackend()
    set_activation_transport_backend(mock)
    assert get_activation_transport_backend() is mock
    # Reset for other tests.
    set_activation_transport_backend(None)


def test_set_backend_none_resets_to_default():
    """``set_activation_transport_backend(None)`` clears the
    singleton; next access constructs a fresh NVSHMEM default."""
    set_activation_transport_backend(_MockBackend())
    assert not isinstance(
        get_activation_transport_backend(), NvshmemActivationTransportBackend
    )
    set_activation_transport_backend(None)
    assert isinstance(
        get_activation_transport_backend(), NvshmemActivationTransportBackend
    )


def test_dispatcher_routes_through_backend_singleton():
    """End-to-end on the dispatcher with the mock backend: a
    cross-shard hop fires the backend's ``send_hidden`` /
    ``receive_hidden`` instead of NVSHMEM directly."""
    from megatron.core.inference.route_dispatcher import RouteDispatcher
    from megatron.rl.inference.route_planner import Route, RouteHop

    mock = _MockBackend()
    set_activation_transport_backend(mock)
    try:
        route = Route(
            hops=(
                RouteHop(shard_idx=0, layer_indices=(0,)),
                RouteHop(shard_idx=1, layer_indices=(1,)),
            )
        )
        # Build dispatcher as shard 0 (sender side).
        d_send = RouteDispatcher(
            route=route,
            my_shard_idx=0,
            my_pe=0,
            my_tp_offset=0,
            shard_tp=[1, 1],
            shard_rank_offset=[0, 1],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        # Drive forward: layer 0 owned, hop exit → triggers send.
        hidden = torch.ones(2, 4)
        out, _ = d_send.dispatch_layer(0, hidden, lambda h: h)
        # Mock backend recorded one send to PE 1.
        assert mock.sends == [(0, 1, torch.Size([2, 4]), 2 * 4 * 4)]
        assert out is None  # SEND clears hidden

        # Build dispatcher as shard 1 (receiver side).
        d_recv = RouteDispatcher(
            route=route,
            my_shard_idx=1,
            my_pe=1,
            my_tp_offset=0,
            shard_tp=[1, 1],
            shard_rank_offset=[0, 1],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        # Drive forward: layer 0 not owned → NOT_MY_REQUEST.
        # Layer 1 owned, hop entry → triggers receive.
        d_recv.dispatch_layer(0, None, lambda h: h)
        d_recv.dispatch_layer(1, None, lambda h: h)
        assert mock.receives == [(1, 0, (2, 4), 2 * 4 * 4)]
    finally:
        set_activation_transport_backend(None)


def test_backend_interface_minimal():
    """The backend protocol is intentionally narrow — just the
    operations the dispatcher needs. Backends can ignore knobs they
    don't have (e.g. NVSHMEM's lane count) at the cost of slightly
    different perf characteristics."""
    backend = _MockBackend()
    assert hasattr(backend, "is_initialized")
    assert hasattr(backend, "init")
    assert hasattr(backend, "stream")
    assert hasattr(backend, "send_hidden")
    assert hasattr(backend, "receive_hidden")
    # All abstract methods are present on the impl (otherwise ABCMeta
    # would have rejected instantiation in the class statement).
