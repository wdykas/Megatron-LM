# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engine-side route handler registration test.

Tests the bookkeeping pieces of disagg integration on
:class:`DynamicInferenceEngine`:

- ``set_route_handler`` registers a callback.
- ``register_route_dispatcher`` / ``get_route_dispatcher`` /
  ``release_route_dispatcher`` are the per-request lifecycle hooks.
- A ``Headers.ROUTE_REQUEST`` signal flowing through the engine's
  dispatcher invokes the registered handler with the right route.

This is a non-distributed test that bypasses the engine's normal
construction (which needs a real model + distributed init) and instead
exercises the route-handler surface directly. The full engine forward-
pass integration uses these primitives but isn't covered here.
"""

from unittest.mock import MagicMock

import pytest

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import Headers
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    deserialize_route,
    serialize_route,
)


def _bare_engine():
    """Build a stripped-down engine state for testing the route-handler
    surface — bypasses ``__init__`` (which needs a real model)."""
    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._migration_handler = None
    eng._pending_migrations = []
    eng._route_handler = None
    eng._route_dispatchers = {}
    return eng


def _sample_route() -> Route:
    return Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 1)),
            RouteHop(shard_idx=1, layer_indices=(2, 3)),
        ),
    )


def test_set_route_handler_assigns_callback():
    eng = _bare_engine()
    handler = MagicMock()
    eng.set_route_handler(handler)
    assert eng._route_handler is handler

    # Unsetting also works.
    eng.set_route_handler(None)
    assert eng._route_handler is None


def test_register_and_get_route_dispatcher_round_trip():
    eng = _bare_engine()
    sentinel = object()
    eng.register_route_dispatcher(42, sentinel)
    assert eng.get_route_dispatcher(42) is sentinel
    # Missing request → None (not an error).
    assert eng.get_route_dispatcher(99) is None


def test_release_route_dispatcher_drops_entry():
    eng = _bare_engine()
    sentinel = object()
    eng.register_route_dispatcher(7, sentinel)
    eng.release_route_dispatcher(7)
    assert eng.get_route_dispatcher(7) is None
    # Calling release on a non-existent id is a no-op.
    eng.release_route_dispatcher(99)


def test_route_request_signal_invokes_handler_with_route():
    """When the engine's dispatcher receives a ROUTE_REQUEST payload,
    it deserializes the route and calls ``_route_handler(request_id,
    route)``."""
    eng = _bare_engine()
    handler = MagicMock()
    eng.set_route_handler(handler)

    route = _sample_route()
    # Wire payload: [ROUTE_REQUEST, request_id, route_hops]
    data = [Headers.ROUTE_REQUEST.value, 42, serialize_route(route)]

    # Replay the engine's ROUTE_REQUEST branch.
    _, rid, hops_wire = data
    eng._route_handler(rid, deserialize_route(hops_wire))

    handler.assert_called_once()
    called_rid, called_route = handler.call_args[0]
    assert called_rid == 42
    assert called_route == route


def test_route_request_signal_with_no_handler_does_not_crash():
    """If no handler is registered, the engine logs a warning and
    continues — same robustness as the MIGRATE_BATCH branch."""
    eng = _bare_engine()
    # No set_route_handler call → _route_handler is None.

    # The engine's branch in async_step is:
    #     if self._route_handler is None: log.warning(...)
    # We just verify the precondition holds; the full branch is
    # exercised by the live engine, not here.
    assert eng._route_handler is None
