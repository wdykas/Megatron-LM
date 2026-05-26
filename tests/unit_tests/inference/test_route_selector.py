# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive per-request route selection.

The disagg producer auto-fans a layout-wide route on every
``SUBMIT_REQUEST``; the per-request override path lets a caller
swap that route on a per-submit basis (e.g. short prompts → small
KV-budget shard, long prompts → large KV-budget shard). The
:mod:`route_selector` module provides the policy layer:
:class:`RouteSelector` is the protocol, :class:`LayoutRouteSelector`
is the no-op default. Production deployments write their own
concrete selectors (size-based, load-aware, etc.) on top of the
protocol.
"""

import pytest

from megatron.rl.inference.route_planner import Route, RouteHop
from megatron.rl.inference.route_selector import (
    LayoutRouteSelector,
    RequestInfo,
    RouteSelector,
)


route_a = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1, 2, 3)),))
route_b = Route(
    hops=(
        RouteHop(shard_idx=0, layer_indices=(0, 1)),
        RouteHop(shard_idx=1, layer_indices=(2, 3)),
    )
)


def test_layout_selector_always_returns_none():
    """Default policy: every request defers to the layout route."""
    sel = LayoutRouteSelector()
    assert sel.select(RequestInfo(prompt_tokens=[1, 2, 3])) is None
    assert sel.select(RequestInfo(prompt_text="hi")) is None
    assert sel.select(RequestInfo()) is None  # no info at all


def test_layout_selector_satisfies_protocol():
    """Default impl is structurally a ``RouteSelector`` — the
    ``runtime_checkable`` Protocol means ``isinstance`` works."""
    assert isinstance(LayoutRouteSelector(), RouteSelector)


def test_request_info_dataclass_optional_fields():
    """``RequestInfo`` is a plain dataclass; all fields are optional
    and default to ``None`` so callers only populate what they have."""
    info = RequestInfo()
    assert info.prompt_tokens is None
    assert info.prompt_text is None
    assert info.sampling_params is None


def test_custom_selector_satisfies_protocol():
    """Arbitrary user-defined selectors plug in without inheriting
    from anything — the ``runtime_checkable`` Protocol accepts any
    object with a ``select`` method of the right shape."""

    class _MySelector:
        def __init__(self):
            self.calls = 0

        def select(self, request_info):
            self.calls += 1
            return route_a if request_info.prompt_text == "fast" else None

    sel = _MySelector()
    assert isinstance(sel, RouteSelector)
    assert sel.select(RequestInfo(prompt_text="fast")) is route_a
    assert sel.select(RequestInfo(prompt_text="slow")) is None
    assert sel.calls == 2
