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
    PromptLengthRouteSelector,
    RequestInfo,
    RouteSelector,
    StickyShardRouteSelector,
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


# ---- PromptLengthRouteSelector --------------------------------------------


def test_prompt_length_selector_picks_short_tier():
    """Short prompt falls into the first tier with a max >= len."""
    sel = PromptLengthRouteSelector([
        (10, route_a),
        (1000, route_b),
    ])
    assert sel.select(RequestInfo(prompt_tokens=[0] * 5)) is route_a


def test_prompt_length_selector_picks_long_tier():
    """Prompt that exceeds the small tier rolls over to the next."""
    sel = PromptLengthRouteSelector([
        (10, route_a),
        (1000, route_b),
    ])
    assert sel.select(RequestInfo(prompt_tokens=[0] * 500)) is route_b


def test_prompt_length_selector_fallback_above_max():
    """Prompts longer than every tier fall back to the highest tier
    (so the selector is never undefined for any input length)."""
    sel = PromptLengthRouteSelector([
        (10, route_a),
        (100, route_b),
    ])
    assert sel.select(RequestInfo(prompt_tokens=[0] * 10_000)) is route_b


def test_prompt_length_selector_fallback_on_unknown_length():
    """If neither ``prompt_tokens`` nor a computable length is set,
    the selector picks the last tier — typically the "be safe" route."""
    sel = PromptLengthRouteSelector([
        (10, route_a),
        (100, route_b),
    ])
    assert sel.select(RequestInfo(prompt_text="raw text")) is route_b
    assert sel.select(RequestInfo()) is route_b


def test_prompt_length_selector_sorts_tiers():
    """Tiers don't need to be pre-sorted; the constructor sorts
    ascending so the per-call match is unambiguous."""
    sel = PromptLengthRouteSelector([
        (1000, route_b),  # larger first
        (10, route_a),
    ])
    assert sel.select(RequestInfo(prompt_tokens=[0] * 5)) is route_a
    assert sel.select(RequestInfo(prompt_tokens=[0] * 500)) is route_b


def test_prompt_length_selector_boundary_inclusive():
    """The boundary is inclusive on the tier max — a prompt of
    exactly the tier max picks that tier, not the next."""
    sel = PromptLengthRouteSelector([
        (10, route_a),
        (100, route_b),
    ])
    assert sel.select(RequestInfo(prompt_tokens=[0] * 10)) is route_a
    assert sel.select(RequestInfo(prompt_tokens=[0] * 11)) is route_b


def test_prompt_length_selector_rejects_empty_tiers():
    """No tiers → no valid route; reject at construction time so the
    error surfaces near the bad config rather than at request time."""
    with pytest.raises(ValueError):
        PromptLengthRouteSelector([])


# ---- StickyShardRouteSelector ---------------------------------------------


def test_sticky_selector_returns_fixed_route():
    """Always returns the route it was constructed with, regardless
    of the request info — useful when a caller wants to override the
    layout route for an entire job (e.g. an A/B benchmark)."""
    sel = StickyShardRouteSelector(route_b)
    assert sel.select(RequestInfo(prompt_tokens=[0])) is route_b
    assert sel.select(RequestInfo(prompt_tokens=[0] * 100_000)) is route_b
    assert sel.select(RequestInfo()) is route_b
