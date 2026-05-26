# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive per-request route selection for layer-kind disagg.

The disagg producer wires per-request routes via
``InferenceClient.add_request(disagg_route=...)`` and the coord
auto-fans ``ROUTE_REQUEST`` with the per-request route override.
This module is the routing policy layer above that protocol: given
a request, decide which route to use.

The default :class:`LayoutRouteSelector` is a no-op — every request
walks the stored layout route. Two concrete selectors ship as
worked examples:

- :class:`PromptLengthRouteSelector` — picks among N pre-built
  routes by prompt length.
- :class:`StickyShardRouteSelector` — pins every request to a
  fixed route (useful for A/B testing, job-level overrides).

Users wanting custom routing (load-aware, prefix-cache-aware,
entropy-aware, etc.) implement :class:`RouteSelector` directly and
plug into ``MegatronLocalMulti.set_route_selector``.

The selector is consulted on each ``base_generate`` call; ``None``
from ``select`` means "use the layout default."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from megatron.rl.inference.route_planner import Route


@dataclass
class RequestInfo:
    """Per-request context the selector sees.

    Only the fields a selector actually needs are populated; ``None``
    is the "unknown / not provided" sentinel. Selectors should
    tolerate missing fields gracefully.

    Attributes:
        prompt_tokens: Tokenized prompt (if available pre-submit).
            ``len()`` gives the prompt length in tokens.
        prompt_text: Raw prompt string if pre-tokenization hasn't
            happened. Selectors that need a token count from text
            should call the tokenizer themselves.
        sampling_params: Sampling parameters dict; useful for
            decisions like "long-tail rollouts go to a different
            route" (via ``num_tokens_to_generate``).
    """

    prompt_tokens: Optional[Sequence[int]] = None
    prompt_text: Optional[str] = None
    sampling_params: Optional[dict] = None


@runtime_checkable
class RouteSelector(Protocol):
    """Decide which route a request should take through a disagg
    layout. User-defined selectors with a ``select`` method that
    returns ``Optional[Route]`` satisfy ``isinstance``.
    """

    def select(self, request_info: RequestInfo) -> Optional[Route]: ...


class LayoutRouteSelector:
    """Default policy: defer to the layout route for every request.
    Behaviourally identical to not setting a selector at all."""

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        return None


class PromptLengthRouteSelector:
    """Pick one of N pre-built routes by prompt length.

    Useful when a layout has multiple disagg shards with different
    KV-budget profiles. Short prompts can take a cheap route
    (e.g. fewer hops, smaller TP); long prompts route through the
    high-KV-budget shard.

    Args:
        tiers: List of ``(max_prompt_tokens, route)`` pairs. The
            first tier whose ``max_prompt_tokens`` is ``>=`` the
            prompt length is selected. If no tier fits, the last
            tier is the fallback. Tiers don't need to be pre-sorted;
            the constructor sorts ascending.

    Example::

        selector = PromptLengthRouteSelector([
            (512, short_route),
            (4096, medium_route),
            (32768, long_route),
        ])
    """

    def __init__(self, tiers: List[Tuple[int, Route]]) -> None:
        if not tiers:
            raise ValueError("PromptLengthRouteSelector needs at least one tier")
        self._tiers: List[Tuple[int, Route]] = sorted(tiers, key=lambda t: t[0])

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        n = _prompt_length(request_info)
        if n is None:
            return self._tiers[-1][1]  # unknown length → fallback
        for max_len, route in self._tiers:
            if n <= max_len:
                return route
        return self._tiers[-1][1]


class StickyShardRouteSelector:
    """Always returns a fixed route. Useful when a caller wants to
    pin every request to a specific path — A/B benchmarks, sticky
    routing for a particular job, or job-level overrides during
    research."""

    def __init__(self, route: Route) -> None:
        self._route = route

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        return self._route


def _prompt_length(info: RequestInfo) -> Optional[int]:
    """Token-count helper. Returns the prompt length in tokens when
    we can determine it without invoking a tokenizer; ``None``
    otherwise (selectors should fall back to a default tier)."""
    if info.prompt_tokens is not None:
        return len(info.prompt_tokens)
    return None
