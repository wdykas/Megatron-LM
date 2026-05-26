# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive per-request route selection for layer-kind disagg.

The disagg producer wires per-request routes via
``InferenceClient.add_request(disagg_route=...)`` and the coord
auto-fans ``ROUTE_REQUEST`` with the per-request route override.
This module is the routing policy layer above that protocol: given
a request, decide which route to use.

The default :class:`LayoutRouteSelector` is a no-op — every request
walks the stored layout route. Smart selectors override that with
per-request decisions:

- :class:`PromptLengthRouteSelector` picks among pre-built routes
  by prompt length (e.g. short prompts → a cheap route over a
  small subset of shards; long prompts → a route through a
  high-KV-budget shard).
- Custom selectors (load-aware, prefix-cache-aware, sticky-routing)
  implement :class:`RouteSelector` and plug into
  ``MegatronLocalMulti.set_route_selector``.

The selector is consulted on each ``base_generate`` call;
``None`` from ``select`` means "use the layout default."
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


@dataclass
class StepInfo:
    """Per-decode-step context the reroute selector sees.

    Attributes:
        request_id: Coord ``server_request_id`` of the in-flight
            request — same id the producer uses with
            :meth:`InferenceClient.reselect_route`.
        decode_step: 0-indexed token position within this request's
            generation (0 = first decode after prefill).
        current_route: Whatever route the request is currently walking
            (so the selector can decide if a swap is worth the cost).
        prompt_length: Token count of the prompt, if known. Useful for
            "long-prompt prefill done → drop the prefill-heavy shard"
            policies that key off prompt length.
    """

    request_id: int
    decode_step: int
    current_route: Route
    prompt_length: Optional[int] = None


@runtime_checkable
class RouteSelector(Protocol):
    """Decide which route a request should take through a disagg
    layout. The minimal protocol is just :meth:`select`; user-defined
    selectors with only that method satisfy ``isinstance``.

    ``select_for_step`` is an optional extension for stateful
    selectors that want to reroute mid-rollout. Callers use
    :func:`select_for_step` (module-level helper) which falls back to
    a no-op if the selector doesn't implement it — so stateless
    selectors (size-based, prefix-cache-based) don't need to.
    """

    def select(self, request_info: RequestInfo) -> Optional[Route]: ...


def select_for_step(
    selector: RouteSelector, step_info: StepInfo
) -> Optional[Route]:
    """Module-level helper that calls ``selector.select_for_step`` when
    present and falls back to ``None`` otherwise.

    The reroute-policy task uses this so user-defined selectors that
    only implement :meth:`RouteSelector.select` keep working — they
    simply never trigger a mid-rollout reroute.
    """
    fn = getattr(selector, "select_for_step", None)
    if fn is None:
        return None
    return fn(step_info)


class LayoutRouteSelector:
    """Default policy: defer to the layout route for every request.
    Behaviourally identical to not setting a selector at all."""

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        return None

    def select_for_step(self, step_info: StepInfo) -> Optional[Route]:
        return None


class PrefillDecodeFlipSelector:
    """Stateful selector that picks one route at submit time and a
    different route after a configurable prefill threshold.

    Use case: a long-context prompt benefits from a large-TP attention
    shard during prefill (compute-bound), but once decoding starts the
    request is bandwidth-bound and the smaller decode shard wins. The
    selector picks ``prefill_route`` at submit, then on decode step
    ``flip_at_step`` switches to ``decode_route``.

    Subsequent steps return ``None`` (no further reroutes) so the
    coord doesn't churn.
    """

    def __init__(
        self,
        prefill_route: Route,
        decode_route: Route,
        *,
        flip_at_step: int = 1,
    ) -> None:
        self._prefill_route = prefill_route
        self._decode_route = decode_route
        self._flip_at_step = flip_at_step

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        return self._prefill_route

    def select_for_step(self, step_info: StepInfo) -> Optional[Route]:
        if step_info.decode_step == self._flip_at_step:
            return self._decode_route
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

    def select_for_step(self, step_info: StepInfo) -> Optional[Route]:
        return None


class StickyShardRouteSelector:
    """Always returns a fixed route. Useful when a caller wants to
    pin a request to a specific path (e.g. "this rollout must take
    the wraparound route, not the fan-out one")."""

    def __init__(self, route: Route) -> None:
        self._route = route

    def select(self, request_info: RequestInfo) -> Optional[Route]:
        return self._route

    def select_for_step(self, step_info: StepInfo) -> Optional[Route]:
        return None


def _prompt_length(info: RequestInfo) -> Optional[int]:
    """Token-count helper. Returns the prompt length in tokens when
    we can determine it without invoking a tokenizer; ``None``
    otherwise (selectors should fall back to a default tier)."""
    if info.prompt_tokens is not None:
        return len(info.prompt_tokens)
    return None
