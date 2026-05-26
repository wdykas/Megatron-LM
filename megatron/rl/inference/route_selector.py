# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adaptive per-request route selection for layer-kind disagg.

The disagg producer wires per-request routes via
``InferenceClient.add_request(disagg_route=...)`` and the coord
auto-fans ``ROUTE_REQUEST`` with the per-request route override.
This module is the routing policy layer above that protocol: given
a request, decide which route to use.

The default :class:`LayoutRouteSelector` is a no-op — every request
walks the stored layout route. Users wanting per-request route
selection (size-based, load-aware, prefix-cache-aware, etc.)
implement :class:`RouteSelector` and plug into
``MegatronLocalMulti.set_route_selector``.

The selector is consulted on each ``base_generate`` call; ``None``
from ``select`` means "use the layout default."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

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
