# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pluggable decode-replica router for disaggregated prefill->decode.

Decides *which decode replica* a request's KV is handed to. A replica is
one internally-uniform model instance (its own TP/PP); the pool may be
heterogeneous (replicas of different parallelism), and the KV is
resharded to whichever replica is chosen via
:func:`kv_shard_layout.plan_kv_reshard`.

Swappable by design, mirroring the ``RouteSelector`` shape on the
``hetero-inference`` branch: a narrow :class:`DecodeRouter` ABC with one
required method, :meth:`select`. Concrete policies ship here
(round-robin, sticky-affinity, least-loaded, prompt-length-tiered) and
custom routers register via :func:`register_router`. Load-aware routers
use the optional :meth:`on_admit` / :meth:`on_complete` hooks.

Routing (which replica) is intentionally separate from transport (how
the KV moves) and resharding (how it is re-partitioned): they compose
via :func:`route_and_plan`.
"""

from __future__ import annotations

import hashlib
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Type

from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout


@dataclass(frozen=True)
class RequestInfo:
    """What a router gets to decide on. Extend ``metadata`` freely;
    routers should access it defensively."""

    request_id: int
    num_prompt_tokens: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class DecodeTarget:
    """One decode replica: an id + its per-rank :class:`KVShardLayout`s
    (internally uniform parallelism). ``max_prompt_tokens`` is an optional
    capacity hint used by the prompt-length-tiered router."""

    target_id: str
    layouts: List[KVShardLayout]
    max_prompt_tokens: Optional[int] = None


class DecodeRouter(ABC):
    """Swappable policy: pick a decode replica for a request.

    Subclasses implement :meth:`select`. Load-aware subclasses also
    override :meth:`on_admit` / :meth:`on_complete`; the coordinator
    calls those around the request's lifetime so the router can track
    in-flight load. The base hooks are no-ops, so stateless routers
    ignore them.
    """

    def __init__(self, targets: List[DecodeTarget]) -> None:
        if not targets:
            raise ValueError("DecodeRouter requires at least one decode target")
        self.targets = list(targets)
        self._by_id = {t.target_id: t for t in self.targets}

    @abstractmethod
    def select(self, request: RequestInfo) -> DecodeTarget:
        """Return the decode replica this request should be handed to."""

    def on_admit(self, request: RequestInfo, target: DecodeTarget) -> None:
        """Called when ``request`` is admitted to ``target`` (optional)."""

    def on_complete(self, request: RequestInfo, target: DecodeTarget) -> None:
        """Called when ``request`` finishes on ``target`` (optional)."""


class RoundRobinRouter(DecodeRouter):
    """Cycle through targets. Stateless, even spread, ignores load."""

    def __init__(self, targets: List[DecodeTarget]) -> None:
        super().__init__(targets)
        self._counter = itertools.count()

    def select(self, request: RequestInfo) -> DecodeTarget:
        return self.targets[next(self._counter) % len(self.targets)]


class StickyHashRouter(DecodeRouter):
    """Affinity routing: a stable key (default ``request_id``) hashes to a
    fixed replica, so a repeated prompt lands on the replica that already
    has its prefix cached. Pass ``key`` to hash on prompt content / a
    session id instead."""

    def __init__(
        self,
        targets: List[DecodeTarget],
        key: Optional[Callable[[RequestInfo], object]] = None,
    ) -> None:
        super().__init__(targets)
        self._key = key or (lambda r: r.request_id)

    def select(self, request: RequestInfo) -> DecodeTarget:
        digest = hashlib.sha256(str(self._key(request)).encode()).digest()
        idx = int.from_bytes(digest[:8], "little") % len(self.targets)
        return self.targets[idx]


class LeastLoadedRouter(DecodeRouter):
    """Pick the replica with the fewest in-flight requests. Requires the
    coordinator to call :meth:`on_admit` / :meth:`on_complete`."""

    def __init__(self, targets: List[DecodeTarget]) -> None:
        super().__init__(targets)
        self._load: Dict[str, int] = {t.target_id: 0 for t in targets}

    def select(self, request: RequestInfo) -> DecodeTarget:
        # Stable tie-break by target order (min is first-min).
        return min(self.targets, key=lambda t: self._load[t.target_id])

    def on_admit(self, request: RequestInfo, target: DecodeTarget) -> None:
        self._load[target.target_id] += 1

    def on_complete(self, request: RequestInfo, target: DecodeTarget) -> None:
        self._load[target.target_id] = max(0, self._load[target.target_id] - 1)


class PromptLengthTieredRouter(DecodeRouter):
    """Route by prompt length to the smallest-capacity replica that fits;
    longest tier is the fallback. Lets long contexts go to higher-TP
    replicas. Targets should set ``max_prompt_tokens`` (a target with
    ``None`` is treated as unbounded)."""

    def __init__(self, targets: List[DecodeTarget]) -> None:
        super().__init__(targets)
        self._sorted = sorted(
            self.targets,
            key=lambda t: (t.max_prompt_tokens if t.max_prompt_tokens is not None else (1 << 62)),
        )

    def select(self, request: RequestInfo) -> DecodeTarget:
        n = request.num_prompt_tokens or 0
        for t in self._sorted:
            if t.max_prompt_tokens is None or n <= t.max_prompt_tokens:
                return t
        return self._sorted[-1]


# --- registry: switch routers by name; register custom ones --------------

_ROUTERS: Dict[str, Type[DecodeRouter]] = {
    "round_robin": RoundRobinRouter,
    "sticky": StickyHashRouter,
    "least_loaded": LeastLoadedRouter,
    "prompt_length": PromptLengthTieredRouter,
}


def register_router(name: str, cls: Type[DecodeRouter]) -> None:
    """Register a custom :class:`DecodeRouter` subclass under ``name`` so
    it can be selected via :func:`make_router` (and a CLI flag)."""
    _ROUTERS[name] = cls


def make_router(name: str, targets: List[DecodeTarget], **kwargs) -> DecodeRouter:
    """Construct a router by name. Raises ``KeyError`` on unknown name so
    a typo'd ``--router`` fails loudly rather than silently defaulting."""
    if name not in _ROUTERS:
        raise KeyError(
            f"unknown router '{name}'; known: {sorted(_ROUTERS)}. "
            "register custom routers with register_router()."
        )
    return _ROUTERS[name](targets, **kwargs)


def route_and_plan(
    router: DecodeRouter, request: RequestInfo, src_layouts: List[KVShardLayout]
) -> Tuple[DecodeTarget, list]:
    """Compose routing + resharding: pick a decode replica, then build the
    KV reshard plan prefill(``src_layouts``) -> that replica. Returns
    ``(target, transfers)``. The caller drives the transport with the
    transfers and calls ``router.on_admit`` once the request is accepted."""
    from megatron.core.inference.disaggregation.kv_shard_layout import plan_kv_reshard

    target = router.select(request)
    transfers = plan_kv_reshard(src_layouts, target.layouts)
    return target, transfers
