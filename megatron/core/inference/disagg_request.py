# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Wire-format bundle for a layer-kind-disaggregated request.

The coord ships one of these via the
:data:`megatron.core.inference.headers.Headers.ROUTE_REQUEST` header
to every shard participating in a request's route. Each shard's
engine consults the bundle's :class:`Route` to know which layers it
owns for this request and where to send/receive activations.

This module is the *contract* between the coord-side fan-out and the
engine-side forward-pass router. It deliberately does **not** depend
on any coord or engine internals so both sides can import it without
pulling in heavyweight modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from megatron.rl.inference.route_planner import (
    Route,
    deserialize_route,
    serialize_route,
)


@dataclass
class DisaggRequestBundle:
    """Per-request descriptor for disaggregated forward passes.

    Mirrors the migration bundle's role for inter-shard request
    migration but is *additive*: a request can be both
    layer-kind-disaggregated (via this bundle) and later get migrated
    between full-model shards (via the existing migration bundle).

    Attributes:
        request_id: Server-side id assigned by the coord. The same id
            is used by activation transport's per-(request, layer) flag
            namespace.
        route: Per-request route plan. Captures the ordered shard hops
            and per-hop layer ranges. Computed by the coord's route
            planner from the layout's shards.
        prompt_tokens: Initial input tokens. Lives only on the entry
            shard (the others receive embedded activations, not tokens).
        sampling_params: Serialized SamplingParams dict, same shape as
            the migration bundle. Applied on the exit shard when
            producing the next token.

    Wire form: msgpack-serializable via :meth:`to_wire` and
    :meth:`from_wire`.
    """

    request_id: int
    route: Route
    prompt_tokens: List[int] = field(default_factory=list)
    sampling_params: dict = field(default_factory=dict)
    # Optional: pre-populated generated tokens (for resume-after-pause
    # workflows). Empty for fresh submits.
    generated_tokens: List[int] = field(default_factory=list)

    def to_wire(self) -> dict:
        """Flatten to a msgpack-compatible dict."""
        return {
            "request_id": self.request_id,
            "route": serialize_route(self.route),
            "prompt_tokens": list(self.prompt_tokens),
            "sampling_params": dict(self.sampling_params),
            "generated_tokens": list(self.generated_tokens),
        }

    @classmethod
    def from_wire(cls, obj: dict) -> "DisaggRequestBundle":
        """Inverse of :meth:`to_wire`."""
        return cls(
            request_id=int(obj["request_id"]),
            route=deserialize_route(obj["route"]),
            prompt_tokens=list(obj.get("prompt_tokens", [])),
            sampling_params=dict(obj.get("sampling_params", {})),
            generated_tokens=list(obj.get("generated_tokens", [])),
        )

    def shards_participating(self) -> Tuple[int, ...]:
        """Distinct shard indices this request touches.

        Used by the coord's fan-out: it sends ROUTE_REQUEST to every
        shard in this set."""
        return tuple({h.shard_idx for h in self.route.hops})

    def expects_reply_from(self) -> int:
        """Shard index that produces the final token + ENGINE_REPLY.

        Coord uses this to route the reply back to the client."""
        return self.route.exit_shard


