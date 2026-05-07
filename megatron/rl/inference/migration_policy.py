# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pluggable migration policies for cross-shard request migration.

Each :class:`MigrationPolicy` decides, on its own per-shard scheduler
tick, which in-flight requests on its watched src shard should be
migrated to a fixed dst shard. The migration mechanism (NVSHMEM
transport, coord forwarding, engine-side surgery) is policy-agnostic
and lives in :mod:`megatron.rl.inference.multi_shard`; this module
just provides the eligibility hook.

Built-in policies:
- :class:`FirstTokenDisaggPolicy` — migrate after prefill (n_generated
  >= 1) when the request carries an opt-in disagg tag.
- :class:`TailCutPolicy` — migrate once a request has produced more
  than ``min_tokens`` decoded tokens.

Adding a new policy is a subclass + an :meth:`is_eligible` method.
The scheduler loop, batching, round-robin dst-DP selection, and
coord dispatch are inherited.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MigrationPolicy:
    """Base class for a per-shard migration policy.

    A policy watches one ``src_shard_index`` and migrates eligible
    requests to a fixed ``dst_shard_index``. Subclasses override
    :meth:`is_eligible`.

    Attributes:
        src_shard_index: Shard whose engine the scheduler polls. The
            policy's asyncio task runs on that shard's
            ``rank_offset`` (decider rank); other ranks idle.
        dst_shard_index: Where to migrate eligible requests.
        poll_interval_s: Seconds between scheduler ticks.
        max_batch_size: Cap on migrations fired per tick.
    """

    src_shard_index: int
    dst_shard_index: int
    poll_interval_s: float = 0.05
    max_batch_size: int = 16

    def is_eligible(self, request: Any) -> bool:
        """Return ``True`` if ``request`` should be migrated this tick.

        ``request`` is the live :class:`DynamicInferenceRequest` (the
        last item in ``RequestEntry.record``). Read whatever request
        attributes you need (``generated_tokens``, custom tags, etc.).

        The default impl always returns ``False`` — subclasses must
        override.
        """
        return False


@dataclass
class FirstTokenDisaggPolicy(MigrationPolicy):
    """Migrate as soon as a request has produced its first decoded
    token, *if* it carries the opt-in ``disagg_dst_shard_index`` tag
    pointing at this policy's ``dst_shard_index``.

    Set by HTTP submits with ``disagg_pair=[src, dst]`` or by
    :meth:`InferenceClient.add_request(disagg_dst_shard_index=...)`.
    Untagged requests stay put.
    """

    def is_eligible(self, request: Any) -> bool:
        n = _generated_token_count(request)
        if n < 1:
            return False
        tag = getattr(request, "disagg_dst_shard_index", None)
        return tag is not None and tag == self.dst_shard_index


@dataclass
class TailCutPolicy(MigrationPolicy):
    """Migrate once a request has accumulated ``min_tokens`` tokens
    on its current shard, *if* it carries the opt-in
    ``late_dst_shard_index`` / ``late_dst_min_tokens`` tags pointing
    at this policy's ``dst_shard_index``.

    Set on every rollout when ``--rl-tail-cut-dst-shard`` and
    ``--rl-tail-cut-min-tokens`` are configured. Designed to pull
    long-tail requests off a throughput shard onto a latency-
    optimized shard.
    """

    min_tokens: int = 128

    def is_eligible(self, request: Any) -> bool:
        n = _generated_token_count(request)
        if n < self.min_tokens:
            return False
        tag = getattr(request, "late_dst_shard_index", None)
        if tag is None or tag != self.dst_shard_index:
            return False
        # Per-request override of the policy threshold (the HTTP
        # ``tail_cut=[dst, n]`` tuple). Lets traffic with different
        # length distributions share a single tail-cut shard.
        per_req_min = getattr(request, "late_dst_min_tokens", None)
        threshold = per_req_min if per_req_min is not None else self.min_tokens
        return n >= threshold


def _generated_token_count(request: Any) -> int:
    """Robust token-count read: handles tensor and list."""
    import torch

    gen = getattr(request, "generated_tokens", None)
    if gen is None:
        return 0
    if isinstance(gen, torch.Tensor):
        return int(gen.numel())
    return len(gen)
