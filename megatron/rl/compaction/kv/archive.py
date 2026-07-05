# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU KV archive + negative-cache index: eviction as demotion, not deletion.

When live compaction evicts a span of KV, the exact tensors move to pinned CPU
memory and a tiny GPU index of the *evicted* content stays behind — one mean-key
centroid per contiguous span per layer/KV-group (the "negative cache": an index
of what is absent from the GPU cache). At decode time a query is scored against
retained keys and against these centroids with the same attention math; a query
that scores higher on a centroid than on any retained key is demonstrably
reaching for dropped content — the trigger — and the winning centroid names the
span to restore (validated on trained-Nano captures: trigger↔oracle correlation
+0.77, top-1 span identification 76.6% vs 1.4% chance).

Entries are keyed by the ENGINE request id (stable across pause/resume slot
swaps); LiveKVCompactor drops a request's entries when it finishes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class _Span:
    positions: list[int]           # original token positions (bookkeeping only)
    keys: torch.Tensor             # (L, T, H, D) pinned CPU
    values: torch.Tensor           # (L, T, H, D) pinned CPU
    centroids: torch.Tensor        # (L, Hkv, D) GPU — mean evicted key per layer/group


class KVArchive:
    """Per-request store of evicted KV spans with a GPU negative-cache index."""

    def __init__(self, max_span: int = 16) -> None:
        if max_span < 1:
            raise ValueError(f"max_span must be >= 1, got {max_span}")
        self.max_span = max_span
        self._spans: dict[int, list[_Span]] = {}
        self.retrievals = 0

    # ------------------------------------------------------------------
    # Store (called at prune time)
    # ------------------------------------------------------------------

    def store_evicted(
        self,
        request_id: int,
        keys: torch.Tensor,          # (L, S, H, D) — the PRE-prune cache
        values: torch.Tensor,
        retained_positions: list[int],
    ) -> None:
        """Archive every evicted span of a just-pruned request."""
        L, S, H, D = keys.shape
        kept = set(retained_positions)
        spans: list[list[int]] = []
        cur: list[int] = []
        for p in range(S):
            if p in kept:
                if cur:
                    spans.append(cur)
                    cur = []
            else:
                cur.append(p)
                if len(cur) == self.max_span:
                    spans.append(cur)
                    cur = []
        if cur:
            spans.append(cur)

        entries = []
        for span in spans:
            idx = torch.tensor(span, device=keys.device)
            k = keys[:, idx]                                   # (L, T, H, D)
            v = values[:, idx]
            entries.append(_Span(
                positions=span,
                keys=k.to("cpu", non_blocking=True).pin_memory(),
                values=v.to("cpu", non_blocking=True).pin_memory(),
                centroids=k.float().mean(dim=1),               # (L, H, D) on GPU
            ))
        self._spans[int(request_id)] = entries

    # ------------------------------------------------------------------
    # Trigger + retrieval (called per decode step)
    # ------------------------------------------------------------------

    def score(
        self,
        request_id: int,
        q_per_layer: list[torch.Tensor],   # per layer (Hq, D) — this step's query
        retained_keys: torch.Tensor,        # (L, C, H, D) — current cache
    ) -> tuple[float, int] | None:
        """Return (margin, best_span_idx) for the negative-cache trigger.

        margin = mean over layers of (max attention logit to any evicted-span
        centroid) − (max logit to any retained key). Higher ⇒ the query is
        reaching for dropped content. Absolute values are model-scale dependent
        and typically negative (a span centroid rarely beats the max over ALL
        retained keys); what matters is the gap between need-steps and idle
        steps (~2 logits on trained Nano), so the firing threshold must be
        calibrated per model. None when the request has no archived spans.
        """
        entries = self._spans.get(int(request_id))
        if not entries:
            return None
        L, C, H, D = retained_keys.shape
        scale = 1.0 / math.sqrt(D)
        margin = 0.0
        span_total = torch.zeros(len(entries), device=retained_keys.device)
        for li in range(L):
            q = q_per_layer[li].float()                        # (Hq, D)
            group = q.shape[0] // H
            neg_l = torch.zeros(len(entries), device=retained_keys.device)
            pos_best = -float("inf")
            for g in range(H):
                qg = q[g * group:(g + 1) * group].mean(dim=0)  # (D,)
                cg = torch.stack([e.centroids[li, g] for e in entries])  # (n, D)
                neg_l += cg @ qg * scale
                pos_best = max(pos_best,
                               float((retained_keys[li, :, g].float() @ qg).max()) * scale)
            margin += float(neg_l.max()) / H - pos_best
            span_total += neg_l
        return margin / L, int(span_total.argmax())

    def take(self, request_id: int, span_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove and return one span's (keys, values), each (L, T, H, D) CPU."""
        entries = self._spans[int(request_id)]
        span = entries.pop(span_idx)
        if not entries:
            del self._spans[int(request_id)]
        self.retrievals += 1
        return span.keys, span.values

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def has(self, request_id: int) -> bool:
        return int(request_id) in self._spans

    @property
    def empty(self) -> bool:
        return not self._spans

    def drop(self, request_id: int) -> None:
        """Free a finished request's archive (CPU tensors + GPU centroids)."""
        self._spans.pop(int(request_id), None)

    def drop_all_except(self, live_ids: set[int]) -> None:
        """Garbage-collect entries whose requests are no longer live."""
        for rid in [r for r in self._spans if r not in live_ids]:
            del self._spans[rid]
