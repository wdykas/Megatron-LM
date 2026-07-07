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
from dataclasses import dataclass, field

import torch

from .transfer import build_span_store


@dataclass
class _Span:
    positions: list[int]           # original token positions (bookkeeping only)
    span_id: int                   # key into the transfer-backend store
    centroids: torch.Tensor        # (L, Hkv, D) GPU — mean evicted key per layer/group
    # Host copy kept ONLY when the flywheel logs (an on-node feature);
    # the authoritative bytes live in the SpanStore.
    cpu_keys: torch.Tensor | None = field(default=None, repr=False)


class KVArchive:
    """Per-request store of evicted KV spans with a GPU negative-cache index."""

    def __init__(self, max_span: int = 16,
                 flywheel_dir: str | None = None,
                 flywheel_max_files: int = 512,
                 transfer: str = "pinned",
                 **transfer_kwargs) -> None:
        if max_span < 1:
            raise ValueError(f"max_span must be >= 1, got {max_span}")
        self.max_span = max_span
        # Byte movement is delegated to the store: 'pinned' host memory
        # on-node, 'nixl' for a remote/disaggregated archive tier.
        self._store = build_span_store(transfer, **transfer_kwargs)
        self._next_span_id = 0
        self._spans: dict[int, list[_Span]] = {}
        self.retrievals = 0
        # Retrieval flywheel: every take() is a proven eviction MISTAKE (the
        # model's own future queries demanded the span back) and every span
        # still archived when its request finishes was a CORRECT eviction.
        # Logged per finished request as scorer training data — the archive
        # self-labels the eviction policy on real traffic. Bounded: filenames
        # rotate modulo flywheel_max_files, so disk usage is capped.
        self.flywheel_dir = flywheel_dir
        self.flywheel_max_files = flywheel_max_files
        self._flywheel_seq = 0
        self._restored: dict[int, list[_Span]] = {}
        # Speculative prefetch staging: at most one span in flight, identified
        # by (request_id, span_idx) — span indices shift on take, so staging is
        # invalidated whenever the request's span list changes.
        self._staged_key: tuple[int, int] | None = None
        self._staged: tuple[torch.Tensor, torch.Tensor, torch.cuda.Event] | None = None
        self.prefetch_hits = 0

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
            sid = self._next_span_id
            self._next_span_id += 1
            self._store.put(sid, k, v)
            entries.append(_Span(
                positions=span,
                span_id=sid,
                centroids=k.float().mean(dim=1),               # (L, H, D) on GPU
                cpu_keys=(k.detach().cpu() if self.flywheel_dir is not None
                          else None),
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

    def prefetch(self, request_id: int, span_idx: int, stream: torch.cuda.Stream) -> None:
        """Start the CPU→GPU copy of one span on a side stream (speculative).

        Called when the trigger margin crosses the *prefetch* threshold but not
        yet the firing threshold: the pinned-memory transfer overlaps subsequent
        decode steps, so a later `take` of the same span costs no PCIe stall.
        At most one span is staged; re-prefetching the staged span is a no-op.
        """
        key = (int(request_id), span_idx)
        if self._staged_key == key:
            return
        span = self._spans[int(request_id)][span_idx]
        gk, gv, event = self._store.get_async(span.span_id, stream)
        self._staged_key = key
        self._staged = (gk, gv, event)

    def take(
        self, request_id: int, span_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Remove and return one span's (keys, values, original_positions).

        keys/values are (L, T, H, D) — the staged GPU copy when this span was
        prefetched (waiting on the copy event, which has typically long
        completed), otherwise the pinned CPU tensors. Original positions let a
        RoPE-renumber caller re-rotate the keys to their new cache positions.
        Any staging for this request is invalidated — span indices shift when
        an entry is popped.
        """
        entries = self._spans[int(request_id)]
        span = entries.pop(span_idx)
        if not entries:
            del self._spans[int(request_id)]
        self.retrievals += 1
        staged = None
        if self._staged_key == (int(request_id), span_idx):
            gk, gv, event = self._staged
            torch.cuda.current_stream().wait_event(event)
            staged = (gk, gv)
            self.prefetch_hits += 1
        if self._staged_key is not None and self._staged_key[0] == int(request_id):
            self._staged_key = None
            self._staged = None
        if self.flywheel_dir is not None:
            self._restored.setdefault(int(request_id), []).append(span)
        if staged is not None:
            k, v = staged
        else:
            k, v = self._store.get(span.span_id)
        self._store.drop(span.span_id)
        return k, v, span.positions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def has(self, request_id: int) -> bool:
        return int(request_id) in self._spans

    @property
    def empty(self) -> bool:
        return not self._spans

    def _flush_flywheel(self, request_id: int,
                        remaining: list[_Span] | None) -> None:
        """Write one finished request's labelled spans (restored=1, unused=0)."""
        if self.flywheel_dir is None:
            return
        restored = self._restored.pop(int(request_id), [])
        events = ([(sp, 1) for sp in restored]
                  + [(sp, 0) for sp in (remaining or [])])
        if not events:
            return
        import os
        os.makedirs(self.flywheel_dir, exist_ok=True)
        path = os.path.join(
            self.flywheel_dir,
            f"events_{self._flywheel_seq % self.flywheel_max_files:05d}.pt")
        self._flywheel_seq += 1
        torch.save({
            "keys": [sp.cpu_keys.clone() for sp, _ in events],   # (L, T, H, D) cpu
            "positions": [sp.positions for sp, _ in events],
            "labels": [lab for _, lab in events],
        }, path)

    def drop(self, request_id: int) -> None:
        """Free a finished request's archive (store bytes + GPU centroids)."""
        remaining = self._spans.pop(int(request_id), None)
        for sp in (remaining or []):
            self._store.drop(sp.span_id)
        self._flush_flywheel(request_id, remaining)
        if self._staged_key is not None and self._staged_key[0] == int(request_id):
            self._staged_key = None
            self._staged = None

    def drop_all_except(self, live_ids: set[int]) -> None:
        """Garbage-collect entries whose requests are no longer live."""
        for rid in [r for r in self._spans if r not in live_ids]:
            dead = self._spans.pop(rid)
            for sp in dead:
                self._store.drop(sp.span_id)
            self._flush_flywheel(rid, dead)
        for rid in [r for r in self._restored if r not in live_ids]:
            self._flush_flywheel(rid, None)
        if self._staged_key is not None and self._staged_key[0] not in live_ids:
            self._staged_key = None
            self._staged = None
