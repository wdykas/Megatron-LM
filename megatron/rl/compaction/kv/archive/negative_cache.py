# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU KV archive + negative-cache index: eviction as demotion, not deletion.

When live compaction evicts a span of KV, the exact tensors move to pinned CPU
memory and a tiny GPU index of the *evicted* content stays behind — one mean-key
centroid per contiguous span per layer/KV-group (the "negative cache": an index
of what is absent from the GPU cache). At decode time, ``span_alphas`` scores
each span's centroid attention-mass fraction α̂ — the estimated share of this
step's attention that WANTS that evicted span, normalized against the retained
keys' mass. The serving trigger (LiveKVCompactor) fires a span whose α̂ spikes
above its own running baseline: scale-free, no per-model calibration, and the
firing span is also the span to restore (identification measured reliable on
needle captures even under mass eviction).

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
        # Overlapped trigger engine: per-request epoch state (see
        # build_trigger_epoch/launch_score/poll_verdict).
        self._trigger_epochs: dict[int, KVArchive._TriggerEpoch] = {}

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

        # One gather + one pinned slab + one async D2H for the whole request
        # (the per-span path costs 2 synchronous pinned allocations per span).
        all_idx = torch.tensor([p for sp in spans for p in sp],
                               device=keys.device)
        bounds = [0]
        for sp in spans:
            bounds.append(bounds[-1] + len(sp))
        k_all = keys[:, all_idx]                               # (L, T_evicted, H, D)
        v_all = values[:, all_idx]
        span_ids = list(range(self._next_span_id,
                              self._next_span_id + len(spans)))
        self._next_span_id += len(spans)
        if hasattr(self._store, "put_bulk"):
            self._store.put_bulk(span_ids, k_all, v_all, bounds)
        else:                                                  # nixl tier
            for i, sid in enumerate(span_ids):
                self._store.put(sid, k_all[:, bounds[i]:bounds[i + 1]],
                                v_all[:, bounds[i]:bounds[i + 1]])
        entries = []
        for i, (span, sid) in enumerate(zip(spans, span_ids)):
            k = k_all[:, bounds[i]:bounds[i + 1]]
            entries.append(_Span(
                positions=span,
                span_id=sid,
                centroids=k.float().mean(dim=1),               # (L, H, D) on GPU
                cpu_keys=(k.detach().cpu() if self.flywheel_dir is not None
                          else None),
            ))
        # APPEND across compaction rounds (recursive compaction re-evicts
        # from a cache that already has archived spans); positions from later
        # rounds are indices into the THEN-current compacted cache, which is
        # fine — positions are only used for RoPE renumbering at restore.
        self._spans.setdefault(int(request_id), []).extend(entries)

    # ------------------------------------------------------------------
    # Trigger + retrieval (called per decode step)
    # ------------------------------------------------------------------

    def span_alphas(
        self,
        request_id: int,
        q_per_layer: list[torch.Tensor],   # per layer (Hq, D) — this step's query
        retained_keys: torch.Tensor,        # (L, C, H, D) — current cache
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Per-span attention-mass fractions α̂ for the retrieval trigger.

        α̂_i = |E_i|·exp(q·c_i) / (Σ_retained exp(q·k) + Σ_j |E_j|·exp(q·c_j)),
        aggregated as the max over (layer, KV group). The centroid estimate is
        deliberately NOT exact span mass: averaging suppresses spans whose heat
        comes from one incoherent outlier key while preserving semantically
        aligned spans — measured on needle captures it separates need (0.25-
        0.37) from idle (0.13-0.15) where exact per-span mass does not
        (chronically hot filler spans). Scale-free in [0, 1): one threshold
        transfers across models and eviction patterns, unlike the old
        max-logit margin (removed — it required per-model calibration and had
        no separation at all under content-blind mass eviction).

        Returns (alphas (n_spans,), span_ids) or None when nothing is archived.
        """
        entries = self._spans.get(int(request_id))
        if not entries:
            return None
        L, C, H, D = retained_keys.shape
        scale = 1.0 / math.sqrt(D)
        counts = torch.tensor([len(e.positions) for e in entries],
                              device=retained_keys.device, dtype=torch.float32)
        best = torch.zeros(len(entries), device=retained_keys.device)
        for li in range(L):
            q = q_per_layer[li].float()                        # (Hq, D)
            group = q.shape[0] // H
            for g in range(H):
                qg = q[g * group:(g + 1) * group].mean(dim=0)  # (D,)
                lr = (retained_keys[li, :, g].float() @ qg) * scale        # (C,)
                cg = torch.stack([e.centroids[li, g] for e in entries])    # (n, D)
                le = (cg @ qg) * scale                                     # (n,)
                mx = torch.maximum(lr.max(), le.max())
                denom_r = torch.exp(lr - mx).sum()
                mass = counts * torch.exp(le - mx)
                alphas = mass / (denom_r + mass.sum())
                best = torch.maximum(best, alphas)
        return best, [e.span_id for e in entries]

    # ------------------------------------------------------------------
    # Overlapped trigger engine (CUDA-graph-era serving path)
    #
    # The eager per-step path (span_alphas + host EMA/CUSUM) costs a device
    # sync, a full retained-KV gather, and an L*H Python matmul loop on the
    # decode critical path EVERY step. The overlapped engine removes all
    # three: per-request state is cached per EPOCH (rebuilt only on
    # compaction/retrieval events or every refresh interval), scoring runs
    # batched on a SIDE stream reading a snapshot of the step's Q, EMA/CUSUM
    # update on-device, and only a 6-scalar verdict crosses to pinned host
    # memory asynchronously. The verdict is consumed ONE STEP LATE — the
    # CUSUM already integrates evidence across steps, so a one-token delay
    # is semantically free. The main stream waits only on the tiny Q-row
    # snapshot, so heavy scoring fully overlaps the next decode step.
    # ------------------------------------------------------------------

    class _TriggerEpoch:
        __slots__ = ("span_ids", "centroids", "counts", "retained", "ema",
                     "cusum", "seeded", "q_scratch", "verdict_dev",
                     "verdict_host", "event", "launched", "age")

    def build_trigger_epoch(self, request_id: int,
                            retained_keys: torch.Tensor) -> None:
        """(Re)build cached trigger state for one request.

        retained_keys (L, C, H, D) is gathered ONCE here — not per step.
        EMA/CUSUM state survives the rebuild for surviving span_ids (the
        chronic-heat suppression must not reset every epoch)."""
        rid = int(request_id)
        entries = self._spans.get(rid)
        if not entries:
            self._trigger_epochs.pop(rid, None)
            return
        dev = retained_keys.device
        ep = KVArchive._TriggerEpoch()
        ep.span_ids = [e.span_id for e in entries]
        # (L, H, n, D) stacked centroids; counts (n,)
        ep.centroids = torch.stack([e.centroids for e in entries], dim=2).float()
        ep.counts = torch.tensor([len(e.positions) for e in entries],
                                 device=dev, dtype=torch.float32)
        ep.retained = retained_keys.float()
        n = len(entries)
        ep.ema = torch.zeros(n, device=dev)
        ep.cusum = torch.zeros(n, device=dev)
        ep.seeded = False
        prev = self._trigger_epochs.get(rid)
        if prev is not None and prev.seeded:
            # carry state for surviving spans (matched by span_id)
            old = {sid: i for i, sid in enumerate(prev.span_ids)}
            keep = [(j, old[sid]) for j, sid in enumerate(ep.span_ids)
                    if sid in old]
            if keep:
                jj = torch.tensor([j for j, _ in keep], device=dev)
                oo = torch.tensor([o for _, o in keep], device=dev)
                ep.ema[jj] = prev.ema[oo]
                ep.cusum[jj] = prev.cusum[oo]
                ep.seeded = True
        ep.q_scratch = None      # (L, Hq, D) allocated on first launch
        # verdict: [fire_idx, fire_alpha, best_idx, best_cusum, alpha_max, n]
        ep.verdict_dev = torch.zeros(6, device=dev)
        ep.verdict_host = torch.zeros(6, device="cpu", pin_memory=True)
        ep.event = torch.cuda.Event()
        ep.launched = False
        ep.age = 0
        self._trigger_epochs[rid] = ep

    def trigger_epoch_valid(self, request_id: int, refresh_every: int = 32) -> bool:
        ep = self._trigger_epochs.get(int(request_id))
        return ep is not None and ep.age < refresh_every

    def invalidate_trigger_epoch(self, request_id: int) -> None:
        self._trigger_epochs.pop(int(request_id), None)

    def launch_score(self, request_id: int, q_rows: list[torch.Tensor],
                     stream: torch.cuda.Stream, alpha_thr: float,
                     cusum_thr: float, ema_decay: float,
                     cusum_drift: float) -> None:
        """Enqueue this step's trigger scoring on `stream` (no sync).

        q_rows: per-layer (Hq, D) views of THIS step's Q (static buffers under
        graphed decode). The main stream is made to wait only on the tiny
        Q snapshot; everything downstream overlaps the next decode step."""
        rid = int(request_id)
        ep = self._trigger_epochs.get(rid)
        if ep is None:
            return
        main = torch.cuda.current_stream()
        L, C, H, D = ep.retained.shape
        with torch.cuda.stream(stream):
            stream.wait_stream(main)                    # replay wrote Q
            q = torch.stack([r.float() for r in q_rows])   # snapshot (L,Hq,D)
            if ep.q_scratch is None:
                ep.q_scratch = torch.empty_like(q)
            ep.q_scratch.copy_(q)
            snap_done = torch.cuda.Event()
            snap_done.record(stream)
            Hq = ep.q_scratch.shape[1]
            group = Hq // H
            qg = ep.q_scratch.view(L, H, group, D).mean(dim=2)   # (L, H, D)
            scale = 1.0 / math.sqrt(D)
            # retained logits (L, C, H) and centroid logits (L, H, n)
            lr = torch.einsum("lchd,lhd->lch", ep.retained, qg) * scale
            le = torch.einsum("lhnd,lhd->lhn", ep.centroids, qg) * scale
            mx = torch.maximum(lr.amax(dim=1),                  # (L, H)
                               le.amax(dim=2))
            denom_r = torch.exp(lr - mx[:, None, :]).sum(dim=1)     # (L, H)
            mass = ep.counts[None, None, :] * torch.exp(le - mx[..., None])
            alphas = (mass / (denom_r[..., None] + mass.sum(dim=2, keepdim=True)))
            a = alphas.amax(dim=(0, 1))                          # (n,)
            if not ep.seeded:
                ep.ema.copy_(a)                                  # seed = first alpha
                ep.seeded = True
            ep.cusum.copy_(torch.clamp(ep.cusum + a - ep.ema - cusum_drift,
                                       min=0.0))
            ep.ema.mul_(ema_decay).add_(a, alpha=1.0 - ema_decay)
            fired = (a >= alpha_thr) | (ep.cusum >= cusum_thr)
            fire_alpha = torch.where(fired, a, torch.zeros_like(a))
            v = ep.verdict_dev
            v[0] = torch.where(fired.any(), fire_alpha.argmax().float(),
                               torch.tensor(-1.0, device=a.device))
            v[1] = fire_alpha.max()
            v[2] = ep.cusum.argmax().float()
            v[3] = ep.cusum.max()
            v[4] = a.max()
            v[5] = float(len(ep.span_ids))
            ep.verdict_host.copy_(v, non_blocking=True)
            ep.event.record(stream)
        # Next replay may overwrite the Q buffers only after the snapshot.
        main.wait_event(snap_done)
        ep.launched = True
        ep.age += 1

    def poll_verdict(self, request_id: int):
        """Non-blocking read of the LAST launched verdict.

        Returns (fire_span_id | None, fire_alpha, best_span_id, best_cusum,
        alpha_max, n_spans) or None when nothing is ready. Never syncs."""
        ep = self._trigger_epochs.get(int(request_id))
        if ep is None or not ep.launched or not ep.event.query():
            return None
        v = ep.verdict_host
        fire_i = int(v[0].item())
        n = len(ep.span_ids)
        best_i = min(max(int(v[2].item()), 0), n - 1)
        return (ep.span_ids[fire_i] if 0 <= fire_i < n else None,
                float(v[1]), ep.span_ids[best_i], float(v[3]),
                float(v[4]), n)

    def remove_span_from_epoch(self, request_id: int, span_id: int) -> None:
        """Drop one span's rows from the cached trigger epoch in place.

        A retrieval removes the span from the archive; rebuilding the whole
        epoch (retained-KV regather + centroid restack + state carry) per
        fire is the expensive path — dropping one column is O(n_spans).
        The retained set also grew by the restored tokens; the denominator
        drift is bounded by the epoch refresh interval."""
        rid = int(request_id)
        ep = self._trigger_epochs.get(rid)
        if ep is None or span_id not in ep.span_ids:
            return
        j = ep.span_ids.index(span_id)
        if len(ep.span_ids) == 1:
            self._trigger_epochs.pop(rid, None)
            return
        keep = torch.tensor([i for i in range(len(ep.span_ids)) if i != j],
                            device=ep.counts.device)
        ep.span_ids.pop(j)
        ep.centroids = ep.centroids[:, :, keep]
        ep.counts = ep.counts[keep]
        ep.ema = ep.ema[keep]
        ep.cusum = ep.cusum[keep]
        ep.launched = False   # verdict indices are stale for the new layout

    def span_index(self, request_id: int, span_id: int) -> int | None:
        entries = self._spans.get(int(request_id), [])
        for i, e in enumerate(entries):
            if e.span_id == span_id:
                return i
        return None

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
        self._trigger_epochs.pop(int(request_id), None)
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
        for rid in [r for r in self._trigger_epochs if r not in live_ids]:
            del self._trigger_epochs[rid]
        for rid in [r for r in self._restored if r not in live_ids]:
            self._flush_flywheel(rid, None)
        if self._staged_key is not None and self._staged_key[0] not in live_ids:
            self._staged_key = None
            self._staged = None
