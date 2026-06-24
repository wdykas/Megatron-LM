# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Server-side per-session belief state compaction for multi-turn inference.

BeliefServerCompactor maintains a compact POMDP belief state keyed by session ID.
On each turn where the caller opts in (compact=True), it:

  1. Reads the current KV matrices from the paged cache (post-prefill).
  2. Merges them with any existing belief via BeliefUpdater.forward(), or
     bootstraps via BeliefUpdater.initial_compress() on the first turn.
  3. Writes the new compact belief back into the paged cache via
     hook.apply_belief_memory_for_request().
  4. Persists the updated BeliefMemory in BeliefSessionStore keyed by session_id.

Integration
-----------
In the request handler, BEFORE calling client.add_request():

    compactor.register(session_id="conv-123", do_compact=True)
    result = await client.add_request(tokens, params)

In the engine generation loop, after each engine.step():

    compactor.maybe_compact(engine)

Register calls must be in the same FIFO order as the engine processes requests.
This is guaranteed for single-DP, single-GPU serving (common case).

Session lifecycle
-----------------
Sessions persist until evicted by TTL (default 1 hour) or deleted explicitly:

    compactor.session_store.delete("conv-123")

The belief step counter (BeliefSession.step) tracks how many compactions have
fired for a session; useful for curriculum-aware serving.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch

log = logging.getLogger("compaction.belief_server")


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

@dataclass
class BeliefSession:
    """One conversation's persisted belief state."""
    session_id: str
    memory: Any          # BeliefMemory | None (None before first compact)
    step: int = 0
    last_updated: float = field(default_factory=time.monotonic)


class BeliefSessionStore:
    """Thread-safe mapping from session_id to BeliefSession with TTL eviction."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._store: dict[str, BeliefSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> BeliefSession | None:
        with self._lock:
            return self._store.get(session_id)

    def put(self, session_id: str, memory: Any) -> None:
        with self._lock:
            existing = self._store.get(session_id)
            self._store[session_id] = BeliefSession(
                session_id=session_id,
                memory=memory,
                step=(existing.step + 1) if existing else 0,
                last_updated=time.monotonic(),
            )

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def evict_expired(self) -> int:
        """Remove sessions not updated within TTL. Returns count evicted."""
        cutoff = time.monotonic() - self._ttl
        with self._lock:
            expired = [sid for sid, s in self._store.items() if s.last_updated < cutoff]
            for sid in expired:
                del self._store[sid]
        if expired:
            log.info(f"BeliefSessionStore: evicted {len(expired)} expired sessions")
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------

class BeliefServerCompactor:
    """Post-prefill belief state compaction for multi-turn inference.

    Parameters
    ----------
    updater:
        Trained BeliefUpdater (wraps a PerceiverCompactor).
    session_store:
        Optional external BeliefSessionStore.  A fresh one is created if omitted.
    min_seq_len:
        Skip compaction when the current KV sequence is shorter than this.
    """

    def __init__(
        self,
        updater: Any,                           # BeliefUpdater
        session_store: BeliefSessionStore | None = None,
        min_seq_len: int = 8,
    ) -> None:
        self.updater = updater
        self.session_store = session_store or BeliefSessionStore()
        self.min_seq_len = min_seq_len

        # FIFO queue of (session_id, do_compact) registered before each add_request
        self._pending: deque[tuple[str, bool]] = deque()
        # id(request_to_kv_block_ids[b_global]) → already processed
        self._seen_ids: set[int] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, session_id: str, do_compact: bool = True) -> None:
        """Register compaction intent for the next request submitted to the engine.

        Must be called BEFORE client.add_request() so the queue stays in sync
        with engine processing order.
        """
        with self._lock:
            self._pending.append((session_id, do_compact))

    def maybe_compact(self, engine: Any) -> None:
        """Fire belief compaction for any request that just completed prefill.

        Call this after every engine.step() in the generation loop.
        """
        try:
            ctx = engine.controller.inference_wrapped_model.inference_context
        except AttributeError:
            return
        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return

        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return

        from megatron.rl.compaction.kv.megatron_hook import MegatronInferenceHook
        hook = MegatronInferenceHook(ctx)
        BS = ctx.block_size_tokens

        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            req_key = id(ctx.request_to_kv_block_ids[b_global])

            with self._lock:
                if req_key in self._seen_ids:
                    continue
                self._seen_ids.add(req_key)

                if not self._pending:
                    log.warning(
                        "BeliefServerCompactor: no pending registration for "
                        f"b_global={b_global}; skipping"
                    )
                    continue
                session_id, do_compact = self._pending.popleft()

            if not do_compact:
                continue

            n_blocks = int(ctx.request_kv_block_counts[b_global].item())
            last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
            seq_len = (n_blocks - 1) * BS + last_offset + 1

            if seq_len < self.min_seq_len:
                log.debug(
                    f"BeliefServerCompactor: seq_len={seq_len} < min={self.min_seq_len}; "
                    "skipping"
                )
                continue

            kv = hook.get_kv_matrices()
            if kv is None:
                log.warning("BeliefServerCompactor: get_kv_matrices() returned None; skipping")
                continue
            keys_per_layer, vals_per_layer = kv

            # Slice to this batch element: (B, S, d) → (1, S, d)
            req_keys = [k[b_local : b_local + 1] for k in keys_per_layer]
            req_vals = [v[b_local : b_local + 1] for v in vals_per_layer]

            session = self.session_store.get(session_id)
            try:
                with torch.no_grad():
                    if session is not None and session.memory is not None:
                        new_memory = self.updater.forward(session.memory, req_keys, req_vals)
                    else:
                        new_memory = self.updater.initial_compress(req_keys, req_vals)
                new_memory = new_memory.detach()
            except Exception as exc:
                log.error(
                    f"BeliefServerCompactor: belief update failed for "
                    f"session={session_id!r}: {exc}"
                )
                continue

            try:
                hook.apply_belief_memory_for_request(b_local, new_memory)
            except Exception as exc:
                log.error(
                    f"BeliefServerCompactor: KV injection failed for "
                    f"session={session_id!r}: {exc}"
                )
                continue

            self.session_store.put(session_id, new_memory)

            prev_step = session.step if session else -1
            log.info(
                f"BeliefServerCompactor: session={session_id!r} "
                f"step={prev_step + 1} "
                f"seq={seq_len} → {new_memory.budget} tokens"
            )

    def reset(self) -> None:
        """Clear per-run tracking state (call between eval runs)."""
        with self._lock:
            self._seen_ids.clear()
            self._pending.clear()
