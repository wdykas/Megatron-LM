# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefix activation cache for layer-kind disaggregation.

Two RL rollouts that share a prompt prefix (e.g. a system prompt
followed by different user turns) produce *identical* per-layer
hidden states through every layer that processes only the shared
prefix tokens. Today the engine re-runs those layers from scratch
on every rollout — wasted compute proportional to ``shared_prefix
× layer_count``.

The activation cache trades a fixed amount of GPU memory for that
work: each layer's output hidden state is stored keyed by the
``(prompt_prefix_hash, layer_idx)`` pair. On a cache hit the
dispatcher skips ``run_local`` for layers covered by the cache and
hands the deepest cached hidden state to the first uncovered
layer.

Scope (v1):

- **Entry-shard caching only.** Cross-shard cache sharing
  (caching activations after a NVSHMEM recv on an intermediate
  shard) is a follow-up. The entry shard sees every layer it owns
  before the first hop-out, which is where most prefix compute
  lives.
- **Deterministic sampling only.** Stochastic sampling
  (temperature > 0 or non-zero top_p, etc.) breaks the
  per-rollout determinism the cache assumes. The
  :class:`PrefixActivationCache` doesn't try to detect this — the
  *caller* (the request submit path) is responsible for skipping
  the cache when sampling is non-deterministic. Same prefix +
  different sampling can still hit on the prefill layers because
  sampling only matters after the lm-head.
- **No cross-request mutation.** Stored tensors are detached and
  cloned at store time so writes by future forward passes can't
  corrupt cache entries.

The implementation is deliberately a plain Python dict + LRU
ordering (``collections.OrderedDict``) — we hit the cache O(1)
times per request, not per layer, so a heavier index structure
wouldn't pay off at the API boundary.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch


def prefix_hash(prompt_tokens: Sequence[int], prefix_len: int) -> bytes:
    """Stable hash of ``prompt_tokens[:prefix_len]``.

    Uses BLAKE2b — fast, cryptographically strong enough that
    collisions in practice are zero, and the digest is short (16
    bytes) so cache keys stay cheap to compare. The hash is over the
    bytes of the underlying integer ids; we don't include the prefix
    length in the hash payload because the caller picks the boundary.

    Returns:
        16-byte digest. Use as a dict key.
    """
    assert prefix_len >= 0
    assert prefix_len <= len(prompt_tokens)
    h = hashlib.blake2b(digest_size=16)
    # 4 bytes per token id is plenty (vocabs are well under 2^32).
    for tok in prompt_tokens[:prefix_len]:
        h.update(int(tok).to_bytes(4, "big", signed=False))
    return h.digest()


@dataclass(frozen=True)
class CacheHit:
    """Result of a successful prefix cache lookup.

    Attributes:
        skip_depth: The forward pass can skip every layer with
            ``layer_idx < skip_depth`` and start with
            :attr:`starting_hidden` as the inbound activation at
            ``layer_idx == skip_depth``. ``0`` means no skipping
            (treat as a miss).
        starting_hidden: Hidden state at the boundary — equal to
            ``output_of_layer(skip_depth - 1)``. Always a fresh
            tensor (cloned at store time) so the caller can write
            to it in place during the forward pass.
    """

    skip_depth: int
    starting_hidden: torch.Tensor


class PrefixActivationCache:
    """LRU cache of per-layer hidden states keyed by prompt prefix.

    The cache layout is:

        (prefix_hash, layer_idx) → hidden_tensor

    where ``prefix_hash`` covers some number of leading prompt tokens
    and ``layer_idx`` identifies the layer whose *output* the tensor
    represents (so the tensor is the input to layer ``layer_idx + 1``).

    Lookup returns the *deepest* layer index cached for the given
    prefix hash, so a request that shares a long prefix with a prior
    rollout skips as many layers as possible.

    Args:
        max_entries: Soft cap on total ``(prefix, layer)`` entries.
            When exceeded, LRU eviction kicks in. Each entry holds
            one hidden tensor (typically ``(batch=1, hidden_dim)``
            for prefill caching), so memory budget ≈ ``max_entries ×
            batch × hidden_dim × dtype_bytes``.
        device: CUDA device on which cache tensors live. Caches
            cross-device would require copying on lookup — kept
            simple in v1.
    """

    def __init__(
        self,
        max_entries: int = 1024,
        device: Optional[torch.device] = None,
    ) -> None:
        assert max_entries > 0
        self._max_entries = max_entries
        self._device = device
        # OrderedDict gives O(1) insertion + LRU move-to-end.
        self._entries: "OrderedDict[Tuple[bytes, int], torch.Tensor]" = OrderedDict()
        # Index from prefix_hash → set of cached layer indices, so
        # lookup_deepest is O(1) on hit (find max of the set). Kept
        # in sync with _entries on insert / evict.
        self._layers_by_prefix: Dict[bytes, set] = {}

    # ---- public API -------------------------------------------------------

    def lookup(
        self,
        prompt_tokens: Sequence[int],
        max_prefix_len: Optional[int] = None,
    ) -> Optional[CacheHit]:
        """Look up the deepest cached layer for the given prompt's
        longest matching prefix.

        Searches the cache for any (prefix_hash, layer_idx) entry
        whose hash matches some prefix of ``prompt_tokens``. The
        ``prefix_hash`` is opaque, so this implementation tries the
        full prompt first and walks back — typical RL rollouts share
        a *long* fixed prefix (system prompt + few-shot examples)
        and diverge near the end, so the full-prompt try usually
        wins or fails fast.

        Args:
            prompt_tokens: The new request's prompt token sequence.
            max_prefix_len: Cap on how many leading tokens to
                consider as part of the prefix. ``None`` means the
                full prompt. Smaller caps make the lookup cheaper at
                the cost of missing longer matches.

        Returns:
            A :class:`CacheHit` with the deepest covered layer, or
            ``None`` on a complete miss.
        """
        cap = len(prompt_tokens) if max_prefix_len is None else max_prefix_len
        cap = min(cap, len(prompt_tokens))
        for prefix_len in range(cap, 0, -1):
            h = prefix_hash(prompt_tokens, prefix_len)
            layers = self._layers_by_prefix.get(h)
            if not layers:
                continue
            deepest = max(layers)
            key = (h, deepest)
            tensor = self._entries.get(key)
            if tensor is None:
                continue  # raced with eviction
            # LRU bump.
            self._entries.move_to_end(key)
            # Return a fresh tensor — callers write to it during forward.
            return CacheHit(
                skip_depth=deepest + 1,
                starting_hidden=tensor.clone(),
            )
        return None

    def store(
        self,
        prompt_tokens: Sequence[int],
        prefix_len: int,
        layer_idx: int,
        hidden: torch.Tensor,
    ) -> None:
        """Store the output of layer ``layer_idx`` for the prompt
        prefix ``prompt_tokens[:prefix_len]``.

        Clones + detaches ``hidden`` so subsequent in-place ops on
        the live forward-pass tensor don't corrupt the cache entry.
        Evicts LRU entries when ``max_entries`` is exceeded.
        """
        h = prefix_hash(prompt_tokens, prefix_len)
        key = (h, int(layer_idx))
        cached = hidden.detach().clone()
        if self._device is not None and cached.device != self._device:
            cached = cached.to(self._device)
        if key in self._entries:
            self._entries.move_to_end(key)
            self._entries[key] = cached
            return
        self._entries[key] = cached
        self._layers_by_prefix.setdefault(h, set()).add(int(layer_idx))
        self._evict_lru_until_under_cap()

    def clear(self) -> None:
        """Drop every entry. Useful between epochs / when weights
        change (cached activations are weight-specific)."""
        self._entries.clear()
        self._layers_by_prefix.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # ---- internals --------------------------------------------------------

    def _evict_lru_until_under_cap(self) -> None:
        while len(self._entries) > self._max_entries:
            key, _ = self._entries.popitem(last=False)
            h, layer_idx = key
            layers = self._layers_by_prefix.get(h)
            if layers is not None:
                layers.discard(layer_idx)
                if not layers:
                    self._layers_by_prefix.pop(h, None)


# Module-level singleton. Caches are weight-specific — clear() on
# weight reload. ``None`` until first ``get_prefix_activation_cache``
# call so callers that don't use the cache pay zero memory cost.
_cache: Optional[PrefixActivationCache] = None


def get_prefix_activation_cache() -> PrefixActivationCache:
    """Return the process-wide activation cache, lazy-constructing
    on first access. Each rank holds its own cache — they only get
    populated by *that rank's* forward pass, so cross-rank coherence
    isn't a concern (matched-prefix rollouts on different ranks each
    compute their own activations the first time and hit afterwards).
    """
    global _cache
    if _cache is None:
        _cache = PrefixActivationCache()
    return _cache


def set_prefix_activation_cache(
    cache: Optional[PrefixActivationCache],
) -> None:
    """Override the singleton (for tests, or to install a custom
    cache with a different memory budget). Pass ``None`` to reset to
    the default lazy-construction behavior."""
    global _cache
    _cache = cache
