# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prefix activation cache — lookup / store / eviction semantics.

Verifies the cache key shape (BLAKE2b over prompt-prefix tokens),
deepest-layer lookup (longest matching prefix wins; deepest cached
layer within that prefix wins), LRU eviction, and isolation of
cached tensors from in-place writes on the live forward-pass
tensor.
"""

import pytest
import torch

from megatron.core.inference.activation_cache import (
    CacheHit,
    PrefixActivationCache,
    get_prefix_activation_cache,
    prefix_hash,
    set_prefix_activation_cache,
)


# ---- prefix_hash ----------------------------------------------------------


def test_prefix_hash_is_deterministic():
    """Same tokens + same prefix_len → same digest. Crucial: cache
    hits depend on this exact equality."""
    assert prefix_hash([1, 2, 3, 4], 3) == prefix_hash([1, 2, 3, 4], 3)


def test_prefix_hash_changes_with_token_change():
    """A different prefix produces a different digest — otherwise the
    cache would return the wrong activation for similar prompts."""
    assert prefix_hash([1, 2, 3], 3) != prefix_hash([1, 2, 4], 3)


def test_prefix_hash_changes_with_length():
    """``prefix_len`` matters: hashing the first 3 of [1,2,3,4] is
    NOT the same as hashing the first 4 (different prefixes)."""
    assert prefix_hash([1, 2, 3, 4], 3) != prefix_hash([1, 2, 3, 4], 4)


def test_prefix_hash_full_prompt_equals_truncated_with_matching_prefix():
    """Hashing the first 3 tokens of any sequence beginning with
    [1, 2, 3] is identical — that's the entire point of the cache
    (shared prefixes hit even when full prompts differ)."""
    assert prefix_hash([1, 2, 3], 3) == prefix_hash([1, 2, 3, 999], 3)


def test_prefix_hash_empty_prefix_is_constant():
    """Empty prefix = empty hash input; same digest for any token
    sequence. Useful only as a sentinel — no cache content keyed at
    length 0 is meaningful."""
    assert prefix_hash([], 0) == prefix_hash([42, 99], 0)


def test_prefix_hash_rejects_overlong_prefix():
    with pytest.raises(AssertionError):
        prefix_hash([1, 2], 3)


# ---- store + lookup ------------------------------------------------------


def test_store_then_lookup_returns_deepest_layer():
    """Cache stores layers 0..K-1 for a prefix; lookup returns
    skip_depth = K (i.e., the request can skip K layers)."""
    cache = PrefixActivationCache()
    prompt = [10, 20, 30, 40, 50]
    for L in range(5):
        cache.store(prompt, prefix_len=5, layer_idx=L,
                    hidden=torch.full((1, 8), float(L)))
    hit = cache.lookup(prompt)
    assert isinstance(hit, CacheHit)
    assert hit.skip_depth == 5  # layers 0..4 cached → start at layer 5
    # Starting hidden = output of layer 4.
    assert torch.allclose(hit.starting_hidden, torch.full((1, 8), 4.0))


def test_lookup_walks_back_to_longest_matching_prefix():
    """Stored under a 10-token prefix; a new prompt that diverges
    at position 8 still hits on the 8-token prefix as long as the
    cache contains an entry under that hash. The lookup walks back
    from longest to shortest prefix and returns the first hit."""
    cache = PrefixActivationCache()
    full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Insert under the 8-token prefix only.
    cache.store(full, prefix_len=8, layer_idx=3, hidden=torch.ones(1, 4))

    # New prompt: shares first 8 tokens but diverges at position 8.
    new_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 99, 100]
    hit = cache.lookup(new_prompt)
    assert hit is not None
    assert hit.skip_depth == 4  # layer 3 cached → skip 0..3, start at 4


def test_lookup_miss_returns_none():
    """No matching prefix in the cache → ``None``."""
    cache = PrefixActivationCache()
    cache.store([1, 2, 3], prefix_len=3, layer_idx=0, hidden=torch.zeros(1, 4))
    assert cache.lookup([99, 100, 101]) is None


def test_lookup_empty_cache_returns_none():
    """Bootstrapping: the first request always misses cleanly."""
    cache = PrefixActivationCache()
    assert cache.lookup([1, 2, 3]) is None


def test_lookup_respects_max_prefix_len():
    """``max_prefix_len`` caps the lookup. Even when a longer match
    exists in the cache, the lookup won't try prefixes longer than
    the cap — useful when the caller wants bounded lookup cost."""
    cache = PrefixActivationCache()
    cache.store([1, 2, 3, 4, 5], prefix_len=5, layer_idx=2,
                hidden=torch.zeros(1, 4))
    # Cache contains a 5-token entry; cap the lookup at 3 → miss.
    assert cache.lookup([1, 2, 3, 4, 5], max_prefix_len=3) is None
    # Cap == 5 → hit.
    assert cache.lookup([1, 2, 3, 4, 5], max_prefix_len=5) is not None


# ---- isolation -----------------------------------------------------------


def test_store_clones_tensor_so_inplace_writes_dont_corrupt_cache():
    """The live forward-pass tensor often gets written in place by
    subsequent layers. The cache must keep its own copy."""
    cache = PrefixActivationCache()
    live = torch.ones(1, 4)
    cache.store([1, 2, 3], prefix_len=3, layer_idx=0, hidden=live)
    live.fill_(999)  # corrupt the live tensor
    hit = cache.lookup([1, 2, 3])
    assert torch.allclose(hit.starting_hidden, torch.ones(1, 4))


def test_lookup_returns_independent_tensor_each_call():
    """The caller may write to the returned ``starting_hidden`` during
    forward. Subsequent lookups must not see those writes."""
    cache = PrefixActivationCache()
    cache.store([1, 2, 3], prefix_len=3, layer_idx=0,
                hidden=torch.full((1, 4), 7.0))
    hit1 = cache.lookup([1, 2, 3])
    hit1.starting_hidden.fill_(0)  # caller mutates the returned tensor
    hit2 = cache.lookup([1, 2, 3])
    assert torch.allclose(hit2.starting_hidden, torch.full((1, 4), 7.0))


# ---- LRU eviction --------------------------------------------------------


def test_eviction_keeps_size_at_or_below_cap():
    """Inserting beyond ``max_entries`` triggers LRU eviction. Cache
    size never exceeds the cap by more than 0 after evictions."""
    cache = PrefixActivationCache(max_entries=3)
    for i in range(5):
        cache.store([i], prefix_len=1, layer_idx=0,
                    hidden=torch.tensor([float(i)]))
    assert len(cache) == 3


def test_eviction_drops_least_recently_used_entry():
    """The least-recently-touched entry goes when capacity is hit.
    Test by inserting 3, touching the first (lookup), then inserting
    a 4th — the 2nd (not the 1st) should evict."""
    cache = PrefixActivationCache(max_entries=3)
    cache.store([1], 1, 0, torch.tensor([1.0]))
    cache.store([2], 1, 0, torch.tensor([2.0]))
    cache.store([3], 1, 0, torch.tensor([3.0]))
    # Touch prompt [1] → it's now most-recently-used; [2] is LRU.
    cache.lookup([1])
    cache.store([4], 1, 0, torch.tensor([4.0]))
    # [2] should be evicted; [1], [3], [4] survive.
    assert cache.lookup([1]) is not None
    assert cache.lookup([2]) is None
    assert cache.lookup([3]) is not None
    assert cache.lookup([4]) is not None


def test_clear_drops_all_entries():
    """`clear()` is the weight-reload hook — cached activations are
    weight-specific so a weight update must invalidate the entire
    cache."""
    cache = PrefixActivationCache()
    for i in range(10):
        cache.store([i], 1, 0, torch.zeros(1, 4))
    assert len(cache) == 10
    cache.clear()
    assert len(cache) == 0
    assert cache.lookup([0]) is None


# ---- module-level singleton ----------------------------------------------


def test_module_singleton_lazy_constructed():
    """The default cache is constructed on first ``get_...`` call,
    not at import time, so callers that don't use the cache pay zero
    memory cost."""
    set_prefix_activation_cache(None)  # reset
    cache = get_prefix_activation_cache()
    assert isinstance(cache, PrefixActivationCache)
    # Second call returns the same instance.
    assert get_prefix_activation_cache() is cache


def test_module_singleton_can_be_overridden():
    """`set_prefix_activation_cache(custom)` redirects subsequent
    lookups — useful for tests / benchmarks with a different memory
    budget."""
    custom = PrefixActivationCache(max_entries=7)
    set_prefix_activation_cache(custom)
    assert get_prefix_activation_cache() is custom
    set_prefix_activation_cache(None)  # reset for downstream tests
