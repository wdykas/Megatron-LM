# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Partial one-sided pull: the decode reuses the longest block prefix it already
has cached and pulls only the missing suffix. These tests exercise the prefix
match + ref-count accounting in isolation (no GPU / NIXL), via a fake KV block
allocator attached to a bare context."""

import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext


class _FakeKVAllocator:
    """Minimal stand-in for the prefix-caching KV block allocator: a
    hash->block registry, per-block ref counts, and a bump-pointer freelist."""

    def __init__(self, total_blocks: int):
        self.enable_prefix_caching = True
        self.kv_hash_to_block_id: dict = {}
        self.block_hashes = torch.full((total_blocks,), -1, dtype=torch.int64)
        self.block_ref_counts = torch.zeros(total_blocks, dtype=torch.int32)
        self._next = 0
        self._total = total_blocks

    def allocate_memory_blocks(self, n: int):
        if n == 0:
            return torch.empty(0, dtype=torch.int64)
        if self._next + n > self._total:
            return None
        ids = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        self.block_ref_counts[ids] += 1  # allocate with ref_count == 1
        return ids

    def release_memory_blocks(self, ids: torch.Tensor):
        self.block_ref_counts[ids] -= 1

    def register_kv_block_hashes(self, block_ids, hashes):
        for b, h in zip(block_ids, hashes):
            self.kv_hash_to_block_id[h] = int(b)
            self.block_hashes[int(b)] = int(h)


def _ctx(total_blocks: int = 64):
    """A bare context with just the fields the pull-alloc path touches."""
    ctx = DynamicInferenceContext.__new__(DynamicInferenceContext)
    ctx.kv_block_allocator = _FakeKVAllocator(total_blocks)
    ctx.memory_buffer = torch.empty((2, 1, total_blocks, 1, 1, 1))
    ctx.is_hybrid_model = False
    return ctx


def _seed_prefix(ctx, hashes, block_ids):
    """Pretend a prior request left these hash->block entries resident at ref 0."""
    ctx.kv_block_allocator.register_kv_block_hashes(block_ids, hashes)


def test_match_prefix_finds_longest_resident_prefix():
    ctx = _ctx()
    # Decode already holds blocks for hashes 10, 11, 12 (prior turn).
    _seed_prefix(ctx, [10, 11, 12], [3, 4, 5])
    # Published prompt: 10,11,12 cached + 13,14 new, trailing partial (-1).
    m = ctx.disagg_pull_match_prefix([10, 11, 12, 13, 14, -1])
    assert m["match_len"] == 3
    assert m["reused_block_ids"] == [3, 4, 5]


def test_match_prefix_stops_at_first_gap():
    ctx = _ctx()
    _seed_prefix(ctx, [10, 12], [3, 5])  # 11 missing -> prefix breaks at index 1
    m = ctx.disagg_pull_match_prefix([10, 11, 12])
    assert m["match_len"] == 1
    assert m["reused_block_ids"] == [3]


def test_match_prefix_no_hit_when_disabled_or_empty():
    ctx = _ctx()
    ctx.kv_block_allocator.enable_prefix_caching = False
    assert ctx.disagg_pull_match_prefix([10, 11]) == {"reused_block_ids": [], "match_len": 0}
    ctx.kv_block_allocator.enable_prefix_caching = True
    assert ctx.disagg_pull_match_prefix([]) == {"reused_block_ids": [], "match_len": 0}
    assert ctx.disagg_pull_match_prefix([999]) == {"reused_block_ids": [], "match_len": 0}


def test_ref_counts_net_zero_after_match_then_commit():
    """match_prefix pins reused blocks; disagg_pull_commit releases every dst
    block once. Reused blocks must return to their cached baseline (ref 0) and
    newly pulled blocks must land at ref 0 (registered, evictable) -- exactly
    the non-partial path's end state."""
    ctx = _ctx()
    _seed_prefix(ctx, [10, 11], [3, 4])
    alloc_ref = ctx.kv_block_allocator.block_ref_counts.clone()

    block_count = 4  # hashes 10,11 cached + 12,13 new
    m = ctx.disagg_pull_match_prefix([10, 11, 12, 13])
    k = m["match_len"]
    assert k == 2
    # Reused blocks are now pinned above baseline.
    assert ctx.kv_block_allocator.block_ref_counts[3] == alloc_ref[3] + 1
    assert ctx.kv_block_allocator.block_ref_counts[4] == alloc_ref[4] + 1

    alloc = ctx.disagg_pull_alloc(block_count - k)
    new_ids = alloc["block_ids"]
    assert len(new_ids) == 2
    dst = list(m["reused_block_ids"]) + list(new_ids)

    ctx.disagg_pull_commit(dst, [10, 11, 12, 13])

    rc = ctx.kv_block_allocator.block_ref_counts
    # Reused blocks back to baseline; new blocks at ref 0 and registered.
    assert rc[3] == alloc_ref[3] and rc[4] == alloc_ref[4]
    for b in new_ids:
        assert rc[int(b)] == 0
    assert ctx.kv_block_allocator.kv_hash_to_block_id[12] == int(new_ids[0])
    assert ctx.kv_block_allocator.kv_hash_to_block_id[13] == int(new_ids[1])


def test_pull_alloc_zero_count_full_prefix_hit():
    """A full prefix hit pulls no KV; disagg_pull_alloc(0) returns an empty
    block list rather than failing."""
    ctx = _ctx()
    alloc = ctx.disagg_pull_alloc(0)
    assert alloc is not None
    assert alloc["block_ids"] == []
    assert alloc["mamba_dst_slot"] == -1


def test_unmatch_releases_transient_pin():
    ctx = _ctx()
    _seed_prefix(ctx, [10, 11], [3, 4])
    baseline = ctx.kv_block_allocator.block_ref_counts.clone()
    m = ctx.disagg_pull_match_prefix([10, 11, 12])
    assert ctx.kv_block_allocator.block_ref_counts[3] == baseline[3] + 1
    ctx.disagg_pull_unmatch(m["reused_block_ids"])
    assert torch.equal(ctx.kv_block_allocator.block_ref_counts, baseline)
