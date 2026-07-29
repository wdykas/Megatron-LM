# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for prefix-cache paused-request resume accounting."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator


def _context(max_requests=4, max_blocks_per_request=4):
    return SimpleNamespace(
        paused_request_count=0,
        total_request_count=0,
        request_kv_block_counts=torch.zeros(max_requests, dtype=torch.int32),
        request_to_kv_block_ids=torch.full(
            (max_requests, max_blocks_per_request), -1, dtype=torch.int32
        ),
        prefix_cache_lru_clock=1,
        enable_prefix_caching=True,
    )


def _lru_allocator(context=None, total_count=6):
    context = context or _context()
    allocator = KVBlockAllocator(
        context,
        total_count=total_count,
        paused_count=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    context.kv_block_allocator = allocator
    return allocator


@pytest.mark.parametrize("needs_new_block", [False, True])
def test_resume_uses_request_demand_and_lru_allocatable_blocks(needs_new_block):
    """A saturated raw pool permits zero-demand resume or one LRU eviction."""
    context = _context()
    context.paused_request_count = 1
    context.total_request_count = 1
    context.block_size_tokens = 4
    context.num_speculative_tokens = 0
    context.max_requests = 4
    context.max_tokens = 16
    context.request_kv_block_counts[0] = 1
    context.request_last_kv_block_offset = torch.tensor(
        [3 if needs_new_block else 1, 0, 0, 0], dtype=torch.int32
    )
    context.request_last_kv_block_id = torch.full((4,), -1, dtype=torch.int32)

    allocator = _lru_allocator(context)
    all_blocks = allocator.allocate_memory_blocks(allocator.total_avail)
    context.request_to_kv_block_ids[0, 0] = all_blocks[0]

    cached_block = int(all_blocks[-1])
    allocator.register_kv_block_hashes([cached_block], [101], parent_hashes=[0])
    allocator.block_ref_counts[cached_block] = 0
    assert allocator.total_avail == 0

    active_count, newly_paused = DynamicInferenceContext.resume_paused_requests(
        context, active_request_count=0, newly_paused_request_ids=torch.tensor([17])
    )

    assert active_count == 1
    assert context.paused_request_count == 0
    assert newly_paused.numel() == 0
    assert context.request_kv_block_counts[0].item() == (2 if needs_new_block else 1)
    if needs_new_block:
        assert 101 not in allocator.kv_hash_to_block_id


def test_allocatable_count_excludes_reserved_lru_matches():
    allocator = _lru_allocator(total_count=5)
    blocks = allocator.allocate_memory_blocks(allocator.total_avail)
    allocator.register_kv_block_hashes(
        blocks[:2].tolist(), [101, 102], parent_hashes=[0, 101]
    )
    allocator.block_ref_counts[blocks[:2]] = 0

    assert allocator.total_avail == 0
    assert allocator.get_allocatable_block_count() == 2
    assert allocator.get_allocatable_block_count(potential_matched_count=1) == 1
    assert allocator.get_allocatable_block_count(potential_matched_count=99) == 0
    with pytest.raises(ValueError):
        allocator.get_allocatable_block_count(potential_matched_count=-1)


def test_resume_does_not_double_count_prefixes_shared_with_active_requests():
    context = _context(max_blocks_per_request=3)
    context.paused_request_count = 2
    context.total_request_count = 3
    context.block_size_tokens = 4
    context.num_speculative_tokens = 0
    context.max_requests = 4
    context.max_tokens = 16
    context.request_kv_block_counts[:3] = 2
    context.request_to_kv_block_ids[:3, :2] = torch.tensor([0, 1], dtype=torch.int32)
    context.request_last_kv_block_offset = torch.tensor([1, 1, 1, 0], dtype=torch.int32)
    context.request_last_kv_block_id = torch.tensor([1, 1, 1, -1], dtype=torch.int32)
    _lru_allocator(context)  # active partition=3 blocks, active unique use=2

    active_count, _ = DynamicInferenceContext.resume_paused_requests(
        context, active_request_count=1, newly_paused_request_ids=torch.tensor([10, 11])
    )

    assert active_count == 3
    assert context.paused_request_count == 0
