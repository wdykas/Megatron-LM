# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only regressions for prefix-cache allocator failure modes.

These intentionally avoid constructing a model or CUDA context so allocator
invariants can be checked in presubmit and on developer hosts without a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.attention_context.mamba_metadata import (
    MambaMetadata,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.contexts.mamba_slot_allocator import (
    MambaSlotAllocator,
    MambaSlotCapacityError,
)


def _context(max_requests=4, max_blocks_per_request=4):
    return SimpleNamespace(
        paused_request_count=0,
        total_request_count=0,
        request_kv_block_counts=torch.zeros(max_requests, dtype=torch.int32),
        request_to_kv_block_ids=torch.full(
            (max_requests, max_blocks_per_request), -1, dtype=torch.int32
        ),
        prefix_cache_lru_clock=1,
    )


def _lru_allocator(context=None, total_count=6):
    context = context or _context()
    context.enable_prefix_caching = True
    allocator = KVBlockAllocator(
        context,
        total_count=total_count,
        paused_count=1,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    context.kv_block_allocator = allocator
    return allocator


def _cpu_mamba_slot_allocator(max_slots=2, total_blocks=5):
    """Construct only the CPU bookkeeping portion of MambaSlotAllocator."""
    context = _context()
    context.max_requests = 4
    context.batch_dimensions = SimpleNamespace(prefill_req_count=0, decode_req_count=0)
    context.prefix_caching_eviction_policy = PrefixCachingEvictionPolicy.LRU
    context.kv_block_allocator = _lru_allocator(context, total_count=total_blocks)

    allocator = object.__new__(MambaSlotAllocator)
    allocator.context = context
    allocator.max_slots = max_slots
    allocator.block_to_slot = torch.full((total_blocks,), -1, dtype=torch.int32)
    allocator.slot_to_block = torch.full((max_slots,), -1, dtype=torch.int32)
    allocator.free_slots = torch.arange(max_slots, dtype=torch.int32)
    allocator.free_count = max_slots
    allocator.hash_to_block_id = {}

    # reset() and commit failure cleanup bookkeeping.
    allocator.intermediate_ssm_out = torch.zeros(1)
    allocator.intermediate_conv_out = torch.zeros(1)
    allocator._intermediate_offsets_cpu = torch.zeros((4, 3), dtype=torch.int32)
    allocator._intermediate_counts_cpu = torch.zeros(4, dtype=torch.int32)
    allocator._intermediate_block_ids_cpu = torch.full((4, 3), -1, dtype=torch.int32)
    allocator._eos_cache_block_id_cpu = torch.full((4,), -1, dtype=torch.int32)
    allocator._has_intermediates = False
    return allocator


def test_kv_reset_under_inference_mode_keeps_mutable_free_pool():
    """Engine reset must not replace block_bag with an inference tensor."""
    allocator = _lru_allocator()
    allocator.allocate_memory_blocks(2)

    with torch.inference_mode():
        allocator.reset()

    blocks = allocator.allocate_memory_blocks(2)
    allocator.release_memory_blocks(blocks)
    assert allocator.total_avail == allocator.total_count - 1
    assert len(set(allocator.block_bag[: allocator.total_avail].tolist())) == (
        allocator.total_avail
    )


def test_mamba_slot_reset_under_inference_mode_keeps_mutable_free_pool():
    allocator = _cpu_mamba_slot_allocator()
    allocator.allocate_slots_batch([0])

    with torch.inference_mode():
        allocator.reset()

    allocator.allocate_slots_batch([1])
    allocator.invalidate_block(1)
    assert allocator.free_count == allocator.max_slots


def test_mamba_live_slot_reset_under_inference_mode_keeps_mutable_free_pool():
    metadata = object.__new__(MambaMetadata)
    metadata.max_requests = 3
    metadata.request_to_mamba_state_idx = torch.full((3,), -1, dtype=torch.int32)
    metadata.mamba_state_free_slots = torch.arange(3, dtype=torch.int32)
    metadata.mamba_state_free_slot_count = 3

    with torch.inference_mode():
        metadata.reset()

    slot = metadata.batch_allocate_slots(1)
    metadata.request_to_mamba_state_idx[0] = slot[0]
    metadata.free_slots(torch.tensor([0], dtype=torch.int64))
    assert metadata.mamba_state_free_slot_count == metadata.max_requests


def test_mamba_slot_capacity_failure_is_atomic():
    allocator = _cpu_mamba_slot_allocator(max_slots=2)
    allocator.context.kv_block_allocator.block_ref_counts.fill_(1)
    allocator.allocate_slots_batch([0])

    free_before = allocator.free_count
    free_pool_before = allocator.free_slots.clone()
    block_map_before = allocator.block_to_slot.clone()
    slot_map_before = allocator.slot_to_block.clone()

    with pytest.raises(MambaSlotCapacityError):
        allocator.allocate_slots_batch([1, 2])

    assert allocator.free_count == free_before
    assert torch.equal(allocator.free_slots, free_pool_before)
    assert torch.equal(allocator.block_to_slot, block_map_before)
    assert torch.equal(allocator.slot_to_block, slot_map_before)


def test_mamba_snapshot_capacity_pressure_does_not_abort_generation():
    """A durable snapshot is optional and must be dropped when every slot is pinned."""
    allocator = _cpu_mamba_slot_allocator(max_slots=1)
    allocator.context.kv_block_allocator.block_ref_counts.fill_(1)
    allocator.allocate_slots_batch([0])
    allocator._has_intermediates = True
    allocator._collect_commit_data = lambda: ([1], [0], [], [], [101])

    allocator.commit_intermediate_states()

    assert allocator.block_to_slot.tolist() == [0, -1, -1, -1, -1]
    assert allocator.free_count == 0
    assert allocator._has_intermediates is False


def test_prefix_match_stops_at_first_missing_ancestor():
    """A discontinuous hash table must be a short match, not a KeyError."""
    context = SimpleNamespace(
        enable_prefix_caching=True,
        kv_block_allocator=SimpleNamespace(kv_hash_to_block_id={101: 7, 103: 9}),
    )
    request = SimpleNamespace(precomputed_block_hashes=[101, 102, 103])

    block_ids, parent_hash = DynamicInferenceContext._find_kv_match_count(
        context, request, 0, 3
    )

    assert block_ids == [7]
    assert parent_hash == 101
