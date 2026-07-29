# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Synthetic CPU stress cases for dynamic-inference prefix caching.

Run from the repository root:

    python tests/performance_tests/prefix_cache_stress.py --quick

The default sizes are intended for performance hosts. ``--quick`` is suitable
for presubmit/smoke use. Results are emitted as JSON so before/after runs can be
compared mechanically.
"""

import argparse
import heapq
import json
import statistics
import time
import tracemalloc
from types import SimpleNamespace

import torch

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.contexts.kv_block_allocator import KVBlockAllocator
from megatron.core.inference.data_parallel_inference_coordinator.coordinator import (
    DataParallelInferenceCoordinator,
)


def _median_ms(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _legacy_reverse_match(hashes, hash_to_block):
    """The pre-fix reverse-scan matcher, retained only as a benchmark baseline."""
    for i in range(len(hashes) - 1, -1, -1):
        if hashes[i] in hash_to_block:
            return [hash_to_block[hashes[j]] for j in range(i + 1)]
    return []


def benchmark_cold_match(blocks, repeats):
    hashes = list(range(1, blocks + 1))
    request = SimpleNamespace(precomputed_block_hashes=hashes)
    context = SimpleNamespace(
        enable_prefix_caching=True,
        kv_block_allocator=SimpleNamespace(kv_hash_to_block_id={}),
    )
    actual = lambda: DynamicInferenceContext._find_kv_match_count(
        context, request, 0, blocks
    )
    legacy = lambda: _legacy_reverse_match(hashes, {})
    return {
        "blocks": blocks,
        "forward_stop_ms": _median_ms(actual, repeats),
        "legacy_reverse_scan_ms": _median_ms(legacy, repeats),
    }


def _seed_chain(blocks):
    max_requests = 4
    context = SimpleNamespace(
        paused_request_count=0,
        total_request_count=0,
        request_kv_block_counts=torch.zeros(max_requests, dtype=torch.int32),
        request_to_kv_block_ids=torch.full((max_requests, 1), -1, dtype=torch.int32),
        prefix_cache_lru_clock=1,
    )
    allocator = KVBlockAllocator(
        context,
        total_count=blocks + 2,
        paused_count=0,
        enable_prefix_caching=True,
        prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
    )
    ids = list(range(blocks))
    hashes = list(range(1, blocks + 1))
    parents = [0] + hashes[:-1]
    allocator.register_kv_block_hashes(ids, hashes, parents)
    allocator.total_avail -= blocks
    allocator.block_timestamps[:blocks] = torch.arange(blocks, dtype=torch.int64)
    return allocator


def _legacy_evict_lru(allocator, num_blocks_needed):
    """The pre-fix full Python forest rebuild, retained as a benchmark baseline."""
    cached_mask = (allocator.block_ref_counts == 0) & (allocator.block_hashes != -1)
    cached_block_ids = torch.nonzero(cached_mask, as_tuple=True)[0]
    timestamps = allocator.block_timestamps[cached_block_ids].tolist()
    block_ids = cached_block_ids.tolist()
    parent_ids = allocator.block_parent_id[cached_block_ids].tolist()
    child_counts = allocator.block_child_count[cached_block_ids].tolist()
    global_to_local = {block_id: i for i, block_id in enumerate(block_ids)}
    parent_local = [global_to_local.get(parent_id, -1) for parent_id in parent_ids]
    heap = [
        (timestamps[i], block_ids[i], i)
        for i in range(len(block_ids))
        if child_counts[i] == 0
    ]
    heapq.heapify(heap)
    evicted_local = []
    while heap and len(evicted_local) < num_blocks_needed:
        _, _, i = heapq.heappop(heap)
        evicted_local.append(i)
        parent = parent_local[i]
        if parent >= 0:
            child_counts[parent] -= 1
            if child_counts[parent] == 0:
                heapq.heappush(heap, (timestamps[parent], block_ids[parent], parent))
    allocator._deregister_blocks(
        cached_block_ids[torch.tensor(evicted_local, dtype=torch.int64)]
    )


def benchmark_lru_single_eviction(blocks, repeats):
    optimized_samples = []
    legacy_samples = []
    for _ in range(repeats):
        allocator = _seed_chain(blocks)
        start = time.perf_counter()
        assert allocator.evict_lru_blocks(1)
        optimized_samples.append((time.perf_counter() - start) * 1e3)

        allocator = _seed_chain(blocks)
        start = time.perf_counter()
        _legacy_evict_lru(allocator, 1)
        legacy_samples.append((time.perf_counter() - start) * 1e3)
    return {
        "blocks": blocks,
        "optimized_leaf_select_ms": statistics.median(optimized_samples),
        "legacy_python_rebuild_ms": statistics.median(legacy_samples),
    }


def benchmark_mamba_oldest_selection(slots, needed, repeats):
    timestamps = torch.randperm(slots, dtype=torch.int64)
    argsort = lambda: torch.argsort(timestamps)[:needed]
    topk = lambda: torch.topk(timestamps, k=needed, largest=False, sorted=False).indices
    return {
        "slots": slots,
        "needed": needed,
        "argsort_ms": _median_ms(argsort, repeats),
        "topk_ms": _median_ms(topk, repeats),
    }


def benchmark_coordinator_shadow_growth(assignments, blocks_per_request):
    """Measure shadow-table growth when every request has a unique prefix.

    The coordinator receives no cache-eviction feedback, so this represents a
    long-lived high-cardinality service even if each engine cache is tiny.
    """
    coordinator = object.__new__(DataParallelInferenceCoordinator)
    coordinator.identity_to_rank_index = {b"rank-0": 0}
    coordinator._hash_assignment_counter = 0
    coordinator._hash_table = {}

    tracemalloc.start()
    start = time.perf_counter()
    for request_idx in range(assignments):
        first = request_idx * blocks_per_request + 1
        hashes = range(first, first + blocks_per_request)
        coordinator._update_rank_hashes(b"rank-0", hashes)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "assignments": assignments,
        "blocks_per_request": blocks_per_request,
        "shadow_hashes": len(coordinator._hash_table),
        "elapsed_ms": elapsed_ms,
        "peak_mib": peak_bytes / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        match_sizes = [1_000, 10_000]
        eviction_sizes = [1_000, 5_000]
        selection_sizes = [10_000, 100_000]
        repeats = 3
        assignments = 2_000
    else:
        match_sizes = [1_000, 10_000, 100_000, 1_000_000]
        eviction_sizes = [1_000, 10_000, 100_000]
        selection_sizes = [10_000, 100_000, 1_000_000]
        repeats = 7
        assignments = 100_000

    results = {
        "cold_prefix_match": [
            benchmark_cold_match(size, repeats) for size in match_sizes
        ],
        "kv_lru_eviction": [
            benchmark_lru_single_eviction(size, repeats) for size in eviction_sizes
        ],
        "mamba_oldest_selection": [
            benchmark_mamba_oldest_selection(size, min(8, size), repeats)
            for size in selection_sizes
        ],
        "coordinator_shadow_growth": benchmark_coordinator_shadow_growth(
            assignments, blocks_per_request=8
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
