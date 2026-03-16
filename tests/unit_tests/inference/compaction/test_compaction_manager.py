# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for CompactionManager (offline + online compaction)."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.compaction.compaction_manager import (
    CompactionConfig,
    CompactionManager,
    PerSequenceCompactionState,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MockBlockAllocator:
    """Minimal block allocator mock for testing."""

    def __init__(self, total_blocks, device):
        self.total_count = total_blocks
        self.total_avail = total_blocks - 1
        self.next_alloc = 100  # Start allocating from block 100
        self.released = []
        self.device = device

    def allocate_memory_blocks(self, num_blocks):
        ids = torch.arange(
            self.next_alloc, self.next_alloc + num_blocks,
            dtype=torch.int32, device=self.device,
        )
        self.next_alloc += num_blocks
        self.total_avail -= num_blocks
        return ids

    def release_memory_blocks(self, blocks):
        self.released.append(blocks)
        self.total_avail += blocks.numel()


@pytest.fixture
def manager_setup(device):
    """Create a CompactionManager with mock allocator."""
    num_layers = 2
    num_blocks = 200
    block_size = 8
    num_heads = 4
    head_dim = 16

    memory_buffer = torch.randn(
        2, num_layers, num_blocks, block_size, num_heads, head_dim,
        dtype=torch.bfloat16, device=device,
    )

    allocator = MockBlockAllocator(num_blocks, device)

    config = CompactionConfig(
        memory_budget=16,
        hot_window=32,
        compact_every_n=32,
        method="top_attention",
        use_mass_matching=False,
        nnls_iters=20,
        num_ref_queries=8,
    )

    manager = CompactionManager(
        config=config,
        memory_buffer=memory_buffer,
        block_allocator=allocator,
        block_size=block_size,
        num_layers=num_layers,
    )

    return manager, allocator, memory_buffer, config


class TestSequenceManagement:
    """Tests for sequence registration and tracking."""

    def test_register_sequence(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)
        assert 1 in manager.seq_states
        assert manager.seq_states[1].logical_pos == 0

    def test_register_with_initial_pos(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1, initial_pos=100)
        assert manager.seq_states[1].logical_pos == 100

    def test_unregister_sequence(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)
        manager.unregister_sequence(1)
        assert 1 not in manager.seq_states

    def test_advance_position(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)
        pos = manager.advance_position(1, 5)
        assert pos == 5
        assert manager.seq_states[1].logical_pos == 5
        assert manager.seq_states[1].tokens_generated == 5

    def test_rope_position_monotonic(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)
        positions = []
        for i in range(10):
            pos = manager.advance_position(1)
            positions.append(pos)

        # Must be strictly monotonically increasing
        for i in range(1, len(positions)):
            assert positions[i] > positions[i-1]

    def test_get_rope_position(self, manager_setup):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)
        manager.advance_position(1, 50)
        assert manager.get_rope_position(1) == 50

    def test_unknown_sequence_raises(self, manager_setup):
        manager, _, _, _ = manager_setup
        with pytest.raises(ValueError):
            manager.get_rope_position(999)


class TestOfflineCompaction:
    """Tests for offline (single-shot) compaction."""

    def test_offline_compact_basic(self, manager_setup, device):
        manager, allocator, memory_buffer, config = manager_setup

        # Use 4 full blocks (32 tokens with block_size=8)
        block_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        token_count = 32

        new_blocks, mem_count, metrics = manager.compact_offline(
            block_ids, token_count,
        )

        assert mem_count == config.memory_budget  # 16
        assert new_blocks.shape[0] == math.ceil(config.memory_budget / 8)  # 2 blocks
        assert "compression_ratio" in metrics
        assert metrics["compression_ratio"] == 32 / 16

    def test_offline_compact_with_queries(self, manager_setup, device):
        manager, _, memory_buffer, config = manager_setup
        H = memory_buffer.shape[4]
        D = memory_buffer.shape[5]
        R = config.num_ref_queries

        block_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        Q_ref = torch.randn(R, H, D, device=device, dtype=torch.bfloat16)

        new_blocks, mem_count, metrics = manager.compact_offline(
            block_ids, 32, Q_ref=Q_ref,
        )

        assert mem_count == 16
        assert metrics["mean_output_error"] < 5.0  # Sanity bound

    def test_offline_compact_metrics(self, manager_setup, device):
        manager, _, _, _ = manager_setup
        block_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        _, _, metrics = manager.compact_offline(block_ids, 32)

        required_keys = [
            "mean_mass_error", "max_mass_error",
            "mean_output_error", "max_output_error",
            "compression_ratio", "original_tokens", "compacted_tokens",
        ]
        for key in required_keys:
            assert key in metrics

    def test_offline_blocks_allocated(self, manager_setup, device):
        manager, allocator, _, _ = manager_setup
        initial_avail = allocator.total_avail
        block_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        new_blocks, _, _ = manager.compact_offline(block_ids, 32)

        # New blocks should have been allocated
        expected_blocks = math.ceil(16 / 8)
        assert allocator.total_avail == initial_avail - expected_blocks


class TestOnlineCompaction:
    """Tests for online (periodic) compaction."""

    def test_should_compact(self, manager_setup):
        manager, _, _, config = manager_setup
        manager.register_sequence(1)
        state = manager.seq_states[1]

        # Not yet time to compact
        state.hot_token_count = config.hot_window
        assert not manager.should_compact(1)

        # Time to compact
        state.hot_token_count = config.hot_window + config.compact_every_n
        assert manager.should_compact(1)

    def test_online_compact_basic(self, manager_setup, device):
        manager, allocator, memory_buffer, config = manager_setup
        manager.register_sequence(1)

        # 10 blocks = 80 tokens
        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        total_tokens = 80

        new_bt, new_kv_len = manager.compact_online(
            seq_id=1,
            all_block_ids=block_ids,
            total_token_count=total_tokens,
        )

        state = manager.seq_states[1]
        M = config.memory_budget
        W = config.hot_window

        assert state.compaction_count == 1
        assert state.mem_token_count == M
        assert new_kv_len == M + state.hot_token_count

    def test_online_compact_logical_pos_monotonic(self, manager_setup, device):
        """Logical position never decreases after compaction."""
        manager, _, _, config = manager_setup
        manager.register_sequence(1)

        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        total_tokens = 80

        manager.compact_online(1, block_ids, total_tokens)
        pos1 = manager.get_rope_position(1)

        # Simulate more tokens
        manager.advance_position(1, 20)

        # Re-compact with more blocks
        block_ids2 = torch.arange(12, device=device, dtype=torch.int32)
        manager.compact_online(1, block_ids2, 100)
        pos2 = manager.get_rope_position(1)

        assert pos2 >= pos1

    def test_online_compact_frees_cold_blocks(self, manager_setup, device):
        """Cold blocks are freed after compaction."""
        manager, allocator, _, config = manager_setup
        manager.register_sequence(1)

        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        manager.compact_online(1, block_ids, 80)

        # Cold blocks should have been released
        assert len(allocator.released) > 0

    def test_block_table_two_tier(self, manager_setup, device):
        """Block table has [mem + hot] layout after compaction."""
        manager, _, _, config = manager_setup
        manager.register_sequence(1)

        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        new_bt, _ = manager.compact_online(1, block_ids, 80)

        state = manager.seq_states[1]
        bt = manager.get_block_table(1)
        assert bt is not None

        mem_blocks = math.ceil(config.memory_budget / 8)
        # Block table should start with memory blocks, then hot blocks
        assert bt.shape[0] >= mem_blocks

    def test_online_skip_if_insufficient_cold(self, manager_setup, device):
        """Skip compaction if cold prefix is smaller than budget."""
        manager, _, _, config = manager_setup
        config.memory_budget = 100  # Larger than sequence
        manager.register_sequence(1)

        block_ids = torch.arange(4, device=device, dtype=torch.int32)
        new_bt, new_len = manager.compact_online(1, block_ids, 32)

        # Should return unchanged
        assert torch.equal(new_bt, block_ids)
        assert new_len == 32


class TestCompactionMetrics:
    """Tests for compaction metrics tracking."""

    def test_metrics_after_compaction(self, manager_setup, device):
        manager, _, _, _ = manager_setup
        manager.register_sequence(1)

        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        manager.compact_online(1, block_ids, 80)

        metrics = manager.get_compaction_metrics(1)
        assert metrics["compaction_count"] == 1
        assert metrics["mem_tokens"] > 0
        assert "mean_mass_error" in metrics
        assert "mean_output_error" in metrics

    def test_physical_kv_len(self, manager_setup, device):
        manager, _, _, config = manager_setup
        manager.register_sequence(1)

        assert manager.get_physical_kv_len(1) == 0

        block_ids = torch.arange(10, device=device, dtype=torch.int32)
        manager.compact_online(1, block_ids, 80)

        phys_len = manager.get_physical_kv_len(1)
        state = manager.seq_states[1]
        assert phys_len == state.mem_token_count + state.hot_token_count
