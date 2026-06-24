# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for MegatronInferenceHook.

Uses a fake DynamicInferenceContext backed by plain torch tensors so no
actual Megatron runtime is required.
"""

from __future__ import annotations

import math

import pytest
import torch

from megatron.rl.compaction.kv.megatron_hook import MegatronInferenceHook
from megatron.rl.compaction.kv.types import KVMask
from megatron.rl.compaction.learned.models.belief import BeliefMemory


# ---------------------------------------------------------------------------
# Fake context helpers
# ---------------------------------------------------------------------------

class _FakeBlockAllocator:
    """Minimal block allocator that returns pre-allocated IDs and tracks frees."""

    def __init__(self, total_blocks: int):
        self._free = list(range(total_blocks))
        self.released: list[int] = []

    def allocate_memory_blocks(self, n: int):
        if len(self._free) < n:
            return None
        ids = self._free[:n]
        self._free = self._free[n:]
        return torch.tensor(ids, dtype=torch.int32)

    def release_memory_blocks(self, blocks: torch.Tensor):
        self.released.extend(blocks.tolist())
        self._free.extend(blocks.tolist())


def _make_context(
    n_layers: int = 2,
    n_heads: int = 1,
    d_head: int = 4,
    block_size: int = 4,
    seq_len: int = 8,
    paused: int = 0,
) -> object:
    """Create a minimal fake DynamicInferenceContext with one active request."""
    total_blocks = 20
    BS = block_size
    H = n_heads
    D = d_head
    B = 1  # one active request

    # memory_buffer: (2, n_layers, total_blocks, BS, H, D)
    # Fill with recognisable values: buf[kv, layer, block, pos, h, d] = layer * 1000 + pos_in_seq
    buf = torch.zeros(2, n_layers, total_blocks, BS, H, D)

    n_blocks = math.ceil(seq_len / BS)
    last_offset = (seq_len - 1) % BS

    # Fill the blocks used by the one request
    block_ids_for_req = list(range(n_blocks))
    for t in range(seq_len):
        blk = t // BS
        pos = t % BS
        for layer in range(n_layers):
            for kv in range(2):
                buf[kv, layer, block_ids_for_req[blk], pos] = float(t + 1)

    allocator = _FakeBlockAllocator(total_blocks)
    # Pre-use the blocks taken by the request so allocator doesn't reassign them
    allocator._free = [i for i in allocator._free if i not in block_ids_for_req]

    request_to_kv_block_ids = torch.full((1 + paused, total_blocks), -1, dtype=torch.int32)
    request_to_kv_block_ids[paused, :n_blocks] = torch.tensor(block_ids_for_req, dtype=torch.int32)

    request_kv_block_counts = torch.zeros(1 + paused, dtype=torch.int32)
    request_kv_block_counts[paused] = n_blocks

    request_last_kv_block_offset = torch.zeros(1 + paused, dtype=torch.int32)
    request_last_kv_block_offset[paused] = last_offset

    ctx = type("FakeCtx", (), {})()
    ctx.memory_buffer = buf
    ctx.cache_mla_latent = False
    ctx.num_attention_layers = n_layers
    ctx.num_attention_heads_per_partition = H
    ctx.hidden_size_per_attention_head = D
    ctx.block_size_tokens = BS
    ctx.paused_request_count = paused
    ctx.total_request_count = paused + B
    ctx.request_to_kv_block_ids = request_to_kv_block_ids
    ctx.request_kv_block_counts = request_kv_block_counts
    ctx.request_last_kv_block_offset = request_last_kv_block_offset
    ctx.block_allocator = allocator
    return ctx


# ---------------------------------------------------------------------------
# get_kv_matrices
# ---------------------------------------------------------------------------

class TestGetKvMatrices:
    def test_returns_correct_shape(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        result = hook.get_kv_matrices()
        assert result is not None
        keys, vals = result
        assert len(keys) == 2
        assert keys[0].shape == (1, 8, 4)   # (B, S, H*D)

    def test_returns_none_when_no_buffer(self):
        ctx = _make_context()
        ctx.memory_buffer = None
        hook = MegatronInferenceHook(ctx)
        assert hook.get_kv_matrices() is None

    def test_returns_none_when_no_active(self):
        ctx = _make_context()
        ctx.total_request_count = 0
        hook = MegatronInferenceHook(ctx)
        assert hook.get_kv_matrices() is None

    def test_returns_none_for_mla(self):
        ctx = _make_context()
        ctx.cache_mla_latent = True
        hook = MegatronInferenceHook(ctx)
        assert hook.get_kv_matrices() is None

    def test_seq_len_partial_block(self):
        # 6 tokens in 2 blocks of size 4: block 0 full, block 1 has 2 tokens
        ctx = _make_context(block_size=4, seq_len=6)
        hook = MegatronInferenceHook(ctx)
        result = hook.get_kv_matrices()
        assert result is not None
        keys, _ = result
        assert keys[0].shape[1] == 6


# ---------------------------------------------------------------------------
# approx_attention_scores
# ---------------------------------------------------------------------------

class TestGetAttentionScores:
    def test_returns_list_of_floats(self):
        ctx = _make_context(seq_len=8)
        hook = MegatronInferenceHook(ctx)
        scores = hook.approx_attention_scores()
        assert isinstance(scores, list)
        assert len(scores) == 8
        assert all(isinstance(s, float) for s in scores)

    def test_returns_empty_when_no_buffer(self):
        ctx = _make_context()
        ctx.memory_buffer = None
        hook = MegatronInferenceHook(ctx)
        assert hook.approx_attention_scores() == []

    def test_returns_empty_when_no_active(self):
        ctx = _make_context()
        ctx.total_request_count = 0
        hook = MegatronInferenceHook(ctx)
        assert hook.approx_attention_scores() == []

    def test_scores_are_positive(self):
        ctx = _make_context(seq_len=10)
        hook = MegatronInferenceHook(ctx)
        scores = hook.approx_attention_scores()
        assert all(s >= 0.0 for s in scores)

    def test_scores_reflect_kv_magnitudes(self):
        # Positions with larger KV values should get higher scores
        ctx = _make_context(n_layers=1, n_heads=1, d_head=4, block_size=4, seq_len=4)
        # Override memory_buffer: make position 3 have very large K
        ctx.memory_buffer[:, :, :, :, :, :] = 0.001
        ctx.memory_buffer[0, 0, 0, 3] = 10.0   # position 3, keys layer 0
        hook = MegatronInferenceHook(ctx)
        scores = hook.approx_attention_scores()
        assert scores[3] > scores[0]


# ---------------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------------

class TestApplyMask:
    def _make_mask(self, retained: list[int], total: int) -> KVMask:
        return KVMask(
            run_id="test",
            step_id=0,
            retained_positions=retained,
            total_positions=total,
            strategy="topk",
        )

    def test_no_op_when_retain_all(self):
        ctx = _make_context(seq_len=8, block_size=4)
        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask(list(range(8)), 8)
        hook.apply_mask(mask)
        assert ctx.request_kv_block_counts[0].item() == 2
        assert ctx.request_last_kv_block_offset[0].item() == 3

    def test_reduces_block_count(self):
        ctx = _make_context(seq_len=8, block_size=4)
        hook = MegatronInferenceHook(ctx)
        # Keep only first 3 tokens → fits in 1 block
        mask = self._make_mask([0, 1, 2], 8)
        hook.apply_mask(mask)
        assert ctx.request_kv_block_counts[0].item() == 1
        assert ctx.request_last_kv_block_offset[0].item() == 2

    def test_frees_excess_blocks(self):
        ctx = _make_context(seq_len=8, block_size=4)
        allocator = ctx.block_allocator
        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask([0, 1, 2], 8)
        hook.apply_mask(mask)
        # block 1 (the second block) should be released
        assert len(allocator.released) > 0

    def test_block_ids_cleaned_up(self):
        ctx = _make_context(seq_len=8, block_size=4)
        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask([0, 1, 2], 8)
        hook.apply_mask(mask)
        # slot 1 in block_ids should be -1 after freeing
        assert ctx.request_to_kv_block_ids[0, 1].item() == -1

    def test_retained_values_preserved(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=8)
        # Write distinct values at each position
        buf = ctx.memory_buffer
        for t in range(8):
            blk, pos = t // 4, t % 4
            buf[0, 0, blk, pos] = float(t + 1)

        hook = MegatronInferenceHook(ctx)
        retained = [2, 5, 7]
        mask = self._make_mask(retained, 8)
        hook.apply_mask(mask)

        # After compaction, token at position 0 should have value of original token 2
        bid0 = ctx.request_to_kv_block_ids[0, 0].item()
        v0 = buf[0, 0, bid0, 0, 0, 0].item()
        assert abs(v0 - 3.0) < 1e-5  # token 2 (0-indexed) had value 3.0

    def test_no_op_when_no_active_requests(self):
        ctx = _make_context(seq_len=8)
        ctx.total_request_count = 0
        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask([0, 1], 8)
        hook.apply_mask(mask)   # must not raise

    def test_no_op_when_no_buffer(self):
        ctx = _make_context(seq_len=8)
        ctx.memory_buffer = None
        hook = MegatronInferenceHook(ctx)
        hook.apply_mask(self._make_mask([0], 8))   # must not raise

    def test_cross_block_boundary(self):
        # Keep tokens 3 and 4 which span block 0 (pos 3) and block 1 (pos 0)
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=8)
        buf = ctx.memory_buffer
        for t in range(8):
            buf[0, 0, t // 4, t % 4] = float(t + 10)  # values 10..17

        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask([3, 4], 8)
        hook.apply_mask(mask)

        assert ctx.request_kv_block_counts[0].item() == 1
        bid0 = ctx.request_to_kv_block_ids[0, 0].item()
        v0 = buf[0, 0, bid0, 0, 0, 0].item()
        v1 = buf[0, 0, bid0, 1, 0, 0].item()
        assert abs(v0 - 13.0) < 1e-5   # token 3 had value 13
        assert abs(v1 - 14.0) < 1e-5   # token 4 had value 14

    def test_retain_single_token(self):
        ctx = _make_context(seq_len=8, block_size=4)
        hook = MegatronInferenceHook(ctx)
        mask = self._make_mask([6], 8)
        hook.apply_mask(mask)
        assert ctx.request_kv_block_counts[0].item() == 1
        assert ctx.request_last_kv_block_offset[0].item() == 0


# ---------------------------------------------------------------------------
# apply_belief_memory
# ---------------------------------------------------------------------------

class TestApplyBeliefMemory:
    def _make_memory(self, n_layers: int, B: int, C: int, d_model: int) -> BeliefMemory:
        keys = torch.ones(n_layers, B, C, d_model) * 42.0
        vals = torch.ones(n_layers, B, C, d_model) * 99.0
        return BeliefMemory(keys=keys, values=vals, step=1)

    def test_basic_inject(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        C = 3
        memory = self._make_memory(n_layers=2, B=1, C=C, d_model=4)
        hook.apply_belief_memory(memory)

        # Should now have ceil(3/4)=1 block with last_offset=2
        assert ctx.request_kv_block_counts[0].item() == 1
        assert ctx.request_last_kv_block_offset[0].item() == 2

    def test_values_written_correctly(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=2, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        C = 2
        keys = torch.tensor([[[[7.0, 8.0], [9.0, 10.0]]]])  # (1, 1, 2, 2) → wrong shape
        # Correct shape: (n_layers, B, C, d_model) = (1, 1, 2, 2)
        keys = torch.tensor([[[[7.0, 8.0], [9.0, 10.0]]]])   # (1, 1, 2, 2)
        vals = keys * 2
        memory = BeliefMemory(keys=keys, values=vals, step=0)
        hook.apply_belief_memory(memory)

        bid0 = ctx.request_to_kv_block_ids[0, 0].item()
        buf = ctx.memory_buffer
        # First token of layer 0: should be [7.0, 8.0] reshaped into (H=1, D=2)
        assert abs(buf[0, 0, bid0, 0, 0, 0].item() - 7.0) < 1e-5
        assert abs(buf[0, 0, bid0, 0, 0, 1].item() - 8.0) < 1e-5

    def test_old_blocks_freed(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=8)
        allocator = ctx.block_allocator
        old_block_ids = ctx.request_to_kv_block_ids[0, :2].tolist()
        hook = MegatronInferenceHook(ctx)
        memory = self._make_memory(n_layers=2, B=1, C=3, d_model=4)
        hook.apply_belief_memory(memory)
        for bid in old_block_ids:
            assert bid in allocator.released

    def test_raises_on_allocator_failure(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=4, block_size=4, seq_len=4)
        # Drain the allocator so it cannot allocate
        ctx.block_allocator._free = []
        hook = MegatronInferenceHook(ctx)
        memory = self._make_memory(n_layers=1, B=1, C=3, d_model=4)
        with pytest.raises(RuntimeError, match="allocator exhausted"):
            hook.apply_belief_memory(memory)

    def test_raises_on_batch_mismatch(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=4, block_size=4, seq_len=4)
        hook = MegatronInferenceHook(ctx)
        # B=2 but only 1 active request
        memory = self._make_memory(n_layers=1, B=2, C=3, d_model=4)
        with pytest.raises(RuntimeError, match="batch size"):
            hook.apply_belief_memory(memory)

    def test_no_op_when_no_active(self):
        ctx = _make_context(seq_len=4)
        ctx.total_request_count = 0
        hook = MegatronInferenceHook(ctx)
        memory = self._make_memory(n_layers=1, B=1, C=3, d_model=4)
        hook.apply_belief_memory(memory)  # must not raise

    def test_budget_larger_than_block_size(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=4, block_size=4, seq_len=4)
        hook = MegatronInferenceHook(ctx)
        C = 9  # needs ceil(9/4) = 3 blocks
        memory = self._make_memory(n_layers=1, B=1, C=C, d_model=4)
        hook.apply_belief_memory(memory)
        assert ctx.request_kv_block_counts[0].item() == 3
        assert ctx.request_last_kv_block_offset[0].item() == 0  # (9-1)%4=0


# ---------------------------------------------------------------------------
# Round-trip: apply_mask then get_kv_matrices
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_get_kv_after_apply_mask(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        retained = [1, 3, 5, 7]
        mask = KVMask(run_id="r", step_id=0, retained_positions=retained,
                      total_positions=8, strategy="topk")
        hook.apply_mask(mask)
        result = hook.get_kv_matrices()
        assert result is not None
        keys, _ = result
        assert keys[0].shape == (1, 4, 4)   # (B=1, S=4, H*D=4)

    def test_get_kv_after_apply_belief_memory(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=8)
        hook = MegatronInferenceHook(ctx)
        C = 3
        memory = BeliefMemory(
            keys=torch.ones(2, 1, C, 4) * 5.0,
            values=torch.ones(2, 1, C, 4) * 6.0,
            step=1,
        )
        hook.apply_belief_memory(memory)
        result = hook.get_kv_matrices()
        assert result is not None
        keys, _ = result
        assert keys[0].shape == (1, C, 4)
        # Verify values came from the compact memory
        assert abs(keys[0][0, 0, 0].item() - 5.0) < 1e-4
