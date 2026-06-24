# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for HookTrajectoryCollector."""

import pytest
import torch

from megatron.rl.compaction.learned.capture.hook_collector import HookTrajectoryCollector
from megatron.rl.compaction.learned.training.data import PipelineConfig
from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe
from megatron.rl.compaction.kv.megatron_hook import NullHook


# ---------------------------------------------------------------------------
# Fake hook that returns controllable KV tensors
# ---------------------------------------------------------------------------

class FakeHook:
    """Returns cumulative KV tensors of shape (B, cumul_len, d_model)."""

    def __init__(self, n_layers=2, d_model=32, batch=1):
        self.n_layers = n_layers
        self.d_model = d_model
        self.batch = batch
        self._cumul_len = 0
        self._enabled = True

    def approx_attention_scores(self):
        return []

    def apply_mask(self, mask):
        pass

    def get_kv_matrices(self):
        if not self._enabled or self._cumul_len == 0:
            return None
        keys = [torch.ones(self.batch, self._cumul_len, self.d_model) for _ in range(self.n_layers)]
        vals = [torch.zeros(self.batch, self._cumul_len, self.d_model) for _ in range(self.n_layers)]
        return keys, vals

    def apply_belief_memory(self, memory):
        pass

    def advance(self, n_tokens: int):
        """Simulate n_tokens being processed."""
        self._cumul_len += n_tokens
        return n_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_LAYERS = 2
D_MODEL  = 32
BATCH    = 1

def _logits(B=1, S=4, V=64):
    return torch.randn(B, S, V)

def _tokens(B=1, S=4, V=64):
    return torch.randint(0, V, (B, S))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHookTrajectoryCollectorBasic:
    def test_empty_rollout_returns_trajectory(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))
        c.begin_rollout()
        traj = c.end_rollout()
        assert isinstance(traj, Trajectory)
        assert traj.n_chunks == 0

    def test_null_hook_returns_empty_trajectory(self):
        c = HookTrajectoryCollector(NullHook(), PipelineConfig(chunk_size=16))
        c.begin_rollout()
        c.on_step(32)   # NullHook.get_kv_matrices() returns None → nothing stored
        traj = c.end_rollout()
        assert traj.n_chunks == 0

    def test_single_chunk_exact(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))
        c.begin_rollout()
        hook.advance(16)
        c.on_step(16)
        traj = c.end_rollout()
        assert traj.n_chunks == 1
        keys, vals = traj.chunks[0]
        assert len(keys) == N_LAYERS
        assert keys[0].shape == (BATCH, 16, D_MODEL)

    def test_two_chunks(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))
        c.begin_rollout()
        hook.advance(32)
        c.on_step(32)
        traj = c.end_rollout()
        assert traj.n_chunks == 2

    def test_partial_final_chunk_included(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))
        c.begin_rollout()
        hook.advance(20)
        c.on_step(20)
        traj = c.end_rollout()
        # 16-token chunk + 4-token partial
        assert traj.n_chunks == 2
        # Last chunk is partial (4 tokens)
        keys, _ = traj.chunks[-1]
        assert keys[0].shape[1] == 4

    def test_begin_rollout_resets_state(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))

        c.begin_rollout()
        hook.advance(16)
        c.on_step(16)
        traj1 = c.end_rollout()
        assert traj1.n_chunks == 1

        # Second rollout from fresh hook
        hook2 = FakeHook()
        c2 = HookTrajectoryCollector(hook2, PipelineConfig(chunk_size=16))
        c2.begin_rollout()
        traj2 = c2.end_rollout()
        assert traj2.n_chunks == 0

    def test_multiple_on_step_calls(self):
        """Prefill (many tokens) then decode (one token at a time)."""
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=8))
        c.begin_rollout()

        # Prefill: 8 tokens in one call
        hook.advance(8)
        c.on_step(8)

        # Decode: 8 more tokens, one at a time
        for _ in range(8):
            hook.advance(1)
            c.on_step(1)

        traj = c.end_rollout()
        # 16 tokens total / 8 per chunk = 2 chunks
        assert traj.n_chunks == 2


class TestHookTrajectoryCollectorProbes:
    def test_probe_created_with_logits(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16, probe_stride=1, probe_query_len=8))
        c.begin_rollout()
        hook.advance(16)
        c.on_step(16, teacher_logits=_logits(S=16), query_tokens=_tokens(S=16))
        traj = c.end_rollout()
        assert 0 in traj.probes_by_chunk
        probe = traj.probes_by_chunk[0][0]
        assert isinstance(probe, TrainingProbe)
        assert probe.query_tokens.shape == (BATCH, 8)
        assert probe.teacher_logits.shape[-1] == 64

    def test_no_probe_without_logits(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16, probe_stride=1))
        c.begin_rollout()
        hook.advance(16)
        c.on_step(16)   # no teacher_logits → no probe
        traj = c.end_rollout()
        assert len(traj.probes_by_chunk) == 0

    def test_probe_stride_2(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=8, probe_stride=2, probe_query_len=4))
        c.begin_rollout()
        hook.advance(32)
        # Feed logits at one big step (4 chunks worth)
        c.on_step(32, teacher_logits=_logits(S=32), query_tokens=_tokens(S=32))
        traj = c.end_rollout()
        # Chunks 0, 1, 2, 3 — probes only at 0 and 2 (stride=2)
        for idx in traj.probes_by_chunk:
            assert idx % 2 == 0

    def test_max_probes_respected(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(
            chunk_size=8, probe_stride=1, probe_query_len=4, max_probes=2
        ))
        c.begin_rollout()
        hook.advance(64)
        c.on_step(64, teacher_logits=_logits(S=64), query_tokens=_tokens(S=64))
        traj = c.end_rollout()
        total_probes = sum(len(ps) for ps in traj.probes_by_chunk.values())
        assert total_probes <= 2

    def test_probe_query_len_truncated_to_step_size(self):
        """probe_query_len > step size → clamp to what's available."""
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(
            chunk_size=16, probe_stride=1, probe_query_len=64
        ))
        c.begin_rollout()
        hook.advance(16)
        c.on_step(16, teacher_logits=_logits(S=16), query_tokens=_tokens(S=16))
        traj = c.end_rollout()
        probe = traj.probes_by_chunk[0][0]
        # Only 16 tokens available in this step
        assert probe.query_tokens.shape[1] <= 16


class TestHookTrajectoryCollectorKVContent:
    def test_kv_shape_correct(self):
        hook = FakeHook(n_layers=3, d_model=64)
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=8))
        c.begin_rollout()
        hook.advance(8)
        c.on_step(8)
        traj = c.end_rollout()
        keys, vals = traj.chunks[0]
        assert len(keys) == 3
        assert keys[0].shape == (BATCH, 8, 64)
        assert vals[0].shape == (BATCH, 8, 64)

    def test_chunks_are_disjoint(self):
        """Each chunk should contain exactly chunk_size tokens worth of KV."""
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=8))
        c.begin_rollout()
        hook.advance(24)
        c.on_step(24)
        traj = c.end_rollout()
        assert traj.n_chunks == 3
        total_tokens = sum(traj.chunks[i][0][0].shape[1] for i in range(traj.n_chunks))
        assert total_tokens == 24

    def test_tensors_are_detached(self):
        hook = FakeHook()
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=16))
        c.begin_rollout()
        hook.advance(16)
        c.on_step(16)
        traj = c.end_rollout()
        keys, vals = traj.chunks[0]
        for k in keys:
            assert not k.requires_grad

    def test_batch_dim_preserved(self):
        hook = FakeHook(batch=4)
        c = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=8))
        c.begin_rollout()
        hook.advance(8)
        c.on_step(8)
        traj = c.end_rollout()
        keys, _ = traj.chunks[0]
        assert keys[0].shape[0] == 4
