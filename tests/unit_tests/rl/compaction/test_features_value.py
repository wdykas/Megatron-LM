# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for ChunkFeatureExtractor, ValueHead, and path_consistency_loss."""

import pytest
import torch
import torch.nn as nn

from megatron.rl.compaction.learned.models.value import (
    ChunkFeatureExtractor,
    ChunkFeatures,
    FEATURE_DIM,
)
from megatron.rl.compaction.learned.models.value import ValueHead
from megatron.rl.compaction.learned.models.belief import BeliefMemory
from megatron.rl.compaction.learned.training.losses import path_consistency_loss
from megatron.rl.compaction.learned.models.belief import BeliefUpdater
from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_keys(n_layers=2, B=1, T=8, d=16):
    return [torch.randn(B, T, d) for _ in range(n_layers)]


def _belief(n_layers=2, B=1, C=4, d=16):
    return BeliefMemory(
        keys=torch.randn(n_layers, B, C, d),
        values=torch.randn(n_layers, B, C, d),
    )


# ---------------------------------------------------------------------------
# ChunkFeatureExtractor
# ---------------------------------------------------------------------------

class TestChunkFeatureExtractor:
    def test_vector_shape(self):
        extractor = ChunkFeatureExtractor(memory_budget=64)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=0)
        assert feat.vector.shape == (FEATURE_DIM,)

    def test_feature_dim_constant(self):
        assert FEATURE_DIM == 4

    def test_chunk_index_norm_in_range(self):
        extractor = ChunkFeatureExtractor(memory_budget=64, max_chunks=100)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=50)
        assert 0.0 <= feat.chunk_index_norm <= 1.0

    def test_memory_pressure_zero_when_empty(self):
        extractor = ChunkFeatureExtractor(memory_budget=64)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=0, current_memory_slots=0)
        assert feat.memory_pressure == 0.0

    def test_memory_pressure_one_when_full(self):
        extractor = ChunkFeatureExtractor(memory_budget=64)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=0, current_memory_slots=64)
        assert feat.memory_pressure == 1.0

    def test_memory_pressure_capped(self):
        extractor = ChunkFeatureExtractor(memory_budget=8)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=0, current_memory_slots=100)
        assert feat.memory_pressure == 1.0

    def test_key_norm_positive(self):
        extractor = ChunkFeatureExtractor(memory_budget=64)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=2)
        assert feat.key_norm_mean >= 0.0
        assert feat.key_norm_std >= 0.0

    def test_batch_extract_returns_tensor(self):
        extractor = ChunkFeatureExtractor(memory_budget=32)
        keys = _chunk_keys()
        vec = extractor.batch_extract(keys, chunk_index=1)
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (FEATURE_DIM,)

    def test_last_chunk_saturates_norm(self):
        extractor = ChunkFeatureExtractor(memory_budget=64, max_chunks=10)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=9)
        assert feat.chunk_index_norm == pytest.approx(0.9)

    def test_returns_chunk_features_dataclass(self):
        extractor = ChunkFeatureExtractor(memory_budget=16)
        keys = _chunk_keys()
        feat = extractor.extract(keys, chunk_index=3)
        assert isinstance(feat, ChunkFeatures)
        assert feat.chunk_index == 3

    def test_different_chunks_different_vectors(self):
        extractor = ChunkFeatureExtractor(memory_budget=64, max_chunks=100)
        keys = _chunk_keys()
        v0 = extractor.batch_extract(keys, chunk_index=0)
        v5 = extractor.batch_extract(keys, chunk_index=50)
        assert not torch.allclose(v0, v5)


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------

class TestValueHead:
    def test_output_shape(self):
        vh = ValueHead(n_layers=2, d_model=16, hidden_dim=32)
        mem = _belief(n_layers=2, B=3, C=4, d=16)
        out = vh(mem)
        assert out.shape == (3,)

    def test_with_features(self):
        vh = ValueHead(n_layers=2, d_model=16, hidden_dim=32, feature_dim=4)
        mem = _belief(n_layers=2, B=2, C=4, d=16)
        feat = torch.randn(2, 4)
        out = vh(mem, features=feat)
        assert out.shape == (2,)

    def test_broadcast_features(self):
        """1D feature vector is broadcast over batch."""
        vh = ValueHead(n_layers=2, d_model=16, hidden_dim=32, feature_dim=4)
        mem = _belief(n_layers=2, B=3, C=4, d=16)
        feat = torch.randn(4)   # no batch dim
        out = vh(mem, features=feat)
        assert out.shape == (3,)

    def test_grad_flows(self):
        vh = ValueHead(n_layers=2, d_model=16, hidden_dim=32)
        mem = _belief(n_layers=2, B=1, C=4, d=16)
        mem.keys.requires_grad_(True)
        out = vh(mem)
        out.sum().backward()
        assert mem.keys.grad is not None

    def test_no_feature_dim_no_error(self):
        vh = ValueHead(n_layers=3, d_model=32, hidden_dim=64, feature_dim=0)
        mem = _belief(n_layers=3, B=2, C=8, d=32)
        out = vh(mem, features=None)
        assert out.shape == (2,)


# ---------------------------------------------------------------------------
# path_consistency_loss
# ---------------------------------------------------------------------------

class TestPathConsistencyLoss:
    def _make_updater(self, n_layers=2, d=16, C=4, n_heads=2):
        cfg = PerceiverConfig(d_kv=d, n_heads=n_heads, n_compress=C, n_attn_layers=n_layers)
        compactor = PerceiverCompactor(cfg)
        return BeliefUpdater(compactor)

    def _student_fn(self, query_tokens, compact_kv):
        B, S_q = query_tokens.shape
        V = 32
        # Produce logits that depend on compact_kv so grad flows
        k0 = compact_kv[0][0]          # (B, C, d)
        flat = k0.reshape(B, -1)
        need = S_q * V
        reps = (need + flat.shape[1] - 1) // flat.shape[1]
        return flat.repeat(1, reps)[:, :need].reshape(B, S_q, V)

    def test_loss_is_scalar(self):
        updater = self._make_updater()
        n_layers, B, T, d = 2, 1, 6, 16
        ka = [torch.randn(B, T, d) for _ in range(n_layers)]
        va = [torch.randn(B, T, d) for _ in range(n_layers)]
        kb = [torch.randn(B, T, d) for _ in range(n_layers)]
        vb = [torch.randn(B, T, d) for _ in range(n_layers)]
        q  = torch.randint(0, 100, (B, 4))
        loss = path_consistency_loss(updater, ka, va, kb, vb, q, self._student_fn)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        updater = self._make_updater()
        n_layers, B, T, d = 2, 1, 6, 16
        ka = [torch.randn(B, T, d) for _ in range(n_layers)]
        va = [torch.randn(B, T, d) for _ in range(n_layers)]
        kb = [torch.randn(B, T, d) for _ in range(n_layers)]
        vb = [torch.randn(B, T, d) for _ in range(n_layers)]
        q  = torch.randint(0, 100, (B, 4))
        loss = path_consistency_loss(updater, ka, va, kb, vb, q, self._student_fn)
        assert loss.item() >= -1e-5   # ~0 by construction; allow GPU fp rounding

    def test_grad_flows_through_sequential_path(self):
        """path_consistency_loss should back-prop into the updater's compactor."""
        updater = self._make_updater()
        n_layers, B, T, d = 2, 1, 6, 16
        ka = [torch.randn(B, T, d) for _ in range(n_layers)]
        va = [torch.randn(B, T, d) for _ in range(n_layers)]
        kb = [torch.randn(B, T, d) for _ in range(n_layers)]
        vb = [torch.randn(B, T, d) for _ in range(n_layers)]
        q  = torch.randint(0, 100, (B, 4))

        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        loss = path_consistency_loss(updater, ka, va, kb, vb, q, self._student_fn)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in updater.parameters())
        assert has_grad

    def test_finite_loss(self):
        """Loss should always be finite and non-NaN."""
        updater = self._make_updater()
        n_layers, B, T, d = 2, 1, 8, 16
        ka = [torch.randn(B, T, d) for _ in range(n_layers)]
        va = [torch.randn(B, T, d) for _ in range(n_layers)]
        q  = torch.randint(0, 100, (B, 4))
        # Same chunk for both paths
        loss = path_consistency_loss(updater, ka, va, ka, va, q, self._student_fn)
        assert torch.isfinite(loss), f"loss = {loss.item()} is not finite"
