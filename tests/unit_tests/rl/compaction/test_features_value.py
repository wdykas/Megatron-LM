# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for path_consistency_loss."""

import pytest
import torch
import torch.nn as nn

from megatron.rl.compaction.learned.training.losses import path_consistency_loss
from megatron.rl.compaction.learned.models.belief import BeliefUpdater
from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor


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
