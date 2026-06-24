# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the Still and Belief-Still implementation.

Covers:
  - PerceiverConfig validation
  - PerceiverCompactor: output shapes, per-layer vs shared weights
  - BeliefMemory: factory, accessors, utilities
  - BeliefUpdater: initial_compress, forward (recurrent update), step counter
  - Loss functions: teacher_kl, future_kl, consistency, task
  - CompactorLosses.compute: full and partial loss configs
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn

from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor
from megatron.rl.compaction.learned.models.belief import BeliefMemory, BeliefUpdater
from megatron.rl.compaction.learned.training.losses import (
    CompactorLosses,
    CompactorLossWeights,
    CompactorLossTerms,
    consistency_loss,
    future_kl_loss,
    task_loss,
    teacher_kl_loss,
)
from megatron.rl.compaction.kv.benchmark import KVCompactionBenchmark
from megatron.rl.compaction.kv import OMPCompressor, TopKCompressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return PerceiverConfig(d_kv=16, n_heads=2, n_compress=4, n_attn_layers=3)


@pytest.fixture
def cfg_shared():
    return PerceiverConfig(
        d_kv=16, n_heads=2, n_compress=4, n_attn_layers=3, share_across_layers=True
    )


@pytest.fixture
def compactor(cfg):
    return PerceiverCompactor(cfg)


@pytest.fixture
def compactor_shared(cfg_shared):
    return PerceiverCompactor(cfg_shared)


def _kv(B, T, d, *, seed=0):
    g = torch.Generator(device='cuda')
    g.manual_seed(seed)
    return (
        torch.randn(B, T, d, generator=g),
        torch.randn(B, T, d, generator=g),
    )


def _kv_list(n_layers, B, T, d, *, seed=0):
    ks, vs = [], []
    for l in range(n_layers):
        k, v = _kv(B, T, d, seed=seed + l)
        ks.append(k)
        vs.append(v)
    return ks, vs


# ---------------------------------------------------------------------------
# PerceiverConfig
# ---------------------------------------------------------------------------

class TestPerceiverConfig:
    def test_d_head(self):
        cfg = PerceiverConfig(d_kv=64, n_heads=8, n_compress=16, n_attn_layers=4)
        assert cfg.d_head == 8

    def test_ff_dim_default(self):
        cfg = PerceiverConfig(d_kv=32, n_heads=4, n_compress=8, n_attn_layers=2)
        assert cfg.ff_dim == 128

    def test_ff_dim_override(self):
        cfg = PerceiverConfig(d_kv=32, n_heads=4, n_compress=8, n_attn_layers=2, d_ff=64)
        assert cfg.ff_dim == 64

    def test_invalid_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            PerceiverConfig(d_kv=15, n_heads=4, n_compress=8, n_attn_layers=2)

    def test_invalid_compress(self):
        with pytest.raises(ValueError, match="n_compress"):
            PerceiverConfig(d_kv=16, n_heads=2, n_compress=0, n_attn_layers=2)


# ---------------------------------------------------------------------------
# PerceiverCompactor
# ---------------------------------------------------------------------------

class TestPerceiverCompactor:
    def test_output_shape_single_layer(self, compactor, cfg):
        B, T = 2, 32
        k, v = _kv(B, T, cfg.d_kv)
        C_k, C_v = compactor(k, v, layer_idx=0)
        assert C_k.shape == (B, cfg.n_compress, cfg.d_kv)
        assert C_v.shape == (B, cfg.n_compress, cfg.d_kv)

    def test_all_layers_single_layer_idx(self, compactor, cfg):
        B, T = 1, 16
        k, v = _kv(B, T, cfg.d_kv)
        for l in range(cfg.n_attn_layers):
            C_k, C_v = compactor(k, v, layer_idx=l)
            assert C_k.shape == (B, cfg.n_compress, cfg.d_kv)

    def test_compress_all_layers_shape(self, compactor, cfg):
        B, T = 2, 20
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        ck_list, cv_list = compactor.compress_all_layers(ks, vs)
        assert len(ck_list) == cfg.n_attn_layers
        for ck in ck_list:
            assert ck.shape == (B, cfg.n_compress, cfg.d_kv)

    def test_per_layer_different_outputs(self, compactor, cfg):
        B, T = 1, 16
        k, v = _kv(B, T, cfg.d_kv)
        outputs = [compactor(k, v, layer_idx=l)[0] for l in range(cfg.n_attn_layers)]
        # Per-layer compactor should produce different outputs for each layer
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    "Per-layer compactor should have different weights per layer"

    def test_shared_compactor_n_params(self, compactor, compactor_shared):
        n_per_layer = sum(p.numel() for p in compactor.parameters())
        n_shared = sum(p.numel() for p in compactor_shared.parameters())
        cfg = compactor.cfg
        assert n_shared < n_per_layer, "Shared compactor should have fewer parameters"

    def test_shared_compactor_output_shape(self, compactor_shared, cfg_shared):
        B, T = 2, 24
        k, v = _kv(B, T, cfg_shared.d_kv)
        C_k, C_v = compactor_shared(k, v, layer_idx=0)
        assert C_k.shape == (B, cfg_shared.n_compress, cfg_shared.d_kv)

    def test_grad_flows_through_compactor(self, compactor, cfg):
        B, T = 1, 8
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        ck_list, cv_list = compactor.compress_all_layers(ks, vs)
        # Sum over all layers to ensure all per-layer params are in the graph
        loss = sum(ck.sum() + cv.sum() for ck, cv in zip(ck_list, cv_list))
        loss.backward()
        for name, p in compactor.named_parameters():
            assert p.grad is not None, f"No gradient for parameter: {name}"

    def test_output_changes_with_input(self, compactor, cfg):
        B, T = 1, 16
        k1, v1 = _kv(B, T, cfg.d_kv, seed=0)
        k2, v2 = _kv(B, T, cfg.d_kv, seed=1)
        C_k1, _ = compactor(k1, v1, layer_idx=0)
        C_k2, _ = compactor(k2, v2, layer_idx=0)
        assert not torch.allclose(C_k1, C_k2), "Different inputs should give different outputs"

    def test_variable_T(self, compactor, cfg):
        B = 1
        for T in [4, 16, 128, 512]:
            k, v = _kv(B, T, cfg.d_kv)
            C_k, C_v = compactor(k, v, layer_idx=0)
            assert C_k.shape == (B, cfg.n_compress, cfg.d_kv), \
                f"Shape mismatch for T={T}"


# ---------------------------------------------------------------------------
# BeliefMemory
# ---------------------------------------------------------------------------

class TestBeliefMemory:
    def test_zero_shape(self):
        m = BeliefMemory.zero(n_layers=3, batch=2, budget=4, d_kv=16)
        assert m.keys.shape == (3, 2, 4, 16)
        assert m.values.shape == (3, 2, 4, 16)
        assert m.step == 0

    def test_zero_is_zero(self):
        m = BeliefMemory.zero(n_layers=2, batch=1, budget=8, d_kv=8)
        assert torch.all(m.keys == 0)
        assert torch.all(m.values == 0)

    def test_shape_accessors(self):
        m = BeliefMemory.zero(n_layers=4, batch=3, budget=6, d_kv=32)
        assert m.n_layers == 4
        assert m.batch_size == 3
        assert m.budget == 6
        assert m.d_model == 32

    def test_layer_accessor(self):
        m = BeliefMemory.zero(n_layers=3, batch=1, budget=4, d_kv=8)
        k, v = m.layer(1)
        assert k.shape == (1, 4, 8)
        assert v.shape == (1, 4, 8)

    def test_detach_breaks_grad(self):
        m = BeliefMemory.zero(n_layers=2, batch=1, budget=4, d_kv=8)
        m = BeliefMemory(m.keys.requires_grad_(True), m.values.requires_grad_(True), 0)
        m_det = m.detach()
        assert not m_det.keys.requires_grad
        assert not m_det.values.requires_grad

    def test_keys_list(self):
        m = BeliefMemory.zero(n_layers=3, batch=2, budget=4, d_kv=8)
        ks = m.keys_list()
        assert len(ks) == 3
        for k in ks:
            assert k.shape == (2, 4, 8)

    def test_step_field(self):
        m = BeliefMemory.zero(n_layers=2, batch=1, budget=4, d_kv=8)
        assert m.step == 0
        m2 = BeliefMemory(m.keys, m.values, step=5)
        assert m2.step == 5


# ---------------------------------------------------------------------------
# BeliefUpdater
# ---------------------------------------------------------------------------

class TestBeliefUpdater:
    def test_initial_compress_shape(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 2, 32
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)
        assert m0.keys.shape == (cfg.n_attn_layers, B, cfg.n_compress, cfg.d_kv)
        assert m0.step == 0

    def test_forward_shape(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 2, 16
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)

        # New chunk
        new_ks, new_vs = _kv_list(cfg.n_attn_layers, B, 8, cfg.d_kv, seed=99)
        m1 = updater(m0, new_ks, new_vs)

        assert m1.keys.shape == m0.keys.shape
        assert m1.values.shape == m0.values.shape

    def test_step_counter_increments(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 1, 8
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)
        assert m0.step == 0

        for expected in range(1, 4):
            new_ks, new_vs = _kv_list(cfg.n_attn_layers, B, 4, cfg.d_kv, seed=expected * 10)
            m0 = updater(m0, new_ks, new_vs)
            assert m0.step == expected

    def test_update_changes_memory(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 1, 16
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)

        # Update with a new, different chunk
        new_ks, new_vs = _kv_list(cfg.n_attn_layers, B, 8, cfg.d_kv, seed=999)
        m1 = updater(m0, new_ks, new_vs)

        assert not torch.allclose(m0.keys, m1.keys), "Memory should change after update"

    def test_gradients_through_update(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 1, 8
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)

        new_ks, new_vs = _kv_list(cfg.n_attn_layers, B, 4, cfg.d_kv, seed=7)
        m1 = updater(m0, new_ks, new_vs)

        loss = m1.keys.sum() + m1.values.sum()
        loss.backward()

        for name, p in compactor.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_variable_chunk_sizes(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B = 1
        ks, vs = _kv_list(cfg.n_attn_layers, B, 64, cfg.d_kv)
        m = updater.initial_compress(ks, vs)

        for T_new in [1, 8, 32, 128]:
            new_ks, new_vs = _kv_list(cfg.n_attn_layers, B, T_new, cfg.d_kv, seed=T_new)
            m_new = updater(m, new_ks, new_vs)
            assert m_new.keys.shape == m.keys.shape, \
                f"Shape should be constant regardless of chunk size T_new={T_new}"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class TestLossFunctions:
    def _logits(self, B=2, S=8, V=32, seed=0):
        torch.manual_seed(seed)
        return torch.randn(B, S, V)

    def test_teacher_kl_zero_when_equal(self):
        logits = self._logits()
        loss = teacher_kl_loss(logits, logits)
        assert loss.item() < 1e-5

    def test_teacher_kl_positive_when_different(self):
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        loss = teacher_kl_loss(full, compact)
        assert loss.item() > 0.0

    def test_teacher_kl_shape_scalar(self):
        logits = self._logits()
        loss = teacher_kl_loss(logits, logits * 0.5)
        assert loss.shape == ()   # scalar

    def test_teacher_kl_temperature_scaling(self):
        import torch.nn.functional as F
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        T = 2.0

        # Verify T² scaling is applied: the loss equals T² * raw_kl
        log_p = F.log_softmax(full / T, dim=-1)
        log_q = F.log_softmax(compact / T, dim=-1)
        raw_kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
        expected = raw_kl * T * T

        actual = teacher_kl_loss(full, compact, temperature=T)
        assert torch.isclose(actual, expected, atol=1e-5)

    def test_teacher_kl_reduction_none(self):
        full = self._logits(B=2, S=4)
        compact = self._logits(B=2, S=4, seed=1)
        loss = teacher_kl_loss(full, compact, reduction="none")
        assert loss.shape == (2, 4)

    def test_future_kl_aggregates_steps(self):
        n = 3
        full_list = [self._logits(seed=i) for i in range(n)]
        compact_list = [self._logits(seed=i + 10) for i in range(n)]
        loss = future_kl_loss(full_list, compact_list)
        assert loss.shape == ()
        assert loss.item() > 0.0

    def test_future_kl_zero_when_all_equal(self):
        n = 3
        logits_list = [self._logits(seed=i) for i in range(n)]
        loss = future_kl_loss(logits_list, logits_list)
        assert loss.item() < 1e-5

    def test_future_kl_step_weights(self):
        n = 3
        full_list = [self._logits(seed=i) for i in range(n)]
        compact_list = [self._logits(seed=i + 5) for i in range(n)]
        # Downweight later steps
        loss_uniform = future_kl_loss(full_list, compact_list)
        loss_weighted = future_kl_loss(full_list, compact_list, step_weights=[0.5, 0.3, 0.2])
        # Both should be positive; they'll differ in value
        assert loss_uniform.item() > 0.0
        assert loss_weighted.item() > 0.0

    def test_consistency_zero_when_same(self, cfg):
        m = BeliefMemory.zero(n_layers=cfg.n_attn_layers, batch=1, budget=cfg.n_compress, d_kv=cfg.d_kv)
        loss = consistency_loss(m, m)
        assert loss.item() == pytest.approx(0.0, abs=1e-9)

    def test_consistency_positive_when_different(self, cfg):
        m0 = BeliefMemory.zero(n_layers=cfg.n_attn_layers, batch=1, budget=cfg.n_compress, d_kv=cfg.d_kv)
        m1 = BeliefMemory(torch.randn_like(m0.keys), torch.randn_like(m0.values), step=1)
        loss = consistency_loss(m0, m1)
        assert loss.item() > 0.0

    def test_task_loss_shape(self):
        B, S, V = 2, 8, 64
        logits = torch.randn(B, S, V)
        targets = torch.randint(0, V, (B, S))
        loss = task_loss(logits, targets)
        assert loss.shape == ()

    def test_task_loss_ignore_index(self):
        B, S, V = 2, 8, 64
        logits = torch.randn(B, S, V)
        targets = torch.randint(0, V, (B, S))
        targets[:, :4] = -100   # mask first 4 positions
        loss_masked = task_loss(logits, targets, ignore_index=-100)
        targets[:, :4] = torch.randint(0, V, (B, 4))  # replace with random
        loss_all = task_loss(logits, targets)
        assert loss_masked.item() != loss_all.item(), "Masking should change the loss"


# ---------------------------------------------------------------------------
# CompactorLosses.compute
# ---------------------------------------------------------------------------

class TestBeliefStillLosses:
    def _logits(self, B=2, S=8, V=64, seed=0):
        torch.manual_seed(seed)
        return torch.randn(B, S, V)

    def test_minimal_config_returns_total(self):
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        terms = CompactorLosses.compute(CompactorLossWeights(), full, compact)
        assert isinstance(terms, CompactorLossTerms)
        assert terms.total.shape == ()
        assert terms.future_kl is None
        assert terms.consistency is None
        assert terms.task is None

    def test_total_matches_weighted_sum(self):
        w = CompactorLossWeights(teacher_kl=2.0, future_kl=0.0, consistency=0.0)
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        terms = CompactorLosses.compute(w, full, compact)
        expected = 2.0 * teacher_kl_loss(full, compact)
        assert torch.isclose(terms.total, expected, atol=1e-6)

    def test_future_kl_included_when_provided(self):
        w = CompactorLossWeights(teacher_kl=1.0, future_kl=0.5)
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        future_full = [self._logits(seed=i) for i in range(2)]
        future_compact = [self._logits(seed=i + 10) for i in range(2)]
        terms = CompactorLosses.compute(
            w, full, compact,
            full_logits_future=future_full,
            compact_logits_future=future_compact,
        )
        assert terms.future_kl is not None
        assert terms.future_kl.item() > 0.0

    def test_consistency_included_when_provided(self, cfg):
        w = CompactorLossWeights(consistency=0.1)
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        m0 = BeliefMemory.zero(cfg.n_attn_layers, 1, cfg.n_compress, cfg.d_kv)
        m1 = BeliefMemory(torch.randn_like(m0.keys), torch.randn_like(m0.values), 1)
        terms = CompactorLosses.compute(w, full, compact, memory_t=m0, memory_t1=m1)
        assert terms.consistency is not None
        assert terms.consistency.item() > 0.0

    def test_as_dict_returns_floats(self):
        full = self._logits(seed=0)
        compact = self._logits(seed=1)
        terms = CompactorLosses.compute(CompactorLossWeights(), full, compact)
        d = terms.as_dict(include_total=True)
        assert isinstance(d["total"], float)
        assert isinstance(d["teacher_kl"], float)
        # total excluded by default to prevent double-counting in loggers
        assert "total" not in terms.as_dict()

    def test_gradients_flow_to_model(self, compactor, cfg):
        updater = BeliefUpdater(compactor)
        B, T = 1, 8
        ks, vs = _kv_list(cfg.n_attn_layers, B, T, cfg.d_kv)
        m0 = updater.initial_compress(ks, vs)

        # Simulate compact forward: use m0 keys/values as "compact logits"
        V_vocab = 32
        full = torch.randn(B, T, V_vocab)
        # Create compact logits by passing through a dummy linear
        linear = nn.Linear(cfg.d_kv, V_vocab, bias=False)
        compact = linear(m0.keys[0].mean(dim=1, keepdim=True).expand(B, T, cfg.d_kv))

        terms = CompactorLosses.compute(CompactorLossWeights(), full, compact)
        terms.total.backward()

        # Compactor params should have gradients
        has_grad = any(p.grad is not None for p in compactor.parameters())
        assert has_grad, "Loss should flow gradients back through compact representation"
