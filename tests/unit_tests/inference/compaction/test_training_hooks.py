# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for compaction training hooks (Phase 3)."""

import pytest
import torch

from megatron.core.inference.compaction.streaming_compactor import (
    StreamingClusterCompactor,
    StreamingCompactorConfig,
)
from megatron.core.inference.compaction.training_hooks import (
    AttentionOutputDistillationLoss,
    CompactionConsistencyLoss,
    CompactionInTheLoopTrainer,
    CompactionRLCost,
    CompactionTrainingConfig,
    TeacherDistillationLoss,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def training_config():
    return CompactionTrainingConfig(
        kl_weight=1.0,
        distill_weight=0.1,
        teacher_every_n_steps=2,
        compact_checkpoint_interval=64,
        teacher_num_ref_queries=8,
    )


class TestConsistencyLoss:
    """Tests for CompactionConsistencyLoss."""

    def test_identical_logits_zero_loss(self, training_config, device):
        loss_fn = CompactionConsistencyLoss(training_config)
        logits = torch.randn(2, 16, 100, device=device)
        loss = loss_fn(logits, logits)
        assert loss.item() < 1e-5

    def test_different_logits_positive_loss(self, training_config, device):
        loss_fn = CompactionConsistencyLoss(training_config)
        logits_full = torch.randn(2, 16, 100, device=device)
        logits_compact = logits_full + torch.randn_like(logits_full) * 0.5
        loss = loss_fn(logits_full, logits_compact)
        assert loss.item() > 0

    def test_with_mask(self, training_config, device):
        loss_fn = CompactionConsistencyLoss(training_config)
        logits_full = torch.randn(2, 16, 100, device=device)
        logits_compact = torch.randn(2, 16, 100, device=device)  # Fully different
        mask = torch.ones(2, 16, device=device)
        mask[:, 8:] = 0  # Mask out second half

        loss = loss_fn(logits_full, logits_compact, mask)
        assert loss.item() > 0.01  # Unrelated logits should have significant KL

    def test_gradient_flow(self, training_config, device):
        loss_fn = CompactionConsistencyLoss(training_config)
        logits_full = torch.randn(1, 8, 50, device=device)
        logits_compact = torch.randn(1, 8, 50, device=device, requires_grad=True)
        loss = loss_fn(logits_full, logits_compact)
        loss.backward()
        assert logits_compact.grad is not None


class TestTeacherDistillation:
    """Tests for TeacherDistillationLoss."""

    def test_identical_kv_zero_loss(self, training_config, device):
        loss_fn = TeacherDistillationLoss(training_config)
        K = torch.randn(16, 4, 32, device=device)
        V = torch.randn(16, 4, 32, device=device)
        loss = loss_fn(K, V, K, V)
        assert loss.item() < 1e-5

    def test_different_kv_positive_loss(self, training_config, device):
        loss_fn = TeacherDistillationLoss(training_config)
        K_s = torch.randn(16, 4, 32, device=device)
        V_s = torch.randn(16, 4, 32, device=device)
        K_t = torch.randn(16, 4, 32, device=device)
        V_t = torch.randn(16, 4, 32, device=device)
        loss = loss_fn(K_s, V_s, K_t, V_t)
        assert loss.item() > 0

    def test_gradient_flow(self, training_config, device):
        loss_fn = TeacherDistillationLoss(training_config)
        K_s = torch.randn(16, 4, 32, device=device, requires_grad=True)
        V_s = torch.randn(16, 4, 32, device=device, requires_grad=True)
        K_t = torch.randn(16, 4, 32, device=device)
        V_t = torch.randn(16, 4, 32, device=device)
        loss = loss_fn(K_s, V_s, K_t, V_t)
        loss.backward()
        assert K_s.grad is not None
        assert V_s.grad is not None


class TestAttentionDistillation:
    """Tests for AttentionOutputDistillationLoss."""

    def test_basic(self, training_config, device):
        loss_fn = AttentionOutputDistillationLoss(training_config)
        R, H, D, M = 8, 4, 16, 8

        Q = torch.randn(R, H, D, device=device)
        K = torch.randn(M, H, D, device=device)
        V = torch.randn(M, H, D, device=device)

        loss = loss_fn(Q, K, V, None, K, V, None)
        assert loss.item() < 1e-5  # Same K/V should give ~0 loss

    def test_gradient_flow(self, training_config, device):
        loss_fn = AttentionOutputDistillationLoss(training_config)
        R, H, D, M = 8, 4, 16, 8

        Q = torch.randn(R, H, D, device=device)
        K_s = torch.randn(M, H, D, device=device, requires_grad=True)
        V_s = torch.randn(M, H, D, device=device, requires_grad=True)
        K_t = torch.randn(M, H, D, device=device)
        V_t = torch.randn(M, H, D, device=device)

        loss = loss_fn(Q, K_s, V_s, None, K_t, V_t, None)
        loss.backward()
        assert K_s.grad is not None
        assert V_s.grad is not None


class TestRLCost:
    """Tests for CompactionRLCost."""

    def test_zero_compaction(self, training_config, device):
        cost_fn = CompactionRLCost(training_config)
        cost = cost_fn(
            compaction_count=0,
            memory_budget=0,
            kl_drift=0.0,
            max_sequence_length=4096,
        )
        assert cost.item() == 0.0

    def test_cost_increases_with_compaction(self, training_config, device):
        cost_fn = CompactionRLCost(training_config)
        cost1 = cost_fn(1, 512, 0.1, 4096)
        cost2 = cost_fn(5, 512, 0.1, 4096)
        assert cost2.item() > cost1.item()


class TestCompactionInTheLoop:
    """Tests for CompactionInTheLoopTrainer."""

    def test_compaction_checkpoint(self, training_config, device):
        config = StreamingCompactorConfig(num_anchors=8, routing="top1")
        student = StreamingClusterCompactor(16, 4, config).to(device)

        consistency = CompactionConsistencyLoss(training_config)
        distill = TeacherDistillationLoss(training_config)

        trainer = CompactionInTheLoopTrainer(
            config=training_config,
            student_compactor=student,
            consistency_loss=consistency,
            distill_loss=distill,
        )

        K = torch.randn(64, 4, 16, device=device)
        V = torch.randn(64, 4, 16, device=device)
        logits_f = torch.randn(32, 100, device=device)
        logits_c = torch.randn(32, 100, device=device)

        result = trainer.compaction_checkpoint(K, V, logits_f, logits_c)

        assert "K_mem" in result
        assert "V_mem" in result
        assert "consistency" in result
        assert result["K_mem"].shape == (8, 4, 16)

    def test_teacher_distillation_periodic(self, training_config, device):
        training_config.teacher_every_n_steps = 2

        config = StreamingCompactorConfig(num_anchors=8, routing="top1")
        student = StreamingClusterCompactor(16, 4, config).to(device)
        consistency = CompactionConsistencyLoss(training_config)
        distill = TeacherDistillationLoss(training_config)

        trainer = CompactionInTheLoopTrainer(
            config=training_config,
            student_compactor=student,
            consistency_loss=consistency,
            distill_loss=distill,
        )

        K = torch.randn(64, 4, 16, device=device)
        V = torch.randn(64, 4, 16, device=device)
        Q = torch.randn(8, 4, 16, device=device)

        # First call (step 1): no distillation
        result1 = trainer.compaction_checkpoint(K, V, Q_ref=Q)
        assert "distill_kv" not in result1

        # Second call (step 2): has distillation
        result2 = trainer.compaction_checkpoint(K, V, Q_ref=Q)
        assert "distill_kv" in result2

    def test_compute_total_loss(self, training_config, device):
        config = StreamingCompactorConfig(num_anchors=8, routing="top1")
        student = StreamingClusterCompactor(16, 4, config).to(device)
        consistency = CompactionConsistencyLoss(training_config)

        trainer = CompactionInTheLoopTrainer(
            config=training_config,
            student_compactor=student,
            consistency_loss=consistency,
        )

        loss_dict = {
            "consistency": torch.tensor(0.5, device=device),
            "K_mem": torch.randn(8, 4, 16, device=device),
            "V_mem": torch.randn(8, 4, 16, device=device),
        }

        total = trainer.compute_total_loss(loss_dict)
        assert total.item() == pytest.approx(0.5, abs=1e-5)
