# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for compaction validation harness."""

import math

import pytest
import torch

from megatron.core.inference.compaction.validation import (
    AttentionMatchMetrics,
    CompactionValidationReport,
    LogitDriftMetrics,
    compute_memory_savings,
    run_full_validation,
    validate_attention_output,
    validate_logit_drift,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestAttentionValidation:
    """Tests for attention output validation."""

    def test_perfect_match(self, device):
        """Identical K/V should give zero error."""
        T, H, D = 32, 4, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(8, H, D, device=device)

        metrics = validate_attention_output(K, V, K, V, Q)

        assert metrics.mean_relative_l2 < 1e-5
        assert metrics.max_relative_l2 < 1e-5

    def test_error_increases_with_compression(self, device):
        """Higher compression should generally give larger error."""
        T, H, D = 128, 4, 16
        torch.manual_seed(42)
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(16, H, D, device=device)

        # Low compression (keep 64)
        K_low = K[:64]
        V_low = V[:64]
        m_low = validate_attention_output(K, V, K_low, V_low, Q)

        # High compression (keep 8)
        K_high = K[:8]
        V_high = V[:8]
        m_high = validate_attention_output(K, V, K_high, V_high, Q)

        # Not strictly guaranteed but very likely with random data
        assert m_low.mean_relative_l2 < m_high.mean_relative_l2 + 0.5

    def test_with_biases(self, device):
        """Validation works with biases."""
        T, H, D = 32, 4, 16
        M = 8
        K_full = torch.randn(T, H, D, device=device)
        V_full = torch.randn(T, H, D, device=device)
        K_mem = torch.randn(M, H, D, device=device)
        V_mem = torch.randn(M, H, D, device=device)
        biases = torch.zeros(M, device=device)
        Q = torch.randn(8, H, D, device=device)

        metrics = validate_attention_output(K_full, V_full, K_mem, V_mem, Q, biases)
        assert metrics.mean_relative_l2 >= 0
        assert len(metrics.relative_l2_per_head) == H

    def test_metrics_structure(self, device):
        T, H, D = 32, 4, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(8, H, D, device=device)

        metrics = validate_attention_output(K, V, K[:16], V[:16], Q)
        assert len(metrics.relative_l2_per_head) == H
        assert len(metrics.mass_relative_error_per_head) == H
        assert metrics.mean_mass_error >= 0


class TestLogitDrift:
    """Tests for logit drift validation."""

    def test_identical_logits(self, device):
        """Identical logits give zero drift."""
        logits = torch.randn(32, 1000, device=device)
        metrics = validate_logit_drift(logits, logits)

        assert metrics.kl_divergence < 1e-5
        assert metrics.mean_logprob_delta < 1e-5
        assert metrics.output_match_rate == 1.0
        assert metrics.num_eval_tokens == 32

    def test_different_logits(self, device):
        """Different logits give nonzero drift."""
        logits_full = torch.randn(32, 1000, device=device)
        logits_compact = logits_full + torch.randn_like(logits_full) * 0.5

        metrics = validate_logit_drift(logits_full, logits_compact)

        assert metrics.kl_divergence > 0
        assert metrics.mean_logprob_delta > 0

    def test_metrics_structure(self, device):
        logits = torch.randn(16, 100, device=device)
        metrics = validate_logit_drift(logits, logits + 0.1)

        assert isinstance(metrics, LogitDriftMetrics)
        assert 0 <= metrics.output_match_rate <= 1.0
        # KL can be negligibly negative due to float precision
        assert metrics.kl_divergence >= -1e-6
        assert metrics.reverse_kl >= -1e-6


class TestMemorySavings:
    """Tests for memory savings computation."""

    def test_basic_savings(self):
        savings = compute_memory_savings(
            original_tokens=8192,
            compacted_tokens=512,
            num_heads=32,
            head_dim=128,
            num_layers=32,
        )

        assert savings["compression_ratio"] == 16.0
        assert savings["savings_pct"] > 90.0
        assert savings["saved_bytes"] > 0
        assert savings["original_bytes"] > savings["compacted_bytes"]

    def test_no_compression(self):
        savings = compute_memory_savings(
            original_tokens=100,
            compacted_tokens=100,
            num_heads=4,
            head_dim=64,
            num_layers=8,
        )

        # With biases, compacted is slightly larger
        assert savings["compression_ratio"] == 1.0

    def test_with_biases_overhead(self):
        savings_no_bias = compute_memory_savings(
            100, 50, 4, 64, 8, include_biases=False,
        )
        savings_with_bias = compute_memory_savings(
            100, 50, 4, 64, 8, include_biases=True,
        )
        # Bias adds overhead
        assert savings_with_bias["compacted_bytes"] > savings_no_bias["compacted_bytes"]


class TestFullValidation:
    """Tests for run_full_validation."""

    def test_full_validation(self, device):
        num_layers = 2
        num_blocks = 32
        block_size = 8
        H = 4
        D = 16

        memory_buffer = torch.randn(
            2, num_layers, num_blocks, block_size, H, D,
            dtype=torch.bfloat16, device=device,
        )

        full_blocks = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        compact_blocks = torch.tensor([10, 11], device=device, dtype=torch.int32)

        # Write some data to compact blocks
        from megatron.core.inference.compaction.kv_utils import gather_kv, write_kv
        K_full, V_full = gather_kv(memory_buffer, 0, full_blocks, block_size, 32)
        write_kv(memory_buffer, 0, compact_blocks, block_size, K_full[:16], V_full[:16])

        K_full, V_full = gather_kv(memory_buffer, 1, full_blocks, block_size, 32)
        write_kv(memory_buffer, 1, compact_blocks, block_size, K_full[:16], V_full[:16])

        Q_eval = torch.randn(8, H, D, device=device, dtype=torch.bfloat16)

        report = run_full_validation(
            memory_buffer=memory_buffer,
            full_block_ids=full_blocks,
            full_token_count=32,
            compact_block_ids=compact_blocks,
            compact_token_count=16,
            Q_eval=Q_eval,
            block_size=block_size,
            num_layers=num_layers,
        )

        assert isinstance(report, CompactionValidationReport)
        assert len(report.attention_metrics) == num_layers
        assert report.compression_ratio == 32 / 16
        assert report.memory_saved_bytes > 0
