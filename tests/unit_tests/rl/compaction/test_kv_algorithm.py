# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.rl.compaction.kv import KVCompressor, CompactionResult
from megatron.rl.compaction.kv.selectors import (
    AttentionSumScorer,
    UniformScorer,
)
from megatron.rl.compaction.kv.types import KVMask


def _kv(n: int, d: int = 8):
    """Make random key/value tensors of shape (n, d)."""
    torch.manual_seed(42)
    return torch.randn(n, d), torch.randn(n, d)


class TestKVCompressorProtocol:
    def test_all_selectors_satisfy_protocol(self):
        selectors = [
            AttentionSumScorer(),
            UniformScorer(),
        ]
        for sel in selectors:
            assert isinstance(sel, KVCompressor)


class TestAttentionSumScorer:
    def test_retains_top_scored_positions_with_queries(self):
        # With ref_queries: high-norm keys at positions 2 and 4 should score high.
        torch.manual_seed(0)
        K, V = _kv(6, d=8)
        K[2] = K[2] * 10   # inflate norm of pos 2
        K[4] = K[4] * 10   # inflate norm of pos 4
        Q_ref = torch.randn(2, 8)
        algo = AttentionSumScorer(min_recent=0)
        result = algo.compress(K, V, 2, ref_queries=Q_ref)
        assert isinstance(result, CompactionResult)
        assert len(result.retained_positions) == 2

    def test_fallback_to_norm_without_queries(self):
        K, V = _kv(10)
        K[3] = K[3] * 100  # very high norm at position 3
        algo = AttentionSumScorer(min_recent=0)
        result = algo.compress(K, V, 1)
        assert result.retained_positions == [3]

    def test_always_retains_recent_positions(self):
        K, V = _kv(10)
        algo = AttentionSumScorer(min_recent=3)
        result = algo.compress(K, V, 4)
        # Last 3 must always be retained
        assert 7 in result.retained_positions
        assert 8 in result.retained_positions
        assert 9 in result.retained_positions

    def test_total_positions_is_context_length(self):
        K, V = _kv(20)
        algo = AttentionSumScorer()
        result = algo.compress(K, V, 10)
        assert result.original_length == 20

    def test_budget_clamped_to_n(self):
        K, V = _kv(5)
        algo = AttentionSumScorer()
        result = algo.compress(K, V, 100)
        assert len(result.retained_positions) <= 5

    def test_strategy_label(self):
        K, V = _kv(10)
        algo = AttentionSumScorer()
        result = algo.compress(K, V, 5)
        assert result.strategy == "attention_sum"

    def test_invalid_min_recent_raises(self):
        with pytest.raises(ValueError):
            AttentionSumScorer(min_recent=-1)

    def test_result_has_compacted_kv(self):
        K, V = _kv(10)
        result = AttentionSumScorer().compress(K, V, 5)
        assert result.compacted_keys.shape == (len(result.retained_positions), K.shape[1])
        assert result.compacted_values.shape == (len(result.retained_positions), V.shape[1])

    def test_bias_is_zero(self):
        K, V = _kv(10)
        result = AttentionSumScorer().compress(K, V, 5)
        assert (result.bias == 0.0).all()


class TestUniformScorer:
    def test_covers_full_range(self):
        K, V = _kv(10)
        algo = UniformScorer()
        result = algo.compress(K, V, 5)
        assert result.retained_positions[0] == 0
        assert result.retained_positions[-1] < 10
        assert len(result.retained_positions) == 5

    def test_full_retention_when_budget_ge_n(self):
        K, V = _kv(5)
        algo = UniformScorer()
        result = algo.compress(K, V, 10)
        assert result.retained_positions == list(range(5))

    def test_strategy_label(self):
        K, V = _kv(10)
        algo = UniformScorer()
        result = algo.compress(K, V, 5)
        assert result.strategy == "uniform"

    def test_compacted_kv_are_subset(self):
        K, V = _kv(10)
        result = UniformScorer().compress(K, V, 5)
        for i, pos in enumerate(result.retained_positions):
            assert torch.allclose(result.compacted_keys[i], K[pos])
            assert torch.allclose(result.compacted_values[i], V[pos])


class TestKVMask:
    def test_retention_ratio(self):
        mask = KVMask(
            run_id="r1", step_id=0,
            retained_positions=[0, 2, 4],
            total_positions=10,
            strategy="uniform",
        )
        assert mask.retention_ratio() == pytest.approx(0.3)

    def test_retention_ratio_empty_context(self):
        mask = KVMask(
            run_id="r1", step_id=0,
            retained_positions=[],
            total_positions=0,
            strategy="uniform",
        )
        assert mask.retention_ratio() == pytest.approx(1.0)

    def test_to_context_str_contains_counts(self):
        mask = KVMask(
            run_id="r1", step_id=0,
            retained_positions=list(range(5)),
            total_positions=10,
            strategy="attention_sum",
        )
        s = mask.to_context_str()
        assert "5" in s
        assert "10" in s

    def test_round_trip(self):
        mask = KVMask(
            run_id="r1", step_id=2,
            retained_positions=[0, 1, 3],
            total_positions=5,
            strategy="recent_window",
        )
        restored = KVMask.from_dict(mask.to_dict())
        assert restored.run_id == mask.run_id
        assert restored.retained_positions == mask.retained_positions
        assert restored.strategy == mask.strategy


# ---------------------------------------------------------------------------
# TopKCompressor — fit_values ridge regression stability
# ---------------------------------------------------------------------------

class TestTopKCompressorFitValues:
    """_fit_values must not produce extreme values when the attention
    distribution is nearly one-hot (peaked logits)."""

    def test_fit_values_stable_for_peaked_attention(self):
        """Peaked attention (near-one-hot) must not cause lstsq explosion."""
        from megatron.rl.compaction.kv import TopKCompressor

        torch.manual_seed(0)
        T, D, budget = 64, 8, 32
        # Make keys with one dominant direction to create peaked attention.
        keys = torch.randn(T, D) * 0.01
        keys[:, 0] = torch.linspace(0, 10, T)  # column 0 dominates dot products
        values = torch.randn(T, D)

        comp = TopKCompressor(fit_bias=True, fit_values=True)
        ref_q = keys[:16]  # queries that will have very peaked attention
        result = comp.compress(keys, values, budget, ref_queries=ref_q)

        # Compacted values must be finite and not astronomically large.
        assert not result.compacted_values.isnan().any(), "nan in compacted_values"
        assert not result.compacted_values.isinf().any(), "inf in compacted_values"
        max_orig = values.abs().max().item()
        max_comp = result.compacted_values.abs().max().item()
        # Allow 1000× original magnitude — ridge regression may amplify slightly
        # for barely-constrained positions, but not to the previous 1e13+ scale.
        assert max_comp < max_orig * 1000, (
            f"compacted_values too large: {max_comp:.2f} vs original {max_orig:.2f}"
        )

    def test_fit_values_false_uses_original_values(self):
        """With fit_values=False the compacted values must be a subset of originals."""
        from megatron.rl.compaction.kv import TopKCompressor

        torch.manual_seed(1)
        T, D = 32, 8
        keys = torch.randn(T, D)
        values = torch.randn(T, D)
        comp = TopKCompressor(fit_bias=False, fit_values=False)
        result = comp.compress(keys, values, 16, ref_queries=keys[:8])

        for row in result.compacted_values:
            assert any(torch.allclose(row, values[i]) for i in range(T)), (
                "compacted value not found in original values"
            )
