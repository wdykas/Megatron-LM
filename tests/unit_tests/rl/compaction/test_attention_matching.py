# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for offline KV cache compressors (compressors.py).

Covers:
  - CompactionResult shape and serialisation
  - TopKCompressor (baseline)
  - OMPCompressor (Algorithm 1 from paper)
  - Value fitting reduces output MSE vs no fitting
  - OMP key selection beats uniform selection on structured data
  - KVCompactionBenchmark runs all variants without error
  - Selectors wrap positional algorithms for benchmark
  - Protocol check (isinstance)
"""

from __future__ import annotations

import math

import pytest
import torch

from megatron.rl.compaction.kv import CompactionResult, KVCompressor
from megatron.rl.compaction.kv.attention_matching import OMPCompressor, TopKCompressor
from megatron.rl.compaction.kv.compressors import (
    _attention_output,
    _fit_bias,
    _fit_values,
    _mass_features,
)
from megatron.rl.compaction.kv.selectors import (
    AttentionSumScorer,
    UniformScorer,
)
from megatron.rl.compaction.kv.benchmark import CompactionBenchmarkResult, KVCompactionBenchmark
from megatron.rl.compaction.kv.types import KVMask


# ---------------------------------------------------------------------------
# Fixtures — CPU tensors (no GPU needed for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_kv():
    torch.manual_seed(42)
    T, d, n = 64, 16, 8
    K = torch.randn(T, d, dtype=torch.float64)
    V = torch.randn(T, d, dtype=torch.float64)
    Q_ref  = torch.randn(n, d, dtype=torch.float64)
    Q_eval = torch.randn(n, d, dtype=torch.float64)
    return K, V, Q_ref, Q_eval


@pytest.fixture
def medium_kv():
    torch.manual_seed(7)
    T, d, n = 256, 32, 16
    K = torch.randn(T, d, dtype=torch.float64)
    V = torch.randn(T, d, dtype=torch.float64)
    Q_ref  = torch.randn(n, d, dtype=torch.float64)
    Q_eval = torch.randn(n, d, dtype=torch.float64)
    return K, V, Q_ref, Q_eval


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

class TestMassFeaturesAndAttention:
    def test_mass_features_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        Phi = _mass_features(Q_ref, K)
        assert Phi.shape == (Q_ref.shape[0], K.shape[0])

    def test_mass_features_non_negative(self, small_kv):
        K, V, Q_ref, _ = small_kv
        Phi = _mass_features(Q_ref, K)
        assert (Phi > 0).all()

    def test_attention_output_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        out, mass = _attention_output(Q_ref, K, V)
        assert out.shape == Q_ref.shape
        assert mass.shape == (Q_ref.shape[0],)

    def test_full_retention_identity(self, small_kv):
        K, V, Q_ref, _ = small_kv
        out1, _ = _attention_output(Q_ref, K, V)
        out2, _ = _attention_output(Q_ref, K, V, bias=torch.zeros(K.shape[0], dtype=K.dtype))
        assert torch.allclose(out1, out2, atol=1e-10)


class TestFitBias:
    def test_fit_bias_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        budget = 16
        C_k = K[:budget]
        beta, w = _fit_bias(K, C_k, Q_ref)
        assert beta.shape == (budget,)
        assert w.shape == (budget,)
        assert (w > 0).all(), "NNLS weights must be non-negative"

    def test_fit_bias_reduces_nnls_objective(self, small_kv):
        K, V, Q_ref, _ = small_kv
        budget = 16
        C_k = K[:budget]
        beta, w = _fit_bias(K, C_k, Q_ref)

        d = K.shape[1]
        logits_orig = Q_ref @ K.T / math.sqrt(d)
        logits_c    = Q_ref @ C_k.T / math.sqrt(d)
        row_max     = logits_orig.max(dim=1, keepdim=True).values
        Phi_orig = torch.exp(logits_orig - row_max)
        Phi_c    = torch.exp(logits_c    - row_max)
        m = Phi_orig.sum(dim=1)

        m_fitted  = Phi_c @ w
        m_uniform = Phi_c.sum(dim=1)

        err_uniform = float(((m - m_uniform) ** 2).mean().item())
        err_fitted  = float(((m - m_fitted) ** 2).mean().item())
        assert err_fitted <= err_uniform + 1e-9

    def test_fit_bias_low_compact_logits_no_nan(self):
        # Regression for bug where compact logits << full-key max caused
        # Phi_c ≈ 0 → NNLS zeros → log(w) = -inf → NaN bias.
        torch.manual_seed(0)
        d = 32
        K = torch.randn(64, d) * 5.0   # large magnitude — high logits
        C_k = torch.randn(4, d) * 0.01  # tiny magnitude — compact logits << row_max
        Q_ref = torch.randn(8, d)
        beta, w = _fit_bias(K, C_k, Q_ref)
        assert not torch.isnan(beta).any(), "bias must not be NaN"
        assert not torch.isinf(beta).any(), "bias must not be ±inf"
        assert (w > 0).all(), "NNLS weights must be positive"


class TestFitValues:
    def test_fit_values_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        budget = 16
        C_k = K[:budget]
        beta = torch.zeros(budget, dtype=K.dtype)
        C_v = _fit_values(K, V, C_k, beta, Q_ref)
        assert C_v.shape == (budget, V.shape[1])

    def test_fit_values_reduces_output_error(self, small_kv):
        K, V, Q_ref, Q_eval = small_kv
        budget = 32
        C_k = K[:budget]
        beta = torch.zeros(budget, dtype=K.dtype)
        C_v_raw = V[:budget]
        C_v_fit = _fit_values(K, V, C_k, beta, Q_ref)

        Y_full, _ = _attention_output(Q_eval, K, V)
        Y_raw, _  = _attention_output(Q_eval, C_k, C_v_raw, beta)
        Y_fit, _  = _attention_output(Q_eval, C_k, C_v_fit, beta)

        mse_raw = float(((Y_full - Y_raw) ** 2).mean().item())
        mse_fit = float(((Y_full - Y_fit) ** 2).mean().item())
        assert mse_fit < mse_raw, "Fitted values should reduce output MSE"


# ---------------------------------------------------------------------------
# TopKCompressor
# ---------------------------------------------------------------------------

class TestTopKCompressor:
    def test_basic_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor().compress(K, V, 16, ref_queries=Q_ref)
        assert len(result.retained_positions) == 16
        assert result.compacted_keys.shape == (16, K.shape[1])
        assert result.compacted_values.shape == (16, V.shape[1])
        assert result.bias.shape == (16,)

    def test_positions_in_bounds(self, small_kv):
        K, V, Q_ref, _ = small_kv
        T = K.shape[0]
        result = TopKCompressor().compress(K, V, 20, ref_queries=Q_ref)
        assert all(0 <= p < T for p in result.retained_positions)
        assert len(set(result.retained_positions)) == len(result.retained_positions)

    def test_no_fit_uses_subset_values(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor(fit_bias=False, fit_values=False).compress(K, V, 16, ref_queries=Q_ref)
        assert (result.bias == 0.0).all()
        for i, pos in enumerate(result.retained_positions):
            assert torch.allclose(result.compacted_values[i], V[pos])

    def test_strategy_string(self):
        assert TopKCompressor().strategy == "topk+bias+values"
        assert TopKCompressor(fit_bias=True, fit_values=False).strategy == "topk+bias"
        assert TopKCompressor(fit_bias=False, fit_values=True).strategy == "topk+values"
        assert TopKCompressor(fit_bias=False, fit_values=False).strategy == "topk"

    def test_to_kv_mask(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor().compress(K, V, 16, ref_queries=Q_ref, run_id="run1", step_id=5)
        mask = result.to_kv_mask()
        assert isinstance(mask, KVMask)
        assert mask.run_id == "run1"
        assert mask.retained_positions == result.retained_positions

    def test_to_dict_serialisable(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor().compress(K, V, 8, ref_queries=Q_ref)
        d = result.to_dict()
        assert isinstance(d["compacted_keys"], list)
        assert d["retention_ratio"] == result.retention_ratio()

    def test_budget_clamp_at_T(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor().compress(K, V, K.shape[0] + 100, ref_queries=Q_ref)
        assert len(result.retained_positions) == K.shape[0]

    def test_budget_clamp_at_1(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = TopKCompressor().compress(K, V, 0, ref_queries=Q_ref)
        assert len(result.retained_positions) == 1


# ---------------------------------------------------------------------------
# OMPCompressor
# ---------------------------------------------------------------------------

class TestOMPCompressor:
    def test_basic_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = OMPCompressor().compress(K, V, 16, ref_queries=Q_ref)
        assert len(result.retained_positions) == 16
        assert result.compacted_keys.shape == (16, K.shape[1])

    def test_positions_unique_and_in_bounds(self, small_kv):
        K, V, Q_ref, _ = small_kv
        T = K.shape[0]
        result = OMPCompressor().compress(K, V, 20, ref_queries=Q_ref)
        assert all(0 <= p < T for p in result.retained_positions)
        assert len(set(result.retained_positions)) == len(result.retained_positions)

    def test_strategy_string(self):
        assert OMPCompressor(keys_per_iter=4).strategy == "omp_k4+values"
        assert OMPCompressor(keys_per_iter=1, fit_values=False).strategy == "omp_k1"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            OMPCompressor(keys_per_iter=0)
        with pytest.raises(ValueError):
            OMPCompressor(nnls_every=0)

    def test_no_fit_values_uses_subset(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = OMPCompressor(fit_values=False).compress(K, V, 16, ref_queries=Q_ref)
        for i, pos in enumerate(result.retained_positions):
            assert torch.allclose(result.compacted_values[i], V[pos])

    def test_omp_better_than_uniform_on_structured_data(self):
        torch.manual_seed(0)
        T, d, n = 128, 16, 8
        K = torch.randn(T, d) * 0.01
        K[:8] = torch.randn(8, d) * 3.0
        V = torch.randn(T, d)
        Q_ref = torch.randn(n, d)
        Q_ref = Q_ref @ K[:8].T @ torch.linalg.pinv(K[:8].T)
        Q_eval = torch.randn(n, d)

        r_omp = OMPCompressor(fit_values=False).compress(K, V, 16, ref_queries=Q_ref)
        r_uni = UniformScorer().compress(K, V, 16)

        Y_full, _ = _attention_output(Q_eval, K, V)
        Y_omp, _  = _attention_output(Q_eval, r_omp.compacted_keys, r_omp.compacted_values, r_omp.bias)
        Y_uni, _  = _attention_output(Q_eval, r_uni.compacted_keys, r_uni.compacted_values, r_uni.bias)

        mse_omp = float(((Y_full - Y_omp) ** 2).mean().item())
        mse_uni = float(((Y_full - Y_uni) ** 2).mean().item())
        assert mse_omp < mse_uni


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_topk_is_protocol(self):
        assert isinstance(TopKCompressor(), KVCompressor)

    def test_omp_is_protocol(self):
        assert isinstance(OMPCompressor(), KVCompressor)

    def test_attention_sum_scorer_is_protocol(self):
        assert isinstance(AttentionSumScorer(), KVCompressor)

    def test_uniform_scorer_is_protocol(self):
        assert isinstance(UniformScorer(), KVCompressor)


# ---------------------------------------------------------------------------
# Selectors (direct compress interface)
# ---------------------------------------------------------------------------

class TestSelectors:
    def test_attention_sum_scorer_shape(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = AttentionSumScorer(min_recent=4).compress(K, V, 16, ref_queries=Q_ref)
        assert len(result.retained_positions) > 0
        assert result.compacted_keys.shape[1] == K.shape[1]
        assert (result.bias == 0.0).all()

    def test_uniform_scorer_values_are_subset(self, small_kv):
        K, V, Q_ref, _ = small_kv
        result = UniformScorer().compress(K, V, 10)
        for i, pos in enumerate(result.retained_positions):
            assert torch.allclose(result.compacted_values[i], V[pos])
            assert torch.allclose(result.compacted_keys[i], K[pos])

    def test_uniform_scorer_full_budget_is_identity(self, small_kv):
        K, V, Q_ref, _ = small_kv
        T = K.shape[0]
        result = UniformScorer().compress(K, V, T)
        assert result.retained_positions == list(range(T))
        assert result.compacted_keys.shape == K.shape


# ---------------------------------------------------------------------------
# KVCompactionBenchmark
# ---------------------------------------------------------------------------

class TestKVCompactionBenchmark:
    def _compressors(self):
        return {
            "omp":       OMPCompressor(fit_values=True),
            "topk_full": TopKCompressor(fit_bias=True, fit_values=True),
            "topk_sel":  TopKCompressor(fit_bias=False, fit_values=False),
            "attn_sum":  AttentionSumScorer(),
            "uniform":   UniformScorer(),
        }

    def test_returns_one_result_per_compressor(self, small_kv):
        K, V, Q_ref, Q_eval = small_kv
        results = KVCompactionBenchmark().run(self._compressors(), K, V, Q_ref, Q_eval, budget=16)
        assert len(results) == len(self._compressors())

    def test_results_sorted_by_mse(self, small_kv):
        K, V, Q_ref, Q_eval = small_kv
        results = KVCompactionBenchmark().run(self._compressors(), K, V, Q_ref, Q_eval, budget=16)
        mses = [r.output_mse for r in results]
        assert mses == sorted(mses)

    def test_result_fields_valid(self, small_kv):
        K, V, Q_ref, Q_eval = small_kv
        results = KVCompactionBenchmark().run({"omp": OMPCompressor()}, K, V, Q_ref, Q_eval, budget=16)
        r = results[0]
        assert 0.0 < r.retention_ratio <= 1.0
        assert r.output_mse >= 0.0
        assert r.wall_time_s >= 0.0

    def test_full_retention_zero_mse(self, small_kv):
        # UniformScorer at budget == T retains all positions (identity),
        # so the benchmark must report ~zero reconstruction error.
        K, V, Q_ref, Q_eval = small_kv
        T = K.shape[0]
        results = KVCompactionBenchmark().run(
            {"full": UniformScorer()}, K, V, Q_ref, Q_eval, budget=T
        )
        assert results[0].output_mse < 1e-10

    def test_value_fitting_helps(self, medium_kv):
        K, V, Q_ref, Q_eval = medium_kv
        results = KVCompactionBenchmark().run(
            {
                "topk_fit":    TopKCompressor(fit_bias=True, fit_values=True),
                "topk_no_fit": TopKCompressor(fit_bias=True, fit_values=False),
            },
            K, V, Q_ref, Q_eval, budget=32,
        )
        by_name = {r.algorithm: r for r in results}
        assert by_name["topk_fit"].output_mse <= by_name["topk_no_fit"].output_mse
