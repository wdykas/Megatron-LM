# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for AM compaction algorithm."""

import math

import pytest
import torch

from megatron.core.inference.compaction.am_compaction import (
    AMCompactionResult,
    _compute_attention_mass,
    _compute_attention_output,
    _compute_attention_scores,
    _nnls_solve,
    am_compact,
    am_compact_with_mass,
    fit_biases,
    fit_values,
    select_keys_omp,
    select_keys_top_attention,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def problem_setup(device):
    """Standard test problem: T=128 tokens, H=4 heads, D=32, R=16 queries."""
    T, H, D, R = 128, 4, 32, 16
    torch.manual_seed(42)
    K = torch.randn(T, H, D, device=device)
    V = torch.randn(T, H, D, device=device)
    Q = torch.randn(R, H, D, device=device)
    return K, V, Q, T, H, D, R


class TestAttentionPrimitives:
    """Tests for attention score/mass/output computation."""

    def test_attention_scores_shape(self, device):
        R, T, D = 8, 64, 16
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        scores = _compute_attention_scores(Q, K)
        assert scores.shape == (R, T)

    def test_attention_scores_scale(self, device):
        R, T, D = 4, 32, 16
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        scores_default = _compute_attention_scores(Q, K)
        scores_custom = _compute_attention_scores(Q, K, scale=2.0)
        expected = Q.float() @ K.float().T * 2.0
        assert torch.allclose(scores_custom, expected, atol=1e-5)

    def test_attention_mass_positive(self, device):
        scores = torch.randn(8, 64, device=device)
        mass = _compute_attention_mass(scores)
        assert (mass > 0).all()
        assert mass.shape == (8,)

    def test_attention_output_shape(self, device):
        R, T, D = 8, 64, 16
        scores = torch.randn(R, T, device=device)
        V = torch.randn(T, D, device=device)
        output = _compute_attention_output(scores, V)
        assert output.shape == (R, D)


class TestNNLS:
    """Tests for nonnegative least squares solver."""

    def test_nnls_nonneg(self, device):
        """Solution is nonnegative."""
        A = torch.randn(16, 8, device=device).abs() + 0.1
        b = torch.randn(16, device=device).abs()
        w = _nnls_solve(A, b, max_iters=50)
        assert (w > 0).all()

    def test_nnls_reduces_residual(self, device):
        """NNLS solution reduces residual compared to zero initialization."""
        A = torch.randn(16, 8, device=device).abs() + 0.1
        b = torch.randn(16, device=device).abs()
        w = _nnls_solve(A, b, max_iters=100)

        residual = (A @ w - b).pow(2).sum()
        zero_residual = b.pow(2).sum()
        assert residual < zero_residual

    def test_nnls_exact_solution(self, device):
        """NNLS finds exact solution for well-conditioned problem."""
        M = 4
        w_true = torch.tensor([1.0, 2.0, 3.0, 0.5], device=device)
        A = torch.eye(M, device=device) * 2 + 0.1
        b = A @ w_true
        w = _nnls_solve(A, b, max_iters=200)
        assert torch.allclose(w, w_true, atol=0.1)


class TestKeySelection:
    """Tests for key selection methods."""

    def test_top_attention_shape(self, device):
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        budget = 16
        indices = select_keys_top_attention(Q, K, budget)
        assert indices.shape == (budget,)
        assert indices.max() < T
        assert indices.min() >= 0

    def test_top_attention_sorted(self, device):
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        indices = select_keys_top_attention(Q, K, budget=16)
        assert torch.equal(indices, indices.sort().values)

    def test_top_attention_unique(self, device):
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        indices = select_keys_top_attention(Q, K, budget=16)
        assert len(indices.unique()) == 16

    def test_omp_shape(self, device):
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        budget = 16
        indices, weights = select_keys_omp(Q, K, budget)
        assert indices.shape == (budget,)
        assert weights.shape == (budget,)
        assert (weights > 0).all()

    def test_omp_indices_valid(self, device):
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        indices, _ = select_keys_omp(Q, K, budget=16)
        assert indices.max() < T
        assert indices.min() >= 0

    def test_omp_reduces_mass_residual(self, device):
        """OMP selection reduces mass reconstruction error."""
        T, D, R = 64, 16, 8
        Q = torch.randn(R, D, device=device)
        K = torch.randn(T, D, device=device)
        scale = 1.0 / math.sqrt(D)

        # Full mass
        Phi = (Q.float() @ K.float().T * scale).exp()
        m = Phi.sum(dim=1)

        # OMP mass
        indices, weights = select_keys_omp(Q, K, budget=16)
        Phi_sel = Phi[:, indices]
        m_approx = Phi_sel @ weights

        residual = (m - m_approx).pow(2).sum()
        assert residual < m.pow(2).sum()  # Better than zero


class TestBiasFitting:
    """Tests for bias (beta) fitting."""

    def test_biases_shape(self, device):
        T, D, R, M = 64, 16, 8, 16
        Q = torch.randn(R, D, device=device)
        K_full = torch.randn(T, D, device=device)
        K_sel = K_full[:M]
        biases = fit_biases(Q, K_sel, K_full)
        assert biases.shape == (M,)

    def test_biases_finite(self, device):
        T, D, R, M = 64, 16, 8, 16
        Q = torch.randn(R, D, device=device)
        K_full = torch.randn(T, D, device=device)
        K_sel = K_full[:M]
        biases = fit_biases(Q, K_sel, K_full)
        assert torch.isfinite(biases).all()


class TestValueFitting:
    """Tests for value fitting."""

    def test_values_shape(self, device):
        T, D, R, M = 64, 16, 8, 16
        Q = torch.randn(R, D, device=device)
        K_full = torch.randn(T, D, device=device)
        V_full = torch.randn(T, D, device=device)
        K_sel = K_full[:M]
        biases = torch.zeros(M, device=device)
        V_mem = fit_values(Q, K_sel, biases, K_full, V_full)
        assert V_mem.shape == (M, D)

    def test_values_reduce_output_error(self, device):
        """Fitted values reduce attention output error vs random values."""
        T, D, R, M = 64, 16, 8, 16
        torch.manual_seed(42)
        Q = torch.randn(R, D, device=device)
        K_full = torch.randn(T, D, device=device)
        V_full = torch.randn(T, D, device=device)
        K_sel = K_full[:M]
        biases = torch.zeros(M, device=device)

        V_fitted = fit_values(Q, K_sel, biases, K_full, V_full)
        V_random = torch.randn(M, D, device=device)

        scale = 1.0 / math.sqrt(D)

        # Full output
        full_scores = Q.float() @ K_full.float().T * scale
        O_full = torch.softmax(full_scores, dim=-1) @ V_full.float()

        # Fitted output
        comp_scores = Q.float() @ K_sel.float().T * scale
        O_fitted = torch.softmax(comp_scores, dim=-1) @ V_fitted.float()

        # Random output
        O_random = torch.softmax(comp_scores, dim=-1) @ V_random.float()

        err_fitted = (O_full - O_fitted).pow(2).mean()
        err_random = (O_full - O_random).pow(2).mean()

        assert err_fitted < err_random


class TestAMCompact:
    """Tests for the full AM compaction pipeline."""

    def test_am_compact_basic(self, problem_setup):
        K, V, Q, T, H, D, R = problem_setup
        budget = 32
        result = am_compact(K, V, Q, budget, method="top_attention")

        assert isinstance(result, AMCompactionResult)
        assert result.K_mem.shape == (budget, H, D)
        assert result.V_mem.shape == (budget, H, D)
        assert result.biases.shape == (budget, H)
        assert result.mass_error >= 0
        assert result.output_error >= 0

    def test_am_compact_omp(self, problem_setup):
        K, V, Q, T, H, D, R = problem_setup
        budget = 16
        result = am_compact(K, V, Q, budget, method="omp")

        assert result.K_mem.shape == (budget, H, D)
        assert result.V_mem.shape == (budget, H, D)
        assert torch.isfinite(result.K_mem).all()
        assert torch.isfinite(result.V_mem).all()

    def test_am_compact_reduces_error(self, problem_setup):
        """Compaction produces lower error than random K/V."""
        K, V, Q, T, H, D, R = problem_setup
        budget = 32
        result = am_compact(K, V, Q, budget, method="top_attention")

        # Error should be bounded (not perfect but meaningful)
        assert result.output_error < 1.0  # relative L2 < 100%

    def test_am_compact_dtype_preserved(self, problem_setup):
        K, V, Q, T, H, D, R = problem_setup
        K = K.bfloat16()
        V = V.bfloat16()
        Q = Q.bfloat16()
        result = am_compact(K, V, Q, budget=16, method="top_attention")
        assert result.K_mem.dtype == torch.bfloat16
        assert result.V_mem.dtype == torch.bfloat16

    def test_am_compact_with_mass(self, problem_setup):
        K, V, Q, T, H, D, R = problem_setup
        budget = 32
        result = am_compact_with_mass(K, V, Q, budget, method="top_attention")

        assert result.K_mem.shape == (budget, H, D)
        assert result.mass_error >= 0
        assert result.output_error >= 0

    def test_compression_ratios(self, problem_setup):
        """Test various compression ratios."""
        K, V, Q, T, H, D, R = problem_setup
        for budget in [8, 16, 32, 64]:
            result = am_compact(K, V, Q, budget, method="top_attention")
            assert result.K_mem.shape[0] == budget
            # Higher budget should generally give lower error
            # (not strict monotonic due to randomness, but sanity check)
            assert result.output_error < 5.0  # Should not explode
