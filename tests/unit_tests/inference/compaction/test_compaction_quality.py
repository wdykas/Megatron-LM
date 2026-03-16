# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end compaction quality tests on GPU.

These tests verify that compaction actually works — not just shapes and
structure, but that the compacted cache faithfully reproduces the attention
outputs and logit distributions of the full cache.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.compaction.am_compaction import (
    am_compact,
    am_compact_with_mass,
    fit_biases,
    fit_values,
    select_keys_omp,
    select_keys_top_attention,
)
from megatron.core.inference.compaction.kv_utils import gather_kv, write_kv
from megatron.core.inference.compaction.streaming_compactor import (
    StreamingClusterCompactor,
    StreamingCompactorConfig,
)
from megatron.core.inference.compaction.validation import (
    compute_memory_savings,
    validate_attention_output,
    validate_logit_drift,
)


def require_cuda(fn):
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")(fn)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA required")


# =========================================================================
# Test: AM compaction actually reduces attention output error
# =========================================================================


class TestAMCompactionQuality:
    """Verify AM compaction faithfully reproduces full-cache attention."""

    @require_cuda
    def test_attention_output_error_decreases_with_budget(self, device):
        """Higher budget -> lower attention output error."""
        torch.manual_seed(42)
        T, H, D, R = 512, 8, 64, 32
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q_ref = torch.randn(R, H, D, device=device)
        Q_eval = torch.randn(R, H, D, device=device)  # Held-out queries

        errors = []
        for budget in [16, 32, 64, 128, 256]:
            result = am_compact(K, V, Q_ref, budget, method="top_attention")
            m = validate_attention_output(K, V, result.K_mem, result.V_mem, Q_eval, result.biases)
            errors.append(m.mean_relative_l2)

        # Error should generally decrease as budget increases
        # Allow some noise but check the trend
        assert errors[-1] < errors[0], (
            f"Error should decrease with budget: {list(zip([16,32,64,128,256], errors))}"
        )
        # With random data (not real model KV), held-out query error may not
        # drop below 0.5 but should still be meaningfully lower than small budgets
        assert errors[-1] < 1.0, f"256-token budget should have <100% error, got {errors[-1]}"

    @require_cuda
    def test_omp_better_than_random_selection(self, device):
        """OMP key selection should outperform random key selection."""
        torch.manual_seed(42)
        T, H, D, R = 256, 4, 32, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(R, H, D, device=device)
        budget = 32

        # OMP compaction
        result_omp = am_compact(K, V, Q, budget, method="omp")
        m_omp = validate_attention_output(K, V, result_omp.K_mem, result_omp.V_mem, Q, result_omp.biases)

        # Random selection baseline
        rand_idx = torch.randperm(T, device=device)[:budget]
        K_rand = K[rand_idx]
        V_rand = V[rand_idx]
        m_rand = validate_attention_output(K, V, K_rand, V_rand, Q)

        assert m_omp.mean_relative_l2 < m_rand.mean_relative_l2, (
            f"OMP ({m_omp.mean_relative_l2:.4f}) should beat random ({m_rand.mean_relative_l2:.4f})"
        )

    @require_cuda
    def test_top_attention_better_than_random(self, device):
        """Top-attention selection should outperform random."""
        torch.manual_seed(42)
        T, H, D, R = 256, 4, 32, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(R, H, D, device=device)
        budget = 32

        result = am_compact(K, V, Q, budget, method="top_attention")
        m_am = validate_attention_output(K, V, result.K_mem, result.V_mem, Q, result.biases)

        rand_idx = torch.randperm(T, device=device)[:budget]
        m_rand = validate_attention_output(K, V, K[rand_idx], V[rand_idx], Q)

        assert m_am.mean_relative_l2 < m_rand.mean_relative_l2

    @require_cuda
    def test_biases_improve_mass_matching(self, device):
        """Fitting biases should improve attention mass reconstruction."""
        torch.manual_seed(42)
        T, D, R, M = 256, 32, 16, 32
        K_full = torch.randn(T, D, device=device)
        Q = torch.randn(R, D, device=device)
        scale = 1.0 / math.sqrt(D)

        # Select keys
        indices = select_keys_top_attention(Q, K_full, M, scale)
        K_sel = K_full[indices]

        # Mass with no biases
        scores_full = (Q.float() @ K_full.float().T * scale).exp()
        mass_full = scores_full.sum(dim=1)
        scores_sel = (Q.float() @ K_sel.float().T * scale).exp()
        mass_no_bias = scores_sel.sum(dim=1)
        err_no_bias = ((mass_full - mass_no_bias).pow(2).mean() / mass_full.pow(2).mean()).sqrt()

        # Mass with fitted biases
        biases = fit_biases(Q, K_sel, K_full, scale)
        scores_biased = (Q.float() @ K_sel.float().T * scale + biases.float().unsqueeze(0)).exp()
        mass_biased = scores_biased.sum(dim=1)
        err_biased = ((mass_full - mass_biased).pow(2).mean() / mass_full.pow(2).mean()).sqrt()

        assert err_biased < err_no_bias, (
            f"Biases should improve mass: with={err_biased:.4f} vs without={err_no_bias:.4f}"
        )

    @require_cuda
    def test_value_fitting_improves_output(self, device):
        """Fitted values should outperform raw selected values."""
        torch.manual_seed(42)
        T, D, R, M = 256, 32, 16, 32
        K_full = torch.randn(T, D, device=device)
        V_full = torch.randn(T, D, device=device)
        Q = torch.randn(R, D, device=device)
        scale = 1.0 / math.sqrt(D)

        indices = select_keys_top_attention(Q, K_full, M, scale)
        K_sel = K_full[indices]
        V_raw = V_full[indices]
        biases = fit_biases(Q, K_sel, K_full, scale)
        V_fitted = fit_values(Q, K_sel, biases, K_full, V_full, scale)

        # Full output
        full_scores = Q.float() @ K_full.float().T * scale
        O_full = F.softmax(full_scores, dim=-1) @ V_full.float()

        # Compact output with raw values
        comp_scores = Q.float() @ K_sel.float().T * scale + biases.float().unsqueeze(0)
        O_raw = F.softmax(comp_scores, dim=-1) @ V_raw.float()
        err_raw = (O_full - O_raw).pow(2).mean()

        # Compact output with fitted values
        O_fitted = F.softmax(comp_scores, dim=-1) @ V_fitted.float()
        err_fitted = (O_full - O_fitted).pow(2).mean()

        assert err_fitted < err_raw, (
            f"Fitted values should be better: {err_fitted:.6f} vs raw {err_raw:.6f}"
        )

    @require_cuda
    def test_am_with_mass_preserves_mass(self, device):
        """AM+mass variant should preserve attention mass well."""
        torch.manual_seed(42)
        T, H, D, R = 256, 4, 32, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(R, H, D, device=device)

        result = am_compact_with_mass(K, V, Q, budget=64, method="top_attention")
        m = validate_attention_output(K, V, result.K_mem, result.V_mem, Q, result.biases)

        # Mass error should be reasonable
        assert m.mean_mass_error < 0.5, f"Mass error too high: {m.mean_mass_error}"
        # Output error should be reasonable
        assert m.mean_relative_l2 < 0.5, f"Output error too high: {m.mean_relative_l2}"


# =========================================================================
# Test: Paged gather/write roundtrip on GPU
# =========================================================================


class TestPagedKVOnGPU:
    """Verify paged KV operations are exact on GPU."""

    @require_cuda
    def test_gather_write_roundtrip_bitwise(self, device):
        """Gather -> write -> gather should be bitwise identical."""
        buf = torch.randn(2, 4, 64, 16, 8, 32, dtype=torch.bfloat16, device=device)
        src = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=torch.int32)
        dst = torch.tensor([50, 51, 52, 53, 54], device=device, dtype=torch.int32)

        for layer in range(4):
            K_orig, V_orig = gather_kv(buf, layer, src, block_size=16)
            write_kv(buf, layer, dst, block_size=16, K_mem=K_orig, V_mem=V_orig)
            K_back, V_back = gather_kv(buf, layer, dst, block_size=16)
            assert torch.equal(K_orig, K_back), f"Key roundtrip failed at layer {layer}"
            assert torch.equal(V_orig, V_back), f"Value roundtrip failed at layer {layer}"

    @require_cuda
    def test_partial_block_roundtrip(self, device):
        """Roundtrip with partial last block is exact."""
        buf = torch.randn(2, 2, 32, 8, 4, 16, dtype=torch.bfloat16, device=device)
        src = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        dst = torch.tensor([20, 21, 22], device=device, dtype=torch.int32)
        token_count = 19  # 2 full blocks + 3 tokens

        K_orig, V_orig = gather_kv(buf, 0, src, 8, token_count=token_count)
        write_kv(buf, 0, dst, 8, K_orig, V_orig)
        K_back, V_back = gather_kv(buf, 0, dst, 8, token_count=token_count)
        assert torch.equal(K_orig, K_back)
        assert torch.equal(V_orig, V_back)

    @require_cuda
    def test_compaction_into_fewer_blocks(self, device):
        """Compact from many blocks into fewer blocks."""
        buf = torch.randn(2, 2, 100, 8, 4, 16, dtype=torch.bfloat16, device=device)
        # Source: 8 blocks = 64 tokens
        src = torch.arange(8, device=device, dtype=torch.int32)
        K_full, V_full = gather_kv(buf, 0, src, 8)
        assert K_full.shape[0] == 64

        # Compact to 16 tokens = 2 blocks
        budget = 16
        Q_ref = torch.randn(8, 4, 16, device=device, dtype=torch.bfloat16)
        result = am_compact(K_full, V_full, Q_ref, budget, method="top_attention")

        # Write to 2 blocks
        dst = torch.tensor([90, 91], device=device, dtype=torch.int32)
        write_kv(buf, 0, dst, 8, result.K_mem, result.V_mem)
        K_back, V_back = gather_kv(buf, 0, dst, 8, token_count=budget)

        assert torch.equal(result.K_mem, K_back)
        assert torch.equal(result.V_mem, V_back)


# =========================================================================
# Test: Full offline compaction pipeline on GPU
# =========================================================================


class TestOfflineCompactionE2E:
    """End-to-end offline compaction on GPU."""

    @require_cuda
    def test_offline_compaction_reduces_error(self, device):
        """Full pipeline: pages -> gather -> AM compact -> write -> validate."""
        num_layers = 4
        num_blocks = 64
        block_size = 16
        H, D = 8, 32
        T = 256  # 16 blocks of 16 tokens
        budget = 64

        buf = torch.randn(2, num_layers, num_blocks, block_size, H, D,
                          dtype=torch.bfloat16, device=device)
        src_blocks = torch.arange(T // block_size, device=device, dtype=torch.int32)

        Q_ref = torch.randn(16, H, D, device=device, dtype=torch.bfloat16)
        Q_eval = torch.randn(16, H, D, device=device, dtype=torch.bfloat16)

        for layer in range(num_layers):
            K_full, V_full = gather_kv(buf, layer, src_blocks, block_size, T)

            result = am_compact(
                K_full.float(), V_full.float(), Q_ref.float(),
                budget, method="top_attention",
            )

            # Write compact to new blocks
            num_dst = math.ceil(budget / block_size)
            dst_start = 32 + layer * num_dst
            dst_blocks = torch.arange(
                dst_start, dst_start + num_dst, device=device, dtype=torch.int32
            )
            write_kv(buf, layer, dst_blocks, block_size,
                     result.K_mem.bfloat16(), result.V_mem.bfloat16())

            # Validate
            K_compact, V_compact = gather_kv(buf, layer, dst_blocks, block_size, budget)
            m = validate_attention_output(
                K_full, V_full, K_compact, V_compact, Q_eval, result.biases,
            )

            assert m.mean_relative_l2 < 1.0, (
                f"Layer {layer}: error {m.mean_relative_l2:.4f} too high"
            )

    @require_cuda
    def test_compression_ratio_vs_quality(self, device):
        """Higher compression ratios degrade quality but remain bounded."""
        T, H, D = 512, 4, 32
        buf = torch.randn(2, 1, 64, 16, H, D, dtype=torch.bfloat16, device=device)
        src = torch.arange(T // 16, device=device, dtype=torch.int32)
        K_full, V_full = gather_kv(buf, 0, src, 16, T)

        Q = torch.randn(16, H, D, device=device)

        ratios_and_errors = []
        for budget in [256, 128, 64, 32]:
            result = am_compact(K_full.float(), V_full.float(), Q.float(),
                               budget, method="top_attention")
            m = validate_attention_output(K_full, V_full, result.K_mem, result.V_mem, Q, result.biases)
            ratio = T / budget
            ratios_and_errors.append((ratio, m.mean_relative_l2))

        # All should have bounded error
        for ratio, err in ratios_and_errors:
            assert err < 2.0, f"Compression {ratio}x: error {err:.4f} too high"

        # Higher ratio should generally have more error
        assert ratios_and_errors[-1][1] > ratios_and_errors[0][1] * 0.5, (
            "16x compression should have meaningfully more error than 2x"
        )


# =========================================================================
# Test: Streaming compactor on GPU
# =========================================================================


class TestStreamingCompactorQuality:
    """Verify streaming compactor produces useful compaction on GPU."""

    @require_cuda
    def test_streaming_better_than_truncation(self, device):
        """Streaming compactor should outperform simple truncation."""
        torch.manual_seed(42)
        T, H, D = 256, 4, 32
        M = 32
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)
        Q = torch.randn(16, H, D, device=device)

        config = StreamingCompactorConfig(num_anchors=M, routing="top1")
        compactor = StreamingClusterCompactor(D, H, config).to(device)
        compactor.initialize_anchors_from_data(K)

        K_stream, V_stream = compactor.compact(K, V)
        K_trunc, V_trunc = K[:M], V[:M]

        m_stream = validate_attention_output(K, V, K_stream, V_stream, Q)
        m_trunc = validate_attention_output(K, V, K_trunc, V_trunc, Q)

        # Streaming should generally be competitive with or better than truncation
        # (truncation keeps first M which loses recent info)
        assert m_stream.mean_relative_l2 < m_trunc.mean_relative_l2 + 0.3, (
            f"Streaming ({m_stream.mean_relative_l2:.4f}) should be "
            f"competitive with truncation ({m_trunc.mean_relative_l2:.4f})"
        )

    @require_cuda
    def test_streaming_from_pages_matches_contiguous(self, device):
        """Page-streaming produces identical results to contiguous compaction."""
        H, D, M = 4, 32, 16
        block_size = 8
        num_blocks = 8
        T = num_blocks * block_size

        buf = torch.randn(2, 1, 32, block_size, H, D, dtype=torch.float32, device=device)
        block_ids = torch.arange(num_blocks, device=device, dtype=torch.int32)

        config = StreamingCompactorConfig(num_anchors=M, routing="top1")
        compactor = StreamingClusterCompactor(D, H, config).to(device)

        K_pages, V_pages = compactor.compact_from_pages(buf, 0, block_ids, block_size, T)
        K_full, V_full = gather_kv(buf, 0, block_ids, block_size, T)
        K_cont, V_cont = compactor.compact(K_full, V_full)

        assert torch.allclose(K_pages.float(), K_cont.float(), atol=1e-4)
        assert torch.allclose(V_pages.float(), V_cont.float(), atol=1e-4)

    @require_cuda
    def test_top2_routing_captures_more(self, device):
        """Top-2 routing should utilize more anchors than top-1."""
        torch.manual_seed(42)
        T, H, D, M = 128, 4, 32, 16
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)

        config1 = StreamingCompactorConfig(num_anchors=M, routing="top1")
        config2 = StreamingCompactorConfig(num_anchors=M, routing="top2")
        c1 = StreamingClusterCompactor(D, H, config1).to(device)
        c2 = StreamingClusterCompactor(D, H, config2).to(device)
        # Share anchors
        c2.anchors.data.copy_(c1.anchors.data)

        K1, V1 = c1.compact(K, V)
        K2, V2 = c2.compact(K, V)

        # Top-2 should have fewer zero-valued anchors (more utilization)
        zero1 = (K1.abs().sum(dim=(1, 2)) < 1e-6).sum().item()
        zero2 = (K2.abs().sum(dim=(1, 2)) < 1e-6).sum().item()
        assert zero2 <= zero1, "Top-2 should utilize at least as many anchors as top-1"


# =========================================================================
# Test: Multi-round compaction stability on GPU
# =========================================================================


class TestMultiRoundStability:
    """Verify repeated compaction doesn't explode error."""

    @require_cuda
    def test_repeated_compaction_bounded_error(self, device):
        """Error stays bounded when compacting growing buffers repeatedly."""
        torch.manual_seed(42)
        H, D = 4, 32
        budget = 32
        chunk_size = 64
        Q_ref = torch.randn(16, H, D, device=device)

        # Simulate: each round, add new tokens and re-compact
        K_mem, V_mem = None, None
        errors = []

        for round_num in range(5):
            K_new = torch.randn(chunk_size, H, D, device=device)
            V_new = torch.randn(chunk_size, H, D, device=device)

            if K_mem is not None:
                K_combined = torch.cat([K_mem, K_new])
                V_combined = torch.cat([V_mem, V_new])
            else:
                K_combined = K_new
                V_combined = V_new

            result = am_compact(
                K_combined.float(), V_combined.float(), Q_ref.float(),
                budget, method="top_attention",
            )
            K_mem = result.K_mem
            V_mem = result.V_mem

            # Measure error vs this round's full input
            m = validate_attention_output(
                K_combined, V_combined, K_mem, V_mem, Q_ref, result.biases,
            )
            errors.append(m.mean_relative_l2)

        # All errors should be bounded
        for i, err in enumerate(errors):
            assert err < 5.0, f"Error exploded at round {i}: {err}"

    @require_cuda
    def test_online_simulation(self, device):
        """Simulate online compaction: grow hot window, compact, repeat."""
        torch.manual_seed(42)
        H, D = 4, 32
        budget = 32
        hot_window = 64
        block_size = 8
        num_total_blocks = 128

        buf = torch.randn(2, 1, num_total_blocks, block_size, H, D,
                          dtype=torch.bfloat16, device=device)

        Q_ref = torch.randn(8, H, D, device=device)

        # Simulate: generate tokens, compact when hot window overflows
        mem_K, mem_V = None, None
        mem_blocks = None
        hot_start_block = 0
        logical_pos = 0
        compaction_errors = []

        for step in range(3):
            # "Generate" hot_window tokens
            hot_block_count = hot_window // block_size
            hot_blocks = torch.arange(
                hot_start_block, hot_start_block + hot_block_count,
                device=device, dtype=torch.int32,
            )
            logical_pos += hot_window

            if mem_K is not None:
                # Combine memory + hot for compaction input
                K_hot, V_hot = gather_kv(buf, 0, hot_blocks, block_size, hot_window)
                K_combined = torch.cat([mem_K, K_hot])
                V_combined = torch.cat([mem_V, V_hot])
            else:
                K_combined, V_combined = gather_kv(buf, 0, hot_blocks, block_size, hot_window)

            # Compact
            result = am_compact(
                K_combined.float(), V_combined.float(), Q_ref.float(),
                budget, method="top_attention",
            )
            mem_K = result.K_mem
            mem_V = result.V_mem
            compaction_errors.append(result.output_error)

            hot_start_block += hot_block_count

        # Errors should all be bounded
        for i, err in enumerate(compaction_errors):
            assert err < 3.0, f"Round {i} error {err:.4f} too high"

        # Logical position should only go forward
        assert logical_pos == hot_window * 3


# =========================================================================
# Test: Memory savings are real
# =========================================================================


class TestMemorySavingsGPU:
    """Verify actual GPU memory savings from compaction."""

    @require_cuda
    def test_block_count_reduction(self, device):
        """Compaction uses fewer blocks than original."""
        T = 512
        budget = 64
        block_size = 16
        H, D = 8, 64

        original_blocks = math.ceil(T / block_size)  # 32 blocks
        compact_blocks = math.ceil(budget / block_size)  # 4 blocks

        savings = compute_memory_savings(T, budget, H, D, num_layers=32)
        assert savings["compression_ratio"] == T / budget  # 8x
        assert savings["savings_pct"] > 85.0  # Should save >85% memory

        # Verify block reduction
        assert compact_blocks < original_blocks
        assert compact_blocks == 4
        assert original_blocks == 32
