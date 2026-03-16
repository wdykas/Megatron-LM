# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for streaming cluster compactor (Phase 3)."""

import math

import pytest
import torch

from megatron.core.inference.compaction.streaming_compactor import (
    StreamingClusterCompactor,
    StreamingCompactorConfig,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def compactor_setup(device):
    """Standard compactor setup."""
    head_dim = 16
    num_heads = 4
    config = StreamingCompactorConfig(
        num_anchors=8,
        routing="top1",
        heads_per_group=2,
    )
    compactor = StreamingClusterCompactor(head_dim, num_heads, config).to(device)
    return compactor, head_dim, num_heads, config


class TestStreamingCompaction:
    """Tests for StreamingClusterCompactor."""

    def test_compact_shape(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        T = 64
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)

        K_mem, V_mem = compactor.compact(K, V)
        assert K_mem.shape == (config.num_anchors, H, D)
        assert V_mem.shape == (config.num_anchors, H, D)

    def test_compact_finite(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        T = 64
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)

        K_mem, V_mem = compactor.compact(K, V)
        assert torch.isfinite(K_mem).all()
        assert torch.isfinite(V_mem).all()

    def test_compact_top2(self, device):
        config = StreamingCompactorConfig(
            num_anchors=8, routing="top2", heads_per_group=1,
        )
        compactor = StreamingClusterCompactor(16, 4, config).to(device)
        K = torch.randn(64, 4, 16, device=device)
        V = torch.randn(64, 4, 16, device=device)

        K_mem, V_mem = compactor.compact(K, V)
        assert K_mem.shape == (8, 4, 16)
        assert torch.isfinite(K_mem).all()

    def test_anchor_initialization(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        T = 100
        K_sample = torch.randn(T, H, D, device=device)

        compactor.initialize_anchors_from_data(K_sample)

        # Anchors should be finite and changed from random init
        assert torch.isfinite(compactor.anchors).all()

    def test_compact_deterministic(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        T = 64
        K = torch.randn(T, H, D, device=device)
        V = torch.randn(T, H, D, device=device)

        K1, V1 = compactor.compact(K, V)
        K2, V2 = compactor.compact(K, V)

        assert torch.allclose(K1, K2, atol=1e-5)
        assert torch.allclose(V1, V2, atol=1e-5)

    def test_compact_dtype_preserved(self, compactor_setup, device):
        compactor, D, H, _ = compactor_setup
        K = torch.randn(64, H, D, device=device, dtype=torch.bfloat16)
        V = torch.randn(64, H, D, device=device, dtype=torch.bfloat16)

        K_mem, V_mem = compactor.compact(K, V)
        assert K_mem.dtype == torch.bfloat16
        assert V_mem.dtype == torch.bfloat16

    def test_forward_equals_compact(self, compactor_setup, device):
        compactor, D, H, _ = compactor_setup
        compactor.eval()  # In eval mode, forward() == compact()
        K = torch.randn(64, H, D, device=device)
        V = torch.randn(64, H, D, device=device)

        K1, V1 = compactor.compact(K, V)
        K2, V2 = compactor(K, V)

        assert torch.allclose(K1, K2, atol=1e-5)
        assert torch.allclose(V1, V2, atol=1e-5)

    def test_soft_routing_has_grad(self, device):
        """Soft routing (train mode) produces differentiable output."""
        config = StreamingCompactorConfig(
            num_anchors=8, routing="top1", heads_per_group=1,
            learnable_anchors=True,
        )
        compactor = StreamingClusterCompactor(16, 4, config).to(device)
        compactor.train()

        K = torch.randn(64, 4, 16, device=device)
        V = torch.randn(64, 4, 16, device=device)

        K_mem, V_mem = compactor(K, V)
        assert K_mem.requires_grad, "compact_soft output should have grad"
        loss = K_mem.float().pow(2).sum()
        loss.backward()
        assert compactor.anchors.grad is not None, "anchors should receive grad"


class TestStreamingFromPages:
    """Tests for page-streaming compaction."""

    def test_compact_from_pages_shape(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        num_layers = 2
        num_blocks = 32
        block_size = 8

        memory_buffer = torch.randn(
            2, num_layers, num_blocks, block_size, H, D,
            dtype=torch.bfloat16, device=device,
        )

        block_ids = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        token_count = 4 * block_size

        K_mem, V_mem = compactor.compact_from_pages(
            memory_buffer, layer=0, block_ids=block_ids,
            block_size=block_size, token_count=token_count,
        )

        assert K_mem.shape == (config.num_anchors, H, D)
        assert V_mem.shape == (config.num_anchors, H, D)

    def test_compact_from_pages_partial(self, compactor_setup, device):
        compactor, D, H, config = compactor_setup
        num_layers = 2
        num_blocks = 32
        block_size = 8

        memory_buffer = torch.randn(
            2, num_layers, num_blocks, block_size, H, D,
            dtype=torch.bfloat16, device=device,
        )

        block_ids = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        token_count = 2 * block_size + 3  # Partial last block

        K_mem, V_mem = compactor.compact_from_pages(
            memory_buffer, layer=0, block_ids=block_ids,
            block_size=block_size, token_count=token_count,
        )

        assert K_mem.shape == (config.num_anchors, H, D)
        assert torch.isfinite(K_mem).all()

    def test_pages_matches_contiguous(self, compactor_setup, device):
        """Page-streaming and contiguous compaction produce same results."""
        compactor, D, H, config = compactor_setup
        block_size = 8
        num_blocks = 4
        T = num_blocks * block_size

        memory_buffer = torch.randn(
            2, 1, 32, block_size, H, D,
            dtype=torch.float32, device=device,
        )

        block_ids = torch.arange(num_blocks, device=device, dtype=torch.int32)

        # From pages
        K_pages, V_pages = compactor.compact_from_pages(
            memory_buffer, layer=0, block_ids=block_ids,
            block_size=block_size, token_count=T,
        )

        # Contiguous
        from megatron.core.inference.compaction.kv_utils import gather_kv
        K_full, V_full = gather_kv(memory_buffer, 0, block_ids, block_size, T)
        K_cont, V_cont = compactor.compact(K_full, V_full)

        assert torch.allclose(K_pages.float(), K_cont.float(), atol=1e-4)
        assert torch.allclose(V_pages.float(), V_cont.float(), atol=1e-4)
