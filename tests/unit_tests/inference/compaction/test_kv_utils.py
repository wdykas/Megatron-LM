# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for paged KV gather/scatter utilities."""

import math

import pytest
import torch

from megatron.core.inference.compaction.kv_utils import (
    gather_kv,
    gather_kv_with_biases,
    write_kv,
    write_kv_with_biases,
)


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cache_params():
    """Standard cache parameters for testing."""
    return {
        "num_layers": 4,
        "num_blocks": 32,
        "block_size": 8,
        "num_heads": 4,
        "head_dim": 16,
        "dtype": torch.bfloat16,
    }


@pytest.fixture
def memory_buffer(cache_params, device):
    """Create a test memory buffer."""
    p = cache_params
    buf = torch.randn(
        2, p["num_layers"], p["num_blocks"], p["block_size"],
        p["num_heads"], p["head_dim"],
        dtype=p["dtype"], device=device,
    )
    return buf


class TestGatherKV:
    """Tests for gather_kv."""

    def test_full_blocks_gather(self, memory_buffer, cache_params, device):
        """Gather from fully occupied blocks gives correct shape."""
        block_ids = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        bs = cache_params["block_size"]
        K, V = gather_kv(memory_buffer, layer=0, block_ids=block_ids, block_size=bs)

        expected_tokens = 3 * bs
        assert K.shape == (expected_tokens, cache_params["num_heads"], cache_params["head_dim"])
        assert V.shape == K.shape

    def test_partial_last_block(self, memory_buffer, cache_params, device):
        """Gather with token_count less than full blocks."""
        block_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
        bs = cache_params["block_size"]
        token_count = bs + 3  # 1 full block + 3 tokens

        K, V = gather_kv(memory_buffer, layer=0, block_ids=block_ids,
                         block_size=bs, token_count=token_count)

        assert K.shape[0] == token_count
        assert V.shape[0] == token_count

    def test_single_block(self, memory_buffer, cache_params, device):
        """Gather from a single block."""
        block_ids = torch.tensor([5], device=device, dtype=torch.int32)
        bs = cache_params["block_size"]

        K, V = gather_kv(memory_buffer, layer=1, block_ids=block_ids, block_size=bs)
        assert K.shape[0] == bs

    def test_gather_matches_direct_read(self, memory_buffer, cache_params, device):
        """Gathered data matches direct indexing into memory buffer."""
        bid = 3
        layer = 2
        bs = cache_params["block_size"]

        block_ids = torch.tensor([bid], device=device, dtype=torch.int32)
        K, V = gather_kv(memory_buffer, layer=layer, block_ids=block_ids, block_size=bs)

        K_direct = memory_buffer[0, layer, bid]  # (block_size, H, D)
        V_direct = memory_buffer[1, layer, bid]

        assert torch.equal(K, K_direct)
        assert torch.equal(V, V_direct)

    def test_multi_block_order_preserved(self, memory_buffer, cache_params, device):
        """Blocks are gathered in the correct order."""
        bs = cache_params["block_size"]
        block_ids = torch.tensor([5, 2, 7], device=device, dtype=torch.int32)

        K, V = gather_kv(memory_buffer, layer=0, block_ids=block_ids, block_size=bs)

        # First block_size tokens should come from block 5
        assert torch.equal(K[:bs], memory_buffer[0, 0, 5])
        # Next block_size from block 2
        assert torch.equal(K[bs:2*bs], memory_buffer[0, 0, 2])
        # Last block_size from block 7
        assert torch.equal(K[2*bs:3*bs], memory_buffer[0, 0, 7])

    def test_contiguous_output(self, memory_buffer, cache_params, device):
        """Output tensors are contiguous."""
        block_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
        bs = cache_params["block_size"]
        K, V = gather_kv(memory_buffer, layer=0, block_ids=block_ids, block_size=bs)
        assert K.is_contiguous()
        assert V.is_contiguous()

    def test_all_layers(self, memory_buffer, cache_params, device):
        """Gather works for all layers."""
        block_ids = torch.tensor([0], device=device, dtype=torch.int32)
        bs = cache_params["block_size"]

        for layer in range(cache_params["num_layers"]):
            K, V = gather_kv(memory_buffer, layer=layer, block_ids=block_ids, block_size=bs)
            assert torch.equal(K, memory_buffer[0, layer, 0])


class TestWriteKV:
    """Tests for write_kv."""

    def test_write_full_blocks(self, memory_buffer, cache_params, device):
        """Write fills blocks correctly."""
        bs = cache_params["block_size"]
        M = 2 * bs  # exactly 2 full blocks
        H = cache_params["num_heads"]
        D = cache_params["head_dim"]

        K_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        V_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        block_ids = torch.tensor([10, 11], device=device, dtype=torch.int32)

        offset = write_kv(memory_buffer, layer=0, block_ids=block_ids,
                          block_size=bs, K_mem=K_mem, V_mem=V_mem)

        assert offset == bs  # last block is full
        assert torch.equal(memory_buffer[0, 0, 10], K_mem[:bs])
        assert torch.equal(memory_buffer[0, 0, 11], K_mem[bs:])

    def test_write_partial_last_block(self, memory_buffer, cache_params, device):
        """Write handles partial last block."""
        bs = cache_params["block_size"]
        M = bs + 3  # 1 full block + 3 tokens
        H = cache_params["num_heads"]
        D = cache_params["head_dim"]

        K_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        V_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        block_ids = torch.tensor([10, 11], device=device, dtype=torch.int32)

        offset = write_kv(memory_buffer, layer=0, block_ids=block_ids,
                          block_size=bs, K_mem=K_mem, V_mem=V_mem)

        assert offset == 3
        assert torch.equal(memory_buffer[0, 0, 10], K_mem[:bs])
        assert torch.equal(memory_buffer[0, 0, 11, :3], K_mem[bs:])
        # Remaining slots zeroed
        assert torch.equal(memory_buffer[0, 0, 11, 3:],
                           torch.zeros(bs - 3, H, D, dtype=cache_params["dtype"], device=device))

    def test_roundtrip_gather_write(self, memory_buffer, cache_params, device):
        """Gather then write roundtrip preserves data."""
        bs = cache_params["block_size"]
        src_blocks = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        dst_blocks = torch.tensor([20, 21, 22], device=device, dtype=torch.int32)

        K_orig, V_orig = gather_kv(memory_buffer, layer=0,
                                    block_ids=src_blocks, block_size=bs)

        write_kv(memory_buffer, layer=0, block_ids=dst_blocks,
                 block_size=bs, K_mem=K_orig, V_mem=V_orig)

        K_back, V_back = gather_kv(memory_buffer, layer=0,
                                    block_ids=dst_blocks, block_size=bs)

        assert torch.equal(K_orig, K_back)
        assert torch.equal(V_orig, V_back)

    def test_roundtrip_partial(self, memory_buffer, cache_params, device):
        """Gather+write roundtrip with partial blocks preserves data."""
        bs = cache_params["block_size"]
        token_count = 2 * bs + 5
        src_blocks = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        dst_blocks = torch.tensor([20, 21, 22], device=device, dtype=torch.int32)

        K_orig, V_orig = gather_kv(memory_buffer, layer=1,
                                    block_ids=src_blocks, block_size=bs,
                                    token_count=token_count)

        write_kv(memory_buffer, layer=1, block_ids=dst_blocks,
                 block_size=bs, K_mem=K_orig, V_mem=V_orig)

        K_back, V_back = gather_kv(memory_buffer, layer=1,
                                    block_ids=dst_blocks, block_size=bs,
                                    token_count=token_count)

        assert torch.equal(K_orig, K_back)
        assert torch.equal(V_orig, V_back)


class TestBiasGatherWrite:
    """Tests for bias-aware gather/write."""

    def test_roundtrip_with_biases(self, memory_buffer, cache_params, device):
        """Gather+write roundtrip with per-head biases preserves all data."""
        bs = cache_params["block_size"]
        num_blocks = cache_params["num_blocks"]
        num_layers = cache_params["num_layers"]
        H = cache_params["num_heads"]

        bias_buffer = torch.randn(
            num_layers, num_blocks, bs, H,
            dtype=cache_params["dtype"], device=device,
        )

        M = bs + 3
        D = cache_params["head_dim"]

        K_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        V_mem = torch.randn(M, H, D, dtype=cache_params["dtype"], device=device)
        biases = torch.randn(M, H, dtype=cache_params["dtype"], device=device)

        block_ids = torch.tensor([15, 16], device=device, dtype=torch.int32)

        write_kv_with_biases(
            memory_buffer, bias_buffer, layer=0,
            block_ids=block_ids, block_size=bs,
            K_mem=K_mem, V_mem=V_mem, biases=biases,
        )

        K_back, V_back, biases_back = gather_kv_with_biases(
            memory_buffer, bias_buffer, layer=0,
            block_ids=block_ids, block_size=bs,
            token_count=M,
        )

        assert torch.equal(K_mem, K_back)
        assert torch.equal(V_mem, V_back)
        assert torch.equal(biases, biases_back)
