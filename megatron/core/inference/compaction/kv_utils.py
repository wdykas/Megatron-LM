# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paged KV cache gather/scatter utilities for compaction.

Provides gather_kv() to read contiguous K/V tensors from paged blocks,
and write_kv() to write compacted K/V back into freshly allocated pages.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def gather_kv(
    memory_buffer: Tensor,
    layer: int,
    block_ids: Tensor,
    block_size: int,
    token_count: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Gather contiguous K and V tensors from paged KV cache blocks.

    Reads tokens from the specified blocks for a single layer and assembles
    them into contiguous [T, H, D] tensors.

    Args:
        memory_buffer: The 6D KV cache tensor
            shape (2, num_layers, num_blocks, block_size, num_heads, head_dim).
        layer: Layer index (0-based).
        block_ids: 1D tensor of block IDs to gather from, in logical order.
        block_size: Number of tokens per block.
        token_count: Total number of valid tokens across all blocks.
            If None, assumes all blocks are fully occupied (len(block_ids) * block_size).

    Returns:
        K_full: [T, H, D] contiguous key tensor.
        V_full: [T, H, D] contiguous value tensor.
    """
    num_blocks = block_ids.shape[0]
    total_slots = num_blocks * block_size
    if token_count is None:
        token_count = total_slots

    # Index into the cache: memory_buffer[kv, layer, block_id] -> (block_size, H, D)
    # Gather all blocks at once using advanced indexing.
    key_cache = memory_buffer[0, layer]  # (num_blocks_total, block_size, H, D)
    val_cache = memory_buffer[1, layer]

    # Gather: (num_blocks_gathered, block_size, H, D)
    K_blocks = key_cache[block_ids.long()]
    V_blocks = val_cache[block_ids.long()]

    # Reshape to (num_blocks * block_size, H, D) then trim to token_count
    num_heads = K_blocks.shape[2]
    head_dim = K_blocks.shape[3]
    K_full = K_blocks.reshape(total_slots, num_heads, head_dim)[:token_count].contiguous()
    V_full = V_blocks.reshape(total_slots, num_heads, head_dim)[:token_count].contiguous()

    return K_full, V_full


def write_kv(
    memory_buffer: Tensor,
    layer: int,
    block_ids: Tensor,
    block_size: int,
    K_mem: Tensor,
    V_mem: Tensor,
) -> int:
    """Write compacted K/V tensors back into paged cache blocks.

    Distributes the M compacted tokens across the provided blocks.

    Args:
        memory_buffer: The 6D KV cache tensor
            shape (2, num_layers, num_blocks, block_size, num_heads, head_dim).
        layer: Layer index (0-based).
        block_ids: 1D tensor of block IDs to write into (must have enough
            capacity: len(block_ids) * block_size >= M).
        block_size: Number of tokens per block.
        K_mem: [M, H, D] compacted key tensor.
        V_mem: [M, H, D] compacted value tensor.

    Returns:
        Number of tokens written into the last block (last_block_offset).
    """
    M = K_mem.shape[0]
    num_blocks_needed = math.ceil(M / block_size)
    assert block_ids.shape[0] >= num_blocks_needed, (
        f"Need {num_blocks_needed} blocks but only {block_ids.shape[0]} provided"
    )

    key_cache = memory_buffer[0, layer]  # (num_blocks_total, block_size, H, D)
    val_cache = memory_buffer[1, layer]

    # Write block by block
    for i in range(num_blocks_needed):
        bid = block_ids[i].long().item()
        start = i * block_size
        end = min(start + block_size, M)
        length = end - start
        key_cache[bid, :length] = K_mem[start:end]
        val_cache[bid, :length] = V_mem[start:end]
        # Zero remaining slots in partial last block
        if length < block_size:
            key_cache[bid, length:] = 0
            val_cache[bid, length:] = 0

    last_block_offset = M % block_size
    if last_block_offset == 0 and M > 0:
        last_block_offset = block_size
    return last_block_offset


def gather_kv_with_biases(
    memory_buffer: Tensor,
    bias_buffer: Tensor,
    layer: int,
    block_ids: Tensor,
    block_size: int,
    token_count: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Gather K, V, and per-head biases from paged cache.

    Like gather_kv but also gathers the per-head bias terms used in AM compaction.

    Args:
        memory_buffer: 6D KV cache tensor.
        bias_buffer: Bias storage tensor
            shape (num_layers, num_blocks, block_size, num_heads).
        layer: Layer index.
        block_ids: Block IDs to gather from.
        block_size: Tokens per block.
        token_count: Number of valid tokens (default: all slots).

    Returns:
        K_full: [T, H, D]
        V_full: [T, H, D]
        biases: [T, H]
    """
    K_full, V_full = gather_kv(memory_buffer, layer, block_ids, block_size, token_count)

    total_slots = block_ids.shape[0] * block_size
    if token_count is None:
        token_count = total_slots

    bias_blocks = bias_buffer[layer][block_ids.long()]  # (num_blocks, block_size, H)
    num_heads = bias_blocks.shape[-1]
    biases = bias_blocks.reshape(total_slots, num_heads)[:token_count].contiguous()

    return K_full, V_full, biases


def write_kv_with_biases(
    memory_buffer: Tensor,
    bias_buffer: Tensor,
    layer: int,
    block_ids: Tensor,
    block_size: int,
    K_mem: Tensor,
    V_mem: Tensor,
    biases: Tensor,
) -> int:
    """Write compacted K/V and per-head biases back into paged cache blocks.

    Args:
        memory_buffer: 6D KV cache tensor.
        bias_buffer: Bias storage tensor shape (num_layers, num_blocks, block_size, num_heads).
        layer: Layer index.
        block_ids: Block IDs to write into.
        block_size: Tokens per block.
        K_mem: [M, H, D] compacted keys.
        V_mem: [M, H, D] compacted values.
        biases: [M, H] per-head biases (log weights).

    Returns:
        Last block offset.
    """
    last_block_offset = write_kv(memory_buffer, layer, block_ids, block_size, K_mem, V_mem)

    M = K_mem.shape[0]
    num_blocks_needed = math.ceil(M / block_size)

    for i in range(num_blocks_needed):
        bid = block_ids[i].long().item()
        start = i * block_size
        end = min(start + block_size, M)
        length = end - start
        bias_buffer[layer, bid, :length] = biases[start:end]
        if length < block_size:
            bias_buffer[layer, bid, length:] = 0

    return last_block_offset
