# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Constexpr-specialized NVLS multicast publish kernel for the
partitioned-state inference path.

In partitioned mode each rank computes mamba locally and then publishes
its ``[local_tokens, hidden]`` output to all peers' copies of the
global symmetric-memory buffer at row offset ``rank * local_tokens``.
The standard ``multimem_all_gather_v`` kernel handles this via runtime
``ep_max_tokens`` / ``rank_token_offset`` pointer loads to support
variable per-rank counts; under EP-token-count sync those values are
identical across ranks at every step, so we can specialize them as
constexpr — saving a few microseconds of pointer-load + dispatcher
metadata work and producing a kernel that is sized for exactly
``local_tokens`` CTAs (no per-rank-max wasted CTAs).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

from .barrier import symm_mem_sync
from .multimem_asm import ld_128, st_128
from .utils import is_device_nvls_capable, sync_threads

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    tl = MagicMock()
    triton.jit = null_decorator
    HAVE_TRITON = False


@triton.jit
def _publish_multicast_kernel(
    local_ptr,
    multicast_ptr,
    signal_pad_ptrs,
    LOCAL_TOKENS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    RANK_TOKEN_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    BITS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Publish ``local_ptr[0:LOCAL_TOKENS, :HIDDEN_SIZE]`` to all peers'
    copies of ``multicast_ptr`` starting at row ``RANK_TOKEN_OFFSET``.

    All counts are constexpr so the kernel is fully specialized per
    shape. Mirrors ``_multimem_all_gather_v_kernel`` but with a fixed
    grid of ``LOCAL_TOKENS`` CTAs (one per token row) and constexpr
    rank-offset.
    """
    pid = tl.program_id(axis=0)
    if pid >= LOCAL_TOKENS:
        return

    tid = tl.arange(0, BLOCK_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    channel_mask = tid < numel_per_token

    for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
        local_offsets = pid * numel_per_token + channel_offset + tid
        token_mask = local_offsets < LOCAL_TOKENS * numel_per_token
        mask = token_mask & channel_mask

        global_offsets = RANK_TOKEN_OFFSET * numel_per_token + local_offsets

        if BITS == 128:
            multicast_ptrs = multicast_ptr.to(tl.pointer_type(tl.uint64)) + global_offsets * 2
            local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets * 2
            (x, y, z, w) = ld_128(local_ptrs, mask=mask, multicast_op=False)
            st_128(multicast_ptrs, x, y, z, w, mask=mask, multicast_op=True)
        else:
            from .multimem_asm import ld_64, st_64
            multicast_ptrs = multicast_ptr.to(tl.pointer_type(tl.uint64)) + global_offsets
            local_ptrs = local_ptr.to(tl.pointer_type(tl.uint64)) + local_offsets
            (x, y) = ld_64(local_ptrs, mask=mask)
            st_64(multicast_ptrs, x, y, mask=mask, multicast_op=True)

    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multicast_publish_constexpr(
    local_tensor: torch.Tensor,
    output_global_buffer: torch.Tensor,
    output_symm_mem_handle,
    rank: int,
    rank_token_offset: int,
) -> torch.Tensor:
    """Publish ``local_tensor`` (already-computed per-rank output) to all
    peers' copies of ``output_global_buffer`` at row ``rank_token_offset``.

    Equivalent to ``multimem_all_gather_v`` but with a constexpr-specialized
    Triton kernel sized exactly for ``local_tensor.shape[0]`` CTAs.

    Args:
        local_tensor: ``[local_tokens, hidden]`` per-rank tensor to publish.
        output_global_buffer: ``[ep_size * local_tokens, hidden]`` symm-mem
            buffer (registered with NVLS).
        output_symm_mem_handle: ``_SymmetricMemory`` handle for the buffer.
        rank: this rank's index in the EP group.
        rank_token_offset: row offset for this rank's contribution.

    Returns: ``output_global_buffer`` (unchanged ref) populated with all
    ranks' contributions after the end-of-kernel ``symm_mem_sync`` barrier.
    """
    assert HAVE_TRITON
    assert local_tensor.dim() == 2 and output_global_buffer.dim() == 2

    local_tokens, hidden_size = local_tensor.shape
    row_bytes = hidden_size * local_tensor.element_size()
    assert row_bytes % 8 == 0
    bits = 128 if row_bytes % 16 == 0 else 64
    numel_per_thread = bits // (local_tensor.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread
    block_size = min(triton.next_power_of_2(numel_per_token), 1024)
    num_warps = max(1, block_size // 32)

    _publish_multicast_kernel[(local_tokens, 1, 1)](
        local_tensor.contiguous(),
        output_symm_mem_handle.multicast_ptr,
        output_symm_mem_handle.signal_pad_ptrs_dev,
        LOCAL_TOKENS=local_tokens,
        HIDDEN_SIZE=hidden_size,
        RANK_TOKEN_OFFSET=rank_token_offset,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        BITS=bits,
        RANK=rank,
        WORLD_SIZE=output_symm_mem_handle.world_size,
        num_warps=num_warps,
    )
    return output_global_buffer
