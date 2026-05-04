# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVLS multicast publish kernel for the partitioned-state inference path.

In partitioned mode each rank computes mamba locally and then publishes
its ``[local_tokens, hidden]`` output to all peers' copies of the
global symmetric-memory buffer at a per-rank prefix-sum offset.

This kernel supports **asymmetric per-rank counts** without an
EP-token-count sync. Each rank passes its own runtime ``local_tokens``
and ``rank_token_offset_ptr`` (filled with the prefix-sum across
ranks). The grid size is fixed to ``per_rank_max_tokens`` (a static
config max), so every rank's captured graph has the same grid; CTAs
with ``pid >= local_tokens`` skip the data work but still participate
in the cross-rank barrier so the barrier slots align across ranks.
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
    local_tokens,             # runtime: this rank's actual count this step
    rank_token_offset_ptr,    # runtime: this rank's prefix-sum offset
    HIDDEN_SIZE: tl.constexpr,
    PER_RANK_MAX: tl.constexpr,    # static grid size; same across all ranks
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    BITS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Publish per-rank rows to all peers at this rank's prefix-sum offset.

    Fixed grid of ``PER_RANK_MAX`` CTAs (same across all ranks).
    CTAs with ``pid >= local_tokens`` skip the load+store but still
    participate in the cross-rank barrier — that is what makes the
    barrier symmetric when peers have different ``local_tokens``.

    Each CTA handles one token row. ``rank_token_offset_ptr`` is read
    at runtime from a tensor populated by the dispatcher's per-step
    metadata pass (prefix-sum of all preceding ranks' local_tokens).
    """
    pid = tl.program_id(axis=0)
    do_work = pid < local_tokens

    tid = tl.arange(0, BLOCK_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    channel_mask = tid < numel_per_token

    if do_work:
        rank_token_offset = tl.load(rank_token_offset_ptr)
        for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = pid * numel_per_token + channel_offset + tid
            token_mask = local_offsets < local_tokens * numel_per_token
            mask = token_mask & channel_mask

            global_offsets = rank_token_offset * numel_per_token + local_offsets

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

    # All CTAs (including those that skipped data work) participate in
    # the barrier so signal-pad slots align across asymmetric ranks.
    sync_threads()
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


def multicast_publish(
    local_tensor: torch.Tensor,
    output_global_buffer: torch.Tensor,
    output_symm_mem_handle,
    rank: int,
    rank_token_offset_tensor: torch.Tensor,
    per_rank_max: int,
) -> torch.Tensor:
    """Publish per-rank rows to all peers' copies of the global buffer.

    Args:
        local_tensor: ``[local_tokens, hidden]`` per-rank tensor.
        output_global_buffer: ``[per_rank_max * ep_size, hidden]`` symm-mem.
        output_symm_mem_handle: ``_SymmetricMemory`` handle.
        rank: this rank's EP index.
        rank_token_offset_tensor: 0-d int32 CUDA tensor holding this rank's
            prefix-sum offset (sum of preceding ranks' local_tokens).
            Populated by the dispatcher each step.
        per_rank_max: the static config max-per-rank token count. Grid
            size; same on every rank so barriers align.
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

    _publish_multicast_kernel[(per_rank_max, 1, 1)](
        local_tensor.contiguous(),
        output_symm_mem_handle.multicast_ptr,
        output_symm_mem_handle.signal_pad_ptrs_dev,
        local_tokens=local_tokens,
        rank_token_offset_ptr=rank_token_offset_tensor,
        HIDDEN_SIZE=hidden_size,
        PER_RANK_MAX=per_rank_max,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        BITS=bits,
        RANK=rank,
        WORLD_SIZE=output_symm_mem_handle.world_size,
        num_warps=num_warps,
    )
    return output_global_buffer
