# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multicast variant of the unpermute step that folds the AR combine in.

The standard unpermute kernel (``permute.py:_unpermute_tokens_kernel``)
``tl.atomic_add``-s expert outputs into a per-rank accumulator buffer,
and a separate ``multimem_all_reduce`` then sums those buffers across
ranks. We can collapse those two steps by replacing ``tl.atomic_add``
with ``multimem.red.add.acc::f32.v4.bf16x2``: the same atomic that
accumulates contributions for the same token across multiple top-k
experts on this rank simultaneously accumulates across all ranks via
the multicast group. After all ranks have unpermuted, every rank's
local view of the buffer holds the global AR sum without a separate
collective.

This trades one explicit AR kernel (~46μs/layer at nano b=4) for one
extra cross-rank barrier inside the unpermute (~5μs), saving ~40μs per
MoE layer × 23 layers ≈ 1ms/step.

Caveats:
  - The destination buffer must be zeroed on every rank before the
    atomic-adds start. Pair with ``_zero_output_rows_kernel`` and a
    cross-rank barrier (this kernel's start-barrier serves that role).
  - ``multimem.red.add`` uses ``relaxed`` memory ordering, so the
    end-of-kernel ``symm_mem_sync`` (release+acquire) is load-bearing
    — it is what makes the accumulated value visible to consumers.
  - Each row of expert output's hidden_dim must be a multiple of 8
    (we step in 128-bit chunks of 8 bf16 each).
"""

from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

from megatron.core.inference.communication.torch_symm_triton.barrier import symm_mem_sync
from megatron.core.inference.communication.torch_symm_triton.fused_collectives import unpack_bf16x2
from megatron.core.inference.communication.torch_symm_triton.multimem_asm import (
    ld_128,
    red_v4_bf16x2,
)
from megatron.core.inference.communication.torch_symm_triton.utils import sync_threads


@triton.jit
def _unpermute_tokens_multicast_kernel(
    expert_out_ptr,        # [output_size, hidden_dim] expert outputs in permuted order, bf16
    probs_ptr,             # [output_size] fp32 routing probabilities (permuted)
    src_idx_ptr,           # [output_size] int32 permutation_map: original token index, or -1
    multicast_ptr,         # multicast pointer to [max_tokens, hidden_dim] bf16 RSV buffer
    signal_pad_ptrs,
    n_used_ptr,            # int32 scalar: inclusive_expert_offsets[-1]
    HIDDEN_DIM: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,    # = 8 for bf16/128-bit chunks
    NUM_BLOCKS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Unpermute expert outputs and atomically accumulate them into all
    peers' RSV via NVLS multicast — the AR combine happens for free.

    Grid: ``NUM_BLOCKS`` CTAs strided over rows of the permuted output.
    Each row contributes ``HIDDEN_DIM // NUMEL_PER_THREAD`` 128-bit
    chunks via ``multimem.red.add.acc::f32.v4.bf16x2``.

    The start barrier ensures every rank has finished zeroing the RSV
    buffer before any rank starts atomic-adding into it (otherwise an
    unfortunately late ``zero`` on one rank would clobber peers'
    contributions). The end barrier ensures atomic-adds are committed
    to peers before subsequent kernels read the result.
    """
    # Wait for all ranks to be ready (they should have just zeroed RSV).
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=True, hasSubsequentMemAccess=True,
    )
    sync_threads()

    pid = tl.program_id(0)
    n_used = tl.load(n_used_ptr)
    if pid >= n_used:
        # CTAs that exit early still need to participate in the end
        # barrier, but since the barrier slot is keyed on block_id and
        # all peers agree on n_used (NUM_BLOCKS exit identically across
        # ranks), correctness is preserved by simply returning here —
        # those slots are never touched. The remaining CTAs hit the
        # end barrier on their own slots only.
        return

    chunks_per_token = HIDDEN_DIM // NUMEL_PER_THREAD

    for row in tl.range(pid, MAX_ROWS, NUM_BLOCKS):
        if row < n_used:
            source_idx = tl.load(src_idx_ptr + row)
            if source_idx >= 0:
                prob = tl.load(probs_ptr + row)  # fp32
                # Walk hidden_dim in 128-bit chunks (each chunk = 8 bf16).
                for chunk_idx_block in range(0, chunks_per_token, 256):
                    chunk_offsets = chunk_idx_block + tl.arange(0, 256)
                    chunk_mask = chunk_offsets < chunks_per_token

                    # Load 8 bf16 per chunk from local expert output.
                    src_ptrs = (
                        expert_out_ptr.to(tl.pointer_type(tl.uint64))
                        + (row * chunks_per_token + chunk_offsets) * 2
                    )
                    (x, y, z, w) = ld_128(src_ptrs, mask=chunk_mask, multicast_op=False)

                    # Apply prob: unpack each uint32 → 2 fp32 → multiply → repack to bf16x2.
                    x_hi, x_lo = unpack_bf16x2(x, chunk_mask)
                    y_hi, y_lo = unpack_bf16x2(y, chunk_mask)
                    z_hi, z_lo = unpack_bf16x2(z, chunk_mask)
                    w_hi, w_lo = unpack_bf16x2(w, chunk_mask)

                    x_hi = x_hi * prob
                    x_lo = x_lo * prob
                    y_hi = y_hi * prob
                    y_lo = y_lo * prob
                    z_hi = z_hi * prob
                    z_lo = z_lo * prob
                    w_hi = w_hi * prob
                    w_lo = w_lo * prob

                    # Pack back: hi → upper 16 bits of uint32, lo → lower 16 bits.
                    x_hi_p = (
                        x_hi.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
                    )
                    x_lo_p = x_lo.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
                    y_hi_p = (
                        y_hi.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
                    )
                    y_lo_p = y_lo.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
                    z_hi_p = (
                        z_hi.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
                    )
                    z_lo_p = z_lo.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)
                    w_hi_p = (
                        w_hi.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32) << 16
                    )
                    w_lo_p = w_lo.cast(tl.bfloat16).cast(tl.uint16, bitcast=True).cast(tl.uint32)

                    x_packed = x_hi_p | x_lo_p
                    y_packed = y_hi_p | y_lo_p
                    z_packed = z_hi_p | z_lo_p
                    w_packed = w_hi_p | w_lo_p

                    # Atomic-accumulate 8 bf16 per chunk to multicast at source_idx.
                    mc_ptrs = (
                        multicast_ptr.to(tl.pointer_type(tl.uint64))
                        + (source_idx * chunks_per_token + chunk_offsets) * 2
                    )
                    red_v4_bf16x2(mc_ptrs, x_packed, y_packed, z_packed, w_packed, mask=chunk_mask)

    sync_threads()
    # Release fence so peers see this rank's atomic-adds before the next kernel.
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=True, hasSubsequentMemAccess=True,
    )


@triton.jit
def _zero_rsv_kernel(
    out_ptr,
    valid_tokens_ptr,
    HIDDEN_DIM: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """Zero rows ``[0, valid_tokens)`` of a bf16 buffer for multicast accumulation."""
    pid = tl.program_id(0)
    valid = tl.load(valid_tokens_ptr)
    if pid >= valid:
        return
    zero = tl.zeros([BLOCK_H], dtype=tl.float32)
    for row in tl.range(pid, MAX_ROWS, NUM_BLOCKS):
        if row < valid:
            for h in tl.range(0, HIDDEN_DIM, BLOCK_H):
                o = h + tl.arange(0, BLOCK_H)
                m = o < HIDDEN_DIM
                tl.store(out_ptr + row * HIDDEN_DIM + o, zero, mask=m)


def _zero_rsv_for_multicast(
    out: torch.Tensor, valid_tokens: torch.Tensor, max_tokens: int
) -> None:
    """Zero ``out[:valid_tokens]`` so multicast atomic-adds accumulate from zero."""
    assert HAVE_TRITON
    hidden_dim = out.shape[1]
    BLOCK_H = min(triton.next_power_of_2(hidden_dim), 1024)
    NUM_BLOCKS = min(max_tokens, 512)
    _zero_rsv_kernel[(NUM_BLOCKS,)](
        out,
        valid_tokens,
        HIDDEN_DIM=hidden_dim,
        MAX_ROWS=max_tokens,
        BLOCK_H=BLOCK_H,
        NUM_BLOCKS=NUM_BLOCKS,
    )


def unpermute_tokens_multicast(
    expert_output: torch.Tensor,
    permuted_probs: torch.Tensor,
    permutation_map: torch.Tensor,
    n_used: torch.Tensor,
    out_symm_handle,
) -> None:
    """Unpermute expert outputs straight into all peers' RSV view via multicast.

    Skips the post-MoE ``multimem_all_reduce`` because the atomic-add IS
    the AR. Caller must have zeroed the RSV buffer on every rank before
    invoking this kernel; the kernel's start barrier waits for that.

    Args:
        expert_output: ``[output_size, hidden_dim]`` bf16 expert outputs in
            permuted order.
        permuted_probs: ``[output_size]`` fp32 routing probabilities.
        permutation_map: ``[output_size]`` int32 — original token index
            for each row, or ``-1`` for alignment-padding rows.
        n_used: scalar int32 CUDA tensor = ``inclusive_expert_offsets[-1]``.
        out_symm_handle: ``_SymmetricMemory`` handle for the RSV buffer
            being accumulated into. The kernel writes via
            ``handle.multicast_ptr``.
    """
    assert HAVE_TRITON, "Triton is required."
    assert expert_output.dtype == torch.bfloat16
    output_size, hidden_dim = expert_output.shape
    assert hidden_dim % 8 == 0, "hidden_dim must be a multiple of 8 for bf16x2 v4 atomics"

    NUMEL_PER_THREAD = 8  # 8 bf16 per 128-bit chunk
    NUM_BLOCKS = min(output_size, 512)

    _unpermute_tokens_multicast_kernel[(NUM_BLOCKS,)](
        expert_output,
        permuted_probs,
        permutation_map,
        out_symm_handle.multicast_ptr,
        out_symm_handle.signal_pad_ptrs_dev,
        n_used,
        HIDDEN_DIM=hidden_dim,
        MAX_ROWS=output_size,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        NUM_BLOCKS=NUM_BLOCKS,
        RANK=out_symm_handle.rank,
        WORLD_SIZE=out_symm_handle.world_size,
        num_warps=8,
    )
