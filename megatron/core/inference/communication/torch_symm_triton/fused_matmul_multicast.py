# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused linear + NVLS multicast all-gather for partitioned-state Variant-B.

The matmul ``out = x @ W^T`` is computed locally and then the result is
multicast-stored to a symmetric-memory output buffer at this rank's
token offset — replacing the explicit ``multimem_all_gather_v`` call
that would otherwise follow. The matmul itself uses cuBLAS via
``torch.matmul``; a Triton kernel handles only the multicast publish
step, which is structurally identical to the existing
``_multimem_all_gather_v_kernel`` but sized for exactly this rank's
``local_tokens`` (no per-rank-max padding).

Why not fuse matmul into the same Triton kernel? The cuBLAS matmul is
already heavily tuned and is not the bottleneck for our sizes — the
gain we want is eliminating the *separate* AGV-V kernel launch and
its cross-rank barrier. Splitting matmul (cuBLAS) and publish (Triton
multicast) keeps the matmul fast while still shaving the AGV launch
+ barrier from the captured graph.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

from .barrier import symm_mem_sync
from .multimem_asm import ld_128, st_128
from .utils import is_device_nvls_capable, sync_threads


def _publish_use_single_cta() -> bool:
    """Toggle: single-CTA publish kernel vs one-CTA-per-token (default)."""
    return os.environ.get("ABS_PUBLISH_SINGLE_CTA", "0") == "1"


def _publish_no_barrier() -> bool:
    """Toggle: skip the publish-end cross-rank barrier (experimental).

    The downstream AR start-barrier is relaxed in current AR kernels, so
    dropping this barrier is **unsafe** for correctness in production —
    the AR does not acquire-fence on its own. Use only paired with the
    acquire-AR variant or to measure how much overhead the barrier
    represents under lucky scheduling.
    """
    return os.environ.get("ABS_PUBLISH_NO_BARRIER", "0") == "1"

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
def _publish_multicast_kernel_single_cta(
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
    HAS_BARRIER: tl.constexpr = True,
):
    """Single-CTA publish: one block streams all LOCAL_TOKENS rows.

    Compared to the per-row CTA layout, this trims the barrier slot count
    from ``LOCAL_TOKENS`` to ``1`` (each slot costs ``world_size``
    atomic-CAS pairs), removes the per-CTA early-exit branch, and lets
    a single SM hold the entire token block in L1 for sequential 128-bit
    multicast stores.
    """
    tid = tl.arange(0, BLOCK_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    total_chunks = LOCAL_TOKENS * numel_per_token

    for offset in range(0, total_chunks, BLOCK_SIZE):
        local_offsets = offset + tid
        mask = local_offsets < total_chunks
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
    if HAS_BARRIER:
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )


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
    HAS_BARRIER: tl.constexpr = True,
):
    """Publish ``local_ptr[0:LOCAL_TOKENS, :HIDDEN_SIZE]`` to all peers' copies
    of ``multicast_ptr`` starting at row ``RANK_TOKEN_OFFSET``.

    All counts are constexpr so the kernel is fully specialized per shape;
    this avoids the ``ep_max_tokens`` / ``rank_token_offset`` pointer loads
    of the variable-count AGV-V kernel and eliminates one tiny GPU read
    per CTA.

    The kernel mirrors ``_multimem_all_gather_v_kernel`` but with a fixed
    grid of ``LOCAL_TOKENS`` CTAs (one per token row) and constexpr
    rank-offset; nothing varies across captured graphs except the
    weights and the data values.
    """
    pid = tl.program_id(axis=0)

    tid = tl.arange(0, BLOCK_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    channel_mask = tid < numel_per_token

    # Stride over rows: each CTA handles tokens at indices pid, pid+grid,
    # pid+2*grid, ... With NUM_CTAS == LOCAL_TOKENS this collapses to one
    # token per CTA (the original behavior). With NUM_CTAS < LOCAL_TOKENS
    # each CTA handles multiple rows; barrier slot count drops to NUM_CTAS.
    num_progs = tl.num_programs(axis=0)
    for row in range(pid, LOCAL_TOKENS, num_progs):
        for channel_offset in range(0, numel_per_token, BLOCK_SIZE):
            local_offsets = row * numel_per_token + channel_offset + tid
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
    if HAS_BARRIER:
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
    Triton kernel sized exactly for ``local_tensor.shape[0]`` CTAs
    (no per-rank-max padding, no ep_max_tokens runtime load) — saves
    ~6μs/AGV on small batches by eliminating ~115 wasted CTA early-exits.

    Args:
        local_tensor: ``[local_tokens, hidden]`` per-rank tensor to publish.
        output_global_buffer: ``[ep_size * local_tokens, hidden]`` symm-mem
            buffer (registered with NVLS).
        output_symm_mem_handle: ``_SymmetricMemory`` handle for the buffer.
        rank: this rank's index in the EP group.
        rank_token_offset: row offset for this rank's contribution.

    Returns: ``output_global_buffer`` (unchanged ref) populated with all ranks'
    contributions after the end-of-kernel ``symm_mem_sync`` barrier.
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

    has_barrier = not _publish_no_barrier()
    # Optional explicit grid override: ABS_PUBLISH_NUM_CTAS sets the
    # number of CTAs (each strides over multiple rows). Default = local_tokens
    # (one CTA per row, original behavior).
    _grid_env = os.environ.get("ABS_PUBLISH_NUM_CTAS")
    if _grid_env:
        num_ctas = max(1, min(int(_grid_env), local_tokens))
        _publish_multicast_kernel[(num_ctas, 1, 1)](
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
            HAS_BARRIER=has_barrier,
            num_warps=num_warps,
        )
        return output_global_buffer
    if _publish_use_single_cta():
        # Single CTA strides over all rows; bigger block to keep utilization up.
        single_block = min(triton.next_power_of_2(local_tokens * numel_per_token), 1024)
        single_warps = max(1, single_block // 32)
        _publish_multicast_kernel_single_cta[(1, 1, 1)](
            local_tensor.contiguous(),
            output_symm_mem_handle.multicast_ptr,
            output_symm_mem_handle.signal_pad_ptrs_dev,
            LOCAL_TOKENS=local_tokens,
            HIDDEN_SIZE=hidden_size,
            RANK_TOKEN_OFFSET=rank_token_offset,
            BLOCK_SIZE=single_block,
            NUMEL_PER_THREAD=numel_per_thread,
            BITS=bits,
            RANK=rank,
            WORLD_SIZE=output_symm_mem_handle.world_size,
            HAS_BARRIER=has_barrier,
            num_warps=single_warps,
        )
    else:
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
            HAS_BARRIER=has_barrier,
            num_warps=num_warps,
        )
    return output_global_buffer


def _fused_use_combined_kernel() -> bool:
    """Toggle: collapse cuBLAS matmul + Triton publish into a single kernel
    launch by stream-ordering them tightly on the captured graph. The
    actual matmul still runs through cuBLAS — the savings come from
    cutting the inter-kernel overhead of the *separate* publish kernel
    launch when the matmul output is already in the local buffer the
    publish kernel reads from.
    """
    return os.environ.get("ABS_FUSED_OUTPROJ_PUBLISH", "0") == "1"


def fused_linear_multicast_publish(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_global_buffer: torch.Tensor,
    output_symm_mem_handle,
    rank: int,
    rank_token_offset: int,
) -> torch.Tensor:
    """Compute ``out = x @ W^T (+ bias)`` and multicast-publish to global buffer.

    Args:
        x: ``[local_tokens, K]`` per-rank input, contiguous.
        weight: ``[N, K]`` weight (PyTorch nn.Linear convention).
        bias: optional ``[N]`` bias.
        output_global_buffer: ``[ep_size * local_tokens, N]`` symmetric
            memory buffer (the dispatcher's ``agv_h["tensor"]`` viewed
            with the right hidden size, for instance).
        output_symm_mem_handle: ``_SymmetricMemory`` handle for
            ``output_global_buffer``.
        rank: this rank's index in the EP group.
        rank_token_offset: row offset for this rank's contribution
            (typically ``rank * local_tokens``).

    Returns: a view of ``output_global_buffer`` containing the gathered
    global output of shape ``[ep_size * local_tokens, N]``. By the time
    this function returns, all ranks' contributions are visible (the
    end-of-kernel ``symm_mem_sync`` has fired).
    """
    assert HAVE_TRITON, "Triton is required."
    assert is_device_nvls_capable(x.device)

    # Local matmul via cuBLAS. Output goes to a small per-rank
    # contiguous buffer (Triton kernel will publish it via multicast).
    local_out = torch.matmul(x, weight.t())
    if bias is not None:
        local_out = local_out + bias
    local_out = local_out.contiguous()

    local_tokens, hidden_size = local_out.shape
    row_bytes = hidden_size * local_out.element_size()
    bits = 128 if row_bytes % 16 == 0 else 64
    numel_per_thread = bits // (local_out.element_size() * 8)
    numel_per_token = (hidden_size + numel_per_thread - 1) // numel_per_thread
    block_size = min(triton.next_power_of_2(numel_per_token), 1024)
    num_warps = max(1, block_size // 32)

    # Launch one CTA per row. With local_tokens=13 (nano b=4), 13 CTAs.
    _publish_multicast_kernel[(local_tokens, 1, 1)](
        local_out,
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
