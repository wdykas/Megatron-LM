# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused AR + residual_add kernel for partitioned-state Variant-B path.

After the MoE combine, the standard sequence runs three separate
launches:

  1. ``multimem_all_reduce(active, active, rsv["handle"])`` — Triton
     ld_reduce + local store.
  2. (return from ``token_combine``)
  3. ``bias_dropout_add(mlp_output, residual)`` in the transformer
     layer's post-mlp path.

Each kernel pays the ~16μs Triton-CUDA-graph launch overhead. By
folding the residual add into the AR's local-store step, we collapse
two launches into one: ``ld_reduce → unpack → +residual → repack →
store``. The AR's start barrier still serves to wait for peers'
expert outputs to be visible before reading them via multicast.

Saves ~16μs/(M,E)-pair × 23 layers ≈ 368μs/step (~4.5% throughput at
nano b=4).
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

from .barrier import symm_mem_sync
from .fused_collectives import apply_norm, sum_sq
from .multimem_asm import add_v8_bf16_from_u32, asm_rsqrt, ld_128, st_128
from .utils import sync_threads


@triton.jit
def _fused_multimem_ar_residual_add_kernel(
    output_ptr,            # [G, hidden] bf16 destination (caller-supplied; same as rsv["tensor"][:G])
    multicast_ptr,         # multicast pointer to rsv["tensor"]
    residual_ptr,          # [G, hidden] bf16 residual to add into the AR result
    signal_pad_ptrs,
    numel,                 # G * hidden, in elements
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,    # = 8 for bf16/128-bit
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-reduce by reading the multicast sum, add bf16 residual, and
    write the result locally. One CTA strides across 128-bit chunks.

    Mirrors ``_multimem_all_reduce_kernel`` (collectives.py) plus the
    bf16 pack/unpack pattern from ``apply_norm`` / ``add_v8_bf16_from_u32``
    in fused_collectives.py — but skips the separate residual-add kernel
    that would normally run after the AR.

    The barrier-on-start semantics match the standalone AR (relaxed CAS
    pair); peers' writes to symm-mem from the previous unpermute are
    already visible because that kernel's atomic_add to bf16 has implicit
    release semantics under Triton's lowering.
    """
    symm_mem_sync(
        signal_pad_ptrs,
        None,
        RANK,
        WORLD_SIZE,
        hasPreviousMemAccess=False,
        hasSubsequentMemAccess=False,
    )
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)

    numel_128 = numel // NUMEL_PER_THREAD  # number of 128-bit chunks
    block_start = pid * BLOCK_SIZE

    while block_start < numel_128:
        offsets = block_start + tid
        mask = offsets < numel_128

        # Read AR sum via multimem.ld_reduce (hardware sums peer copies)
        mc_ptrs = multicast_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        (x, y, z, w) = ld_128(mc_ptrs, mask=mask, multicast_op=True)

        # Read residual locally and add (8 bf16 in 4 uint32)
        res_ptrs = residual_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        (rx, ry, rz, rw) = ld_128(res_ptrs, mask=mask, multicast_op=False)
        (x, y, z, w) = add_v8_bf16_from_u32(x, y, z, w, rx, ry, rz, rw)

        # Write the residual-added AR result locally.
        out_ptrs = output_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
        st_128(out_ptrs, x, y, z, w, mask=mask, multicast_op=False)

        block_start += tl.num_programs(axis=0) * BLOCK_SIZE


@triton.jit
def _fused_multimem_ar_residual_norm_kernel(
    output_ptr,            # [G, hidden] bf16 destination, normalized result
    multicast_ptr,         # multicast pointer to AR input symm-mem
    residual_ptr,          # [G, hidden] bf16 residual to add
    norm_weights_ptr,      # [hidden] bf16 RMSNorm weights for the next layer
    signal_pad_ptrs,
    num_tokens,            # G — number of token rows
    eps,                   # f32 epsilon for rsqrt
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """All-reduce + residual_add + RMSNorm in one kernel pass.

    One CTA per token row. Phase 1 reads the AR sum (multimem.ld_reduce),
    adds the residual, accumulates sum-of-squares, and stores the
    pre-norm value to ``output_ptr``. Phase 2 reads it back, applies
    RMSNorm using next-layer weights, and overwrites ``output_ptr``.

    Mirrors ``_multimem_reduce_scatter_residual_add_kernel`` from
    fused_collectives.py but uses AR (no scatter, full result on every
    rank) and writes locally only (no AG).
    """
    symm_mem_sync(
        signal_pad_ptrs, None, RANK, WORLD_SIZE,
        hasPreviousMemAccess=False, hasSubsequentMemAccess=False,
    )
    sync_threads()

    pid = tl.program_id(axis=0)
    tid = tl.arange(0, BLOCK_SIZE)
    numel_per_token = tl.cdiv(HIDDEN_SIZE, NUMEL_PER_THREAD)
    thread_mask = tid < numel_per_token

    for token_offset in range(pid, num_tokens, tl.num_programs(axis=0)):
        program_offset = token_offset * numel_per_token

        # Phase 1: AR + residual_add + accumulate sum-sq, store pre-norm
        sq_sum_ = 0.0
        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            offsets = program_offset + thread_offset + tid
            mask = ((offsets - program_offset) < numel_per_token) & thread_mask

            mc_ptrs = multicast_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
            res_ptrs = residual_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
            out_ptrs = output_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2

            (x, y, z, w) = ld_128(mc_ptrs, mask=mask, multicast_op=True)
            (rx, ry, rz, rw) = ld_128(res_ptrs, mask=mask, multicast_op=False)
            (x, y, z, w) = add_v8_bf16_from_u32(x, y, z, w, rx, ry, rz, rw)
            st_128(out_ptrs, x, y, z, w, mask=mask, multicast_op=False)
            sq_sum_ += sum_sq(x, y, z, w, mask=mask)

        mean_sq = sq_sum_ / HIDDEN_SIZE
        rrms = asm_rsqrt(mean_sq, eps)

        # Phase 2: read back, apply norm, overwrite output
        for thread_offset in range(0, numel_per_token, BLOCK_SIZE):
            offsets = program_offset + thread_offset + tid
            mask = ((offsets - program_offset) < numel_per_token) & thread_mask

            out_ptrs = output_ptr.to(tl.pointer_type(tl.uint64)) + offsets * 2
            wt_ptrs = norm_weights_ptr.to(tl.pointer_type(tl.uint64)) + (thread_offset + tid) * 2

            (rx, ry, rz, rw) = ld_128(out_ptrs, mask=mask, multicast_op=False)
            (wx, wy, wz, ww) = ld_128(wt_ptrs, mask=mask, multicast_op=False)
            (nx, ny, nz, nw) = apply_norm(rx, ry, rz, rw, wx, wy, wz, ww, rrms, mask)
            st_128(out_ptrs, nx, ny, nz, nw, mask=mask, multicast_op=False)


def fused_multimem_ar_residual_norm(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    norm_weights: torch.Tensor,
    symm_mem_hdl,
    eps: float = 1e-5,
    **kwargs,
) -> torch.Tensor:
    """Fused AR + residual_add + RMSNorm. Result is normalized [G, hidden]
    on every rank — equivalent to (AR(input) + residual) → RMSNorm.

    Saves two extra kernel launches (bias_dropout_add + next layer's
    input_layernorm) on the partitioned + Variant-B path.
    """
    assert HAVE_TRITON
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.shape == output_tensor.shape == residual_tensor.shape
    assert norm_weights.shape[0] == input_tensor.shape[-1]
    assert input_tensor.is_contiguous() and residual_tensor.is_contiguous()
    assert norm_weights.is_contiguous()

    G, hidden = input_tensor.shape
    NUMEL_PER_THREAD = 8  # bf16, 128-bit chunks
    numel_per_token = hidden // NUMEL_PER_THREAD
    BLOCK_SIZE = min(triton.next_power_of_2(numel_per_token), 1024)
    num_warps = max(1, BLOCK_SIZE // 32)
    # At larger G the per-CTA barrier overhead × NUM_BLOCKS dominates;
    # cap at 8 CTAs and stride. Tuned across b=8/16/32 to balance
    # per-CTA work against barrier/launch overhead.
    NUM_BLOCKS = min(G, kwargs.get("max_num_blocks", 8))

    _fused_multimem_ar_residual_norm_kernel[(NUM_BLOCKS,)](
        output_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        residual_tensor.data_ptr(),
        norm_weights.data_ptr(),
        symm_mem_hdl.signal_pad_ptrs_dev,
        num_tokens=G,
        eps=eps,
        HIDDEN_SIZE=hidden,
        BLOCK_SIZE=BLOCK_SIZE,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    return output_tensor


def fused_multimem_ar_residual_add(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    residual_tensor: torch.Tensor,
    symm_mem_hdl,
    **kwargs,
) -> torch.Tensor:
    """All-reduce ``input_tensor`` across EP ranks AND add ``residual_tensor``
    in a single Triton kernel. Writes the result to ``output_tensor``.

    Args:
        output_tensor: bf16 [G, hidden] (or flat) destination.
        input_tensor: same shape as output, must be in symm-mem (AR source).
        residual_tensor: same shape as output, the residual to add.
        symm_mem_hdl: ``_SymmetricMemory`` handle for ``input_tensor``.
        **kwargs: optional ``BLOCK_SIZE`` / ``max_num_blocks`` overrides.
    """
    assert HAVE_TRITON
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.numel() == output_tensor.numel() == residual_tensor.numel()
    assert input_tensor.is_contiguous()
    assert residual_tensor.is_contiguous()

    MAX_NUM_BLOCKS = kwargs.get("max_num_blocks", 128)
    DEFAULT_BLOCK_SIZE = kwargs.get("BLOCK_SIZE", 1024)
    WARP_SIZE = 32

    NUMEL_PER_THREAD = 128 // (input_tensor.element_size() * 8)  # = 8 for bf16
    numel = input_tensor.numel()
    num_threads = triton.cdiv(numel // NUMEL_PER_THREAD, 1)
    num_blocks = min(triton.cdiv(num_threads, DEFAULT_BLOCK_SIZE), MAX_NUM_BLOCKS)
    num_warps = max(1, DEFAULT_BLOCK_SIZE // WARP_SIZE)

    _fused_multimem_ar_residual_add_kernel[(num_blocks, 1, 1)](
        output_tensor.data_ptr(),
        symm_mem_hdl.multicast_ptr,
        residual_tensor.data_ptr(),
        symm_mem_hdl.signal_pad_ptrs_dev,
        numel=numel,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
        NUMEL_PER_THREAD=NUMEL_PER_THREAD,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )
    return output_tensor
