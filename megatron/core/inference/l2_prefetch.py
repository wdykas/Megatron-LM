# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""L2 cache prefetch utility for inference.

Prefetches a list of tensors into L2 cache by issuing strided reads on a
side stream. Used at wide EP where per-rank expert/state working sets fit
in L2 (~50-80 MB on Hopper/Blackwell). At narrow EP the working set
exceeds L2 and prefetching just thrashes — gate the call accordingly.

Typical use:

    if rank_working_set_fits_in_l2():
        prefetch_into_l2([w0, w1, ...], stream=prefetch_stream)
        # ... main stream does some other compute concurrently ...
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        # ... main kernel reads weights, hits warm L2 ...
"""

from __future__ import annotations

from typing import Iterable, Optional
from unittest.mock import MagicMock

import torch

from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


# 128-byte cache lines on Hopper/Blackwell. One u32 read per line is enough
# to fault the line into L2.
_CACHE_LINE_BYTES = 128
_LINES_PER_CTA = 32  # 32 lines * 128B = 4 KB per CTA


@triton.jit
def _l2_prefetch_kernel(
    ptr,
    num_lines,
    LINES_PER_CTA: tl.constexpr,
):
    """Prefetch ``num_lines`` cache lines (128 B each) into L2.

    Each CTA touches ``LINES_PER_CTA`` consecutive lines via one u32
    load per line. The load uses ``evict_last`` to bias L2 toward
    keeping this data resident across subsequent kernels. The returned
    value is unused — the side effect is the L2 fill.
    """
    pid = tl.program_id(axis=0)
    base = pid * LINES_PER_CTA
    offsets = base + tl.arange(0, LINES_PER_CTA)
    mask = offsets < num_lines
    # Each line is 128 B; load 1 u32 per line at offset 0 of the line.
    # Pointer is u32-typed, stride = 32 (128 B / 4 B per u32).
    u32_offsets = offsets * 32
    _ = tl.load(ptr + u32_offsets, mask=mask, eviction_policy='evict_last')


def prefetch_into_l2(
    tensors: Iterable[torch.Tensor],
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Prefetch the given tensors' bytes into L2 on the given stream.

    Returns immediately after enqueueing the kernel(s). Caller must
    synchronize the consumer stream with ``stream`` before reading
    the data on the consumer (otherwise the prefetch may not have
    completed).

    Args:
        tensors: list of CUDA tensors to fault into L2.
        stream: stream to issue the prefetch on (defaults to current).
    """
    if not HAVE_TRITON:
        return
    target_stream = stream if stream is not None else torch.cuda.current_stream()
    with torch.cuda.stream(target_stream):
        for t in tensors:
            if t is None or not t.is_cuda:
                continue
            # Treat the tensor's underlying storage as a flat u32 buffer.
            # Use the storage's data_ptr (handles non-contiguous tensors
            # with offset; we only need to fault their pages).
            n_bytes = t.numel() * t.element_size()
            num_lines = (n_bytes + _CACHE_LINE_BYTES - 1) // _CACHE_LINE_BYTES
            if num_lines == 0:
                continue
            ptr_u32 = t.contiguous().view(torch.uint8).view(-1)[: num_lines * _CACHE_LINE_BYTES]
            ptr_u32 = ptr_u32.view(torch.int32)
            grid = ((num_lines + _LINES_PER_CTA - 1) // _LINES_PER_CTA,)
            _l2_prefetch_kernel[grid](
                ptr_u32,
                num_lines,
                LINES_PER_CTA=_LINES_PER_CTA,
                num_warps=1,
            )


def collect_expert_weight_tensors(experts_module) -> list:
    """Gather per-expert linear weight tensors from a TEGroupedMLP-style
    experts module. Returns ``linear_fc1.weight0``..``weightN``,
    ``linear_fc2.weight0``..``weightN``. Skips missing/None entries.
    """
    out = []
    for sub_name in ('linear_fc1', 'linear_fc2'):
        sub = getattr(experts_module, sub_name, None)
        if sub is None:
            continue
        i = 0
        while True:
            w = getattr(sub, f'weight{i}', None)
            if w is None:
                break
            out.append(w)
            i += 1
    return out


def collect_mamba_weight_tensors(mixer_module) -> list:
    """Gather Mamba mixer weight tensors. Includes in_proj/out_proj
    parameters and the small SSM scalars (A_log, D, dt_bias, conv1d
    weight). Skips missing entries.
    """
    out = []
    for sub_name in ('in_proj', 'out_proj', 'conv1d'):
        sub = getattr(mixer_module, sub_name, None)
        if sub is None:
            continue
        w = getattr(sub, 'weight', None)
        if w is not None:
            out.append(w)
        b = getattr(sub, 'bias', None)
        if b is not None:
            out.append(b)
    for attr in ('A_log', 'D', 'dt_bias'):
        w = getattr(mixer_module, attr, None)
        if w is not None and isinstance(w, torch.Tensor):
            out.append(w)
    return out
