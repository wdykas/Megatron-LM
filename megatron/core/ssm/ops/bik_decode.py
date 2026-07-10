# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode.

Single-token decode that is bitwise identical to a full-sequence chunked
scan (what the training recompute runs). Each slot keeps a buffer of the
inputs since its last chunk boundary; every decode step re-runs the chunked
scan over buffer + new token, so the token lands at the same intra-chunk
position the full scan would give it.

The scan goes through mamba_chunk_scan_decode_rows, which gates the kernels
down to the one output row per slot (and, on boundary crossings, the chunk
state) that a decode step actually uses. Everything is fixed-shape with no
host syncs, so the whole step is CUDA-graph capturable.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_decode_rows


@dataclass
class BikDecodeBuffers:
    """Per-slot persistent state for the buffered decode scan."""

    chunk_size: int
    x: torch.Tensor          # (max_batch + 1, chunk_size, nh, p)
    dt: torch.Tensor         # (max_batch + 1, chunk_size, nh)
    B: torch.Tensor          # (max_batch + 1, chunk_size, ng, n)
    C: torch.Tensor          # (max_batch + 1, chunk_size, ng, n)
    count: torch.Tensor      # (max_batch + 1,) int32, write cursor per slot
    # 1.0 normally; 0.0 while a slot's prefill hasn't crossed a chunk
    # boundary yet, meaning its cached ssm state is stale and the kernels
    # must treat the initial state as zero.
    state_scale: torch.Tensor  # (max_batch + 1,) fp32
    # Per-lane target-row output, allocated once and sliced per step.
    out: torch.Tensor        # (max_batch + 1, nh, p)


def make_bik_decode_buffers(
    max_batch: int,
    chunk_size: int,
    nh: int,
    p: int,
    ng: int,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BikDecodeBuffers:
    """Allocate the per-slot decode buffers.

    No z buffer: batch-invariant mode only supports rmsnorm models, where
    the gate is applied outside the scan and z is never read.

    Buffers get max_batch + 1 rows: the extra row is a write sink for
    inactive lanes (batch_indices < 0). With inactive lanes redirected
    there, every scatter can write unconditionally; duplicate indices only
    occur among inactive lanes writing identical values, so there is no
    write-order race.
    """
    rows = max_batch + 1
    return BikDecodeBuffers(
        chunk_size=chunk_size,
        x=torch.zeros(rows, chunk_size, nh, p, device=device, dtype=dtype),
        dt=torch.zeros(rows, chunk_size, nh, device=device, dtype=dtype),
        B=torch.zeros(rows, chunk_size, ng, n, device=device, dtype=dtype),
        C=torch.zeros(rows, chunk_size, ng, n, device=device, dtype=dtype),
        count=torch.zeros(rows, device=device, dtype=torch.int32),
        state_scale=torch.ones(rows, device=device, dtype=torch.float32),
        out=torch.empty(rows, nh, p, device=device, dtype=dtype),
    )


def seed_bik_decode_buffers(
    bufs: BikDecodeBuffers,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
) -> None:
    """Seed each request's decode buffer with its prefill's partial-chunk tail.

    Decode replays the buffer so the new token sits at the same intra-chunk
    position a full scan would give it. When prefill_len < chunk_size no
    boundary was crossed, so the whole prefill goes into the buffer and
    state_scale[slot] is set to 0.0: the state the prefill left in the cache
    is mid-chunk and must not be used, so the kernels zero the initial state
    instead.

    No host syncs or Python loops; the engine captures prefill steps into
    CUDA graphs, so this has to be capture-legal. Buffer positions past the
    tail get a duplicated row, which is fine: the scan is causal and count
    marks the valid prefix.
    """
    chunk = bufs.chunk_size
    device = x.device
    nseq = cu_seqlens.numel() - 1

    starts = cu_seqlens[:-1].to(torch.long)
    ends = cu_seqlens[1:].to(torch.long)
    plens = ends - starts
    # Covers every case: plen < chunk gives plen, aligned gives 0.
    tails = plens % chunk

    # Redirect inactive lanes (batch_indices < 0) to the trash row so all
    # writes below are unconditional.
    trash = bufs.count.shape[0] - 1
    if batch_indices is not None:
        slots_raw = batch_indices[:nseq].to(torch.long)
    else:
        slots_raw = torch.arange(nseq, device=device, dtype=torch.long)
    active = slots_raw >= 0
    slots = torch.where(active, slots_raw, torch.full_like(slots_raw, trash))

    # Row-gather indices for each sequence's tail, (nseq, chunk). Positions
    # past the tail alias trailing rows; clamped and never consumed.
    j = torch.arange(chunk, device=device, dtype=torch.long)
    idx = ((ends - tails).unsqueeze(1) + j.unsqueeze(0)).clamp(max=x.shape[0] - 1)

    bufs.x[slots] = x[idx]
    bufs.dt[slots] = dt[idx]
    bufs.B[slots] = B[idx]
    bufs.C[slots] = C[idx]

    # Keep the trash row's count pinned at 0 so its buffer writes stay in
    # bounds.
    bufs.count[slots] = torch.where(
        active, tails, torch.zeros_like(tails)
    ).to(torch.int32)

    # Short prefills never produced a boundary state; tell the kernels to
    # zero the initial state for those slots. Multiplying by 0.0/1.0 is
    # exact, and this avoids writing into the engine-owned cache.
    bufs.state_scale[slots] = torch.where(
        active & (plens < chunk),
        torch.zeros_like(plens, dtype=torch.float32),
        torch.ones_like(plens, dtype=torch.float32),
    )


def bik_decode_buffered_scan(
    bufs: BikDecodeBuffers,
    x: torch.Tensor,           # (B_dec, 1, nh, p)
    dt: torch.Tensor,          # (B_dec, 1, nh)
    B: torch.Tensor,            # (B_dec, 1, ng, n)
    C: torch.Tensor,            # (B_dec, 1, ng, n)
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    """One decode step, bitwise identical to a full-sequence chunked scan.

    Why not something simpler: selective_state_update drifts ~6e-5 vs the
    chunked scan (different fp arithmetic), and running the chunked scan on
    just the new token drifts ~2.4e-4 because the token sits at intra-chunk
    position 0 instead of where the full scan has it (measured at
    nemotron6_3b_moe dims in the unit tests). Re-scanning the buffered
    partial chunk puts the token at the right position with the right
    preceding inputs, so the result matches exactly. When a buffer fills to
    chunk_size the returned state is a real boundary state; it is written to
    ssm_state and the buffer restarts.

    The kernels read the buffers and ssm_state in place (each lane's chunk
    is the window at slot * chunk_size) and only compute the rows/states a
    step consumes. All shapes are fixed per decode batch size and there are
    no host syncs, so the step captures into CUDA graphs.

    Returns y of shape (B_dec, 1, nh, p). Mutates bufs and ssm_state.
    """
    B_dec, S_dec, nh, p = x.shape
    n = B.shape[-1]
    dev = x.device
    chunk = bufs.chunk_size
    assert S_dec == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )
    assert B_dec < bufs.out.shape[0] + 1, (
        f"decode batch of {B_dec} lanes exceeds the preallocated scratch "
        f"({bufs.out.shape[0]} lanes = max_batch + 1); increase max_batch."
    )

    # Redirect inactive lanes (batch_indices < 0) to the trash row so the
    # buffer writes below are unconditional.
    trash = bufs.count.shape[0] - 1
    if batch_indices is not None:
        slots_raw = batch_indices.to(torch.long)
    else:
        slots_raw = torch.arange(B_dec, device=dev, dtype=torch.long)
    is_active = slots_raw >= 0                          # (B_dec,)
    slots = torch.where(is_active, slots_raw, torch.full_like(slots_raw, trash))
    # ssm_state is engine-owned and has no trash row: clamp for reads, and
    # writes only happen for crossing slots (in-kernel), which never alias.
    state_slots = slots.clamp(max=trash - 1)

    count_per_batch = bufs.count[slots].to(torch.long)  # (B_dec,)

    # Write the new token at (slot, count).
    bufs.x[slots, count_per_batch] = x[:, 0]
    bufs.dt[slots, count_per_batch] = dt[:, 0]
    bufs.B[slots, count_per_batch] = B[:, 0]
    bufs.C[slots, count_per_batch] = C[:, 0]

    chunk_starts = (slots * chunk).to(torch.int32)
    state_slots_i32 = state_slots.to(torch.int32)
    out = bufs.out[:B_dec]

    target_rows = count_per_batch.to(torch.int32)
    crossing = ((count_per_batch + 1 == chunk) & is_active).to(torch.int32)

    # State passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    mamba_chunk_scan_decode_rows(
        bufs.x.view(-1, nh, p),
        bufs.dt.view(-1, nh),
        A,
        bufs.B.view(-1, bufs.B.shape[-2], n),
        bufs.C.view(-1, bufs.C.shape[-2], n),
        chunk,
        chunk_starts,
        state_slots_i32,
        target_rows,
        crossing,
        ssm_state,
        out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_dtype=ssm_state.dtype,
        # view, not flatten: the in-kernel snapshot must write the cache
        # itself, and view fails loudly if the cache isn't viewable while
        # flatten would silently copy.
        dst_states=ssm_state.view(ssm_state.shape[0], ssm_state.shape[1], -1),
        init_scale=bufs.state_scale,
    )

    # The scan stored each lane's target row at out[i]; padding lanes
    # return zeros.
    y_per_batch = out * is_active.view(-1, 1, 1).to(out.dtype)  # (B_dec, nh, p)
    y = y_per_batch.unsqueeze(1)                                # (B_dec, 1, nh, p)

    # Crossed slots restart their buffer; the rest advance. Inactive lanes
    # write 0 to the trash row, keeping its cursor pinned in bounds.
    new_count_per_batch = torch.where(
        crossing.bool() | ~is_active,
        torch.zeros_like(count_per_batch),
        count_per_batch + 1,
    )
    bufs.count[slots] = new_count_per_batch.to(torch.int32)

    # Crossed slots now have a valid boundary state in the cache, so their
    # init scale goes back to 1.0. Everyone else rewrites their current
    # value.
    old_scale = bufs.state_scale[slots]
    bufs.state_scale[slots] = torch.where(
        crossing.bool(), torch.ones_like(old_scale), old_scale
    )

    return y
