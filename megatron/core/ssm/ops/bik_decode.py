# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode helpers.

A CG-compatible buffered chunk-scan that makes single-token decode produce
output bitwise-identical to a full (prefill + decode) scan, by keeping a
per-slot buffer of inputs since the last chunk boundary and re-running the
chunked-scan pipeline over that buffer each step.

The scan runs through the row-gated repo kernel pipeline
(`mamba_chunk_scan_decode_rows`): only the output row each slot actually
consumes (and, on boundary-crossing steps, the chunk state) is computed.
The surviving kernel blocks execute the exact same instructions as the
ungated kernels, so outputs stay bitwise-identical to a full scan.

See `bik_decode_buffered_scan` for the algorithm.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_decode_rows


@dataclass
class BikDecodeBuffers:
    """Per-slot persistent state for the buffered decode scan.

    Sized at first call and reused across decode steps. All tensors live on
    the GPU; the dataclass exists to keep the function signatures short.
    """

    chunk_size: int
    x: torch.Tensor          # (max_batch, chunk_size, nh, p)
    dt: torch.Tensor         # (max_batch, chunk_size, nh)
    B: torch.Tensor          # (max_batch, chunk_size, ng, n)
    C: torch.Tensor          # (max_batch, chunk_size, ng, n)
    z: Optional[torch.Tensor]  # (max_batch, chunk_size, nh, p); None when the
    #   model applies gating outside the scan (rmsnorm=True) — the scan never
    #   reads z then, so the buffer would be pure memory waste.
    count: torch.Tensor      # (max_batch,) int32 — write cursor per slot
    # Per-decode-batch-size scratch (cu_chunk_seqlens, seq_idx, out, states
    # workspace), lazily created and cached so repeated steps (and CUDA-graph
    # capture) reuse fixed storage.
    scratch: Dict[int, Tuple[torch.Tensor, ...]] = field(default_factory=dict)


def make_bik_decode_buffers(
    max_batch: int,
    chunk_size: int,
    nh: int,
    p: int,
    ng: int,
    n: int,
    device: torch.device,
    x_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    B_dtype: torch.dtype,
    C_dtype: torch.dtype,
    z_dtype: torch.dtype,
    has_z: bool = True,
) -> BikDecodeBuffers:
    """Allocate the per-slot decode-side scan buffers.

    The buffers hold `(x, dt, B, C, z)` since the last chunk boundary plus a
    write cursor per slot. Used by `seed_bik_decode_buffers` and
    `bik_decode_buffered_scan`.

    has_z=False skips the z buffer (same size as the x buffer — the single
    largest allocation) for models where gating happens outside the scan
    (rmsnorm=True): the scan is always called with z=None there.

    Buffers are allocated with max_batch + 1 rows: the last row is a write
    sink for inactive batch lanes (batch_indices < 0). Redirecting inactive
    lanes there lets every scatter write unconditionally — duplicate indices
    then only occur among inactive lanes writing identical values, so there
    is no write-order race with real slots.
    """
    rows = max_batch + 1
    return BikDecodeBuffers(
        chunk_size=chunk_size,
        x=torch.zeros(rows, chunk_size, nh, p, device=device, dtype=x_dtype),
        dt=torch.zeros(rows, chunk_size, nh, device=device, dtype=dt_dtype),
        B=torch.zeros(rows, chunk_size, ng, n, device=device, dtype=B_dtype),
        C=torch.zeros(rows, chunk_size, ng, n, device=device, dtype=C_dtype),
        z=(
            torch.zeros(rows, chunk_size, nh, p, device=device, dtype=z_dtype)
            if has_z
            else None
        ),
        count=torch.zeros(rows, device=device, dtype=torch.int32),
    )


def seed_bik_decode_buffers(
    bufs: BikDecodeBuffers,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
) -> None:
    """Seed each request's decode buffer with its prefill's partial-chunk tail.

    `bik_decode_buffered_scan` needs this so the new decode token sits at the
    same intra-chunk position the full scan would place it, which is what
    makes the decode output bitwise-equal to a full prefill+decode_token scan.

    When prefill_len < chunk_size, the whole prefill goes into the buffer and
    `ssm_state[slot]` is zeroed: no chunk boundary was crossed, so the state
    the prefill kernel wrote there (state at prompt end) must not be used —
    the buffer replays the sequence from position 0 with a zero initial
    state, exactly like the full scan.

    Fully vectorized: no host syncs, no per-request Python loop. Buffer
    positions past each tail are filled with a duplicated (finite) row; they
    are never consumed — the scan is causal and `count` marks the valid
    prefix.
    """
    chunk = bufs.chunk_size
    device = x.device
    nseq = cu_seqlens.numel() - 1

    starts = cu_seqlens[:-1].to(torch.long)
    ends = cu_seqlens[1:].to(torch.long)
    plens = ends - starts
    # Uniform in all cases: plen < chunk → plen; aligned → 0; else remainder.
    tails = plens % chunk

    # Inactive lanes (batch_indices < 0) redirect to the trash row so all
    # buffer writes are unconditional (duplicates only among inactive lanes,
    # writing identical values).
    trash = bufs.count.shape[0] - 1
    if batch_indices is not None:
        slots_raw = batch_indices[:nseq].to(torch.long)
    else:
        slots_raw = torch.arange(nseq, device=device, dtype=torch.long)
    active = slots_raw >= 0
    slots = torch.where(active, slots_raw, torch.full_like(slots_raw, trash))

    # Row-gather indices for each sequence's tail: (nseq, chunk). Positions
    # past the tail alias trailing rows (clamped — finite, never consumed).
    j = torch.arange(chunk, device=device, dtype=torch.long)
    idx = ((ends - tails).unsqueeze(1) + j.unsqueeze(0)).clamp(max=x.shape[0] - 1)

    bufs.x[slots] = x[idx]
    bufs.dt[slots] = dt[idx]
    bufs.B[slots] = B[idx]
    bufs.C[slots] = C[idx]
    if bufs.z is not None:
        z_flat = z.squeeze(0) if z.dim() == 4 else z
        bufs.z[slots] = z_flat[idx]

    # Trash-row count pinned to 0 (identical duplicate writes are benign).
    bufs.count[slots] = torch.where(
        active, tails, torch.zeros_like(tails)
    ).to(torch.int32)

    # No boundary was crossed for prefills shorter than a chunk: decode must
    # start from a zero state, so overwrite whatever the prefill kernel left
    # in the cache for those slots. Boolean compression gives unique real
    # slot indices (no aliasing); seeding runs on the prefill path, outside
    # CUDA-graph capture, so the data-dependent shape is fine.
    zero_slots = slots_raw[(plens < chunk) & active]
    ssm_state[zero_slots] = 0


def _get_scratch(bufs, B_dec, nh, p, n, device, x_dtype):
    """Fixed per-(B_dec) scratch: chunk layout metadata + output/state buffers."""
    key = B_dec
    if key not in bufs.scratch:
        chunk = bufs.chunk_size
        bufs.scratch[key] = (
            torch.arange(B_dec + 1, device=device, dtype=torch.int32) * chunk,
            torch.arange(B_dec, device=device, dtype=torch.int32),
            torch.empty(B_dec, nh, p, device=device, dtype=x_dtype),
            torch.zeros(B_dec, nh, p, n, device=device, dtype=torch.float32),
        )
    return bufs.scratch[key]


def bik_decode_buffered_scan(
    bufs: BikDecodeBuffers,
    x: torch.Tensor,           # (B_dec, 1, nh, p)
    dt: torch.Tensor,          # (B_dec, 1, nh)
    B: torch.Tensor,            # (B_dec, 1, ng, n)
    C: torch.Tensor,            # (B_dec, 1, ng, n)
    z: Optional[torch.Tensor],  # (B_dec, 1, nh, p) or None
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_indices: Optional[torch.Tensor],
    ssm_state: torch.Tensor,
) -> torch.Tensor:
    """CG-compat batched buffered `chunk_scan` for decode, bitwise to a full scan.

    Drift values below are empirical max-abs differences vs a full
    prefill+decode scan, observed in the batch-invariant Mamba decode
    test (nemotron6_3b_moe dims: nheads=128, headdim=64, d_state=128,
    chunk_size=256; bf16 weights and bf16 inputs; single decode step after
    a prefill of varying length).

    Default decode uses `selective_state_update`, which differs from
    `mamba_chunk_scan_combined` in bf16 by ~6e-5 in that test. Single-token
    `chunk_scan(new_tok, init=ssm_state)` also drifts (~2.4e-4) because its
    intra-chunk position differs from the full scan's. Calling
    `chunk_scan(buffer + new_tok, init=ssm_state)` where the buffer is the
    prefill's partial-chunk tail (seeded by `seed_bik_decode_buffers`)
    restores the same intra-chunk position and gives bitwise-identical
    output. When the buffer fills to `chunk_size`, we snapshot the returned
    state as the new ssm_state and reset the buffer.

    Execution: the active slots' buffers are gathered into a dense
    (B_dec, chunk_size, ...) layout and run through the row-gated repo
    kernel pipeline (`mamba_chunk_scan_decode_rows`) — only the output row
    each slot consumes this step is computed, and the chunk-state matmul
    runs only for slots crossing their boundary. O(B_dec · BLOCK_M) work
    instead of O(max_batch · chunk_size). Bitwise-identical because the
    surviving kernel blocks execute the exact same instructions as the
    ungated kernels.

    No host syncs → CG-capturable (all shapes fixed per decode batch size).

    Returns y of shape (B_dec, 1, nh, p). Mutates `bufs` and `ssm_state`.
    """
    B_dec, S_dec, nh, p = x.shape
    n = B.shape[-1]
    dev = x.device
    chunk = bufs.chunk_size
    assert S_dec == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )

    # --- Slot indices + active mask ---
    # Inactive lanes (batch_indices < 0) are redirected to the trash row
    # (the buffers' extra last row), so every buffer write below is
    # unconditional and race-free.
    trash = bufs.count.shape[0] - 1
    if batch_indices is not None:
        slots_raw = batch_indices.to(torch.long)
    else:
        slots_raw = torch.arange(B_dec, device=dev, dtype=torch.long)
    is_active = slots_raw >= 0                          # (B_dec,)
    slots = torch.where(is_active, slots_raw, torch.full_like(slots_raw, trash))
    # ssm_state is engine-owned with max_batch rows (no trash row) — reads of
    # it use a clamped index; writes to it happen only for active crossing
    # slots (inside the fused snapshot), which never alias.
    state_slots = slots.clamp(max=trash - 1)

    # Each batch position's current write-cursor (per-slot count). The trash
    # row's count is pinned to 0 by the update below, so its writes stay
    # in bounds.
    count_per_batch = bufs.count[slots].to(torch.long)  # (B_dec,)

    # --- Write new tokens into persistent buffer at (slot, count[slot]) ---
    bufs.x[slots, count_per_batch] = x[:, 0]
    bufs.dt[slots, count_per_batch] = dt[:, 0]
    bufs.B[slots, count_per_batch] = B[:, 0]
    bufs.C[slots, count_per_batch] = C[:, 0]
    if z is not None and bufs.z is not None:
        bufs.z[slots, count_per_batch] = z[:, 0]

    # --- Gather active slots' buffers into a dense flattened layout ---
    # Each gathered buffer row becomes one fixed-size chunk that is its own
    # sequence (seq_idx = arange), so the scan reads `initial_states`
    # directly. `ssm_state[slot]` is guaranteed valid here: it is either a
    # chunk-boundary snapshot or the zeros written by seeding for prefills
    # shorter than a chunk.
    x_g = bufs.x[slots].reshape(B_dec * chunk, nh, p)
    dt_g = bufs.dt[slots].reshape(B_dec * chunk, nh)
    B_g = bufs.B[slots].reshape(B_dec * chunk, -1, n)
    C_g = bufs.C[slots].reshape(B_dec * chunk, -1, n)
    z_g = (
        bufs.z[slots].reshape(B_dec * chunk, nh, p)
        if (z is not None and bufs.z is not None)
        else None
    )
    init_g = ssm_state[state_slots]

    cu, seq_idx, out, states_ws = _get_scratch(bufs, B_dec, nh, p, n, dev, x_g.dtype)

    target_rows = count_per_batch.to(torch.int32)
    crossing = ((count_per_batch + 1 == chunk) & is_active).to(torch.int32)

    # The fused snapshot in state passing writes crossing slots' boundary
    # states straight into the ssm_state cache (each crossing chunk writes
    # its own slot — no duplicate indices, no separate scatter pass).
    mamba_chunk_scan_decode_rows(
        x_g,
        dt_g,
        A,
        B_g,
        C_g,
        chunk,
        cu,
        seq_idx,
        target_rows,
        crossing,
        init_g,
        out,
        D=D,
        z=z_g,
        dt_bias=dt_bias,
        dt_softplus=True,
        states_workspace=states_ws,
        state_dtype=ssm_state.dtype,
        # view (not flatten) so a non-viewable cache fails loudly instead of
        # silently copying — the in-kernel snapshot writes must hit the cache.
        dst_states=ssm_state.view(ssm_state.shape[0], ssm_state.shape[1], -1),
        dst_indices=state_slots.to(torch.int32),
    )

    # --- y: the scan stored each chunk's target row compactly at out[i] ---
    y_per_batch = out * is_active.view(-1, 1, 1).to(out.dtype)  # (B_dec, nh, p)
    y = y_per_batch.unsqueeze(1)                                # (B_dec, 1, nh, p)

    # --- Per-slot count update ---
    # A slot "crosses" the chunk boundary on this step iff count+1==chunk_size
    # (== `crossing` above). Crossed: reset count to 0 (the boundary state was
    # already persisted in-kernel by the fused snapshot). Uncrossed: count+=1.
    # Inactive lanes write 0 to the trash row — identical values, so the
    # duplicate indices are benign, and the trash row's cursor stays pinned
    # at 0 (keeping its unconditional buffer writes in bounds).
    new_count_per_batch = torch.where(
        crossing.bool() | ~is_active,
        torch.zeros_like(count_per_batch),
        count_per_batch + 1,
    )
    bufs.count[slots] = new_count_per_batch.to(torch.int32)

    return y
