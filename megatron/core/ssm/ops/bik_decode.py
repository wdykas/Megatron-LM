# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Batch-invariant Mamba decode helpers.

A CG-compatible buffered chunk-scan that makes single-token decode produce
output bitwise-identical to a full (prefill + decode) scan, by keeping a
per-slot buffer of inputs since the last chunk boundary and re-running the
batched `mamba_chunk_scan_combined` kernel over that buffer each step.

See `bik_decode_buffered_scan` for the algorithm.
"""

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None


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
    state_is_zero: torch.Tensor  # (max_batch,) bool — pass init=None for slot


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
    count and a "state at position 0 is zero" flag per slot. Used by
    `seed_bik_decode_buffers` and `bik_decode_buffered_scan`.

    has_z=False skips the z buffer (same size as the x buffer — the single
    largest allocation) for models where gating happens outside the scan
    (rmsnorm=True): the scan is always called with z=None there.
    """
    return BikDecodeBuffers(
        chunk_size=chunk_size,
        x=torch.zeros(max_batch, chunk_size, nh, p, device=device, dtype=x_dtype),
        dt=torch.zeros(max_batch, chunk_size, nh, device=device, dtype=dt_dtype),
        B=torch.zeros(max_batch, chunk_size, ng, n, device=device, dtype=B_dtype),
        C=torch.zeros(max_batch, chunk_size, ng, n, device=device, dtype=C_dtype),
        z=(
            torch.zeros(max_batch, chunk_size, nh, p, device=device, dtype=z_dtype)
            if has_z
            else None
        ),
        count=torch.zeros(max_batch, device=device, dtype=torch.int32),
        # True when prefill_len < chunk_size for this slot: decode must pass
        # initial_states=None on its first call instead of reading ssm_state.
        state_is_zero=torch.zeros(max_batch, device=device, dtype=torch.bool),
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
) -> None:
    """Seed each request's decode buffer with its prefill's partial-chunk tail.

    `bik_decode_buffered_scan` needs this so the new decode token sits at the
    same intra-chunk position the full scan would place it, which is what
    makes the decode output bitwise-equal to a full prefill+decode_token scan.
    When prefill_len < chunk_size, the whole prefill goes into the buffer and
    `state_is_zero[slot]` flags decode to use a zero initial state on its
    first call.
    """
    z_flat = z.squeeze(0) if z.dim() == 4 else z
    num_prefill = cu_seqlens.numel() - 1
    for r in range(num_prefill):
        slot = int(batch_indices[r].item()) if batch_indices is not None else r
        if slot < 0:
            continue
        start = int(cu_seqlens[r].item())
        end = int(cu_seqlens[r + 1].item())
        plen = end - start
        # No boundary state was computed when prefill is shorter than a chunk;
        # decode reads `ssm_state[slot]` only when this is False.
        if plen < bufs.chunk_size:
            tail = plen
            bufs.state_is_zero[slot] = True
        else:
            tail = plen % bufs.chunk_size
            bufs.state_is_zero[slot] = False
        bufs.count[slot] = tail
        if tail > 0:
            tail_start = end - tail
            bufs.x[slot, :tail] = x[tail_start:end]
            bufs.dt[slot, :tail] = dt[tail_start:end]
            bufs.B[slot, :tail] = B[tail_start:end]
            bufs.C[slot, :tail] = C[tail_start:end]
            if bufs.z is not None:
                bufs.z[slot, :tail] = z_flat[tail_start:end]


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

    Vectorization: the active slots' buffers are gathered into a dense
    (B_dec, chunk_size, ...) tensor and scanned in one batched chunk_scan —
    O(B_dec) work instead of O(max_batch), bitwise-identical because the
    kernel computes each batch row independently with the same tile shapes.
    We rely on the scan being causal — y[i, count[i]] only depends on
    positions 0..count[i] in the buffer, so stale data at positions >
    count[i] doesn't corrupt the new token's output. Final states are only
    used for slots that filled their chunk this step (count+1 == chunk_size),
    where the entire buffer is fresh and final_states is the boundary state
    we want.

    No host syncs → CG-capturable.

    Returns y of shape (B_dec, 1, nh, p). Mutates `bufs` and `ssm_state`.
    """
    B_dec, S_dec, nh, p = x.shape
    dev = x.device
    assert S_dec == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )

    # --- Slot indices + active mask ---
    if batch_indices is not None:
        slots_raw = batch_indices.to(torch.long)
    else:
        slots_raw = torch.arange(B_dec, device=dev, dtype=torch.long)
    is_active = slots_raw >= 0                          # (B_dec,)
    slots = slots_raw.clamp(min=0)                      # (B_dec,) safe for indexing

    # Each batch position's current write-cursor (per-slot count).
    count_per_batch = bufs.count[slots].to(torch.long)  # (B_dec,)

    # --- Write new tokens into persistent buffer at (slot, count[slot]) ---
    # For inactive batch positions (slot<0 → clamped to 0), guard the write
    # with a torch.where so we don't clobber slot 0's data.
    def _safe_write(buf: torch.Tensor, new_vals: torch.Tensor) -> None:
        # buf: (max_batch, chunk_size, *trailing); new_vals: (B_dec, *trailing)
        old_vals = buf[slots, count_per_batch]
        mask = is_active.view([-1] + [1] * (new_vals.ndim - 1))
        buf[slots, count_per_batch] = torch.where(mask, new_vals, old_vals)

    _safe_write(bufs.x, x[:, 0])
    _safe_write(bufs.dt, dt[:, 0])
    _safe_write(bufs.B, B[:, 0])
    _safe_write(bufs.C, C[:, 0])
    if z is not None:
        _safe_write(bufs.z, z[:, 0])

    # --- Gather the active slots' buffers into a dense (B_dec, chunk, ...) ---
    # The kernel parallelizes the batch dim with identical per-row tile
    # shapes, so scanning the gathered rows is bitwise-identical to scanning
    # them in place inside the full max_batch buffer — but costs O(B_dec)
    # instead of O(max_batch). Gather/scatter with the (unique) slot indices
    # is deterministic, and all shapes are fixed per decode batch size, so
    # this stays CUDA-graph capturable.
    x_g = bufs.x[slots]
    dt_g = bufs.dt[slots]
    B_g = bufs.B[slots]
    C_g = bufs.C[slots]
    z_g = bufs.z[slots] if (z is not None and bufs.z is not None) else None

    # --- Mask initial state to zero where state_is_zero ---
    # (per-slot: True means "treat as zero initial state on this scan call")
    init_g = ssm_state[slots] * (
        ~bufs.state_is_zero[slots]
    ).view(-1, 1, 1, 1).to(ssm_state.dtype)

    # --- Single batched chunk_scan over the dense (B_dec, chunk_size, ...) ---
    y_dense, new_state_dense = mamba_chunk_scan_combined(
        x_g,
        dt_g,
        A,
        B_g,
        C_g,
        bufs.chunk_size,
        D=D,
        z=z_g,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=init_g.contiguous(),
        return_final_states=True,
    )
    # y_dense: (B_dec, chunk_size, nh, p)
    # new_state_dense: (B_dec, nh, p, n)

    # --- Per-batch y extraction: y[i, count[i]] ---
    batch_arange = torch.arange(B_dec, device=dev, dtype=torch.long)
    y_per_batch = y_dense[batch_arange, count_per_batch]    # (B_dec, nh, p)
    y_per_batch = y_per_batch * is_active.view(-1, 1, 1).to(y_per_batch.dtype)
    y = y_per_batch.unsqueeze(1)                            # (B_dec, 1, nh, p)

    # --- Per-slot state updates (vectorized via torch.where masking) ---
    # A slot "crosses" the chunk boundary on this step iff count+1==chunk_size.
    # Crossed: snapshot final state, reset count to 0, clear state_is_zero.
    # Uncrossed: count+=1, state_is_zero unchanged.
    # Inactive batch positions: no update.
    crossed_per_batch = (count_per_batch + 1 == bufs.chunk_size) & is_active  # (B_dec,)
    new_count_per_batch = torch.where(
        crossed_per_batch, torch.zeros_like(count_per_batch), count_per_batch + 1
    )
    new_szero_per_batch = torch.where(
        crossed_per_batch,
        torch.zeros_like(crossed_per_batch),
        bufs.state_is_zero[slots],
    )

    # Apply to the persistent buffers, gated by is_active (so we don't touch
    # slots not in this decode batch).
    old_count = bufs.count[slots]
    bufs.count[slots] = torch.where(
        is_active, new_count_per_batch.to(torch.int32), old_count
    )
    old_szero = bufs.state_is_zero[slots]
    bufs.state_is_zero[slots] = torch.where(is_active, new_szero_per_batch, old_szero)

    # ssm_state: snapshot only at slots that crossed AND are active.
    snapshot_mask = (is_active & crossed_per_batch).view(-1, 1, 1, 1)
    old_state = ssm_state[slots]
    new_state_dtype = new_state_dense.to(ssm_state.dtype)
    ssm_state[slots] = torch.where(snapshot_mask, new_state_dtype, old_state)

    return y
