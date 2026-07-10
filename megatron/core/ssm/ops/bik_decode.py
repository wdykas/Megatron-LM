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

    @property
    def trash_row(self) -> int:
        """Write sink for inactive lanes (the buffers' extra last row)."""
        return self.count.shape[0] - 1

    @property
    def max_lanes(self) -> int:
        """Largest decode batch the preallocated buffers support."""
        return self.count.shape[0]


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
    if batch_indices is not None:
        requested = batch_indices[:nseq].to(torch.long)
    else:
        requested = torch.arange(nseq, device=device, dtype=torch.long)
    active = requested >= 0
    slots = torch.where(
        active, requested, torch.full_like(requested, bufs.trash_row)
    )

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
    chunk = bufs.chunk_size
    assert S_dec == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )
    assert B_dec <= bufs.max_lanes, (
        f"decode batch of {B_dec} lanes exceeds the preallocated buffers "
        f"({bufs.max_lanes} lanes); increase max_batch."
    )

    # Redirect inactive lanes (batch_indices < 0) to the trash row so the
    # buffer writes below are unconditional.
    if batch_indices is not None:
        requested = batch_indices.to(torch.long)
    else:
        requested = torch.arange(B_dec, device=x.device, dtype=torch.long)
    is_active = requested >= 0
    slots = torch.where(is_active, requested, torch.full_like(requested, bufs.trash_row))
    # ssm_state is engine-owned and has no trash row: clamp for reads. Its
    # only writes happen in-kernel for crossing slots, which never alias.
    state_slots = slots.clamp(max=bufs.trash_row - 1)

    # Write the new token at each slot's cursor.
    counts = bufs.count[slots].to(torch.long)
    bufs.x[slots, counts] = x[:, 0]
    bufs.dt[slots, counts] = dt[:, 0]
    bufs.B[slots, counts] = B[:, 0]
    bufs.C[slots, counts] = C[:, 0]

    # A slot crosses its chunk boundary when this token fills the buffer.
    crossed = (counts + 1 == chunk) & is_active
    out = bufs.out[:B_dec]

    # Run the gated pipeline over the buffers and ssm_state in place. State
    # passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    mamba_chunk_scan_decode_rows(
        bufs.x.view(-1, nh, p),
        bufs.dt.view(-1, nh),
        A,
        bufs.B.view(-1, bufs.B.shape[-2], n),
        bufs.C.view(-1, bufs.C.shape[-2], n),
        chunk,
        chunk_starts=(slots * chunk).to(torch.int32),
        slots=state_slots.to(torch.int32),
        target_rows=counts.to(torch.int32),
        chunk_flags=crossed.to(torch.int32),
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        state_dtype=ssm_state.dtype,
        # view, not flatten: the in-kernel snapshot must write the cache
        # itself. view fails loudly if the cache isn't viewable; flatten
        # would silently copy.
        dst_states=ssm_state.view(ssm_state.shape[0], ssm_state.shape[1], -1),
        init_scale=bufs.state_scale,
    )

    # The scan stored each lane's target row at out[i]; padding lanes
    # return zeros.
    y = (out * is_active.view(-1, 1, 1).to(out.dtype)).unsqueeze(1)

    # Crossed slots restart their buffer; the rest advance. Inactive lanes
    # write 0 to the trash row, keeping its cursor pinned in bounds.
    bufs.count[slots] = torch.where(
        crossed | ~is_active, torch.zeros_like(counts), counts + 1
    ).to(torch.int32)

    # Crossed slots now have a valid boundary state in the cache, so their
    # init scale goes back to 1.0. Everyone else rewrites their current
    # value.
    old_scale = bufs.state_scale[slots]
    bufs.state_scale[slots] = torch.where(
        crossed, torch.ones_like(old_scale), old_scale
    )

    return y


class MambaBikDecode:
    """Adapter between a MambaMixer and the buffered decode.

    Owns the decode buffers and translates the mixer's conventions (flat
    layouts, context-parallel projections, config flags) into the tensor-op
    API above, so the mixer itself only carries two call sites. Uses the
    mixer by duck typing; the import direction stays mixer -> bik_decode.
    """

    def __init__(self, mixer):
        # The gate is applied outside the scan (RMSNormGated), so the
        # buffers carry no z. Enforced here because the decode path would
        # otherwise silently drop it.
        assert mixer.rmsnorm, "batch_invariant_mode requires rmsnorm=True"
        self.mixer = mixer
        self.bufs: Optional[BikDecodeBuffers] = None

    def _get_bufs(self, max_batch, nh, p, ng, n, device, dtype) -> BikDecodeBuffers:
        if self.bufs is None:
            self.bufs = make_bik_decode_buffers(
                max_batch, self.mixer.chunk_size, nh, p, ng, n, device, dtype
            )
        return self.bufs

    def seed(self, x, dt, B, C, cu_seqlens, batch_indices, ssm_state) -> None:
        """Seed from the prefill tail. x: (total, nh, p); B/C: (total, ng, n)."""
        nh, p = x.shape[-2], x.shape[-1]
        ng, n = B.shape[-2], B.shape[-1]
        bufs = self._get_bufs(ssm_state.shape[0], nh, p, ng, n, x.device, x.dtype)
        seed_bik_decode_buffers(bufs, x, dt, B, C, cu_seqlens, batch_indices)

    def step(self, x, dt, B, C, batch_indices, ssm_state) -> torch.Tensor:
        """One decode step. Inputs in the mixer's flat layout:
        x (b, 1, nh*p), dt (b, 1, nh), B/C (b, 1, ng*n). Returns (b, 1, nh*p).
        """
        m = self.mixer
        b = x.shape[0]
        x = x.view(b, 1, -1, m.headdim)
        B = B.view(b, 1, m.ngroups_local_tp, -1)
        C = C.view(b, 1, m.ngroups_local_tp, -1)

        A = -torch.exp(m.cp.get_A_log().float())
        D = m.cp.get_D()
        if m.D_has_hdim:
            D = D.float().view(-1, m.headdim)
        dt_bias = m.cp.get_dt_bias().float()

        nh, p = x.shape[-2], x.shape[-1]
        ng, n = B.shape[-2], B.shape[-1]
        bufs = self._get_bufs(ssm_state.shape[0], nh, p, ng, n, x.device, x.dtype)

        y = bik_decode_buffered_scan(
            bufs, x, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y.reshape(b, 1, -1)
