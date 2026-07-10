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
class BatchInvariantDecodeBuffers:
    """Per-slot persistent state for the buffered decode scan."""

    chunk_size: int
    x: torch.Tensor            # (max_batch + 1, chunk_size, nheads, headdim)
    dt: torch.Tensor           # (max_batch + 1, chunk_size, nheads)
    B: torch.Tensor            # (max_batch + 1, chunk_size, ngroups, dstate)
    C: torch.Tensor            # (max_batch + 1, chunk_size, ngroups, dstate)
    # Tokens buffered since the slot's last chunk boundary; doubles as the
    # write cursor for the next token.
    num_buffered: torch.Tensor  # (max_batch + 1,) int32
    # 1.0 normally; 0.0 while a slot's prefill hasn't crossed a chunk
    # boundary yet, meaning its cached ssm state is stale and the kernels
    # must treat the initial state as zero.
    state_scale: torch.Tensor  # (max_batch + 1,) fp32
    # Per-lane target-row output, allocated once and sliced per step.
    out: torch.Tensor          # (max_batch + 1, nheads, headdim)

    @property
    def trash_row(self) -> int:
        """Write sink for inactive lanes (the buffers' extra last row)."""
        return self.num_buffered.shape[0] - 1

    @property
    def max_lanes(self) -> int:
        """Largest decode batch the preallocated buffers support."""
        return self.num_buffered.shape[0]


def make_batch_invariant_decode_buffers(
    max_batch: int,
    chunk_size: int,
    nheads: int,
    headdim: int,
    ngroups: int,
    dstate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> BatchInvariantDecodeBuffers:
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
    return BatchInvariantDecodeBuffers(
        chunk_size=chunk_size,
        x=torch.zeros(rows, chunk_size, nheads, headdim, device=device, dtype=dtype),
        dt=torch.zeros(rows, chunk_size, nheads, device=device, dtype=dtype),
        B=torch.zeros(rows, chunk_size, ngroups, dstate, device=device, dtype=dtype),
        C=torch.zeros(rows, chunk_size, ngroups, dstate, device=device, dtype=dtype),
        num_buffered=torch.zeros(rows, device=device, dtype=torch.int32),
        state_scale=torch.ones(rows, device=device, dtype=torch.float32),
        out=torch.empty(rows, nheads, headdim, device=device, dtype=dtype),
    )


def seed_batch_invariant_decode_buffers(
    bufs: BatchInvariantDecodeBuffers,
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    cu_seqlens: torch.Tensor,
    batch_indices: torch.Tensor,
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
    tail get a duplicated row, which is fine: the scan is causal and
    num_buffered marks the valid prefix.
    """
    chunk_size = bufs.chunk_size
    num_seqs = cu_seqlens.numel() - 1

    seq_starts = cu_seqlens[:-1].to(torch.long)
    seq_ends = cu_seqlens[1:].to(torch.long)
    prefill_lens = seq_ends - seq_starts
    # Covers every case: prefill_len < chunk_size gives prefill_len,
    # boundary-aligned gives 0.
    tail_lens = prefill_lens % chunk_size

    # Redirect inactive lanes (batch_indices < 0) to the trash row so all
    # writes below are unconditional.
    requested_slots = batch_indices[:num_seqs].to(torch.long)
    is_active = requested_slots >= 0
    slots = torch.where(
        is_active, requested_slots, torch.full_like(requested_slots, bufs.trash_row)
    )

    # Row-gather indices for each sequence's tail, (num_seqs, chunk_size).
    # Positions past the tail alias trailing rows; clamped and never
    # consumed.
    offsets = torch.arange(chunk_size, device=x.device, dtype=torch.long)
    tail_token_idx = (
        (seq_ends - tail_lens).unsqueeze(1) + offsets.unsqueeze(0)
    ).clamp(max=x.shape[0] - 1)

    bufs.x[slots] = x[tail_token_idx]
    bufs.dt[slots] = dt[tail_token_idx]
    bufs.B[slots] = B[tail_token_idx]
    bufs.C[slots] = C[tail_token_idx]

    # Keep the trash row's count pinned at 0 so its buffer writes stay in
    # bounds.
    bufs.num_buffered[slots] = torch.where(
        is_active, tail_lens, torch.zeros_like(tail_lens)
    ).to(torch.int32)

    # Short prefills never produced a boundary state; tell the kernels to
    # zero the initial state for those slots. Multiplying by 0.0/1.0 is
    # exact, and this avoids writing into the engine-owned cache.
    bufs.state_scale[slots] = torch.where(
        is_active & (prefill_lens < chunk_size),
        torch.zeros_like(prefill_lens, dtype=torch.float32),
        torch.ones_like(prefill_lens, dtype=torch.float32),
    )


def batch_invariant_decode_buffered_scan(
    bufs: BatchInvariantDecodeBuffers,
    x: torch.Tensor,           # (num_lanes, 1, nheads, headdim)
    dt: torch.Tensor,          # (num_lanes, 1, nheads)
    B: torch.Tensor,           # (num_lanes, 1, ngroups, dstate)
    C: torch.Tensor,           # (num_lanes, 1, ngroups, dstate)
    A: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_indices: torch.Tensor,
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

    Returns y of shape (num_lanes, 1, nheads, headdim). Mutates bufs and
    ssm_state.
    """
    num_lanes, tokens_per_lane, nheads, headdim = x.shape
    dstate = B.shape[-1]
    chunk_size = bufs.chunk_size
    assert tokens_per_lane == 1, (
        "batch-invariant Mamba decode assumes one new token per request "
        "per call (no speculative decoding)."
    )
    assert num_lanes <= bufs.max_lanes, (
        f"decode batch of {num_lanes} lanes exceeds the preallocated buffers "
        f"({bufs.max_lanes} lanes); increase max_batch."
    )

    # Redirect inactive lanes (batch_indices < 0) to the trash row so the
    # buffer writes below are unconditional.
    requested_slots = batch_indices.to(torch.long)
    is_active = requested_slots >= 0
    slots = torch.where(
        is_active, requested_slots, torch.full_like(requested_slots, bufs.trash_row)
    )
    # ssm_state is engine-owned and has no trash row: clamp for reads. Its
    # only writes happen in-kernel for crossing slots, which never alias.
    state_slots = slots.clamp(max=bufs.trash_row - 1)

    # Write the new token at each slot's cursor; write_pos is also the
    # token's intra-chunk row, the one row the scan must produce.
    write_pos = bufs.num_buffered[slots].to(torch.long)
    bufs.x[slots, write_pos] = x[:, 0]
    bufs.dt[slots, write_pos] = dt[:, 0]
    bufs.B[slots, write_pos] = B[:, 0]
    bufs.C[slots, write_pos] = C[:, 0]

    # A slot crosses its chunk boundary when this token fills the buffer.
    crossed = (write_pos + 1 == chunk_size) & is_active
    out = bufs.out[:num_lanes]

    # Run the gated pipeline over the buffers and ssm_state in place. State
    # passing writes crossing slots' boundary states straight into
    # ssm_state, so no scatter is needed afterwards.
    mamba_chunk_scan_decode_rows(
        bufs.x.view(-1, nheads, headdim),
        bufs.dt.view(-1, nheads),
        A,
        bufs.B.view(-1, bufs.B.shape[-2], dstate),
        bufs.C.view(-1, bufs.C.shape[-2], dstate),
        chunk_size,
        chunk_starts=(slots * chunk_size).to(torch.int32),
        slots=state_slots.to(torch.int32),
        target_rows=write_pos.to(torch.int32),
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
    bufs.num_buffered[slots] = torch.where(
        crossed | ~is_active, torch.zeros_like(write_pos), write_pos + 1
    ).to(torch.int32)

    # Crossed slots now have a valid boundary state in the cache, so their
    # init scale goes back to 1.0. Everyone else rewrites their current
    # value.
    prev_scale = bufs.state_scale[slots]
    bufs.state_scale[slots] = torch.where(
        crossed, torch.ones_like(prev_scale), prev_scale
    )

    return y


class MambaBatchInvariantDecode:
    """Adapter between a MambaMixer and the buffered decode.

    Owns the decode buffers and translates the mixer's conventions (flat
    layouts, context-parallel projections, config flags) into the tensor-op
    API above, so the mixer itself only carries two call sites. Uses the
    mixer by duck typing; the import direction stays
    mixer -> batch_invariant_decode.
    """

    def __init__(self, mixer):
        # The gate is applied outside the scan (RMSNormGated), so the
        # buffers carry no z. Enforced here because the decode path would
        # otherwise silently drop it.
        assert mixer.rmsnorm, "batch_invariant_mode requires rmsnorm=True"
        self.mixer = mixer
        self.bufs: Optional[BatchInvariantDecodeBuffers] = None

    def _get_bufs(
        self, max_batch, nheads, headdim, ngroups, dstate, device, dtype
    ) -> BatchInvariantDecodeBuffers:
        if self.bufs is None:
            self.bufs = make_batch_invariant_decode_buffers(
                max_batch, self.mixer.chunk_size, nheads, headdim, ngroups, dstate,
                device, dtype,
            )
        return self.bufs

    def seed(self, x, dt, B, C, cu_seqlens, batch_indices, ssm_state) -> None:
        """Seed from the prefill tail. x: (total, nheads, headdim);
        B/C: (total, ngroups, dstate)."""
        nheads, headdim = x.shape[-2], x.shape[-1]
        ngroups, dstate = B.shape[-2], B.shape[-1]
        bufs = self._get_bufs(
            ssm_state.shape[0], nheads, headdim, ngroups, dstate, x.device, x.dtype
        )
        seed_batch_invariant_decode_buffers(bufs, x, dt, B, C, cu_seqlens, batch_indices)

    def step(self, x, dt, B, C, batch_indices, ssm_state) -> torch.Tensor:
        """One decode step. Inputs in the mixer's flat layout:
        x (batch, 1, nheads*headdim), dt (batch, 1, nheads),
        B/C (batch, 1, ngroups*dstate). Returns (batch, 1, nheads*headdim).
        """
        mixer = self.mixer
        batch = x.shape[0]
        x = x.view(batch, 1, -1, mixer.headdim)
        B = B.view(batch, 1, mixer.ngroups_local_tp, -1)
        C = C.view(batch, 1, mixer.ngroups_local_tp, -1)

        A = -torch.exp(mixer.cp.get_A_log().float())
        D = mixer.cp.get_D()
        if mixer.D_has_hdim:
            D = D.float().view(-1, mixer.headdim)
        dt_bias = mixer.cp.get_dt_bias().float()

        nheads, headdim = x.shape[-2], x.shape[-1]
        ngroups, dstate = B.shape[-2], B.shape[-1]
        bufs = self._get_bufs(
            ssm_state.shape[0], nheads, headdim, ngroups, dstate, x.device, x.dtype
        )

        y = batch_invariant_decode_buffered_scan(
            bufs, x, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
        return y.reshape(batch, 1, -1)
