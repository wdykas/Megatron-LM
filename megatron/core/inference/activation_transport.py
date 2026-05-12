# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM transport for hidden-state activations across disaggregated shards.

Activations flow between shards in the layer-kind-disaggregated forward
pass; this module is the on-the-wire mechanism.

Reuses the NVSHMEM runtime primitives from :mod:`nvshmem_migration` (one
PE identity per process, shared pool builders and put/signal wrappers)
but allocates its own symmetric pools — slot sizing and lifecycle differ
from migration's:

- Migration: per-request handoff, ~MB per op, dozens of ops per second.
  Slots are claimed for the duration of one migration then implicitly
  freed (stream-order releases them across migrations).
- Activations: per-layer per-token, ~KB per op, *thousands* of ops per
  second. Slot count cannot scale with op count — must be a small ring
  buffer with explicit ``dst → src`` ack-flag recycling.

The pool is laned by ``(src_pe, dst_pe)``: each src→dst PE pair gets its
own contiguous block of ``pool_depth`` slots. Within a lane, slots cycle
0 → 1 → ... → pool_depth-1 → 0. The src tracks "next slot to use" per
lane on its own; the dst tracks "next slot to wait on" per lane;
because both walk activations in the same model order, the counters
stay in sync without explicit messaging — same invariant as the
migration ``StagingArena``.

Recycling: before src reuses slot N (i.e., counter wraps past N), it
must observe that the dst's prior scatter on slot N has completed. That
fact is delivered as a ``dst → src`` signal on ``ack_flag[N]``. The src
signals_wait on the ack before reusing the slot. The dst's
``send_activation_ack`` raises the ack after its scatter is done.

The module is split into:

- ``maybe_init_activation_transport`` — collective init; allocates the
  pool + flags + activation stream.
- ``lane_for(src_pe, dst_pe)`` — deterministic lane index.
- ``next_slot_for(lane)`` — increment the per-lane sender counter,
  return the absolute slot index in the pool.
- ``put_activation`` / ``wait_activation`` / ``ack_activation`` —
  primitives that compose into a send-receive-ack cycle.

This module is *transport only*. The forward-pass router (in
``dynamic_engine`` extensions) decides when to call put / wait / ack
based on the request's route plan.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.inference import nvshmem_migration as _nv

logger = logging.getLogger(__name__)


# Module-level state. All `_a_*` names belong to activation transport;
# `_nv` module state is reused but not duplicated here.
_initialized: bool = False
_activation_stream: Optional[torch.cuda.Stream] = None

# Pool layout: ``_a_num_lanes × _a_pool_depth`` total slots. Slot index
# of (lane, slot_in_lane) is ``lane * pool_depth + slot_in_lane``. Same
# encoding for fwd_flag / ack_flag pools.
_a_num_lanes: int = 0
_a_pool_depth: int = 0
_a_slot_bytes: int = 0
_a_max_pes: int = 0
# Encoding base for ``lane_for(src, dst) = src * stride + dst``.
# Defaults to ``world_size`` so ``num_lanes = stride²`` lines up.
_a_lane_stride: int = 0

# Symmetric pools (one bytetensor per slot / per flag).
_a_slots: list = []
_a_fwd_flags: list = []
_a_fwd_flag_bufs: list = []
_a_ack_flags: list = []
_a_ack_flag_bufs: list = []

# Dedicated 1-byte symmetric tensor used as the payload for ``ack_activation``.
# We cannot use a slice of the activation slot itself: NVSHMEM's
# ``put_signal`` tracks the underlying bytetensor handle and writes the
# full transferred-bytes count of *whatever* it interprets as the buffer
# size — empirically, slicing to ``[:1]`` on a 64 KB slot caused the put
# to clobber the whole slot. A separate, always-zero 1-byte buffer keeps
# the ack payload from touching real slot data on the source side.
_a_ack_payload: Optional[torch.Tensor] = None

# Per-lane sender + receiver counters. Both walk in lockstep through
# the lane's ring; src increments before each put, dst increments after
# each wait. Same invariant as the migration StagingArena.
_lane_send_counter: list = []  # len = num_lanes
_lane_recv_counter: list = []  # len = num_lanes


DEFAULT_POOL_DEPTH = 64
DEFAULT_SLOT_BYTES = 4 * 1024 * 1024
# Encoding upper bound on PE id; ``num_lanes`` defaults to
# ``world_size²`` so the actual allocation tracks the live cluster size.
DEFAULT_MAX_PES = 64


def activation_stream() -> torch.cuda.Stream:
    """Dedicated CUDA stream for activation puts.

    Kept separate from :func:`nvshmem_migration.migration_stream` so
    that migrations don't head-of-line-block activations and vice versa
    (migrations are per-request, low-frequency; activations are
    per-layer-per-token, high-frequency).
    """
    assert _initialized, "activation_transport not initialized"
    assert _activation_stream is not None
    return _activation_stream


def is_initialized() -> bool:
    """Whether :func:`maybe_init_activation_transport` has run."""
    return _initialized


def maybe_init_activation_transport(
    *,
    num_lanes: Optional[int] = None,
    pool_depth: Optional[int] = None,
    slot_bytes: Optional[int] = None,
    max_pes: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Initialize the activation-transport pools. Idempotent.

    Calls :func:`nvshmem_migration.maybe_init_nvshmem` first so the
    process-wide PE identity is set up. Then allocates this module's
    own symmetric pools and stream.

    All sizing knobs are also reachable via env vars
    (``ACTIVATION_POOL_DEPTH``, ``ACTIVATION_SLOT_BYTES``,
    ``ACTIVATION_MAX_PES``, ``ACTIVATION_NUM_LANES``); explicit kwargs
    take precedence.
    """
    global _initialized, _activation_stream
    global _a_num_lanes, _a_pool_depth, _a_slot_bytes, _a_max_pes, _a_lane_stride
    global _a_slots, _a_fwd_flags, _a_fwd_flag_bufs
    global _a_ack_flags, _a_ack_flag_bufs
    global _lane_send_counter, _lane_recv_counter

    if _initialized:
        return

    # NVSHMEM availability + version are checked inside maybe_init_nvshmem;
    # it raises with a clear message if either is missing.
    _nv.maybe_init_nvshmem(group=group)

    _a_max_pes = int(
        max_pes
        if max_pes is not None
        else os.environ.get("ACTIVATION_MAX_PES", DEFAULT_MAX_PES)
    )
    # Default num_lanes to ``world_size²`` (actual upper bound on
    # ordered PE pairs in this run) rather than ``max_pes²`` (a
    # conservative ceiling). For a 4-PE cluster that's 16 lanes
    # instead of 4096, saving ~250× symmetric memory. Callers can
    # still override for sparse-connectivity layouts via the kwarg
    # or ``ACTIVATION_NUM_LANES``.
    env_num_lanes = os.environ.get("ACTIVATION_NUM_LANES")
    if num_lanes is not None:
        _a_num_lanes = int(num_lanes)
        _a_lane_stride = _a_max_pes
    elif env_num_lanes is not None:
        _a_num_lanes = int(env_num_lanes)
        _a_lane_stride = _a_max_pes
    else:
        world_size = dist.get_world_size(group) if dist.is_initialized() else _a_max_pes
        _a_num_lanes = world_size * world_size
        _a_lane_stride = world_size
    _a_pool_depth = int(
        pool_depth
        if pool_depth is not None
        else os.environ.get("ACTIVATION_POOL_DEPTH", DEFAULT_POOL_DEPTH)
    )
    _a_slot_bytes = int(
        slot_bytes
        if slot_bytes is not None
        else os.environ.get("ACTIVATION_SLOT_BYTES", DEFAULT_SLOT_BYTES)
    )

    _activation_stream = torch.cuda.Stream()

    n_slots = _a_num_lanes * _a_pool_depth
    _a_slots = _nv.allocate_slot_pool(n_slots, _a_slot_bytes)

    # Two flag pools paired 1-to-1 with slots: fwd (src signals dst that
    # data has landed) and ack (dst signals src that scatter is done and
    # slot is free for reuse). ack flags initialize to 1 ("first use is
    # free") so the first round-trip on each slot doesn't deadlock on a
    # never-issued ack.
    _a_fwd_flags, _a_fwd_flag_bufs = _nv.allocate_flag_pool(n_slots)
    _a_ack_flags, _a_ack_flag_bufs = _nv.allocate_flag_pool(
        n_slots, initial_value=1
    )

    _lane_send_counter = [0] * _a_num_lanes
    _lane_recv_counter = [0] * _a_num_lanes

    # Dedicated 1-byte payload for ack puts — see the comment on
    # _a_ack_payload at module top for why this can't be a slice of a
    # real slot.
    global _a_ack_payload
    _a_ack_payload = _nv.allocate_slot_pool(1, 1)[0]
    _a_ack_payload.zero_()

    _nv.barrier_all_and_sync()
    _initialized = True
    logger.info(
        "[activation-transport] initialized: lanes=%d, depth=%d, "
        "slot_bytes=%d, total_symm_per_pe=%d MB",
        _a_num_lanes,
        _a_pool_depth,
        _a_slot_bytes,
        (_a_num_lanes * _a_pool_depth * _a_slot_bytes) // (1024 * 1024),
    )


def lane_for(src_pe: int, dst_pe: int) -> int:
    """Deterministic lane index from a ``(src_pe, dst_pe)`` pair.

    Both endpoints compute the same lane id so they coordinate without
    explicit messaging. Returned lane id is in ``[0, num_lanes)``.
    """
    assert _initialized
    assert 0 <= src_pe < _a_lane_stride, (
        f"src_pe={src_pe} out of range; raise ACTIVATION_MAX_PES."
    )
    assert 0 <= dst_pe < _a_lane_stride, (
        f"dst_pe={dst_pe} out of range; raise ACTIVATION_MAX_PES."
    )
    lane = src_pe * _a_lane_stride + dst_pe
    assert lane < _a_num_lanes, (
        f"lane id {lane} from (src={src_pe}, dst={dst_pe}) exceeds "
        f"num_lanes={_a_num_lanes}; raise ACTIVATION_NUM_LANES."
    )
    return lane


def _slot_index(lane: int, lane_offset: int) -> int:
    return lane * _a_pool_depth + (lane_offset % _a_pool_depth)


def next_send_slot(lane: int) -> int:
    """Advance the lane's send counter and return the absolute slot
    index in the pool. Caller (the sender) gates the actual put on the
    ack flag for this slot, which is what enforces ring-buffer recycling.
    """
    assert _initialized
    idx = _lane_send_counter[lane]
    _lane_send_counter[lane] = idx + 1
    return _slot_index(lane, idx)


def next_recv_slot(lane: int) -> int:
    """Advance the lane's receive counter and return the absolute slot
    index. The receiver signal_waits on the fwd flag at this slot."""
    assert _initialized
    idx = _lane_recv_counter[lane]
    _lane_recv_counter[lane] = idx + 1
    return _slot_index(lane, idx)


def activation_slot(slot_idx: int) -> torch.Tensor:
    """Return the symmetric uint8 slot for the given absolute index."""
    assert _initialized
    assert 0 <= slot_idx < len(_a_slots), (
        f"slot {slot_idx} out of range [0, {len(_a_slots)})"
    )
    return _a_slots[slot_idx]


def fwd_flag_buffer(slot_idx: int):
    assert _initialized
    return _a_fwd_flag_bufs[slot_idx]


def ack_flag_buffer(slot_idx: int):
    assert _initialized
    return _a_ack_flag_bufs[slot_idx]


def put_activation(
    slot_idx: int,
    dst_pe: int,
    *,
    nbytes: int,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Source-side put: wait for ack of the slot's prior owner, then
    issue an NVSHMEM ``put_signal`` carrying ``nbytes`` from this PE's
    slot ``slot_idx`` to ``dst_pe``'s slot ``slot_idx``, setting the
    fwd flag at ``slot_idx`` on the destination.

    Stream-ordered: the ack-wait, the put, and the fwd-signal all run
    on the activation stream so no host-side sync is required.

    Caller is responsible for writing the payload into the slot before
    calling this (typically with a ``.copy_`` on the activation stream).
    """
    assert _initialized
    s = stream or _activation_stream

    # Wait until the prior user of this slot acked. Pre-credited at
    # init (ack=1) so the very first put doesn't block.
    _nv.signal_wait(_a_ack_flag_bufs[slot_idx], stream=s)
    # Reset the ack so the next round-trip can re-trigger. The reset
    # MUST run on the activation stream — without the explicit
    # ``with torch.cuda.stream`` context, ``.zero_()`` falls back to
    # the default stream and races with the put_signal below.
    with torch.cuda.stream(s):
        _a_ack_flags[slot_idx].zero_()

    # Put data + signal fwd on dst.
    _nv.put_signal(
        _a_slots[slot_idx],
        _a_fwd_flag_bufs[slot_idx],
        dst_pe,
        nbytes=nbytes,
        stream=s,
    )


def wait_activation(
    slot_idx: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Destination-side wait: block on the activation stream until the
    forward signal at ``slot_idx`` reaches 1, then reset it for the next
    round.

    The data is in this PE's slot ``slot_idx`` after this returns.
    Caller is responsible for scattering it out of the slot (typically
    a ``.copy_`` to the engine's activation buffer) before calling
    :func:`ack_activation`.
    """
    assert _initialized
    s = stream or _activation_stream
    _nv.signal_wait(_a_fwd_flag_bufs[slot_idx], stream=s)
    # The reset must be stream-ordered after the signal_wait so the
    # next round's put_signal sees a clean 0 → 1 transition. Without
    # the stream context, ``.zero_()`` runs on the default stream and
    # can race with the next iteration's signal_wait.
    with torch.cuda.stream(s):
        _a_fwd_flags[slot_idx].zero_()


def ack_activation(
    slot_idx: int,
    src_pe: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Destination-side ack: signal the source PE that ``slot_idx`` is
    safe to reuse. Issued after the destination has scattered the
    activation out of the slot.

    Uses a dedicated 1-byte payload (``_a_ack_payload``), never a slice
    of the data slot: a slice would race with the source's pending fill
    / put_signal on the same slot and corrupt the data.
    """
    assert _initialized
    _nv.put_signal(
        _a_ack_payload,
        _a_ack_flag_bufs[slot_idx],
        src_pe,
        stream=stream or _activation_stream,
    )


# ---- High-level hidden-state send/receive ----------------------------------
#
# Combine lane selection, slot allocation, the symmetric-tensor view, and
# the underlying put/wait/ack into single calls per direction. The route
# dispatcher (the only production caller of activation transport) talks
# in shard-PE pairs and hidden tensors — it shouldn't need to touch slot
# indexing or symmetric-memory views.


def send_hidden(
    my_pe: int,
    dst_pe: int,
    hidden: torch.Tensor,
    payload_nbytes: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Send a hidden-state tensor from ``my_pe`` to ``dst_pe`` over the
    activation lane for that pair.

    The activation stream serializes: copy hidden into the symmetric
    slot view, then ``put_activation`` (which itself stream-waits for
    the slot's prior ack before issuing the put + fwd signal).
    """
    lane = lane_for(my_pe, dst_pe)
    slot = next_send_slot(lane)
    sym = activation_slot(slot)
    s = stream or _activation_stream
    with torch.cuda.stream(s):
        view = sym[:payload_nbytes].view(hidden.dtype).reshape(hidden.shape)
        view.copy_(hidden)
    put_activation(slot, dst_pe=dst_pe, nbytes=payload_nbytes, stream=s)


def receive_hidden(
    my_pe: int,
    src_pe: int,
    hidden_shape: tuple,
    hidden_dtype: torch.dtype,
    payload_nbytes: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Receive a hidden-state tensor from ``src_pe`` over the activation
    lane for that pair. Returns a fresh tensor (cloned out of the
    symmetric slot) so the slot can be acked + reused immediately.
    """
    lane = lane_for(src_pe, my_pe)
    slot = next_recv_slot(lane)
    s = stream or _activation_stream
    wait_activation(slot, stream=s)
    s.synchronize()
    sym = activation_slot(slot)
    view = sym[:payload_nbytes].view(hidden_dtype).reshape(hidden_shape)
    out = view.clone()
    ack_activation(slot, src_pe=src_pe, stream=s)
    return out


def reset_lane_counters(lane: int) -> None:
    """Reset both send and recv counters for a lane to 0. Used between
    independent runs in tests; not invoked in normal operation."""
    assert _initialized
    _lane_send_counter[lane] = 0
    _lane_recv_counter[lane] = 0


def pool_stats() -> dict:
    """Diagnostic snapshot. Used by tests + the bench validation path."""
    assert _initialized
    return {
        "num_lanes": _a_num_lanes,
        "pool_depth": _a_pool_depth,
        "slot_bytes": _a_slot_bytes,
        "max_pes": _a_max_pes,
        "total_slots": len(_a_slots),
        "symm_per_pe_bytes": _a_num_lanes * _a_pool_depth * _a_slot_bytes,
        "send_counters": list(_lane_send_counter),
        "recv_counters": list(_lane_recv_counter),
    }


# ---- Test-only ----------------------------------------------------------------


def _init_state_for_test(
    *,
    num_lanes: int,
    pool_depth: int,
    slot_bytes: int = 1024,
    max_pes: int = 8,
) -> None:
    """Set up the Python-side bookkeeping (counters, sizing) without
    allocating any NVSHMEM buffers or starting the CUDA stream. Used
    by unit tests that exercise lane encoding / counter advancement
    without needing a real GPU + multi-rank world.

    Calling :func:`maybe_init_activation_transport` after this raises;
    explicitly :func:`_reset_state_for_test` first to switch modes.
    """
    global _initialized, _a_num_lanes, _a_pool_depth, _a_slot_bytes, _a_max_pes, _a_lane_stride
    global _lane_send_counter, _lane_recv_counter
    if _initialized:
        raise RuntimeError(
            "activation_transport already initialized; call "
            "_reset_state_for_test() first."
        )
    _a_num_lanes = num_lanes
    _a_pool_depth = pool_depth
    _a_slot_bytes = slot_bytes
    _a_max_pes = max_pes
    _a_lane_stride = max_pes
    _lane_send_counter = [0] * num_lanes
    _lane_recv_counter = [0] * num_lanes
    _initialized = True


def _reset_state_for_test() -> None:
    """Tear down test-only state. NOT safe to call after a real
    NVSHMEM init — would leak the symmetric allocations."""
    global _initialized, _a_num_lanes, _a_pool_depth, _a_slot_bytes, _a_max_pes, _a_lane_stride
    global _lane_send_counter, _lane_recv_counter
    _initialized = False
    _a_num_lanes = 0
    _a_pool_depth = 0
    _a_slot_bytes = 0
    _a_max_pes = 0
    _a_lane_stride = 0
    _lane_send_counter = []
    _lane_recv_counter = []
