# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM transport for hidden-state activations across disaggregated shards.

Activations flow between shards in the layer-kind-disaggregated forward
pass; this module is the on-the-wire mechanism.

Reuses the NVSHMEM runtime primitives from :mod:`nvshmem_runtime` (one
PE identity per process, shared pool builders and put/signal wrappers)
but allocates its own symmetric pools — slot sizing and lifecycle differ
from migration's:

- Migration: per-request handoff, ~MB per op, dozens of ops per second.
  Slots are claimed for the duration of one migration then implicitly
  freed (stream-order releases them across migrations).
- Activations: per-layer per-token, ~KB per op, *thousands* of ops per
  second. Slot count cannot scale with op count — must be a small ring
  buffer with explicit ``dst → src`` ack-flag recycling.

Lifecycle: same active-pair model as :mod:`migration_transport`.

1. ``maybe_init_activation_transport()`` — runtime + stream only; no
   pools allocated.
2. ``register_activation_pair(src_pe, dst_pe)`` (or the
   ``register_activation_shard_pair`` / ``register_activation_route``
   helpers) — reserves a lane for every PE pair that any registered
   route will use. Bounded by topology, not by ``world_size²``.
3. ``realize_activation_pools()`` — collective, one-shot; allocates
   slot + fwd + ack pools sized by the registered active-pair set.

After realize, the per-lane ring works exactly like before: each
``(src_pe, dst_pe)`` pair owns a contiguous block of ``pool_depth``
slots; within a lane, slots cycle ``0 → 1 → ... → pool_depth-1 → 0``.
Both endpoints walk activations in model order so the per-lane
send/recv counters stay in sync without explicit messaging — same
invariant as the migration ``StagingArena``.

Recycling: before src reuses slot N (i.e., counter wraps past N), it
must observe that the dst's prior scatter on slot N has completed. That
fact is delivered as a ``dst → src`` signal on ``ack_flag[N]``. The src
signals_wait on the ack before reusing the slot. The dst's
``ack_activation`` raises the ack after its scatter is done.

The module is split into:

- ``maybe_init_activation_transport`` — collective init for runtime + stream.
- ``register_activation_pair`` / ``_shard_pair`` / ``_route`` — declare
  the active-pair set.
- ``realize_activation_pools`` — collective one-shot pool allocation.
- ``lane_for(src_pe, dst_pe)`` — dict lookup into the active-pair
  registry; raises on unregistered pairs.
- ``put_activation`` / ``wait_activation`` / ``ack_activation`` —
  primitives that compose into a send-receive-ack cycle.

This module is *transport only*. The forward-pass router (in
``dynamic_engine`` extensions) decides when to call put / wait / ack
based on the request's route plan.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.inference import nvshmem_runtime as _nv

logger = logging.getLogger(__name__)


# Module-level state. All `_a_*` names belong to activation transport;
# `_nv` module state is reused but not duplicated here.
_initialized: bool = False
_pools_realized: bool = False
_activation_stream: Optional[torch.cuda.Stream] = None

# Active-pair registry. Mirrors migration_transport: instead of
# pre-allocating ``world_size²`` lanes, we collect the ``(src_pe, dst_pe)``
# pairs that any registered route will actually use and allocate only
# those. ``_active_pairs[i] == (src, dst)`` owns lane ``i``;
# ``_pair_to_lane_idx`` is the inverse lookup used by :func:`lane_for`.
_active_pairs: List[Tuple[int, int]] = []
_pair_to_lane_idx: Dict[Tuple[int, int], int] = {}

# Sizing knobs (set in maybe_init_activation_transport, before realize).
_a_pool_depth: int = 0
_a_slot_bytes: int = 0

# Derived after realize: total slot count = ``len(_active_pairs) × _a_pool_depth``.
_a_num_slots: int = 0

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
_lane_send_counter: list = []  # len = len(_active_pairs)
_lane_recv_counter: list = []  # len = len(_active_pairs)


DEFAULT_POOL_DEPTH = 64
DEFAULT_SLOT_BYTES = 4 * 1024 * 1024


def activation_stream() -> torch.cuda.Stream:
    """Dedicated CUDA stream for activation puts.

    Kept separate from :func:`migration_transport.migration_stream` so
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
    pool_depth: Optional[int] = None,
    slot_bytes: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Initialize the activation-transport runtime, stream, and sizing
    knobs. **Does not** allocate the slot or flag pools — those are
    deferred to :func:`realize_activation_pools` once the active-pair
    set is known via :func:`register_activation_pair` (or the
    ``_shard_pair`` / ``_route`` helpers).

    Calls :func:`nvshmem_runtime.maybe_init_nvshmem` first so the
    shared runtime is up. Collective: every rank must call.
    Idempotent — subsequent calls are no-ops.

    Sizing knobs: ``pool_depth`` and ``slot_bytes`` also reachable via
    env vars (``ACTIVATION_POOL_DEPTH`` / ``ACTIVATION_SLOT_BYTES``);
    kwargs take precedence. These are fixed at this call — they cannot
    change between this call and :func:`realize_activation_pools`.
    """
    global _initialized, _activation_stream
    global _a_pool_depth, _a_slot_bytes

    if _initialized:
        return

    # NVSHMEM availability + version are checked inside maybe_init_nvshmem;
    # it raises with a clear message if either is missing.
    _nv.maybe_init_nvshmem(group=group)

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
    _initialized = True
    logger.info(
        "[activation-transport] runtime initialized: depth=%d, slot_bytes=%d; "
        "pools deferred to realize_activation_pools()",
        _a_pool_depth,
        _a_slot_bytes,
    )


def register_activation_pair(src_pe: int, dst_pe: int) -> None:
    """Reserve an activation lane for the ``(src_pe, dst_pe)`` pair.

    Must be called before :func:`realize_activation_pools`. Idempotent —
    duplicates are no-ops.

    Cross-rank consistency: every rank must register the same set of
    pairs in the same order so each rank assigns the same lane index
    per pair. In practice, callers compute pairs as a deterministic
    function of registered routes (which are themselves collective), so
    this is automatic.
    """
    if _pools_realized:
        raise RuntimeError(
            f"activation pools already realized; cannot register new pair "
            f"({src_pe}, {dst_pe}). Register all routes before "
            f"realize_activation_pools()."
        )
    pair = (src_pe, dst_pe)
    if pair in _pair_to_lane_idx:
        return
    _pair_to_lane_idx[pair] = len(_active_pairs)
    _active_pairs.append(pair)


def register_activation_shard_pair(
    src_pes: Iterable[int], dst_pes: Iterable[int]
) -> None:
    """Register every ``(s, d)`` in ``src_pes × dst_pes`` — convenience
    for hop transitions where any rank in the src shard might send to
    any rank in the dst shard. Over-registers slightly compared to the
    actual TP-coupling pattern, but the cost is symmetric-memory only
    (lanes with no traffic just go unused) and the implementation is
    layout-agnostic."""
    for s in src_pes:
        for d in dst_pes:
            register_activation_pair(s, d)


def register_activation_route(route, shards) -> None:
    """Walk every hop transition in ``route`` and register
    ``(src_pes, dst_pes)`` for each, using ``shards`` to resolve which
    PEs belong to each shard. ``shards`` is a sequence of objects with
    a ``ranks()`` method (matching :class:`InferenceShard`).

    Includes both linear-chain successors (``hop_pos + 1``) and DAG
    fan-out / fan-in edges via the route's ``successors`` /
    ``predecessors`` methods.
    """
    for hop_pos, hop in enumerate(route.hops):
        src_shard = shards[hop.shard_idx]
        for succ_pos in route.successors(hop_pos):
            dst_shard = shards[route.hops[succ_pos].shard_idx]
            register_activation_shard_pair(src_shard.ranks(), dst_shard.ranks())


def realize_activation_pools() -> None:
    """Allocate slot + fwd-flag + ack-flag pools for all registered
    pairs. Collective: every rank must call. Idempotent — second call
    is a no-op. After this returns, :func:`register_activation_pair`
    rejects new pairs (NVSHMEM symmetric allocation is one-shot per
    pool — the size is fixed at realize time).

    Pool size is ``pool_depth × len(active_pairs)`` slots, plus the
    matching fwd + ack flag pools. Empty (no registered pairs) is
    allowed and is a no-op apart from the ack payload.
    """
    global _a_slots, _a_fwd_flags, _a_fwd_flag_bufs
    global _a_ack_flags, _a_ack_flag_bufs, _a_ack_payload
    global _a_num_slots, _pools_realized
    global _lane_send_counter, _lane_recv_counter

    assert _initialized, (
        "activation_transport not initialized; call "
        "maybe_init_activation_transport() first"
    )
    if _pools_realized:
        return

    n_lanes = len(_active_pairs)
    _a_num_slots = n_lanes * _a_pool_depth

    if _a_num_slots > 0:
        _a_slots = _nv.allocate_slot_pool(_a_num_slots, _a_slot_bytes)
        # Two flag pools paired 1-to-1 with slots: fwd (src signals dst
        # that data has landed) and ack (dst signals src that scatter is
        # done and slot is free for reuse). ack flags initialize to 1
        # ("first use is free") so the first round-trip on each slot
        # doesn't deadlock on a never-issued ack.
        _a_fwd_flags, _a_fwd_flag_bufs = _nv.allocate_flag_pool(_a_num_slots)
        _a_ack_flags, _a_ack_flag_bufs = _nv.allocate_flag_pool(
            _a_num_slots, initial_value=1
        )

    _lane_send_counter = [0] * n_lanes
    _lane_recv_counter = [0] * n_lanes

    # Dedicated 1-byte payload for ack puts — see the comment on
    # _a_ack_payload at module top for why this can't be a slice of a
    # real slot.
    _a_ack_payload = _nv.allocate_slot_pool(1, 1)[0]
    _a_ack_payload.zero_()

    _nv.barrier_all_and_sync()
    _pools_realized = True
    logger.info(
        "[activation-transport] pools realized: active_pairs=%d, depth=%d, "
        "slot_bytes=%d, total_symm_per_pe=%d MB",
        n_lanes,
        _a_pool_depth,
        _a_slot_bytes,
        (_a_num_slots * _a_slot_bytes) // (1024 * 1024),
    )


def lane_for(src_pe: int, dst_pe: int) -> int:
    """Lane index for the ``(src_pe, dst_pe)`` pair.

    Returns the pair's index in the active-pair registry. Different
    pairs land in disjoint slot ranges so the ack handshake on flag
    ``F`` is unambiguous — only this pair's prior use of slot ``F`` can
    have raised the ack.

    Raises if the pair wasn't registered before
    :func:`realize_activation_pools` — usually that means a route is
    trying to send/receive on a hop transition that wasn't included in
    any registered route's hop set.
    """
    assert _initialized
    pair = (src_pe, dst_pe)
    idx = _pair_to_lane_idx.get(pair)
    if idx is None:
        raise RuntimeError(
            f"activation pair ({src_pe}, {dst_pe}) not registered. "
            f"Every (src_pe, dst_pe) used by a route must be passed to "
            f"register_activation_pair() before realize_activation_pools()."
        )
    return idx


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

    Pipelining: this function does NOT host-sync. The signal wait,
    the clone, and the ack all run on the activation stream
    (stream-ordered). The caller's current stream is made to wait
    for the activation stream's completion via ``wait_stream`` — a
    GPU-side cross-stream barrier that does NOT block the host.
    This means the host can continue issuing other work (e.g. the
    next batch's send / the next layer's compute) while the GPU
    waits for the inbound activation. Pre-change behavior was a
    host-side ``s.synchronize()`` which forced the host to wait
    for the receive to fully complete before doing anything else
    — measured as the single biggest pipelining blocker on the
    receiver side (see :doc:`DISAGG_DESIGN.md` Pipelining section).
    """
    lane = lane_for(src_pe, my_pe)
    slot = next_recv_slot(lane)
    s = stream or _activation_stream
    wait_activation(slot, stream=s)
    # Clone the slot data on the activation stream so the clone is
    # ordered after wait_activation. The returned tensor will be
    # produced on the activation stream; the caller's current
    # stream waits via the wait_stream barrier below.
    with torch.cuda.stream(s):
        sym = activation_slot(slot)
        view = sym[:payload_nbytes].view(hidden_dtype).reshape(hidden_shape)
        out = view.clone()
    ack_activation(slot, src_pe=src_pe, stream=s)
    # Cross-stream sync: caller's current stream must see the
    # clone before reading ``out``. This is a GPU-side barrier,
    # not a host block — the host returns immediately and can
    # issue the next batch's commands.
    torch.cuda.current_stream().wait_stream(s)
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
        "active_pairs": len(_active_pairs),
        "pool_depth": _a_pool_depth,
        "slot_bytes": _a_slot_bytes,
        "total_slots": _a_num_slots,
        "symm_per_pe_bytes": _a_num_slots * _a_slot_bytes,
        "send_counters": list(_lane_send_counter),
        "recv_counters": list(_lane_recv_counter),
        "pools_realized": _pools_realized,
    }


# ---- Test-only ----------------------------------------------------------------


def _init_state_for_test(
    *,
    n_pes: int,
    pool_depth: int,
    slot_bytes: int = 1024,
    register_all_pairs: bool = True,
    realize: bool = True,
) -> None:
    """Set up the Python-side bookkeeping (active-pair registry, counters,
    sizing) without allocating any NVSHMEM buffers or starting the CUDA
    stream. Used by unit tests that exercise lane encoding / counter
    advancement without needing a real GPU + multi-rank world.

    By default, registers every ``(src, dst)`` pair in ``range(n_pes)``
    and marks the pool realized — existing tests can call ``lane_for``
    and the counter helpers without further setup. Pass
    ``register_all_pairs=False`` and/or ``realize=False`` to exercise
    the registration / realize lifecycle directly.
    """
    global _initialized, _pools_realized, _a_pool_depth, _a_slot_bytes
    global _a_num_slots
    global _lane_send_counter, _lane_recv_counter
    if _initialized:
        raise RuntimeError(
            "activation_transport already initialized; call "
            "_reset_state_for_test() first."
        )
    _nv._init_state_for_test(n_pes=n_pes)
    _a_pool_depth = pool_depth
    _a_slot_bytes = slot_bytes
    _initialized = True
    if register_all_pairs:
        for s in range(n_pes):
            for d in range(n_pes):
                register_activation_pair(s, d)
    n_lanes = len(_active_pairs)
    _a_num_slots = n_lanes * _a_pool_depth
    _lane_send_counter = [0] * n_lanes
    _lane_recv_counter = [0] * n_lanes
    if realize:
        _pools_realized = True


def _reset_state_for_test() -> None:
    """Tear down test-only state. NOT safe to call after a real
    NVSHMEM init — would leak the symmetric allocations."""
    global _initialized, _pools_realized
    global _active_pairs, _pair_to_lane_idx
    global _a_pool_depth, _a_slot_bytes, _a_num_slots
    global _lane_send_counter, _lane_recv_counter
    global _a_slots, _a_fwd_flags, _a_fwd_flag_bufs, _a_ack_flags, _a_ack_flag_bufs
    global _a_ack_payload, _activation_stream
    _initialized = False
    _pools_realized = False
    _active_pairs = []
    _pair_to_lane_idx = {}
    _a_pool_depth = 0
    _a_slot_bytes = 0
    _a_num_slots = 0
    _lane_send_counter = []
    _lane_recv_counter = []
    _a_slots = []
    _a_fwd_flags = []
    _a_fwd_flag_bufs = []
    _a_ack_flags = []
    _a_ack_flag_bufs = []
    _a_ack_payload = None
    _activation_stream = None
    _nv._reset_state_for_test()
