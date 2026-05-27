# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Direct NVSHMEM transport for cross-shard request migration.

One-sided ``put_signal`` from src lands KV bytes plus a completion
flag in a single op; dst's ``signal_wait`` on the same flag implies
all bytes have arrived. Both sides issue everything on a dedicated
migration stream, so neither engine has to pause its run loop.

KV / Mamba buffers themselves are *not* in the symmetric heap; only
the fixed-size symmetric staging slots (one ``bytetensor`` per slot,
allocated up front in :func:`maybe_init_nvshmem`) and per-op flags
are. The migration handler stages each op into a slot,
``put_signal``s the slot to the matching offset on the destination's
symmetric region, and the destination scatters the slot back into
its local buffer.

Cross-migration safety: the flag pool is partitioned by ``(src_pe,
dst_pe)`` lane (so different pairs can't alias on the same dst flag)
and reuse within a lane is serialized by an ack handshake — src
``signal_wait``s on a per-flag ack before re-issuing a put; dst raises
the ack after its scatter completes. This is the same discipline
that :mod:`activation_transport` uses, applied to the bulk-migration
traffic shape (one transport, two access patterns; identical
synchronization invariant).
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


try:
    import nvshmem.core

    _HAVE_NVSHMEM = True
except ImportError:  # pragma: no cover
    _HAVE_NVSHMEM = False


_initialized: bool = False
_my_pe: int = -1
_n_pes: int = -1
_migration_stream: Optional[torch.cuda.Stream] = None

# Fwd-flag pool: ``ops_per_pair × n_pes²`` slots, partitioned by
# ``(src_pe, dst_pe)`` lane. Set by put_signal on the dst PE; observed
# by dst's signal_wait. Each slot is its own 8-byte ``bytetensor``
# (NVSHMEM signal_var requires Buffer >= 8 bytes).
_flag_pool: List[torch.Tensor] = []
_flag_pool_buffers: List[object] = []  # cuda.core Buffer per slot

# Ack pool: same shape as flag pool, parallel allocation. Set by dst
# (via put_signal targeting src's ack-flag) after scatter completes;
# observed by src's signal_wait before re-issuing a put on that flag.
# Pre-credited at init (initial_value=1) so the first put on a flag
# doesn't block.
_ack_pool: List[torch.Tensor] = []
_ack_pool_buffers: List[object] = []

# 1-byte symmetric payload used as the bulk-data argument of ack
# ``put_signal``s. Same trick as activation_transport's _a_ack_payload:
# a dedicated buffer keeps the ack put from clobbering the staging slot
# on the src side (NVSHMEM's put copies the full src buffer's tracked
# size, not just the bytes the caller cares about).
_ack_payload: Optional[torch.Tensor] = None

# Pool dimensions. Pool size = ops_per_pair × n_pes².
_ops_per_pair: int = 0
_num_lanes: int = 0  # = n_pes²
_flag_pool_size: int = 0  # = _ops_per_pair × _num_lanes (derived)

_staging_slots: List[torch.Tensor] = []  # pool of symmetric uint8[SLOT_BYTES]
_staging_slot_bytes: int = 0
_staging_num_slots: int = 0


# Per-(src, dst) ops budget. Within one migration, at most
# ``DEFAULT_OPS_PER_PAIR`` ops can target a given pair without the
# allocator wrapping; wraps within the pair recycle via the ack
# handshake. A KV migration with deep PP × big batch hits ~30 ops/pair
# at worst; mamba adds a few more. 64 leaves comfortable headroom for
# typical configs and stays small enough that the symmetric alloc
# count doesn't dominate startup (each flag is its own ``bytetensor``
# allocation). Tunable via ``MIGRATION_OPS_PER_PAIR``.
DEFAULT_OPS_PER_PAIR = 64

# NVSHMEM only tracks the full tensor handle returned by ``bytetensor``;
# slices with non-zero offsets aren't tracked, so we pre-allocate a
# fixed pool of independently-tracked slot tensors. Slot count scales
# with ``batch × overlap_pairs`` for both KV and mamba (mamba's four
# block kinds pack into one slot per overlap in the migration handler).
DEFAULT_STAGING_SLOT_BYTES = 16 * 1024 * 1024  # 16 MB per slot
DEFAULT_STAGING_NUM_SLOTS = 256  # 4 GB total per PE


def migration_stream() -> torch.cuda.Stream:
    """Dedicated CUDA stream for migration puts so they don't serialize
    against the engine's compute stream.

    Singleton: every cross-migration safety argument in this module
    assumes all migration ops issue on this one stream, so they're
    GPU-serialized regardless of which Python task submitted them.
    Submitting migration ops on a different stream would break the
    ack-handshake → put ordering and is not supported.
    """
    assert _initialized, "NVSHMEM not initialized"
    assert _migration_stream is not None
    return _migration_stream


# ---- Shared runtime primitives ---------------------------------------------
#
# Both `nvshmem_migration` and `activation_transport` build symmetric flag
# pools, slot pools, and call `put_signal` / `signal_wait` on the same
# underlying NVSHMEM API. These helpers factor out the duplication so each
# transport module only owns its policy code (slot indexing, recycling,
# ack flow), not the NVSHMEM call boilerplate.


def allocate_flag_pool(
    n_flags: int, *, initial_value: int = 0
) -> Tuple[List[torch.Tensor], List[object]]:
    """Allocate ``n_flags`` independently-tracked 8-byte symmetric flag
    tensors. NVSHMEM ``put_signal`` / ``signal_wait`` require Buffer ≥ 8
    bytes, and slices with non-zero offsets aren't tracked, so each flag
    has to be its own root allocation.

    Returns parallel ``(tensors, cuda.core Buffers)`` lists — the tensor
    for local zero/read, the Buffer for the NVSHMEM call.

    ``initial_value`` (0 or 1) seeds the local flag; use ``1`` for
    pre-credited "first use is free" slots (e.g., the ack pool in the
    activation ring buffer).
    """
    assert _initialized, "NVSHMEM not initialized"
    tensors: List[torch.Tensor] = []
    buffers: List[object] = []
    for _ in range(n_flags):
        f = nvshmem.core.interop.torch.bytetensor((8,), dtype=torch.uint8)
        f.zero_()
        if initial_value:
            f[0] = initial_value
        buf, _, _ = nvshmem.core.tensor_get_buffer(f)
        tensors.append(f)
        buffers.append(buf)
    return tensors, buffers


def allocate_slot_pool(n_slots: int, slot_bytes: int) -> List[torch.Tensor]:
    """Allocate ``n_slots`` independently-tracked symmetric uint8 slot
    tensors. Each slot is its own bytetensor — NVSHMEM tracks slots by
    their root handle, and non-zero-offset slices would silently corrupt
    the transfer.
    """
    assert _initialized, "NVSHMEM not initialized"
    return [
        nvshmem.core.interop.torch.bytetensor((slot_bytes,), dtype=torch.uint8)
        for _ in range(n_slots)
    ]


def barrier_all_and_sync() -> None:
    """Stream-ordered cross-PE barrier + host sync. Call once after all
    symmetric allocations land so every PE sees a consistent state
    before the first put/wait.
    """
    assert _initialized
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def put_signal(
    slot: torch.Tensor,
    flag_buf: object,
    dst_pe: int,
    *,
    signal_value: int = 1,
    nbytes: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Stream-ordered NVSHMEM ``put_signal``: copies ``slot`` (or its
    first ``nbytes``) from this PE to the same symmetric tensor on
    ``dst_pe``, then atomically sets the flag at ``flag_buf`` on
    ``dst_pe`` to ``signal_value``. The signal lands strictly after the
    data, so a ``signal_wait`` on the same flag implies all bytes have
    arrived.

    The same tensor is passed for src and dst because NVSHMEM addresses
    symmetric memory by handle; non-zero-offset slices of a slot would
    not be tracked.
    """
    assert _initialized
    if nbytes is not None and nbytes < slot.numel():
        slot = slot[:nbytes]
    nvshmem.core.put_signal(
        slot,
        slot,
        flag_buf,
        signal_value,
        nvshmem.core.SignalOp.SIGNAL_SET,
        dst_pe,
        stream=stream,
    )


def signal_wait(
    flag_buf: object,
    *,
    expected_value: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Stream-ordered NVSHMEM ``signal_wait``: blocks ``stream``
    (GPU-side) until the local flag at ``flag_buf`` reaches
    ``expected_value``. Pair with :func:`put_signal` on the sender.
    """
    assert _initialized
    nvshmem.core.signal_wait(
        flag_buf,
        expected_value,
        nvshmem.core.ComparisonType.CMP_EQ,
        stream=stream,
    )


def maybe_init_nvshmem(group: Optional[dist.ProcessGroup] = None) -> None:
    """Initialize NVSHMEM on the world group (or ``group`` if given).

    Collective: every rank must call. Idempotent — subsequent calls
    are no-ops. Sets up the migration stream and the symmetric flag /
    staging pools.

    Raises ``RuntimeError`` if ``nvshmem.core`` is not importable —
    cross-shard migration depends on it, so this is a hard dependency
    rather than a graceful fallback.
    """
    global _initialized, _my_pe, _n_pes, _migration_stream
    global _flag_pool_size

    if _initialized:
        return
    if not _HAVE_NVSHMEM:
        raise RuntimeError(
            "nvshmem.core is not importable but is required for cross-shard "
            "request migration. Install nvidia-nvshmem-cu12>=3.6.5."
        )

    min_version = (3, 6, 5)
    allow_unsupported = (
        os.environ.get("MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION", "0") == "1"
    )
    pkg_version = metadata.version("nvidia-nvshmem-cu12")
    parts = tuple(int(p) for p in pkg_version.split(".")[:3])
    if parts < min_version and not allow_unsupported:
        raise RuntimeError(
            f"nvidia-nvshmem-cu12=={pkg_version} is older than required "
            f"minimum {'.'.join(map(str, min_version))}. Set "
            "MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION=1 to override."
        )

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before NVSHMEM init")

    try:
        from cuda.core import Device  # type: ignore
    except ImportError:
        # cuda-core <0.x kept Device under the experimental namespace.
        from cuda.core.experimental import Device  # type: ignore

    local_rank = torch.cuda.current_device()
    device = Device(local_rank)
    device.set_current()

    num_ranks = group.size() if group is not None else dist.get_world_size()
    rank_id = group.rank() if group is not None else dist.get_rank()

    if rank_id == 0:
        bcast = [nvshmem.core.get_unique_id()]
    else:
        bcast = [None]
    dist.broadcast_object_list(bcast, src=0, group=group)
    dist.barrier(group=group)

    nvshmem.core.init(
        device=device,
        uid=bcast[0],
        rank=rank_id,
        nranks=num_ranks,
        initializer_method="uid",
    )
    _my_pe = nvshmem.core.my_pe()
    _n_pes = nvshmem.core.n_pes()

    _migration_stream = torch.cuda.Stream()

    _initialized = True

    # Flag + ack pools: lane-partitioned by (src_pe, dst_pe). Pool size
    # is ``ops_per_pair × n_pes²``. Ack pool is pre-credited so the
    # first put on each flag doesn't block waiting for a never-issued
    # prior-round ack.
    global _flag_pool, _flag_pool_buffers
    global _ack_pool, _ack_pool_buffers, _ack_payload
    global _ops_per_pair, _num_lanes, _flag_pool_size
    _ops_per_pair = int(
        os.environ.get("MIGRATION_OPS_PER_PAIR", DEFAULT_OPS_PER_PAIR)
    )
    _num_lanes = _n_pes * _n_pes
    _flag_pool_size = _ops_per_pair * _num_lanes
    _flag_pool, _flag_pool_buffers = allocate_flag_pool(_flag_pool_size)
    _ack_pool, _ack_pool_buffers = allocate_flag_pool(
        _flag_pool_size, initial_value=1
    )
    _ack_payload = allocate_slot_pool(1, 1)[0]
    _ack_payload.zero_()

    global _staging_slots, _staging_slot_bytes, _staging_num_slots
    _staging_slot_bytes = int(
        os.environ.get("MIGRATION_STAGING_SLOT_BYTES", DEFAULT_STAGING_SLOT_BYTES)
    )
    _staging_num_slots = int(
        os.environ.get("MIGRATION_STAGING_NUM_SLOTS", DEFAULT_STAGING_NUM_SLOTS)
    )
    _staging_slots = allocate_slot_pool(_staging_num_slots, _staging_slot_bytes)

    barrier_all_and_sync()
    logger.info(
        "[nvshmem-migration] initialized: PE %d/%d, ops_per_pair=%d, "
        "lanes=%d, flag_pool_size=%d",
        _my_pe,
        _n_pes,
        _ops_per_pair,
        _num_lanes,
        _flag_pool_size,
    )


def lane_for(src_pe: int, dst_pe: int) -> int:
    """Lane index for the ``(src_pe, dst_pe)`` pair.

    Mirrors :func:`activation_transport.lane_for` so the two transports
    share the same lane-encoding discipline. Lane = ``src * n_pes + dst``;
    each lane gets ``_ops_per_pair`` contiguous flag/ack slots in the
    pool. Different pairs use disjoint slot ranges so an ack from
    ``dst_a → src_a`` can't be misread as an ack from ``dst_b → src_b``.
    """
    assert _initialized
    assert 0 <= src_pe < _n_pes and 0 <= dst_pe < _n_pes
    return src_pe * _n_pes + dst_pe


def flag_buffer(slot: int) -> object:
    """Return the cached ``cuda.core.Buffer`` for the given fwd-flag
    slot — ``put_signal`` / ``signal_wait`` accept Buffer, not tensor.
    """
    assert _initialized
    assert 0 <= slot < len(_flag_pool_buffers)
    return _flag_pool_buffers[slot]


def ack_buffer(slot: int) -> object:
    """Return the cached ``cuda.core.Buffer`` for the given ack-flag
    slot. Same indexing space as :func:`flag_buffer`."""
    assert _initialized
    assert 0 <= slot < len(_ack_pool_buffers)
    return _ack_pool_buffers[slot]


def put_slot_with_signal(
    slot_idx: int,
    flag_slot: int,
    dst_pe: int,
    *,
    nbytes: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
    signal_value: int = 1,
) -> None:
    """Src-side migration op: wait for the prior round's ack on this
    flag, then put the staging slot to ``dst_pe`` and signal the fwd
    flag. All three steps are stream-ordered on the migration stream.

    The ack-wait at the top is what makes cross-migration reuse of the
    same flag slot safe: ``dst_pe`` only acks after its scatter of the
    previous round has completed, so by the time the wait returns the
    flag is genuinely free to overwrite. Pre-credited at init
    (``initial_value=1``) so the first put on each flag doesn't block.
    """
    assert _initialized
    s = stream or _migration_stream

    # Wait for dst's ack on this flag slot's previous round. Pre-credited
    # at init so first use is free.
    signal_wait(ack_buffer(flag_slot), expected_value=1, stream=s)
    # Reset the local ack so the next round's ack-put causes a 0 → 1
    # transition. The reset MUST run on the migration stream — without
    # the explicit context, ``.zero_()`` falls back to the default
    # stream and races with the put below.
    with torch.cuda.stream(s):
        _ack_pool[flag_slot].zero_()

    put_signal(
        staging_slot(slot_idx),
        flag_buffer(flag_slot),
        dst_pe,
        signal_value=signal_value,
        nbytes=nbytes,
        stream=s,
    )


def wait_slot_signal(
    flag_slot: int,
    *,
    expected_value: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Dst-side wait: block the migration stream until the local fwd
    flag reaches ``expected_value``, then reset it to 0 (stream-
    ordered) for the next round.

    Caller is responsible for scattering the staged slot out *before*
    calling :func:`send_ack` — sending the ack frees src to overwrite
    the slot, so the scatter must finish first.
    """
    assert _initialized
    s = stream or _migration_stream
    signal_wait(flag_buffer(flag_slot), expected_value=expected_value, stream=s)
    # Reset must be stream-ordered after the signal_wait so the next
    # round's put_signal sees a clean 0 → 1 transition.
    with torch.cuda.stream(s):
        _flag_pool[flag_slot].zero_()


def send_ack(
    flag_slot: int,
    src_pe: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Dst-side ack: signal ``src_pe`` that ``flag_slot`` is safe to
    reuse. Issued after the destination has scattered the staged
    slot out — sending the ack lets src overwrite the slot, so the
    scatter must complete first (which it does, because both are
    stream-ordered on the migration stream).

    Uses a dedicated 1-byte symmetric payload (:data:`_ack_payload`),
    not a slice of the staging slot: NVSHMEM tracks slot tensors by
    their root handle and would interpret a slice as the full slot,
    racing with src's pending fill on the same slot.
    """
    put_signal(
        _ack_payload,
        ack_buffer(flag_slot),
        src_pe,
        stream=stream or _migration_stream,
    )


def staging_slot(slot_idx: int) -> torch.Tensor:
    """Return the symmetric ``uint8`` slot tensor for ``slot_idx``."""
    assert _initialized
    assert 0 <= slot_idx < len(_staging_slots), (
        f"slot {slot_idx} out of range [0, {len(_staging_slots)})"
    )
    return _staging_slots[slot_idx]


class StagingArena:
    """Per-migration linear allocator over the staging slot pool.

    Both src and dst walk the migration ops in the same order, so
    they end up assigning the same slot index to each op without
    explicit coordination — the invariant that makes
    ``put(slot[i], slot[i], dst_pe)`` land in the right place.

    Cross-migration safety relies on stream ordering: every migration
    starts a fresh arena at slot 0, so two migrations both write into
    slot 0 first. This is only safe because all gather-into-slot copies
    and ``put_slot_with_signal`` calls submit on the singleton
    :func:`migration_stream`, which serializes them GPU-side — migration
    N's put completes before migration N+1's copy_ overwrites the slot.
    Submitting migrations on different streams (or off-stream host-side
    writes into the slot) would race and corrupt the transfer.
    """

    def __init__(self):
        self._next = 0

    def take(self, nbytes: int) -> int:
        """Reserve a slot for ``nbytes`` bytes; returns the slot index.
        Raises if the requested size exceeds the slot capacity or the
        pool is exhausted."""
        if nbytes > _staging_slot_bytes:
            raise RuntimeError(
                f"NVSHMEM migration: op size {nbytes} > slot size "
                f"{_staging_slot_bytes}; raise MIGRATION_STAGING_SLOT_BYTES."
            )
        if self._next >= _staging_num_slots:
            raise RuntimeError(
                f"NVSHMEM migration: staging slot pool exhausted "
                f"({self._next}/{_staging_num_slots}); raise "
                "MIGRATION_STAGING_NUM_SLOTS."
            )
        idx = self._next
        self._next += 1
        return idx


class FlagArena:
    """Per-migration allocator over the lane-partitioned flag/ack pool.

    For each ``(src_pe, dst_pe)`` pair, hands out a unique flag-slot
    index in the lane reserved for that pair. Both src and dst call
    ``take(src_pe, dst_pe)`` once per op while walking the migration
    plan in identical order, so they pick the same flag index for the
    same op without coordination — the same invariant the
    :class:`StagingArena` relies on, applied to flags.

    Lane partitioning makes the ack handshake unambiguous: an ack from
    ``dst_pe → src_pe`` on slot ``S`` refers to *this* pair's prior
    use of slot ``S``, never to some other pair's flag that happened
    to hash to the same global index. Within a lane, the per-pair
    counter wraps modulo :data:`_ops_per_pair`; the ack handshake
    inside :func:`put_slot_with_signal` is what makes wrap-around safe
    — the src signal_waits on the previous round's ack before
    reissuing on the wrapped index.

    No ``MAX_OPS_PER_REQ`` semantics: flag identity is per-pair
    per-op, not per-request. Multiple ops in one bundle, multiple
    bundles in one migration, and multiple migrations across time
    all share the same per-pair lane and recycle via acks rather
    than via a key-based collision-avoidance scheme.
    """

    def __init__(self):
        # Per-(src_pe, dst_pe) op counter for this migration. Resets
        # on each new arena; cross-migration safety is the ack
        # handshake's responsibility, not the arena's.
        self._counters: Dict[Tuple[int, int], int] = {}

    def take(self, src_pe: int, dst_pe: int) -> int:
        """Reserve the next flag-slot in the ``(src_pe, dst_pe)`` lane;
        returns the absolute pool index.

        Wraps modulo :data:`_ops_per_pair` within the lane. Wrap-around
        is correct because :func:`put_slot_with_signal`'s ack-wait
        serializes a new put on a slot with its previous round's
        completion on the same slot.
        """
        assert _initialized
        key = (src_pe, dst_pe)
        n = self._counters.get(key, 0)
        self._counters[key] = n + 1
        lane = lane_for(src_pe, dst_pe)
        return lane * _ops_per_pair + (n % _ops_per_pair)


# ---- Test hooks -----------------------------------------------------------
#
# The NVSHMEM init path requires a real GPU + multi-rank distributed world.
# These hooks let unit tests exercise the pure-Python bookkeeping (lane
# encoding, FlagArena counter discipline) without those dependencies.
# Mirror activation_transport's ``_init_state_for_test`` / ``_reset_state_for_test``.


def _init_state_for_test(*, n_pes: int, ops_per_pair: int = 16) -> None:
    """Set up the module state ``FlagArena`` / ``lane_for`` need, without
    allocating any NVSHMEM buffers or starting the migration stream.
    Tests that only exercise the per-pair flag allocator use this in
    place of :func:`maybe_init_nvshmem`.

    Calling :func:`maybe_init_nvshmem` after this raises (the module
    already thinks it's initialized); call :func:`_reset_state_for_test`
    to flip back.
    """
    global _initialized, _n_pes, _ops_per_pair, _num_lanes, _flag_pool_size
    if _initialized:
        raise RuntimeError(
            "nvshmem_migration already initialized; call "
            "_reset_state_for_test() first."
        )
    _n_pes = n_pes
    _ops_per_pair = ops_per_pair
    _num_lanes = n_pes * n_pes
    _flag_pool_size = _ops_per_pair * _num_lanes
    _initialized = True


def _reset_state_for_test() -> None:
    """Tear down whatever ``_init_state_for_test`` set up. Idempotent;
    safe to call from a pytest fixture teardown."""
    global _initialized, _my_pe, _n_pes, _migration_stream
    global _flag_pool, _flag_pool_buffers
    global _ack_pool, _ack_pool_buffers, _ack_payload
    global _ops_per_pair, _num_lanes, _flag_pool_size
    global _staging_slots, _staging_slot_bytes, _staging_num_slots
    _initialized = False
    _my_pe = -1
    _n_pes = -1
    _migration_stream = None
    _flag_pool = []
    _flag_pool_buffers = []
    _ack_pool = []
    _ack_pool_buffers = []
    _ack_payload = None
    _ops_per_pair = 0
    _num_lanes = 0
    _flag_pool_size = 0
    _staging_slots = []
    _staging_slot_bytes = 0
    _staging_num_slots = 0
