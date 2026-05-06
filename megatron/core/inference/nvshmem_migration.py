# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Direct NVSHMEM transport for cross-shard request migration.

Replaces the collective NCCL/NVSHMEM ``CopyService`` path used for KV
migration with raw one-sided ``nvshmem.core.put`` calls and a flag-
based completion handshake. Designed so that **neither** the src nor
dst engine has to pause its run loop:

- Src issues all KV puts on a dedicated migration stream, then a
  single ``quiet()`` (drains the puts to the remote symmetric heap),
  then a small flag put. The handler returns immediately. The engine
  later polls a CUDA event tied to the post-quiet point and lazily
  frees the migrated blocks once the event fires.
- Dst's handler allocates fresh KV blocks, records a "pending
  migration" entry containing the dst block ids, the flag offset, and
  the bundle metadata, and returns immediately. The engine polls the
  flag each tick; when raised, the request is promoted into the
  active set and decode resumes naturally.

NVSHMEM ordering: puts to the same dst PE issued from the same src PE
on the same stream are observed in source order at the dst. So data-
puts followed by a flag-put guarantee that, when the flag is seen on
dst, the data is also visible. ``quiet`` after the data puts ensures
local completion before the flag put is enqueued.

Symmetric heap: the engine's KV ``memory_buffer`` (and Mamba state
buffers) are allocated via ``nvshmem.core.interop.torch.bytetensor``
sized to ``MAX(local_size)`` across all PEs. Smaller-shard ranks
waste the trailing bytes; in exchange, the migration code can issue
``put`` directly on KV slices without staging.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from typing import List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


try:
    import nvshmem.core

    _HAVE_NVSHMEM = True
except ImportError:  # pragma: no cover
    _HAVE_NVSHMEM = False


# Module-level singleton state.
#
# We initialize NVSHMEM exactly once per process (the first
# :func:`maybe_init_nvshmem` call) and reuse it for the lifetime of
# the inference interface. Re-init is not supported by NVSHMEM, so
# subsequent callers see ``_initialized=True`` and return.
_initialized: bool = False
_my_pe: int = -1
_n_pes: int = -1
_migration_stream: Optional[torch.cuda.Stream] = None
# Each flag is its own 8-byte ``bytetensor`` (NVSHMEM signal_var
# requires Buffer >= 8 bytes). The pool stores both the tensor view
# (for local read/clear) and the Buffer (for ``put_signal`` /
# ``signal_wait``).
_flag_pool: List[torch.Tensor] = []
_flag_pool_buffers: List[object] = []  # cuda.core Buffer per slot
_flag_pool_next: int = 0
_flag_pool_size: int = 0
_staging_slots: List[torch.Tensor] = []  # pool of symmetric uint8[SLOT_BYTES]
_staging_slot_bytes: int = 0
_staging_num_slots: int = 0


# Number of int32 flags pre-allocated in the symmetric heap. Each
# in-flight migration consumes one flag; we cycle through the pool
# (slot reuse is safe because the dst engine clears the flag after
# consuming it). 4096 = ample for any realistic batch of concurrent
# migrations and only 16 KB per PE.
DEFAULT_FLAG_POOL_SIZE = 4096

# Staging slot pool for KV/Mamba slices. NVSHMEM only tracks the full
# tensor handle returned by ``bytetensor`` — slices with non-zero
# offsets aren't tracked, so we can't carve a single big buffer into
# arbitrary ranges. Instead we pre-allocate a fixed pool of
# independently-tracked slot tensors, each of ``DEFAULT_STAGING_SLOT_BYTES``
# bytes, and hand them out per-op. Tunable via env.
DEFAULT_STAGING_SLOT_BYTES = 16 * 1024 * 1024  # 16 MB per slot
DEFAULT_STAGING_NUM_SLOTS = 128  # 2 GB total per PE


def have_nvshmem() -> bool:
    """True iff ``nvshmem.core`` was importable at module load."""
    return _HAVE_NVSHMEM


def is_initialized() -> bool:
    return _initialized


def my_pe() -> int:
    assert _initialized, "NVSHMEM not initialized"
    return _my_pe


def n_pes() -> int:
    assert _initialized, "NVSHMEM not initialized"
    return _n_pes


def migration_stream() -> torch.cuda.Stream:
    """Dedicated CUDA stream for migration puts. All KV/state/flag
    puts go through this stream so they don't serialize against the
    engine's compute stream."""
    assert _initialized, "NVSHMEM not initialized"
    assert _migration_stream is not None
    return _migration_stream


def maybe_init_nvshmem(group: Optional[dist.ProcessGroup] = None) -> None:
    """Initialize NVSHMEM on the world group (or ``group`` if given).

    Collective: every rank must call. Idempotent — subsequent calls
    are no-ops. Sets up the migration stream and the symmetric flag
    pool. Safe to call before allocating any symmetric KV memory.

    Raises ``RuntimeError`` if ``nvshmem.core`` is not importable —
    cross-shard migration is required for the current heterogeneous-
    inference auto-disagg path, so this is a hard dependency rather
    than a graceful fallback.
    """
    global _initialized, _my_pe, _n_pes, _migration_stream
    global _flag_pool_buffer, _flag_pool_size, _flag_pool_next

    if _initialized:
        return
    if not _HAVE_NVSHMEM:
        raise RuntimeError(
            "nvshmem.core is not importable but is required for cross-shard "
            "request migration. Install nvidia-nvshmem-cu12>=3.6.5."
        )

    # Version guard mirrors the resharding init path.
    min_version = (3, 6, 5)
    allow_unsupported = (
        os.environ.get("MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION", "0") == "1"
    )
    try:
        pkg_version = metadata.version("nvidia-nvshmem-cu12")
        parts = tuple(int(p) for p in pkg_version.split(".")[:3])
        if parts < min_version and not allow_unsupported:
            raise RuntimeError(
                f"nvidia-nvshmem-cu12=={pkg_version} is older than required "
                f"minimum {'.'.join(map(str, min_version))}. Set "
                "MEGATRON_NVSHMEM_ALLOW_UNSUPPORTED_VERSION=1 to override."
            )
    except metadata.PackageNotFoundError:
        logger.warning("Could not determine nvidia-nvshmem-cu12 version.")

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before NVSHMEM init")

    from cuda.core.experimental import Device  # type: ignore

    local_rank = torch.cuda.current_device()
    device = Device(local_rank)
    device.set_current()

    num_ranks = group.size() if group is not None else dist.get_world_size()
    rank_id = group.rank() if group is not None else dist.get_rank()

    # Broadcast unique id from rank 0 → all.
    uniqueid = nvshmem.core.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.core.get_unique_id()
        bcast = [uniqueid]
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

    # Flip ``_initialized`` *before* the flag pool / staging buffer
    # allocations so the ``symmetric_empty`` / ``symmetric_zeros``
    # helpers don't see a half-initialized module. NVSHMEM proper is
    # already up at this point — these allocations only need the
    # module-level state set.
    _initialized = True

    # Allocate the signal flag pool. Each flag is an 8-byte symmetric
    # ``bytetensor`` (NVSHMEM ``signal_var`` requires Buffer >= 8 bytes).
    # Each is individually tracked so we can call ``put_signal`` /
    # ``signal_wait`` on it. We cache the corresponding ``Buffer``
    # handles too (``signal_var`` arg expects Buffer, not tensor).
    global _flag_pool, _flag_pool_buffers
    _flag_pool_size = int(os.environ.get("MIGRATION_FLAG_POOL_SIZE", DEFAULT_FLAG_POOL_SIZE))
    _flag_pool = []
    _flag_pool_buffers = []
    for _ in range(_flag_pool_size):
        flag = nvshmem.core.interop.torch.bytetensor((8,), dtype=torch.uint8)
        flag.zero_()
        buf, _, _ = nvshmem.core.tensor_get_buffer(flag)
        _flag_pool.append(flag)
        _flag_pool_buffers.append(buf)
    _flag_pool_next = 0

    # Allocate the staging slot pool. Each slot is an independent
    # ``bytetensor`` because NVSHMEM only tracks the full tensor handle
    # returned by ``bytetensor``; slices with non-zero offsets aren't
    # accepted by ``put``. Collective per-slot.
    global _staging_slots, _staging_slot_bytes, _staging_num_slots
    _staging_slot_bytes = int(
        os.environ.get("MIGRATION_STAGING_SLOT_BYTES", DEFAULT_STAGING_SLOT_BYTES)
    )
    _staging_num_slots = int(
        os.environ.get("MIGRATION_STAGING_NUM_SLOTS", DEFAULT_STAGING_NUM_SLOTS)
    )
    _staging_slots = []
    for _ in range(_staging_num_slots):
        slot = nvshmem.core.interop.torch.bytetensor(
            (_staging_slot_bytes,), dtype=torch.uint8
        )
        _staging_slots.append(slot)

    # Final barrier so every PE sees the flag pool before any migration.
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    logger.info(
        "[nvshmem-migration] initialized: PE %d/%d, flag_pool_size=%d",
        _my_pe,
        _n_pes,
        _flag_pool_size,
    )


def _max_bytes_across_ranks(local_bytes: int) -> int:
    """Collective: returns ``max(local_bytes)`` across the world."""
    t = torch.tensor([local_bytes], dtype=torch.int64, device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def symmetric_empty(shape, *, dtype: torch.dtype) -> torch.Tensor:
    """Allocate a torch tensor of the given shape/dtype backed by the
    NVSHMEM symmetric heap.

    Collective: all PEs must call with consistent shape/dtype semantics.
    The actual byte allocation is the **max** required across PEs — every
    PE gets a buffer of that size, then views the prefix as ``shape``.
    Lets heterogeneous shards share one symmetric region without padding
    the on-device shape.

    Returns a tensor of ``shape`` (this PE's view); the underlying
    bytetensor handle is held in module state to keep it alive.
    """
    assert _initialized, "call maybe_init_nvshmem() before symmetric_empty"

    elem_size = torch.empty((), dtype=dtype).element_size()
    local_numel = 1
    for d in shape:
        local_numel *= d
    local_bytes = local_numel * elem_size
    max_bytes = _max_bytes_across_ranks(local_bytes)

    # ``bytetensor((N,), dtype=torch.uint8)`` is collective and allocates
    # an N-byte symmetric region on every PE. We allocate the world max
    # so heterogeneous local sizes can share a single symmetric region.
    bytetensor = nvshmem.core.interop.torch.bytetensor(
        (max_bytes,), dtype=torch.uint8
    )
    # View the prefix as the local shape/dtype. Pin the underlying byte
    # tensor on the typed view so it isn't garbage-collected.
    typed = bytetensor[:local_bytes].view(dtype).reshape(shape)
    typed._nvshmem_byte_handle = bytetensor  # noqa: SLF001 — intentional pin
    return typed


def symmetric_zeros(shape, *, dtype: torch.dtype) -> torch.Tensor:
    """Like :func:`symmetric_empty` but zero-initialized, with a barrier
    before returning so peer reads cannot observe uninitialized values.
    """
    t = symmetric_empty(shape, dtype=dtype)
    t.zero_()
    torch.cuda.synchronize()
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return t


def acquire_flag_slot() -> int:
    """Reserve one slot in the symmetric flag pool.

    **Must be called collectively on every PE** so the per-PE
    counters stay aligned — both src and dst need the same slot
    index for a given migration. Use :func:`reset_flag_pool` to wipe
    the counter at any synchronization point if PEs got out of step
    (e.g. in tests where some ranks skip a phase).
    """
    global _flag_pool_next
    assert _initialized
    slot = _flag_pool_next
    _flag_pool_next = (_flag_pool_next + 1) % _flag_pool_size
    return slot


def reset_flag_pool() -> None:
    """Reset the flag pool counter to 0. Local-only — caller must
    ensure all PEs invoke this together (typically as part of a
    higher-level synchronization point) so counters stay aligned.
    """
    global _flag_pool_next
    assert _initialized
    _flag_pool_next = 0


def flag_tensor(slot: int) -> torch.Tensor:
    """Return the per-slot symmetric int32 flag tensor (uint8-backed,
    4 bytes). Each slot is independently tracked by NVSHMEM, so it
    can be passed directly to ``put``.
    """
    assert _initialized
    assert 0 <= slot < len(_flag_pool)
    return _flag_pool[slot]


def put_tensor_async(
    src_view: torch.Tensor,
    dst_pe: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Issue ``nvshmem.core.put`` from this PE's local symmetric view
    to the same offset on ``dst_pe``. Both ``src_view`` and the
    implicit dst region must be backed by the symmetric heap (i.e.
    obtained via :func:`symmetric_empty` / :func:`symmetric_zeros`
    or a view thereof).

    Non-blocking on the issuing PE: enqueues the put on the migration
    stream and returns immediately.
    """
    assert _initialized
    s = stream or _migration_stream
    # nvshmem.core.put(dst, src, dst_pe, stream) — dst is the *local
    # view* of where the bytes should land; NVSHMEM uses the symmetric
    # heap relationship to find the actual address on dst_pe.
    nvshmem.core.put(src_view, src_view, dst_pe, stream=s)


def put_to_dst_view(
    dst_view: torch.Tensor,
    src_view: torch.Tensor,
    dst_pe: int,
    *,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Issue ``nvshmem.core.put`` where the dst region differs from
    the src region (different offsets / shapes within the symmetric
    heap). ``dst_view`` is a *local* view of where the data should
    land on the remote PE; the local PE doesn't actually write to
    its own copy of ``dst_view``.

    Both ``src_view`` and ``dst_view`` must be backed by the
    symmetric heap.
    """
    assert _initialized
    assert dst_view.numel() * dst_view.element_size() == src_view.numel() * src_view.element_size(), (
        "dst_view and src_view must have the same byte size"
    )
    s = stream or _migration_stream
    nvshmem.core.put(dst_view, src_view, dst_pe, stream=s)


def quiet_on_migration_stream() -> None:
    """``nvshmem.core.quiet`` on the migration stream — drains all
    pending puts so the next stream-ordered operation observes
    remote completion.
    """
    assert _initialized
    nvshmem.core.quiet(stream=_migration_stream)


def flag_buffer(slot: int) -> object:
    """Return the cached ``cuda.core.Buffer`` for ``flag_tensor(slot)``.
    Required by ``nvshmem.core.put_signal`` / ``signal_wait`` whose
    ``signal_var`` argument expects a Buffer, not a tensor."""
    assert _initialized
    assert 0 <= slot < len(_flag_pool_buffers)
    return _flag_pool_buffers[slot]


def put_slot_with_signal(
    slot_idx: int,
    flag_slot: int,
    dst_pe: int,
    *,
    nbytes: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
    signal_value: int = 1,
) -> None:
    """Atomic put + signal: copies the staging slot to ``dst_pe`` and
    sets ``flag_tensor(flag_slot)`` to ``signal_value`` on ``dst_pe``
    in one stream-ordered NVSHMEM op. The signal arrives strictly
    after the data on the destination, so a ``signal_wait`` on the
    same flag implies all bytes have landed.
    """
    assert _initialized
    s = stream or _migration_stream
    slot = staging_slot(slot_idx)
    if nbytes is not None and nbytes < slot.numel():
        slot = slot[:nbytes]
    flag_buf = flag_buffer(flag_slot)
    nvshmem.core.put_signal(
        slot,
        slot,
        flag_buf,
        signal_value,
        nvshmem.core.SignalOp.SIGNAL_SET,
        dst_pe,
        stream=s,
    )


def wait_slot_signal(
    flag_slot: int,
    *,
    expected_value: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Stream-side wait: blocks the migration stream (GPU-side) until
    the local ``flag_tensor(flag_slot)`` reaches ``expected_value``.
    Pair with :func:`put_slot_with_signal` on src.
    """
    assert _initialized
    s = stream or _migration_stream
    flag_buf = flag_buffer(flag_slot)
    nvshmem.core.signal_wait(
        flag_buf,
        expected_value,
        nvshmem.core.ComparisonType.CMP_EQ,
        stream=s,
    )


def reset_flag(slot: int) -> None:
    """Reset ``flag_tensor(slot) = 0`` locally so it can be reused
    on the next migration."""
    assert _initialized
    _flag_pool[slot].zero_()


def staging_slot(slot_idx: int) -> torch.Tensor:
    """Return the symmetric ``uint8`` slot tensor for ``slot_idx``.

    Each slot is an independent ``bytetensor`` allocation; passing it
    as both args to ``nvshmem.core.put`` writes the bytes to the same
    slot on the destination PE. Slot size is fixed at init
    (``MIGRATION_STAGING_SLOT_BYTES``).
    """
    assert _initialized
    assert 0 <= slot_idx < len(_staging_slots), (
        f"slot {slot_idx} out of range [0, {len(_staging_slots)})"
    )
    return _staging_slots[slot_idx]


def staging_slot_bytes() -> int:
    return _staging_slot_bytes


def staging_num_slots() -> int:
    return _staging_num_slots


class StagingArena:
    """Per-migration linear allocator over the staging slot pool.

    Both src and dst walk the migration ops in the same order, so
    they end up assigning the same slot index to each op without
    explicit coordination — the invariant that makes
    ``put(slot[i], slot[i], dst_pe)`` land in the right place.
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


def put_slot_to_dst(
    slot_idx: int,
    dst_pe: int,
    *,
    nbytes: Optional[int] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Put this PE's slot[``slot_idx``] to the same slot on ``dst_pe``.

    If ``nbytes`` is given, only the first ``nbytes`` bytes are sent
    (the rest of the slot is unused). NVSHMEM tracks the slot tensor
    as a whole; passing a prefix-slice with starting offset 0 is
    accepted by the runtime.
    """
    assert _initialized
    s = stream or _migration_stream
    slot = staging_slot(slot_idx)
    if nbytes is not None and nbytes < slot.numel():
        slot = slot[:nbytes]  # offset-0 prefix is tracked by NVSHMEM
    nvshmem.core.put(slot, slot, dst_pe, stream=s)
