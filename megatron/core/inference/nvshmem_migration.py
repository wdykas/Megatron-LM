# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Direct NVSHMEM transport for cross-shard request migration.

One-sided ``put_signal`` from src lands KV bytes plus a completion flag
in a single op; dst's ``signal_wait`` on the same flag implies all bytes
have arrived. Both sides issue everything on a dedicated migration
stream, so neither engine has to pause its run loop.

KV / Mamba buffers themselves are *not* in the symmetric heap; only the
fixed-size symmetric staging slots (one ``bytetensor`` per slot,
allocated up front in :func:`maybe_init_nvshmem`) and per-op flags are.
The migration handler stages each op into a slot, ``put_signal``s the
slot to the matching offset on the destination's symmetric region, and
the destination scatters the slot back into its local buffer.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from typing import List, Optional, Tuple

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
# Each flag is its own 8-byte ``bytetensor`` (NVSHMEM signal_var
# requires Buffer >= 8 bytes). The pool stores both the tensor view
# (for local read/clear) and the Buffer (for ``put_signal`` /
# ``signal_wait``).
_flag_pool: List[torch.Tensor] = []
_flag_pool_buffers: List[object] = []  # cuda.core Buffer per slot
_flag_pool_size: int = 0
_staging_slots: List[torch.Tensor] = []  # pool of symmetric uint8[SLOT_BYTES]
_staging_slot_bytes: int = 0
_staging_num_slots: int = 0


# 1<<14 = 16384 slots, 128 KB per PE. Aliasing under
# ``flag_slot_for(req_id * MAX_OPS_PER_REQ + op_idx)`` is bounded by
# how many flags can be in-flight simultaneously — with the singleton
# migration stream serializing migrations and at most
# ``batch_size * MAX_OPS_PER_REQ`` flags unresolved per migration, even
# 4096 slots have headroom. 16384 keeps a comfortable margin and
# initializes in seconds (each flag is its own NVSHMEM symmetric
# allocation, so a 1M pool stalled startup for minutes). Tunable via
# env.
DEFAULT_FLAG_POOL_SIZE = 1 << 14

# NVSHMEM only tracks the full tensor handle returned by ``bytetensor``;
# slices with non-zero offsets aren't tracked, so we pre-allocate a
# fixed pool of independently-tracked slot tensors. Slot count scales
# with ``batch × overlap_pairs`` for both KV and mamba (mamba's four
# block kinds pack into one slot per overlap in the migration handler).
DEFAULT_STAGING_SLOT_BYTES = 16 * 1024 * 1024  # 16 MB per slot
DEFAULT_STAGING_NUM_SLOTS = 256  # 4 GB total per PE


def migration_stream() -> torch.cuda.Stream:
    """Dedicated CUDA stream for migration puts so they don't serialize
    against the engine's compute stream."""
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

    global _flag_pool, _flag_pool_buffers
    _flag_pool_size = int(os.environ.get("MIGRATION_FLAG_POOL_SIZE", DEFAULT_FLAG_POOL_SIZE))
    _flag_pool, _flag_pool_buffers = allocate_flag_pool(_flag_pool_size)

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
        "[nvshmem-migration] initialized: PE %d/%d, flag_pool_size=%d",
        _my_pe,
        _n_pes,
        _flag_pool_size,
    )


def flag_slot_for(key: int) -> int:
    """Deterministic flag-slot index from ``key``.

    Both src and dst compute the slot index from a value they both
    see (typically ``request_id * MAX_OPS_PER_REQ + op_index``), so
    no per-PE counter sync is needed even when only a subset of dst
    replicas participates in a given migration. Wraps around the
    pool — caller should ensure ``key`` doesn't recycle within the
    pool's TTL.
    """
    assert _initialized
    return key % _flag_pool_size


def flag_buffer(slot: int) -> object:
    """Return the cached ``cuda.core.Buffer`` for the given flag slot —
    ``put_signal`` / ``signal_wait`` accept Buffer, not tensor."""
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
    sets the flag to ``signal_value`` on ``dst_pe`` in one stream-
    ordered NVSHMEM op. The signal arrives strictly after the data, so
    a ``signal_wait`` on the same flag implies all bytes have landed.
    """
    put_signal(
        staging_slot(slot_idx),
        flag_buffer(flag_slot),
        dst_pe,
        signal_value=signal_value,
        nbytes=nbytes,
        stream=stream or _migration_stream,
    )


def wait_slot_signal(
    flag_slot: int,
    *,
    expected_value: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Stream-side wait: blocks the migration stream (GPU-side) until
    the local flag reaches ``expected_value``. Pair with
    :func:`put_slot_with_signal` on src.
    """
    signal_wait(
        flag_buffer(flag_slot),
        expected_value=expected_value,
        stream=stream or _migration_stream,
    )


def reset_flag(slot: int) -> None:
    """Reset the local flag to 0 so it can be reused on the next migration."""
    assert _initialized
    _flag_pool[slot].zero_()


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
