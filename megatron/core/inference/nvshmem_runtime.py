# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared NVSHMEM runtime layer.

Process-wide NVSHMEM init (one PE identity per rank), symmetric pool
builders, and the ``put_signal`` / ``signal_wait`` wrappers used by
every NVSHMEM-based transport. The two transports built on top —
:mod:`migration_transport` (bulk request state) and
:mod:`activation_transport` (per-layer hidden states) — own their
own pools, streams, and addressing; this module owns only the bits
they share.

KV / Mamba buffers and activation slots must themselves live on the
symmetric heap (allocated via :func:`nvshmem.core.interop.torch.bytetensor`
or the like) so that put_signal can write them directly. That means
:func:`maybe_init_nvshmem` must run *before* engine construction; both
transports' own init functions call this first as a hard prerequisite.
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


def is_initialized() -> bool:
    """Whether :func:`maybe_init_nvshmem` has completed on this PE."""
    return _initialized


def my_pe() -> int:
    """This rank's NVSHMEM PE id. Equals the torch global rank in our setup."""
    assert _initialized, "NVSHMEM not initialized"
    return _my_pe


def n_pes() -> int:
    """Total PEs in the NVSHMEM team. Equals the torch world size in our setup."""
    assert _initialized, "NVSHMEM not initialized"
    return _n_pes


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
    pre-credited "first use is free" slots (e.g., ack pools).
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
    are no-ops. Allocates no transport pools; each transport
    (:mod:`migration_transport`, :mod:`activation_transport`) handles
    its own pool init after this returns.

    Raises ``RuntimeError`` if ``nvshmem.core`` is not importable —
    NVSHMEM-based transports are a hard dependency, not a graceful
    fallback.
    """
    global _initialized, _my_pe, _n_pes

    if _initialized:
        return
    if not _HAVE_NVSHMEM:
        raise RuntimeError(
            "nvshmem.core is not importable but is required for cross-shard "
            "request migration / activation transport. Install "
            "nvidia-nvshmem-cu12>=3.6.5."
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
    _initialized = True

    logger.info("[nvshmem-runtime] initialized: PE %d/%d", _my_pe, _n_pes)
