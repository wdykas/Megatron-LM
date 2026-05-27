# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM transport for cross-shard request migration (bulk KV / Mamba state).

One-sided ``put_signal`` from src lands KV bytes plus a completion
flag in a single op; dst's ``signal_wait`` on the same flag implies
all bytes have arrived. Both sides issue everything on a dedicated
migration stream, so neither engine has to pause its run loop.

KV / Mamba buffers themselves are *not* in the symmetric heap; only
the fixed-size symmetric staging slots (one ``bytetensor`` per slot,
allocated up front in :func:`maybe_init_migration_transport`) and per-op
flags are. The migration handler stages each op into a slot,
``put_signal``s the slot to the matching offset on the destination's
symmetric region, and the destination scatters the slot back into
its local buffer.

Cross-migration safety: the flag pool is partitioned by ``(src_pe,
dst_pe)`` lane (so different pairs can't alias on the same dst flag)
and reuse within a lane is serialized by an ack handshake — src
``signal_wait``s on a per-flag ack before re-issuing a put; dst raises
the ack after its scatter completes. This is the same discipline
:mod:`activation_transport` uses, applied to the bulk-migration traffic
shape (one substrate, two access patterns; identical synchronization
invariant).

Shared NVSHMEM primitives (init, pool builders, put/wait wrappers)
live in :mod:`nvshmem_runtime` and are imported here.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.inference import nvshmem_runtime as _rt

logger = logging.getLogger(__name__)


_initialized: bool = False
_migration_stream: Optional[torch.cuda.Stream] = None

# Active-pair registry. Migration only happens between specific shard
# pairs (defined by registered migration policies), so the flag/ack
# pool is sized for the actual pairs in use — not the full ``n_pes²``
# Cartesian product, which would scale quadratically with cluster
# size and dominate startup at 32+ GPUs (each flag is its own NVSHMEM
# symmetric allocation, and symmetric alloc is a collective).
#
# ``_active_pairs[i] == (src_pe, dst_pe)`` is the pair owning lane ``i``.
# ``_pair_to_lane_idx`` is the inverse lookup used by :func:`lane_for`.
_active_pairs: List[Tuple[int, int]] = []
_pair_to_lane_idx: Dict[Tuple[int, int], int] = {}
_pools_realized: bool = False

# Fwd-flag pool: ``ops_per_pair × len(active_pairs)`` slots, partitioned
# by ``(src_pe, dst_pe)`` lane. Set by put_signal on the dst PE; observed
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

# Pool dimensions. Pool size = ops_per_pair × len(active_pairs).
_ops_per_pair: int = 0
_flag_pool_size: int = 0  # = _ops_per_pair × len(_active_pairs) (derived)

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
    assert _initialized, "migration_transport not initialized"
    assert _migration_stream is not None
    return _migration_stream


def maybe_init_migration_transport(
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Initialize the migration transport's runtime, stream, and
    pair-independent staging slots. **Does not** allocate the flag/ack
    pool — that's deferred to :func:`realize_migration_pools` once the
    set of communicating PE pairs is known.

    Calls :func:`nvshmem_runtime.maybe_init_nvshmem` first so the
    shared runtime is up. Collective: every rank must call.
    Idempotent — subsequent calls are no-ops.
    """
    global _initialized, _migration_stream

    if _initialized:
        return
    _rt.maybe_init_nvshmem(group=group)

    _migration_stream = torch.cuda.Stream()
    _initialized = True

    # Staging slots are pair-independent — the StagingArena allocator
    # is linear over the whole pool, not partitioned by lane. Allocate
    # here so the migration handler can run gather/scatter even before
    # flag/ack pools exist (rare but possible during setup races).
    global _staging_slots, _staging_slot_bytes, _staging_num_slots
    _staging_slot_bytes = int(
        os.environ.get("MIGRATION_STAGING_SLOT_BYTES", DEFAULT_STAGING_SLOT_BYTES)
    )
    _staging_num_slots = int(
        os.environ.get("MIGRATION_STAGING_NUM_SLOTS", DEFAULT_STAGING_NUM_SLOTS)
    )
    _staging_slots = _rt.allocate_slot_pool(_staging_num_slots, _staging_slot_bytes)

    _rt.barrier_all_and_sync()
    logger.info(
        "[migration-transport] runtime + staging initialized; "
        "flag/ack pools deferred to realize_migration_pools()"
    )


def register_migration_pair(src_pe: int, dst_pe: int) -> None:
    """Reserve a flag-pool lane for the ``(src_pe, dst_pe)`` pair.

    Must be called before :func:`realize_migration_pools` (which is
    when the pools are actually allocated). Idempotent — calling
    twice with the same pair is a no-op.

    Cross-rank consistency: callers must register the *same* set of
    pairs on every rank, in the same order, so each rank assigns the
    same lane index to each pair. The migration policy registration
    path in :mod:`multi_shard` enforces this because it's already
    collective.
    """
    if _pools_realized:
        raise RuntimeError(
            f"migration pools already realized; cannot register new pair "
            f"({src_pe}, {dst_pe}). Register all pairs before "
            f"realize_migration_pools()."
        )
    pair = (src_pe, dst_pe)
    if pair in _pair_to_lane_idx:
        return
    _pair_to_lane_idx[pair] = len(_active_pairs)
    _active_pairs.append(pair)


def register_migration_shard_pair(
    src_pes: Iterable[int], dst_pes: Iterable[int]
) -> None:
    """Register every ``(s, d)`` for ``s in src_pes, d in dst_pes`` —
    convenience for the typical case where migration is between two
    shards and any rank in the src shard might send to any rank in
    the dst shard."""
    for s in src_pes:
        for d in dst_pes:
            register_migration_pair(s, d)


def realize_migration_pools() -> None:
    """Allocate flag + ack pools for all registered ``(src, dst)`` pairs.

    Collective: every rank must call. Idempotent — second call is a
    no-op. After this returns, :func:`register_migration_pair` rejects
    new pairs (the pool sizes are fixed at realize time because
    NVSHMEM symmetric allocation is one-shot per pool).

    Pool size is ``ops_per_pair × len(active_pairs)`` — bounded by the
    actual migration topology, not by ``n_pes²``. Empty pool (no
    registered pairs) is allowed and is cheap; trivializes to a
    barrier with zero allocations.
    """
    global _flag_pool, _flag_pool_buffers
    global _ack_pool, _ack_pool_buffers, _ack_payload
    global _ops_per_pair, _flag_pool_size, _pools_realized

    assert _initialized, (
        "migration_transport not initialized; call "
        "maybe_init_migration_transport() first"
    )
    if _pools_realized:
        return

    _ops_per_pair = int(
        os.environ.get("MIGRATION_OPS_PER_PAIR", DEFAULT_OPS_PER_PAIR)
    )
    _flag_pool_size = _ops_per_pair * len(_active_pairs)

    if _flag_pool_size > 0:
        _flag_pool, _flag_pool_buffers = _rt.allocate_flag_pool(_flag_pool_size)
        _ack_pool, _ack_pool_buffers = _rt.allocate_flag_pool(
            _flag_pool_size, initial_value=1
        )
        _ack_payload = _rt.allocate_slot_pool(1, 1)[0]
        _ack_payload.zero_()
        _rt.barrier_all_and_sync()

    _pools_realized = True
    logger.info(
        "[migration-transport] pools realized: active_pairs=%d, "
        "ops_per_pair=%d, flag_pool_size=%d",
        len(_active_pairs),
        _ops_per_pair,
        _flag_pool_size,
    )


def lane_for(src_pe: int, dst_pe: int) -> int:
    """Lane index for the ``(src_pe, dst_pe)`` pair.

    Returns the pair's index in the registry, which the flag pool
    uses as the lane's offset multiplier. Different pairs land in
    disjoint slot ranges so an ack from ``dst_a → src_a`` can't be
    misread as an ack from ``dst_b → src_b``.

    Raises if the pair wasn't registered before
    :func:`realize_migration_pools` — usually that means the migration
    handler is firing for a shard pair whose policy wasn't registered.
    """
    assert _initialized
    pair = (src_pe, dst_pe)
    idx = _pair_to_lane_idx.get(pair)
    if idx is None:
        raise RuntimeError(
            f"migration pair ({src_pe}, {dst_pe}) not registered. "
            f"Every (src_pe, dst_pe) used by a migration must be passed "
            f"to register_migration_pair() before realize_migration_pools()."
        )
    return idx


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
    _rt.signal_wait(ack_buffer(flag_slot), expected_value=1, stream=s)
    # Reset the local ack so the next round's ack-put causes a 0 → 1
    # transition. The reset MUST run on the migration stream — without
    # the explicit context, ``.zero_()`` falls back to the default
    # stream and races with the put below.
    with torch.cuda.stream(s):
        _ack_pool[flag_slot].zero_()

    _rt.put_signal(
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
    _rt.signal_wait(flag_buffer(flag_slot), expected_value=expected_value, stream=s)
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
    _rt.put_signal(
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
                f"migration_transport: op size {nbytes} > slot size "
                f"{_staging_slot_bytes}; raise MIGRATION_STAGING_SLOT_BYTES."
            )
        if self._next >= _staging_num_slots:
            raise RuntimeError(
                f"migration_transport: staging slot pool exhausted "
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

    Flag identity is per-pair per-op — no per-request keying. Multiple
    ops in one bundle, multiple bundles in one migration, and multiple
    migrations across time all share the same per-pair lane and recycle
    via acks rather than via a key-based collision-avoidance scheme.
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


def _init_state_for_test(
    *,
    n_pes: int,
    ops_per_pair: int = 16,
    register_all_pairs: bool = True,
    realize: bool = True,
) -> None:
    """Set up the module state ``FlagArena`` / ``lane_for`` need, without
    allocating any NVSHMEM buffers or starting the migration stream.
    Tests that only exercise the per-pair flag allocator use this in
    place of :func:`maybe_init_migration_transport`.

    By default, registers every ``(src, dst)`` pair in ``range(n_pes)``
    and marks the pool realized — existing tests can call
    ``FlagArena.take(s, d)`` without further setup. Pass
    ``register_all_pairs=False`` and/or ``realize=False`` to exercise
    the registration / realize lifecycle directly.
    """
    global _initialized, _ops_per_pair, _flag_pool_size, _pools_realized
    if _initialized:
        raise RuntimeError(
            "migration_transport already initialized; call "
            "_reset_state_for_test() first."
        )
    _rt._init_state_for_test(n_pes=n_pes)
    _ops_per_pair = ops_per_pair
    _initialized = True
    if register_all_pairs:
        for s in range(n_pes):
            for d in range(n_pes):
                register_migration_pair(s, d)
    _flag_pool_size = _ops_per_pair * len(_active_pairs)
    if realize:
        _pools_realized = True  # pretend pools exist for FlagArena.take


def _reset_state_for_test() -> None:
    """Tear down whatever ``_init_state_for_test`` set up. Idempotent;
    safe to call from a pytest fixture teardown."""
    global _initialized, _migration_stream
    global _flag_pool, _flag_pool_buffers
    global _ack_pool, _ack_pool_buffers, _ack_payload
    global _ops_per_pair, _flag_pool_size, _pools_realized
    global _active_pairs, _pair_to_lane_idx
    global _staging_slots, _staging_slot_bytes, _staging_num_slots
    _initialized = False
    _migration_stream = None
    _flag_pool = []
    _flag_pool_buffers = []
    _ack_pool = []
    _ack_pool_buffers = []
    _ack_payload = None
    _ops_per_pair = 0
    _flag_pool_size = 0
    _pools_realized = False
    _active_pairs = []
    _pair_to_lane_idx = {}
    _staging_slots = []
    _staging_slot_bytes = 0
    _staging_num_slots = 0
    _rt._reset_state_for_test()
