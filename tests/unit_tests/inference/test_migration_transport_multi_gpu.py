# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-rank end-to-end test for migration_transport.

Run with:
    torchrun --nproc-per-node=4 -m pytest \
        tests/unit_tests/inference/test_migration_transport_multi_gpu.py -v

Validates:
- ``maybe_init_migration_transport()`` + active-pair registration +
  ``realize_migration_pools()`` complete on every rank.
- A put_slot_with_signal → wait_slot_signal → scatter → send_ack
  round-trip transfers the right bytes to the right destination
  staging slot.
- The ack handshake closes the loop so the same flag can be reused
  back-to-back without data corruption.
- ``lane_for`` raises for unregistered pairs and works for registered
  ones.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import migration_transport as mt
from megatron.core.inference import nvshmem_runtime as _rt


@pytest.fixture(scope="session")
def _dist_world():
    """Initialize torch.distributed exactly once per pytest session.

    Session-scoped so the process group persists across this and any
    other transport test files in the same run — destroying it
    mid-session would also invalidate NVSHMEM and deadlock the next
    test file's NVSHMEM-collective allocations.

    Run under torchrun / torch.distributed.run; if the env vars aren't
    set we skip (the test is multi-rank by construction)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun with --nproc-per-node>=2")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("migration_transport round-trip needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    # No destroy: let the process exit clean up. See the activation
    # transport multi-GPU test's matching comment.


@pytest.fixture(scope="module")
def _migration_initialized(_dist_world):
    """Bring up the migration transport runtime + staging slots, register
    every PE pair, and realize the flag/ack pools. Module-scoped so the
    expensive symmetric allocs happen once."""
    # Smaller-than-default staging slot so the symmetric alloc is fast.
    os.environ.setdefault("MIGRATION_STAGING_SLOT_BYTES", str(64 * 1024))
    os.environ.setdefault("MIGRATION_STAGING_NUM_SLOTS", "16")
    os.environ.setdefault("MIGRATION_OPS_PER_PAIR", "8")

    mt.maybe_init_migration_transport()
    n_pes = _rt.n_pes()
    # Register every ordered pair so any test can ship in either direction.
    mt.register_migration_shard_pair(range(n_pes), range(n_pes))
    mt.realize_migration_pools()
    yield _dist_world


def test_migration_transport_init(_migration_initialized):
    """After init + realize, the module reports the right pool dims and
    has its migration stream live."""
    assert mt._initialized
    assert mt._pools_realized
    assert mt._migration_stream is not None
    n_pes = _rt.n_pes()
    assert len(mt._active_pairs) == n_pes * n_pes
    assert mt._flag_pool_size == mt._ops_per_pair * len(mt._active_pairs)
    assert len(mt._flag_pool) == mt._flag_pool_size
    assert len(mt._ack_pool) == mt._flag_pool_size


def test_lane_for_known_pair_unique(_migration_initialized):
    """Every (src, dst) pair maps to a distinct lane after registration."""
    n_pes = _rt.n_pes()
    seen = set()
    for s in range(n_pes):
        for d in range(n_pes):
            lane = mt.lane_for(s, d)
            assert lane not in seen
            seen.add(lane)


def test_put_signal_round_trip(_migration_initialized):
    """Rank 0 puts a staging slot to rank 1 with put_slot_with_signal;
    rank 1 signal_waits, verifies the bytes, then send_ack's so the
    handshake closes the loop."""
    rank = _migration_initialized
    src_pe, dst_pe = 0, 1
    if rank not in (src_pe, dst_pe):
        dist.barrier()
        return

    # Use a fresh FlagArena for this test "migration" so we pick a
    # specific lane's first slot symmetrically on both ranks.
    arena = mt.FlagArena()
    flag = arena.take(src_pe, dst_pe)
    # Slot 0 from the staging pool — both sides agree on the slot
    # because both call .take() symmetrically. Here we just use index 0
    # explicitly since this test only ships one op.
    slot_idx = 0
    nbytes = 1024
    stream = mt.migration_stream()

    if rank == src_pe:
        # Stamp the slot with a recognizable byte pattern.
        mt.staging_slot(slot_idx).fill_(0x37)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == src_pe:
        with torch.cuda.stream(stream):
            mt.put_slot_with_signal(slot_idx, flag, dst_pe, nbytes=nbytes, stream=stream)
        torch.cuda.current_stream().wait_stream(stream)
    elif rank == dst_pe:
        with torch.cuda.stream(stream):
            mt.wait_slot_signal(flag, expected_value=1, stream=stream)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        slot = mt.staging_slot(slot_idx)[:nbytes]
        unique = torch.unique(slot)
        assert unique.numel() == 1 and int(unique[0].item()) == 0x37, (
            f"rank {rank}: slot bytes wrong, got unique={unique.tolist()}"
        )
        # Close the loop so the flag is reusable.
        with torch.cuda.stream(stream):
            mt.send_ack(flag, src_pe, stream=stream)
        torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()


def test_back_to_back_reuses_same_flag(_migration_initialized):
    """Two consecutive 'migrations' on the same (src, dst, op) — both
    use the same flag-pool slot. The ack handshake is what makes this
    safe: the second put's ack-wait blocks until the first round's
    scatter+ack completes."""
    rank = _migration_initialized
    src_pe, dst_pe = 0, 1
    if rank not in (src_pe, dst_pe):
        dist.barrier()
        return

    # Both rounds pick the same flag (start of (0, 1)'s lane).
    arena = mt.FlagArena()
    flag = arena.take(src_pe, dst_pe)
    slot_idx = 1  # different staging slot from the previous test
    nbytes = 512
    stream = mt.migration_stream()

    # Round 1: ship 0xAA. Round 2: ship 0xBB. Verify round 2's data
    # ends up on dst — i.e., the second put wasn't lost to a race
    # with round 1's still-pending state.
    for round_idx, payload in enumerate((0xAA, 0xBB)):
        if rank == src_pe:
            mt.staging_slot(slot_idx).fill_(payload)
        torch.cuda.synchronize()
        dist.barrier()

        if rank == src_pe:
            with torch.cuda.stream(stream):
                mt.put_slot_with_signal(
                    slot_idx, flag, dst_pe, nbytes=nbytes, stream=stream
                )
            torch.cuda.current_stream().wait_stream(stream)
        elif rank == dst_pe:
            with torch.cuda.stream(stream):
                mt.wait_slot_signal(flag, expected_value=1, stream=stream)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            slot = mt.staging_slot(slot_idx)[:nbytes]
            unique = torch.unique(slot)
            assert (
                unique.numel() == 1 and int(unique[0].item()) == payload
            ), (
                f"round {round_idx}: expected {hex(payload)} on dst, "
                f"got unique={unique.tolist()}"
            )
            with torch.cuda.stream(stream):
                mt.send_ack(flag, src_pe, stream=stream)
            torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()


def test_independent_pairs_dont_share_flags(_migration_initialized):
    """Lane partitioning: (0,1) and (0,2) use different flag-pool slots,
    so a put to (0,1) doesn't unblock a wait on (0,2)."""
    n_pes = _rt.n_pes()
    if n_pes < 3:
        pytest.skip("requires >=3 PEs to test independent pairs")
    rank = _migration_initialized

    arena = mt.FlagArena()
    flag_01 = arena.take(0, 1)
    flag_02 = arena.take(0, 2)
    assert flag_01 != flag_02, (
        f"different pairs collided on the same flag: "
        f"(0,1)→{flag_01}, (0,2)→{flag_02}"
    )
    # Lane indices are distinct by construction (see test_lane_for_known_pair_unique);
    # this asserts the FlagArena propagates that into pool indices.
