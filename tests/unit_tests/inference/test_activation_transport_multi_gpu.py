# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-rank end-to-end test for activation_transport.

Run with:
    torchrun --nproc-per-node=2 -m pytest \
        tests/unit_tests/inference/test_activation_transport_multi_gpu.py -v

Validates:
- ``maybe_init_activation_transport()`` + ``register_activation_pair`` +
  ``realize_activation_pools()`` complete on every rank.
- A put → wait round-trip transfers the right bytes to the right slot.
- The ack handshake closes the loop so the same slot can be reused
  without data corruption.
- Two distinct ``(src, dst)`` lanes don't interfere.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import activation_transport as at
from megatron.core.inference import nvshmem_runtime as _nv


@pytest.fixture(scope="session")
def _dist_world():
    """Initialize torch.distributed exactly once per pytest session.

    Session-scoped (rather than module-scoped) so the process group
    persists across both transport test files — destroying it between
    files would also tear down state NVSHMEM depends on, which we'd
    then have to re-bootstrap.

    Run under torchrun / torch.distributed.run; if the env vars aren't
    set we skip (the test is multi-rank by construction)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun with --nproc-per-node>=2")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("activation_transport round-trip needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    # No destroy_process_group: NVSHMEM keeps state tied to the
    # distributed runtime; tearing it down mid-session deadlocks the
    # next test file's NVSHMEM-collective allocations. Let the process
    # exit clean up.


@pytest.fixture(scope="module")
def _activation_initialized(_dist_world):
    """Bring up the activation transport, register every PE pair, and
    realize the pools. Module-scoped so the symmetric allocs happen
    once and the round-trip tests can share the state.

    Note: ``maybe_init_*`` and ``realize_*`` are idempotent. We do NOT
    call ``_reset_state_for_test()`` here because that resets the
    nvshmem_runtime state (``_initialized=False``), which would make a
    real NVSHMEM-backed run think it must re-init the C-level
    NVSHMEM library — deadlocks. The unit-test resets are CPU-only
    fixtures and should not run after real NVSHMEM init.
    """
    # Small pool: 4 KB per slot × 4 slots × world_size² pairs.
    at.maybe_init_activation_transport(pool_depth=4, slot_bytes=64 * 1024)
    n_pes = _nv.n_pes()
    at.register_activation_shard_pair(range(n_pes), range(n_pes))
    at.realize_activation_pools()
    yield _dist_world


def test_activation_transport_init_succeeds(_activation_initialized):
    """Init + register + realize all complete; pool sizing matches
    registered pairs."""
    assert at.is_initialized()
    n_pes = _nv.n_pes()
    stats = at.pool_stats()
    assert stats["active_pairs"] == n_pes * n_pes
    assert stats["pool_depth"] == 4
    assert stats["total_slots"] == n_pes * n_pes * 4
    assert stats["pools_realized"] is True
    # The activation stream must be a real CUDA stream.
    s = at.activation_stream()
    assert isinstance(s, torch.cuda.Stream)


def test_activation_transport_one_way_round_trip(_activation_initialized):
    """Rank 0 puts a known payload into rank 1's slot; rank 1 waits,
    verifies bytes, acks."""
    rank = _activation_initialized
    src_pe = 0
    dst_pe = 1
    lane = at.lane_for(src_pe, dst_pe)
    stream = at.activation_stream()
    payload_nbytes = 4096

    if rank == src_pe:
        slot = at.next_send_slot(lane)
        sym = at.activation_slot(slot)
        with torch.cuda.stream(stream):
            sym[:payload_nbytes].fill_(0xAB)
        at.put_activation(slot, dst_pe=dst_pe, nbytes=payload_nbytes, stream=stream)
        stream.synchronize()
    elif rank == dst_pe:
        slot = at.next_recv_slot(lane)
        at.wait_activation(slot, stream=stream)
        stream.synchronize()
        sym = at.activation_slot(slot)
        observed = sym[:payload_nbytes].cpu()
        assert observed.eq(0xAB).all(), (
            f"slot {slot} expected 0xAB; got first 8 bytes "
            f"{observed[:8].tolist()}"
        )
        at.ack_activation(slot, src_pe=src_pe, stream=stream)
        stream.synchronize()

    dist.barrier()


def test_activation_transport_slot_recycle(_activation_initialized):
    """Fill the lane's ring + 1 to force a slot reuse. The ack handshake
    must gate the reuse so no data races."""
    rank = _activation_initialized
    src_pe = 0
    dst_pe = 1
    lane = at.lane_for(src_pe, dst_pe)
    at.reset_lane_counters(lane)
    stream = at.activation_stream()
    payload_nbytes = 1024
    num_iter = 5  # depth=4 → forces slot 0 to be reused

    if rank == src_pe:
        for i in range(num_iter):
            slot = at.next_send_slot(lane)
            sym = at.activation_slot(slot)
            with torch.cuda.stream(stream):
                sym[:payload_nbytes].fill_(i)
            at.put_activation(slot, dst_pe=dst_pe, nbytes=payload_nbytes, stream=stream)
        stream.synchronize()
    elif rank == dst_pe:
        for i in range(num_iter):
            slot = at.next_recv_slot(lane)
            at.wait_activation(slot, stream=stream)
            stream.synchronize()
            sym = at.activation_slot(slot)
            observed = sym[:payload_nbytes].cpu()
            assert observed.eq(i).all(), (
                f"iter {i}: slot {slot} expected {i}, got "
                f"first 8 bytes {observed[:8].tolist()}"
            )
            at.ack_activation(slot, src_pe=src_pe, stream=stream)
        stream.synchronize()

    dist.barrier()


def test_activation_transport_independent_lanes(_activation_initialized):
    """Two distinct ``(src, dst)`` pairs don't share lanes — verified
    by their distinct lane indices."""
    rank = _activation_initialized
    n_pes = _nv.n_pes()
    if n_pes < 3:
        # With only 2 PEs the only outbound pairs from PE 0 are (0,1)
        # and (0,0); use those.
        lane_a = at.lane_for(0, 1)
        lane_b = at.lane_for(0, 0)
    else:
        lane_a = at.lane_for(0, 1)
        lane_b = at.lane_for(0, 2)
    assert lane_a != lane_b, (
        f"distinct (src, dst) pairs collided on lane {lane_a}"
    )
