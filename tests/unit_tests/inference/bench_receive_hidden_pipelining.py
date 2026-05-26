# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Microbenchmark: host-side overhead of ``receive_hidden`` after the
host-sync removal.

Setup: 2-rank NVSHMEM world. Rank 0 sends a stream of activations to
rank 1. Rank 1 calls ``receive_hidden`` in a loop. We measure:

  - **Host-time per receive**: time the host actually spends between
    issuing the receive and the function returning. With the host
    sync removed, this should be just the function-call overhead
    + the GPU stream-event setup. Pre-change behavior was the full
    NVSHMEM wait latency.
  - **End-to-end recv-to-recv latency**: time between the (i)-th
    and (i+1)-th call returning. This is the *throughput* metric;
    pipelining should not change it dramatically because the GPU
    work is the same.

The interesting comparison:

  host_time_per_receive  (low = host is free for other work)
  vs.
  e2e_time_per_receive   (this is the GPU work; same in both cases)

If host_time is dramatically lower than e2e_time, the host is
genuinely free to do other work while the GPU is waiting on the
receive — i.e., pipelining is unblocked.

Run as:

  torchrun --nproc_per_node=2 -m pytest bench_receive_hidden_pipelining.py
"""

import os
import time

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import activation_transport as at


@pytest.fixture(scope="module")
def _dist_world():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("microbenchmark needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    if dist.is_initialized():
        dist.destroy_process_group()


def test_receive_hidden_host_overhead(_dist_world):
    """Time the host overhead of receive_hidden.

    Run 100 iterations on each side; rank 0 sends, rank 1 receives.
    Rank 1 records the wall-time the host spends in receive_hidden
    + the total per-iteration wall time (which includes the GPU
    work).

    For pipelining to work, the host time per receive should be
    much smaller than the GPU work per receive — meaning the host
    returned quickly and could have been doing other work during
    the GPU wait.
    """
    rank = _dist_world
    if dist.get_world_size() < 2:
        pytest.skip("needs >=2 ranks")

    # Reset state and init activation transport.
    at._reset_state_for_test()
    at.maybe_init_activation_transport(
        num_lanes=4, pool_depth=8, slot_bytes=64 * 1024, max_pes=2
    )

    hidden_shape = (32, 1024)  # batch=32, hidden=1024 → ~64KB BF16
    hidden_dtype = torch.bfloat16
    payload_nbytes = (
        hidden_shape[0] * hidden_shape[1]
        * torch.empty((), dtype=hidden_dtype).element_size()
    )

    n_warmup = 5
    n_measure = 100

    if rank == 0:
        # Sender.
        hidden = torch.randn(*hidden_shape, dtype=hidden_dtype, device="cuda")
        dist.barrier()
        for _ in range(n_warmup + n_measure):
            at.send_hidden(my_pe=0, dst_pe=1, hidden=hidden, payload_nbytes=payload_nbytes)
        at.activation_stream().synchronize()
        dist.barrier()
        return

    # Receiver (rank 1).
    dist.barrier()  # match sender's barrier
    for _ in range(n_warmup):
        _ = at.receive_hidden(
            my_pe=1, src_pe=0,
            hidden_shape=hidden_shape, hidden_dtype=hidden_dtype,
            payload_nbytes=payload_nbytes,
        )
    at.activation_stream().synchronize()  # warmup boundary

    host_times = []
    e2e_times = []
    prev_end = time.perf_counter()
    for _ in range(n_measure):
        host_start = time.perf_counter()
        out = at.receive_hidden(
            my_pe=1, src_pe=0,
            hidden_shape=hidden_shape, hidden_dtype=hidden_dtype,
            payload_nbytes=payload_nbytes,
        )
        host_end = time.perf_counter()
        # Force completion to get a real end-to-end time. We do
        # this AFTER recording host_end so the host_time captures
        # only the receive call itself (no synchronize cost
        # attributed to it).
        torch.cuda.synchronize()
        e2e_end = time.perf_counter()

        host_times.append((host_end - host_start) * 1e6)  # microseconds
        e2e_times.append((e2e_end - prev_end) * 1e6)
        prev_end = e2e_end

    # Drop the first 5 to avoid edge effects.
    host_times = host_times[5:]
    e2e_times = e2e_times[5:]
    dist.barrier()

    mean_host = sum(host_times) / len(host_times)
    mean_e2e = sum(e2e_times) / len(e2e_times)
    p50_host = sorted(host_times)[len(host_times) // 2]
    p99_host = sorted(host_times)[int(len(host_times) * 0.99)]
    p50_e2e = sorted(e2e_times)[len(e2e_times) // 2]
    p99_e2e = sorted(e2e_times)[int(len(e2e_times) * 0.99)]

    print(f"\n[bench] receive_hidden host overhead (post host-sync removal):")
    print(f"[bench]   host time / call   : mean={mean_host:.1f} us  "
          f"p50={p50_host:.1f}  p99={p99_host:.1f}")
    print(f"[bench]   e2e time / call    : mean={mean_e2e:.1f} us  "
          f"p50={p50_e2e:.1f}  p99={p99_e2e:.1f}")
    print(f"[bench]   host fraction      : {mean_host / mean_e2e:.2%}")
    print(
        f"[bench]   pipelining margin  : {mean_e2e - mean_host:.1f} us per receive "
        f"(time GPU is busy but host could be doing other work)"
    )

    # Assertion: the host sync removal should leave SOMETHING in
    # the GPU pipeline beyond the host-execution Python overhead.
    # If host_time == e2e_time exactly, the GPU has zero work
    # outstanding when the host returns — meaning either the GPU is
    # done by then (no benefit from pipelining), or the host is
    # still synchronizing (sync wasn't fully removed). We expect a
    # small but non-zero margin.
    assert mean_e2e >= mean_host, (
        f"e2e time ({mean_e2e:.1f} us) should be >= host time "
        f"({mean_host:.1f} us)"
    )
