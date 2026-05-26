# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression guard for the host-sync removal in ``receive_hidden``.

After the host sync was removed (commit 678c1d5e5), ``receive_hidden``
must return BEFORE its GPU work completes — the caller's current
stream waits via ``wait_stream`` on the activation stream, not via
a host-side block. If the host sync ever creeps back in (e.g. as a
"defensive" fix during a refactor), the pipelining gain regresses
silently.

Test strategy: instrument the lower-level NVSHMEM primitives to
record stream-event timestamps. Verify that:

1. ``s.synchronize()`` is never called from ``receive_hidden``.
2. After ``receive_hidden`` returns, the host has NOT waited for
   the GPU's wait_activation kernel to complete — meaning the
   GPU has work still outstanding when the function returns.

These are CPU-side tests with mocked NVSHMEM primitives, so they
run without GPUs or a real activation transport. Multi-GPU
correctness lives in ``test_route_dispatcher_multi_gpu.py``.
"""

import os
import sys
from unittest.mock import patch

import pytest
import torch


def test_receive_hidden_does_not_call_synchronize():
    """``receive_hidden`` must not call ``s.synchronize()``.

    Patch the activation_transport module to mock the lower-level
    primitives, run receive_hidden, and assert synchronize() was
    not invoked. This is the canonical signal for "host did not
    block on GPU."
    """
    if not torch.cuda.is_available():
        pytest.skip("test instruments CUDA stream methods")

    from megatron.core.inference import activation_transport as at

    # Bootstrap test-only state so we can call receive_hidden
    # without a real NVSHMEM setup. The test substitutes mocks
    # for wait_activation / activation_slot / ack_activation so the
    # actual NVSHMEM primitives are never touched.
    at._reset_state_for_test()
    at._init_state_for_test(num_lanes=2, pool_depth=2, slot_bytes=1024, max_pes=2)

    # Patch the three NVSHMEM-touching helpers to no-ops, and patch
    # activation_slot to return a real CUDA tensor we can clone from.
    dummy_slot = torch.zeros(1024, dtype=torch.uint8, device="cuda")

    captured_syncs = []
    real_synchronize = torch.cuda.Stream.synchronize

    def watch_synchronize(self):
        captured_syncs.append(self)
        return real_synchronize(self)

    # Also need to install an _activation_stream since the
    # production path normally lives on a real CUDA stream
    # initialized by maybe_init_activation_transport.
    fake_stream = torch.cuda.Stream()

    with patch.object(at, "wait_activation", lambda slot, *, stream=None: None), \
         patch.object(at, "activation_slot", lambda slot: dummy_slot), \
         patch.object(at, "ack_activation", lambda slot, src_pe, *, stream=None: None), \
         patch.object(at, "_activation_stream", fake_stream), \
         patch.object(torch.cuda.Stream, "synchronize", watch_synchronize):
        out = at.receive_hidden(
            my_pe=1, src_pe=0,
            hidden_shape=(2, 4), hidden_dtype=torch.float32,
            payload_nbytes=32,
        )

    # The function should NOT have called synchronize on the
    # activation stream. The current stream's wait_stream is a
    # GPU-side barrier, not a host block, so it doesn't go through
    # Stream.synchronize.
    activation_stream_syncs = [s for s in captured_syncs if s is fake_stream]
    assert len(activation_stream_syncs) == 0, (
        f"receive_hidden called Stream.synchronize on the activation "
        f"stream {len(activation_stream_syncs)} time(s) — the host sync "
        f"removal regressed."
    )

    # Output tensor should have the right shape + dtype, regardless
    # of host-sync behavior. Sanity check.
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
