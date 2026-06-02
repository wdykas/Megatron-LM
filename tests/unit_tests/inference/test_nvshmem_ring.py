# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit-test the decode-gated credit-ring protocol of
:class:`NvshmemTransportBackend` against an in-memory symmetric-memory
simulator -- no NVSHMEM hardware needed. Verifies:

  * data integrity across ring wrap (depth-k pipelining),
  * ack-credit recycling lets transfers continue past `depth`,
  * backpressure: a (depth+1)-th send with no decode consumption blocks
    on the ack credit (decode-gating).
"""

import pytest
import torch

from megatron.core.inference.disaggregation.transfer_backends.nvshmem import (
    NvshmemSymmetricOps,
    NvshmemTransportBackend,
)


class _WouldBlock(Exception):
    pass


class _SimFabric:
    """Shared symmetric memory across simulated PEs (single process).
    ``mem[pe]`` holds data slots, data flags, ack flags by index."""

    def __init__(self):
        self.mem = {}

    def pe(self, p):
        return self.mem.setdefault(
            p, {"data": {}, "dflag": {}, "aflag": {}}
        )


class _SimOps(NvshmemSymmetricOps):
    """Per-PE view onto a shared :class:`_SimFabric`. ``signal_wait``
    raises ``_WouldBlock`` instead of spinning, so a blocked wait is
    observable (used to assert backpressure)."""

    def __init__(self, fabric, my_pe, n_pes):
        self._f = fabric
        self._my = my_pe
        self._n = n_pes

    def my_pe(self):
        return self._my

    def n_pes(self):
        return self._n

    def alloc(self, n_data, slot_bytes, n_ack):
        m = self._f.pe(self._my)
        # ack flags pre-credited to 1 (first `depth` sends free)
        for i in range(n_ack):
            m["aflag"][i] = 1

    def write_data(self, idx, tensor):
        self._f.pe(self._my)["data"][idx] = tensor.clone()
        return tensor.numel() * tensor.element_size()

    def read_data(self, idx, nbytes, dtype, shape):
        return self._f.pe(self._my)["data"][idx].clone()

    def put_data(self, slot_idx, flag_idx, dst, nbytes):
        d = self._f.pe(dst)
        d["data"][slot_idx] = self._f.pe(self._my)["data"][slot_idx].clone()
        d["dflag"][flag_idx] = 1

    def put_ack(self, flag_idx, dst):
        self._f.pe(dst)["aflag"][flag_idx] = 1

    def wait_data(self, flag_idx):
        if self._f.pe(self._my)["dflag"].get(flag_idx, 0) != 1:
            raise _WouldBlock(f"data flag {flag_idx} not set on pe {self._my}")

    def wait_ack(self, flag_idx):
        if self._f.pe(self._my)["aflag"].get(flag_idx, 0) != 1:
            raise _WouldBlock(f"ack flag {flag_idx} not credited on pe {self._my}")

    def reset_data_flag(self, idx):
        self._f.pe(self._my)["dflag"][idx] = 0

    def reset_ack_flag(self, idx):
        self._f.pe(self._my)["aflag"][idx] = 0


def _backend(fabric, pe, n, depth):
    b = NvshmemTransportBackend(depth=depth, ops=_SimOps(fabric, pe, n))
    b.init()
    return b


def test_pipeline_and_ring_wrap_data_integrity():
    """Send 2*depth distinct tensors 0->1; with interleaved decode
    consumption the ring wraps and every tensor arrives intact in order."""
    depth = 3
    fab = _SimFabric()
    src = _backend(fab, 0, 2, depth)   # prefill = PE 0
    dst = _backend(fab, 1, 2, depth)   # decode  = PE 1

    n = 2 * depth
    sent = [torch.full((4,), float(i)) for i in range(n)]
    for i in range(n):
        src.send(sent[i], dst=1)                 # one-way put (ack pre-credited / recycled)
        got = dst.recv((4,), torch.float32, src=0)  # decode consumes -> acks -> recycles credit
        assert torch.equal(got, sent[i]), f"transfer {i} corrupted"


def test_backpressure_blocks_after_depth_without_consumption():
    """Without any decode consumption, exactly `depth` sends succeed
    (pre-credited), and the (depth+1)-th blocks on the ack credit."""
    depth = 2
    fab = _SimFabric()
    src = _backend(fab, 0, 2, depth)
    _ = _backend(fab, 1, 2, depth)

    for i in range(depth):
        src.send(torch.full((4,), float(i)), dst=1)  # fills the ring, no acks yet
    with pytest.raises(_WouldBlock):
        src.send(torch.full((4,), 99.0), dst=1)       # decode-gated: no credit left


def test_credit_recycle_unblocks():
    """After backpressure, a single decode consume frees one credit and
    lets exactly one more send through."""
    depth = 2
    fab = _SimFabric()
    src = _backend(fab, 0, 2, depth)
    dst = _backend(fab, 1, 2, depth)

    for i in range(depth):
        src.send(torch.full((4,), float(i)), dst=1)
    # decode consumes the first -> recycles its credit
    got0 = dst.recv((4,), torch.float32, src=0)
    assert torch.equal(got0, torch.full((4,), 0.0))
    # now exactly one more send fits
    src.send(torch.full((4,), 42.0), dst=1)
    got1 = dst.recv((4,), torch.float32, src=0)
    assert torch.equal(got1, torch.full((4,), 1.0))
    got2 = dst.recv((4,), torch.float32, src=0)
    assert torch.equal(got2, torch.full((4,), 42.0))
