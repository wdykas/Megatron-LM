# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-hardware validation of the NVSHMEM credit-ring KV backend.

Spawns two PEs on two GPUs, bootstraps NVSHMEM over torch.distributed,
and pushes a sequence of KV blobs through :class:`NvshmemTransportBackend`
with ``depth=2``. Sending 4 blobs through a depth-2 ring forces the
ack-credit path to recycle twice (send #2 blocks until decode acks #0),
so a correct, in-order result validates the one-way put + decode-gated
credit recycling on the actual interconnect.

Requires nvshmem4py + >=2 CUDA devices, i.e. the real NVSHMEM transport on
GPUs. That is the only environment this backend runs in, so it is the right
(and only) place to test it -- no CPU stand-in. This and the NVSHMEM config of
``test_disagg_e2e`` cover the credit-ring backend on hardware.
"""

import os

import pytest
import torch

_HAVE_NVSHMEM = False
try:
    import nvshmem.core  # noqa: F401

    _HAVE_NVSHMEM = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not (_HAVE_NVSHMEM and torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    reason="requires nvshmem4py + >=2 CUDA devices",
)


def _worker(rank, world, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from megatron.core.inference import nvshmem_runtime as nv
        from megatron.core.inference.disaggregation.transfer_backends.nvshmem import (
            NvshmemTransportBackend,
        )

        nv.maybe_init_nvshmem()
        be = NvshmemTransportBackend(slot_bytes=4096, depth=2)
        be.init()
        dev = torch.device(f"cuda:{rank}")
        n = 4
        if rank == 0:
            for i in range(n):
                be.send(torch.full((4,), float(i), device=dev), dst=1).wait()
            q.put(("send", "ok"))
        else:
            got = [
                be.recv((4,), torch.float32, src=0).wait().float().mean().item()
                for _ in range(n)
            ]
            q.put(("recv", got))
        dist.barrier()
    except Exception:
        import traceback

        q.put((f"rank{rank}-ERROR", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        # Let process teardown reclaim symmetric memory (avoid explicit
        # finalize with live pools).


def test_nvshmem_credit_ring_roundtrip_2gpu():
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, 29557, q)) for r in range(2)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(2):
        k, v = q.get(timeout=180)
        out[k] = v
    for p in procs:
        p.join(timeout=60)
    assert "send" in out and out["send"] == "ok", out
    assert out.get("recv") == [0.0, 1.0, 2.0, 3.0], out
