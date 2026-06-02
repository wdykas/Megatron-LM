# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""NVSHMEM depth-k credit-ring KV transfer backend (fast path)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)


def _nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


class NvshmemTransportBackend(KVTransportBackend):
    """Most-performant decode-gated KV transfer over NVSHMEM.

    A **depth-k credit ring** per channel: the bulk KV moves as a single
    one-way ``put_signal`` (data + completion in one op -- lower latency
    than a pull/``get`` round-trip), while decode gates flow via
    **per-slot ack credits**. Ack flags are pre-credited so the first
    ``depth`` sends per channel fire with no handshake on the critical
    path; prefill only stalls when it outruns decode by more than
    ``depth`` (correct backpressure -- decode is the scarce resource).

    This is the NVSHMEM analog of a depth-k pipelined rendezvous:
    decode-gated, one-way puts, pipelined to hide decode's consume
    latency.

    Indexing (pools sized ``n_pes * depth``):
      * data from src in ring pos ``w`` -> index ``src*depth + w``
        (slot + data-flag on the destination);
      * ack for (dst, ring pos ``w``) -> index ``dst*depth + w``
        (ack-flag on the source). Source and destination advance their
        per-pair ring cursor in lockstep, so ``w`` matches on both sides.

    Requires the ``nvshmem`` package + an NVSHMEM-capable interconnect;
    import-safe without them but raises at :meth:`init`.
    """

    def __init__(self, slot_bytes: int = 256 * 1024 * 1024, depth: int = 4) -> None:
        self._init = False
        self._slot_bytes = slot_bytes
        self._depth = depth
        self._nv = None
        self._data_slots = self._data_ft = self._data_fb = None
        self._ack_slots = self._ack_ft = self._ack_fb = None
        self._stream = None
        self._my_pe = -1
        self._n = 0
        self._wcursor: dict = {}  # dst -> count of sends issued
        self._rcursor: dict = {}  # src -> count of recvs completed

    def is_initialized(self) -> bool:
        return self._init

    def init(
        self, *, group: Optional[object] = None, slot_bytes: Optional[int] = None,
        depth: Optional[int] = None, **kwargs,
    ) -> None:
        from megatron.core.inference import nvshmem_runtime as nv

        if not getattr(nv, "_HAVE_NVSHMEM", False):
            raise RuntimeError(
                "NvshmemTransportBackend: the 'nvshmem' package is not available. "
                "Install nvidia-nvshmem-cu12, or use NcclTransportBackend (default)."
            )
        if slot_bytes is not None:
            self._slot_bytes = slot_bytes
        if depth is not None:
            self._depth = depth

        nv.maybe_init_nvshmem(group=group)
        self._nv = nv
        self._my_pe = nv.my_pe()
        self._n = nv.n_pes()
        n_slots = self._n * self._depth
        # Ack flags are pre-credited (initial_value=1) so the first ``depth``
        # sends per channel run without waiting.
        self._data_slots = nv.allocate_slot_pool(n_slots, self._slot_bytes)
        self._data_ft, self._data_fb = nv.allocate_flag_pool(n_slots, initial_value=0)
        self._ack_slots = nv.allocate_slot_pool(n_slots, 8)
        self._ack_ft, self._ack_fb = nv.allocate_flag_pool(n_slots, initial_value=1)
        self._stream = torch.cuda.Stream()
        nv.barrier_all_and_sync()
        self._init = True

    def stream(self) -> Optional[torch.cuda.Stream]:
        return self._stream

    def _check(self, nbytes: int) -> None:
        if nbytes > self._slot_bytes:
            raise RuntimeError(
                f"NvshmemTransportBackend: payload {nbytes} B exceeds slot_bytes "
                f"{self._slot_bytes}. Re-init with a larger slot_bytes."
            )

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        t = tensor.contiguous()
        nbytes = t.numel() * t.element_size()
        self._check(nbytes)
        w = self._wcursor.get(dst, 0) % self._depth
        ack_idx = dst * self._depth + w
        # Decode-gating / backpressure: wait until decode freed this ring slot
        # (ack credit), then consume the credit.
        self._nv.signal_wait(self._ack_fb[ack_idx], expected_value=1, stream=self._stream)
        self._stream.synchronize()
        self._ack_ft[ack_idx].zero_()
        data_idx = self._my_pe * self._depth + w
        self._data_slots[data_idx][:nbytes].copy_(t.reshape(-1).view(torch.uint8))
        self._nv.put_signal(  # one-way bulk put: data + completion in one op
            self._data_slots[data_idx], self._data_fb[data_idx], dst,
            nbytes=nbytes, stream=self._stream,
        )
        self._wcursor[dst] = self._wcursor.get(dst, 0) + 1
        return TransferHandle(wait_fn=self._stream.synchronize)

    def recv(
        self, shape: Tuple[int, ...], dtype: torch.dtype, src: int, tag: int = 0,
        *, device: Optional[torch.device] = None,
    ) -> TransferHandle:
        nbytes = _nbytes(shape, dtype)
        self._check(nbytes)
        r = self._rcursor.get(src, 0) % self._depth
        data_idx = src * self._depth + r
        self._nv.signal_wait(self._data_fb[data_idx], expected_value=1, stream=self._stream)
        self._stream.synchronize()
        out = self._data_slots[data_idx][:nbytes].view(dtype).reshape(shape).clone()
        self._data_ft[data_idx].zero_()
        # Recycle the credit: tell src this ring slot is free again.
        ack_idx = self._my_pe * self._depth + r
        self._nv.put_signal(
            self._ack_slots[ack_idx], self._ack_fb[ack_idx], src, nbytes=8, stream=self._stream,
        )
        self._rcursor[src] = self._rcursor.get(src, 0) + 1
        return TransferHandle(wait_fn=None, tensor=out)
