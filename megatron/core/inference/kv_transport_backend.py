# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pluggable transport backend for native disaggregated KV handoff.

This is the minimal, KV-blob-oriented sibling of the activation
transport backend on the ``hetero-inference`` branch. Where that one
moves per-layer *hidden states* between layer-kind shards
(``send_hidden`` / ``receive_hidden``), this one moves *KV-cache
blobs* between a prefill worker and a decode worker for
prefill->decode disaggregation, reusing the existing
``DynamicInferenceContext.export_request_kv`` /
``import_request_kv`` staging hooks.

Design (kept deliberately small for a first MR):

* The interface is a narrow tensor point-to-point: blocking
  ``send`` / ``recv`` plus non-blocking ``isend`` / ``irecv`` that
  return a :class:`TransferHandle` the caller waits on later. The
  non-blocking pair is what makes request migration non-blocking:
  the prefill engine stages + ``isend``s a request's KV and keeps
  generating while the bytes are in flight; the decode engine
  ``irecv``s and waits only when it is ready to admit the request.
* Two backends ship here:
    - :class:`NcclTransportBackend` — built on ``torch.distributed``
      point-to-point (``isend`` / ``irecv``). Works with the ``nccl``
      backend on GPU and ``gloo`` on CPU, so it runs in CI without any
      special hardware. This is the default.
    - :class:`NvshmemTransportBackend` — bulk KV transfer built directly
      on the shared ``nvshmem_runtime`` primitives (symmetric slot/flag
      pools + ``put_signal`` / ``signal_wait``). Import-safe without
      NVSHMEM; raises a clear error at :meth:`init` if the runtime is
      absent. It deliberately does NOT use ``activation_transport`` —
      that is the per-layer per-token mover for layer-kind
      disaggregation (a separate, deferred feature); a KV handoff is a
      few bulk transfers per request, which the runtime's put/signal
      serves directly.

Larger pieces from ``hetero-inference`` (route DAG dispatcher,
compiled routes, layer-wise activation transport) are intentionally
left for future MRs.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TransferHandle:
    """Opaque handle for an in-flight non-blocking transfer.

    ``wait()`` blocks until the transfer completes and, for receives,
    returns the received tensor. Backends populate ``tensor`` for
    receives so the caller need not pre-know where the bytes landed.
    """

    wait_fn: object  # Callable[[], None]
    tensor: Optional[torch.Tensor] = None

    def wait(self) -> Optional[torch.Tensor]:
        if self.wait_fn is not None:
            self.wait_fn()
        return self.tensor


class KVTransportBackend(abc.ABC):
    """Backend interface for moving KV-cache blobs between workers.

    PE identity is the backend's choice (NCCL uses the process group's
    rank space; NVSHMEM uses global PE ids — equal in our setup). All
    ops are point-to-point between a ``(src, dst)`` pair and tagged so
    multiple per-layer / per-request transfers can be multiplexed.
    """

    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Whether :meth:`init` has run."""

    @abc.abstractmethod
    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        """One-shot, idempotent init. ``group`` scopes the collective
        for backends that need it (NCCL); ignored otherwise."""

    @abc.abstractmethod
    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        """Blocking send of ``tensor`` to ``dst``."""

    @abc.abstractmethod
    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Blocking receive of a tensor of the given shape/dtype."""

    @abc.abstractmethod
    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        """Non-blocking send; returns a handle to wait on. The caller
        must keep ``tensor`` alive until the handle completes."""

    @abc.abstractmethod
    def irecv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> TransferHandle:
        """Non-blocking receive; ``handle.wait()`` returns the tensor."""

    def stream(self) -> Optional[torch.cuda.Stream]:
        """Optional dedicated stream; default ``None`` (use current)."""
        return None


class NcclTransportBackend(KVTransportBackend):
    """``torch.distributed`` point-to-point transport.

    Uses ``isend`` / ``irecv`` so it works identically under the
    ``nccl`` backend (GPU) and ``gloo`` backend (CPU/CI). Tensors are
    moved on whatever device they live on; the receive side allocates
    the destination buffer.
    """

    def __init__(self, group: Optional[object] = None) -> None:
        self._group = group
        self._init = False

    def is_initialized(self) -> bool:
        return self._init

    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "NcclTransportBackend.init: torch.distributed is not initialized; "
                "the prefill/decode workers must share a process group."
            )
        if group is not None:
            self._group = group
        self._init = True

    def _dist(self):
        import torch.distributed as dist

        return dist

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        self._dist().send(tensor.contiguous(), dst=dst, tag=tag, group=self._group)

    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        buf = torch.empty(shape, dtype=dtype, device=device)
        self._dist().recv(buf, src=src, tag=tag, group=self._group)
        return buf

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        t = tensor.contiguous()
        work = self._dist().isend(t, dst=dst, tag=tag, group=self._group)
        # Keep a reference to ``t`` so it isn't freed before the send drains.
        return TransferHandle(wait_fn=lambda _t=t, _w=work: _w.wait())

    def irecv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> TransferHandle:
        buf = torch.empty(shape, dtype=dtype, device=device)
        work = self._dist().irecv(buf, src=src, tag=tag, group=self._group)
        return TransferHandle(wait_fn=work.wait, tensor=buf)


class NvshmemSymmetricOps(abc.ABC):
    """Symmetric-memory primitives the credit-ring backend needs.

    Pulled behind an interface so the bug-prone ring/credit protocol in
    :class:`NvshmemTransportBackend` can be unit-tested against an
    in-memory simulator without real NVSHMEM. The default impl wraps
    :mod:`nvshmem_runtime`. Pools are addressed by integer index; a
    transfer from PE ``p`` writing slot index ``i`` lands in PE ``dst``'s
    slot index ``i`` (symmetric handle) and raises ``dst``'s flag ``i``.
    """

    @abc.abstractmethod
    def my_pe(self) -> int: ...
    @abc.abstractmethod
    def n_pes(self) -> int: ...
    @abc.abstractmethod
    def alloc(self, n_data: int, slot_bytes: int, n_ack: int) -> None: ...
    @abc.abstractmethod
    def write_data(self, idx: int, tensor: torch.Tensor) -> int: ...
    @abc.abstractmethod
    def read_data(self, idx: int, nbytes: int, dtype, shape) -> torch.Tensor: ...
    @abc.abstractmethod
    def put_data(self, slot_idx: int, flag_idx: int, dst: int, nbytes: int) -> None: ...
    @abc.abstractmethod
    def put_ack(self, flag_idx: int, dst: int) -> None: ...
    @abc.abstractmethod
    def wait_data(self, flag_idx: int) -> None: ...
    @abc.abstractmethod
    def wait_ack(self, flag_idx: int) -> None: ...
    @abc.abstractmethod
    def reset_data_flag(self, idx: int) -> None: ...
    @abc.abstractmethod
    def reset_ack_flag(self, idx: int) -> None: ...
    def stream(self) -> Optional[torch.cuda.Stream]:
        return None


class _RuntimeNvshmemOps(NvshmemSymmetricOps):
    """Default ops: wrap :mod:`nvshmem_runtime`. Ack flags are
    pre-credited (initial_value=1) so the first ``depth`` sends per
    channel run without waiting."""

    def __init__(self):
        self._nv = None
        self._data_slots = self._data_ft = self._data_fb = None
        self._ack_slots = self._ack_ft = self._ack_fb = None
        self._stream = None

    def _init_runtime(self, group):
        from megatron.core.inference import nvshmem_runtime as _nv

        if not getattr(_nv, "_HAVE_NVSHMEM", False):
            raise RuntimeError(
                "NvshmemTransportBackend: the 'nvshmem' package is not available. "
                "Install nvidia-nvshmem-cu12, or use NcclTransportBackend (default)."
            )
        _nv.maybe_init_nvshmem(group=group)
        self._nv = _nv

    def my_pe(self) -> int:
        return self._nv.my_pe()

    def n_pes(self) -> int:
        return self._nv.n_pes()

    def alloc(self, n_data, slot_bytes, n_ack):
        self._data_slots = self._nv.allocate_slot_pool(n_data, slot_bytes)
        self._data_ft, self._data_fb = self._nv.allocate_flag_pool(n_data, initial_value=0)
        self._ack_slots = self._nv.allocate_slot_pool(n_ack, 8)
        self._ack_ft, self._ack_fb = self._nv.allocate_flag_pool(n_ack, initial_value=1)
        self._stream = torch.cuda.Stream()
        self._nv.barrier_all_and_sync()

    def stream(self):
        return self._stream

    def write_data(self, idx, tensor):
        t = tensor.contiguous()
        nbytes = t.numel() * t.element_size()
        self._data_slots[idx][:nbytes].copy_(t.reshape(-1).view(torch.uint8))
        return nbytes

    def read_data(self, idx, nbytes, dtype, shape):
        return self._data_slots[idx][:nbytes].view(dtype).reshape(shape).clone()

    def put_data(self, slot_idx, flag_idx, dst, nbytes):
        self._nv.put_signal(
            self._data_slots[slot_idx], self._data_fb[flag_idx], dst,
            nbytes=nbytes, stream=self._stream,
        )

    def put_ack(self, flag_idx, dst):
        self._nv.put_signal(
            self._ack_slots[flag_idx], self._ack_fb[flag_idx], dst,
            nbytes=8, stream=self._stream,
        )

    def wait_data(self, flag_idx):
        self._nv.signal_wait(self._data_fb[flag_idx], expected_value=1, stream=self._stream)
        self._stream.synchronize()

    def wait_ack(self, flag_idx):
        self._nv.signal_wait(self._ack_fb[flag_idx], expected_value=1, stream=self._stream)
        self._stream.synchronize()

    def reset_data_flag(self, idx):
        self._data_ft[idx].zero_()

    def reset_ack_flag(self, idx):
        self._ack_ft[idx].zero_()


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
    latency. It replaces the earlier single-slot push, which was neither
    pipelined nor safe for back-to-back transfers on a pair.

    Indexing (pools sized ``n_pes * depth``):
      * data from src in ring pos ``w`` -> index ``src*depth + w``
        (slot + data-flag on the destination);
      * ack for (dst, ring pos ``w``) -> index ``dst*depth + w``
        (ack-flag on the source). Source and destination advance their
        per-pair ring cursor in lockstep, so ``w`` matches on both sides.

    Does **not** use ``activation_transport`` (the per-layer per-token
    mover for layer-kind disaggregation -- a separate, deferred feature);
    a KV handoff is a few bulk transfers per request, which the runtime's
    put/signal serves directly.

    NOTE: requires the ``nvshmem`` package + NVSHMEM-capable interconnect;
    import-safe without them but raises at :meth:`init`. Not exercised in
    CPU CI (no NVSHMEM there); the ring/credit protocol is unit-tested
    against an in-memory simulator, and NCCL is the CI-tested default.
    """

    def __init__(
        self,
        slot_bytes: int = 256 * 1024 * 1024,
        depth: int = 4,
        ops: Optional[NvshmemSymmetricOps] = None,
    ) -> None:
        self._init = False
        self._slot_bytes = slot_bytes
        self._depth = depth
        self._ops = ops or _RuntimeNvshmemOps()
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
        if slot_bytes is not None:
            self._slot_bytes = slot_bytes
        if depth is not None:
            self._depth = depth
        if isinstance(self._ops, _RuntimeNvshmemOps):
            self._ops._init_runtime(group)
        self._my_pe = self._ops.my_pe()
        self._n = self._ops.n_pes()
        n_slots = self._n * self._depth
        self._ops.alloc(n_slots, self._slot_bytes, n_slots)
        self._init = True

    def stream(self) -> Optional[torch.cuda.Stream]:
        return self._ops.stream()

    def _check(self, nbytes: int) -> None:
        if nbytes > self._slot_bytes:
            raise RuntimeError(
                f"NvshmemTransportBackend: payload {nbytes} B exceeds slot_bytes "
                f"{self._slot_bytes}. Re-init with a larger slot_bytes."
            )

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        t = tensor.contiguous()
        nbytes = t.numel() * t.element_size()
        self._check(nbytes)
        w = self._wcursor.get(dst, 0) % self._depth
        ack_idx = dst * self._depth + w
        # Decode-gating / backpressure: wait until decode freed this ring
        # slot (ack credit), then consume the credit.
        self._ops.wait_ack(ack_idx)
        self._ops.reset_ack_flag(ack_idx)
        data_idx = self._my_pe * self._depth + w
        self._ops.write_data(data_idx, t)
        self._ops.put_data(data_idx, data_idx, dst, nbytes)  # one-way bulk put
        self._wcursor[dst] = self._wcursor.get(dst, 0) + 1

    def recv(
        self, shape: Tuple[int, ...], dtype: torch.dtype, src: int, tag: int = 0,
        *, device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        nbytes = _nbytes(shape, dtype)
        self._check(nbytes)
        r = self._rcursor.get(src, 0) % self._depth
        data_idx = src * self._depth + r
        self._ops.wait_data(data_idx)
        out = self._ops.read_data(data_idx, nbytes, dtype, shape)
        self._ops.reset_data_flag(data_idx)
        # Recycle the credit: tell src this ring slot is free again.
        ack_idx = self._my_pe * self._depth + r
        self._ops.put_ack(ack_idx, src)
        self._rcursor[src] = self._rcursor.get(src, 0) + 1
        return out

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        self.send(tensor, dst, tag)  # stream-ordered: ack-wait + put on the stream
        strm = self.stream()
        return TransferHandle(wait_fn=(strm.synchronize if strm is not None else None))

    def irecv(
        self, shape: Tuple[int, ...], dtype: torch.dtype, src: int, tag: int = 0,
        *, device: Optional[torch.device] = None,
    ) -> TransferHandle:
        t = self.recv(shape, dtype, src, tag, device=device)
        return TransferHandle(wait_fn=None, tensor=t)


def _nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


# Module-level singleton. Defaults to NCCL (portable / CI-friendly).
_backend: Optional[KVTransportBackend] = None


def get_kv_transport_backend() -> KVTransportBackend:
    """Return the active backend, constructing the default (NCCL) on
    first call."""
    global _backend
    if _backend is None:
        _backend = NcclTransportBackend()
    return _backend


def set_kv_transport_backend(backend: Optional[KVTransportBackend]) -> None:
    """Override the active backend. ``None`` resets to the NCCL default
    on next access. Used by tests, or to select NVSHMEM at startup."""
    global _backend
    _backend = backend
