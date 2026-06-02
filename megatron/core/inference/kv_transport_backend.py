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


class NvshmemTransportBackend(KVTransportBackend):
    """Bulk KV transfer over NVSHMEM, built directly on the shared
    :mod:`nvshmem_runtime` primitives (init + symmetric slot/flag pools +
    ``put_signal`` / ``signal_wait``).

    This is a *bulk* point-to-point mover -- a few large transfers per
    request -- which is what a prefill->decode KV handoff needs. It does
    **not** use ``activation_transport`` (that is the per-layer
    per-token ring-buffer mover for layer-kind disaggregation, a
    separate, deferred feature).

    Addressing: symmetric memory is addressed by handle, so a transfer
    from PE ``s`` lands in the destination's slot/flag at index ``s``.
    Slot/flag pools are sized to ``n_pes`` (one inbound channel per
    source PE), so transfers between *distinct* (src, dst) pairs run
    concurrently. The limitation is **one in-flight transfer per (src,
    dst) pair** -- single-buffered per channel; the KV handoff sends one
    sub-block per pair per request, so this suffices. Multi-buffering
    per pair is a future MR.

    NOTE: requires the ``nvshmem`` package + NVSHMEM-capable interconnect;
    it is import-safe without them but raises at :meth:`init`. It is not
    exercised in CPU CI (no NVSHMEM there) -- the NCCL backend is the
    CI-tested default.
    """

    def __init__(self, slot_bytes: int = 256 * 1024 * 1024) -> None:
        self._init = False
        self._slot_bytes = slot_bytes
        self._slots = None       # symmetric uint8 slot tensors, one per src PE
        self._flag_t = None      # flag tensors (for local zero/read)
        self._flag_b = None      # flag cuda.core Buffers (for the NVSHMEM call)
        self._stream = None
        self._my_pe = -1

    def is_initialized(self) -> bool:
        return self._init

    def init(self, *, group: Optional[object] = None, slot_bytes: Optional[int] = None, **kwargs) -> None:
        from megatron.core.inference import nvshmem_runtime as _nv

        if not getattr(_nv, "_HAVE_NVSHMEM", False):
            raise RuntimeError(
                "NvshmemTransportBackend.init: the 'nvshmem' package is not "
                "available. Install nvidia-nvshmem-cu12, or use "
                "NcclTransportBackend (the default)."
            )
        if slot_bytes is not None:
            self._slot_bytes = slot_bytes
        _nv.maybe_init_nvshmem(group=group)
        n = _nv.n_pes()
        self._my_pe = _nv.my_pe()
        # One inbound channel per source PE -> distinct pairs don't collide.
        self._slots = _nv.allocate_slot_pool(n, self._slot_bytes)
        self._flag_t, self._flag_b = _nv.allocate_flag_pool(n)
        self._stream = torch.cuda.Stream()
        _nv.barrier_all_and_sync()
        self._init = True

    def stream(self) -> Optional[torch.cuda.Stream]:
        return self._stream

    def _check(self, nbytes: int) -> None:
        if nbytes > self._slot_bytes:
            raise RuntimeError(
                f"NvshmemTransportBackend: payload {nbytes} B exceeds slot_bytes "
                f"{self._slot_bytes}. Re-init with a larger slot_bytes."
            )

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        from megatron.core.inference import nvshmem_runtime as _nv

        t = tensor.contiguous()
        nbytes = t.numel() * t.element_size()
        self._check(nbytes)
        # Write into our own slot[my_pe]; put_signal lands it in dst's
        # slot[my_pe] (same symmetric handle) and sets dst's flag[my_pe].
        slot = self._slots[self._my_pe]
        slot[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
        _nv.put_signal(
            slot, self._flag_b[self._my_pe], dst, nbytes=nbytes, stream=self._stream
        )

    def recv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        from megatron.core.inference import nvshmem_runtime as _nv

        nbytes = _nbytes(shape, dtype)
        self._check(nbytes)
        _nv.signal_wait(self._flag_b[src], expected_value=1, stream=self._stream)
        self._stream.synchronize()
        out = (
            self._slots[src][:nbytes].view(dtype).reshape(shape).clone()
        )
        self._flag_t[src].zero_()  # recycle the channel for the next transfer
        return out

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        self.send(tensor, dst, tag)  # stream-ordered put
        return TransferHandle(wait_fn=self._stream.synchronize)

    def irecv(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        src: int,
        tag: int = 0,
        *,
        device: Optional[torch.device] = None,
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
