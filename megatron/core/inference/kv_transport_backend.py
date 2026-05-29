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
    - :class:`NvshmemTransportBackend` — adapts the branch's tested
      NVSHMEM activation transport (``activation_transport`` /
      ``nvshmem_runtime``) to move KV blobs. Import-safe without
      NVSHMEM; raises a clear error at :meth:`init` if the runtime is
      absent.

Larger pieces from ``hetero-inference`` (route DAG dispatcher,
compiled routes, hetero-TP / MoE routing) are intentionally left for
future MRs.
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
    """Adapts the branch's NVSHMEM activation transport to move KV blobs.

    Treats a KV staging tensor as a ``hidden`` payload and routes it
    through ``activation_transport.send_hidden`` / ``receive_hidden``.
    NVSHMEM has no separate non-blocking API exposed here, so
    ``isend`` / ``irecv`` fall back to stream-ordered blocking ops on
    the transport's dedicated stream (still off the compute stream).
    """

    def __init__(self) -> None:
        self._init = False

    def is_initialized(self) -> bool:
        try:
            from megatron.core.inference import activation_transport as _at
        except Exception:
            return False
        return self._init and _at.is_initialized()

    def init(self, *, group: Optional[object] = None, **kwargs) -> None:
        from megatron.core.inference import activation_transport as _at
        from megatron.core.inference import nvshmem_runtime as _nv

        if not getattr(_nv, "_HAVE_NVSHMEM", False):
            raise RuntimeError(
                "NvshmemTransportBackend.init: the 'nvshmem' Python package is not "
                "available. Build NVSHMEM, or use NcclTransportBackend instead."
            )
        _at.maybe_init_activation_transport(group=group, **kwargs)
        self._init = True

    def stream(self) -> Optional[torch.cuda.Stream]:
        from megatron.core.inference import activation_transport as _at

        return _at.activation_stream()

    def send(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> None:
        from megatron.core.inference import activation_transport as _at
        import torch.distributed as dist

        my_pe = dist.get_rank()
        _at.send_hidden(
            my_pe=my_pe,
            dst_pe=dst,
            hidden=tensor.contiguous(),
            payload_nbytes=tensor.numel() * tensor.element_size(),
            stream=self.stream(),
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
        from megatron.core.inference import activation_transport as _at
        import torch.distributed as dist

        my_pe = dist.get_rank()
        nbytes = _nbytes(shape, dtype)
        return _at.receive_hidden(
            my_pe=my_pe,
            src_pe=src,
            hidden_shape=shape,
            hidden_dtype=dtype,
            payload_nbytes=nbytes,
            stream=self.stream(),
        )

    def isend(self, tensor: torch.Tensor, dst: int, tag: int = 0) -> TransferHandle:
        # Stream-ordered put; "wait" syncs the transport stream.
        self.send(tensor, dst, tag)
        strm = self.stream()
        return TransferHandle(wait_fn=(strm.synchronize if strm is not None else None))

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
        strm = self.stream()
        return TransferHandle(
            wait_fn=(strm.synchronize if strm is not None else None), tensor=t
        )


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
