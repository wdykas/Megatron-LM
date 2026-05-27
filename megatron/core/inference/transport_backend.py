# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pluggable transport backend for cross-shard activation transport.

The route dispatcher used to call ``activation_transport.send_hidden`` /
``receive_hidden`` directly, which hardcoded NVSHMEM as the transport
primitive. This module introduces a small backend interface so
alternative communication libraries (NCCL, Gloo, MPI, custom RDMA
verbs, etc.) can plug in without touching the dispatcher.

The default backend is :class:`NvshmemActivationTransportBackend`,
which delegates to the existing :mod:`activation_transport` module's
NVSHMEM-backed implementation. Tests + benchmarks can swap in their
own backend via :func:`set_activation_transport_backend`.

The interface is deliberately narrow: just the five operations the
dispatcher needs (init / is_initialized / stream / send_hidden /
receive_hidden). Backend-specific knobs like NVSHMEM lane counts or
NCCL communicator topologies live on the concrete backend's
constructor.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import torch


class ActivationTransportBackend(abc.ABC):
    """Backend interface for moving hidden-state activations between
    shards mid-forward-pass.

    A backend must:

    - Be process-wide initializable (typically world-collective; the
      ``group`` arg restricts the collective to a subgroup if needed).
    - Provide a stream the dispatcher submits puts / waits on so they
      don't serialize against the engine's compute stream.
    - Provide point-to-point ``send_hidden`` / ``receive_hidden``
      between PE pairs. The PE identity space is the backend's
      choice (NVSHMEM uses global ranks; an NCCL backend might use
      a separate communicator's rank space).

    Backends are stateful — they typically allocate symmetric pools
    or communicators at init and reuse them across requests.
    """

    @abc.abstractmethod
    def is_initialized(self) -> bool:
        """Whether :meth:`init` has run."""

    @abc.abstractmethod
    def init(
        self,
        *,
        pool_depth: Optional[int] = None,
        slot_bytes: Optional[int] = None,
        group: Optional[object] = None,
    ) -> None:
        """One-shot collective init. Idempotent on subsequent calls.

        ``pool_depth`` and ``slot_bytes`` set per-pair ring depth and
        per-slot byte capacity; backends without those concepts should
        accept and ignore them. ``group`` is a
        ``torch.distributed.ProcessGroup`` for backends that need to
        scope the collective.
        """

    @abc.abstractmethod
    def stream(self) -> torch.cuda.Stream:
        """Dedicated CUDA stream activation puts / waits run on.
        Kept off the engine's compute stream so activations don't
        head-of-line-block compute and vice versa.
        """

    @abc.abstractmethod
    def send_hidden(
        self,
        my_pe: int,
        dst_pe: int,
        hidden: torch.Tensor,
        payload_nbytes: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Send ``hidden`` from this PE to ``dst_pe`` over the
        transport's lane / channel for that pair. Stream-ordered.
        ``payload_nbytes`` is the byte count to actually transfer
        (the slot buffer may be larger than the tensor).
        """

    @abc.abstractmethod
    def receive_hidden(
        self,
        my_pe: int,
        src_pe: int,
        hidden_shape: Tuple[int, ...],
        hidden_dtype: torch.dtype,
        payload_nbytes: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Receive a hidden-state tensor from ``src_pe``. Returns a
        fresh tensor (cloned out of the backend's slot buffer so
        the backend is free to recycle the slot immediately).
        """


class NvshmemActivationTransportBackend(ActivationTransportBackend):
    """NVSHMEM-backed implementation. Delegates to the existing
    :mod:`activation_transport` module which carries the
    lane-encoded ring-buffer slot pool, fwd / ack flag pools, and
    dedicated stream. See :doc:`DISAGG_DESIGN.md` for the design.
    """

    def is_initialized(self) -> bool:
        from megatron.core.inference import activation_transport as _at
        return _at.is_initialized()

    def init(
        self,
        *,
        pool_depth: Optional[int] = None,
        slot_bytes: Optional[int] = None,
        group: Optional[object] = None,
    ) -> None:
        from megatron.core.inference import activation_transport as _at
        _at.maybe_init_activation_transport(
            pool_depth=pool_depth,
            slot_bytes=slot_bytes,
            group=group,
        )

    def stream(self) -> torch.cuda.Stream:
        from megatron.core.inference import activation_transport as _at
        return _at.activation_stream()

    def send_hidden(
        self,
        my_pe: int,
        dst_pe: int,
        hidden: torch.Tensor,
        payload_nbytes: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        from megatron.core.inference import activation_transport as _at
        _at.send_hidden(
            my_pe=my_pe,
            dst_pe=dst_pe,
            hidden=hidden,
            payload_nbytes=payload_nbytes,
            stream=stream,
        )

    def receive_hidden(
        self,
        my_pe: int,
        src_pe: int,
        hidden_shape: Tuple[int, ...],
        hidden_dtype: torch.dtype,
        payload_nbytes: int,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        from megatron.core.inference import activation_transport as _at
        return _at.receive_hidden(
            my_pe=my_pe,
            src_pe=src_pe,
            hidden_shape=hidden_shape,
            hidden_dtype=hidden_dtype,
            payload_nbytes=payload_nbytes,
            stream=stream,
        )


# Module-level singleton. Defaults to NVSHMEM; tests / benchmarks can
# swap via ``set_activation_transport_backend``.
_backend: Optional[ActivationTransportBackend] = None


def get_activation_transport_backend() -> ActivationTransportBackend:
    """Return the active backend, constructing the default (NVSHMEM)
    on first call."""
    global _backend
    if _backend is None:
        _backend = NvshmemActivationTransportBackend()
    return _backend


def set_activation_transport_backend(
    backend: Optional[ActivationTransportBackend],
) -> None:
    """Override the active backend. Pass ``None`` to reset to the
    default (NVSHMEM) on next access. Useful for tests that want a
    mock or for experiments with alternative transports.
    """
    global _backend
    _backend = backend
