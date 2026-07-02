# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""KV transfer backend interface + the active-backend factory."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class TransferHandle:
    """Handle for an in-flight non-blocking transfer; ``wait()`` blocks until it
    completes. Received data lands in the buffers :meth:`KVTransportBackend.batch`
    returned."""

    wait_fn: Callable[[], None]

    def wait(self) -> None:
        self.wait_fn()


@dataclass
class PullRegion:
    """A paged buffer registered for one-sided remote READ, whose entries (KV
    blocks, Mamba slots) are addressed by index along ``index_axis``.

    Entry ``i``'s bytes are ``num_outer`` slices (product of dims before
    ``index_axis``), each ``inner_bytes`` long (product of dims after), spaced
    ``outer_stride_bytes`` apart -- so slice ``o`` lives at
    ``base_addr + o*outer_stride_bytes + i*inner_bytes``. This is the stride math
    the pull backend uses to READ entries without a staging copy."""

    tensor: torch.Tensor
    index_axis: int

    def layout(self) -> dict:
        """Per-region layout (plain ints) a remote peer uses to compute
        addresses; crosses the control plane, so no tensors/dtypes."""
        shape = self.tensor.shape
        elem = self.tensor.element_size()
        num_outer = 1
        for d in shape[: self.index_axis]:
            num_outer *= int(d)
        inner = 1
        for d in shape[self.index_axis + 1 :]:
            inner *= int(d)
        return {
            "base_addr": self.tensor.data_ptr(),
            "num_outer": num_outer,
            "outer_stride_bytes": int(shape[self.index_axis]) * inner * elem,
            "inner_bytes": inner * elem,
            "device_id": self.tensor.device.index,
        }


class KVTransportBackend(abc.ABC):
    """Interface for moving KV-cache blobs between workers. Two families,
    distinguished by :attr:`is_pull`:

    * **Push** (two-sided: NCCL). Both peers post one coalesced group of
      point-to-point ops (:meth:`batch`); transfers on a ``(src, dst)`` pair
      match by POST-ORDER, so both sides must enumerate them in the same order.
    * **One-sided** (RDMA: NIXL). Each rank registers its buffers once
      (:meth:`register_regions` / :meth:`export_regions_meta`); one rank then
      READs entries -- whole entries via :meth:`begin_pull` or raw byte
      fragments via :meth:`begin_pull_raw` -- with no peer action. No staging
      copy, no per-request registration.

    A backend implements one family and leaves the other raising
    ``NotImplementedError``; callers branch on :attr:`is_pull`.
    """

    # True for one-sided (pull) backends, False for the two-sided batch.
    is_pull: bool = False

    @abc.abstractmethod
    def init(self) -> None:
        """One-shot, idempotent init."""

    # --- push family (two-sided) ------------------------------------------
    def batch(self, sends, recvs, *, device: Optional[torch.device] = None):
        """(push) Issue one request's point-to-point ops as a single coalesced
        group, returning ``(handle, recv_buffers)``. ``sends``: ``(tensor, dst)``;
        ``recvs``: ``(shape, dtype, src)`` -- buffers are allocated here and
        returned in order. One group, never per-op send/recv: dozens of concurrent
        ungrouped P2P ops to one peer race on NCCL (illegal access)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the push interface")

    # --- one-sided family (RDMA) ------------------------------------------
    def register_regions(self, regions: dict) -> None:
        """(one-sided) Register this rank's KV buffers once for remote READ.
        ``regions`` maps a name to a :class:`PullRegion`."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def export_regions_meta(self) -> dict:
        """(one-sided) Metadata a remote peer needs to READ this rank's
        regions (agent metadata + per-region layout). Exported once."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_pull(self, peer_meta: dict, transfers: list):
        """(one-sided) Remote READ of whole entries from a peer's regions into
        this rank's. ``transfers``: ``(region_name, peer_src_index,
        local_dst_index)``. Returns a pollable handle."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")

    def begin_pull_raw(self, peer_meta: dict, region_name: str, descriptors: list):
        """(one-sided) Remote READ of raw byte fragments from one peer region:
        the hot path for resharded hand-offs. ``descriptors``:
        ``(peer_offset_bytes, local_addr, num_bytes)``. Returns a pollable
        handle."""
        raise NotImplementedError(f"{type(self).__name__} does not implement the one-sided interface")


def construct_kv_transport_backend(name: str) -> KVTransportBackend:
    """Build a KV transport backend by explicit name: ``"nccl"`` (two-sided push)
    or ``"nixl"`` (one-sided pull). Lazy imports avoid a base<->backend cycle and
    keep NIXL an optional dependency."""
    if name == "nccl":
        from megatron.core.inference.disaggregation.transfer_backends.nccl import (
            NcclTransportBackend,
        )
        return NcclTransportBackend()
    if name == "nixl":
        from megatron.core.inference.disaggregation.transfer_backends.nixl import (
            NixlTransportBackend,
        )
        return NixlTransportBackend()
    raise ValueError(f"Unknown KV transfer backend {name!r}; expected 'nccl' or 'nixl'")
