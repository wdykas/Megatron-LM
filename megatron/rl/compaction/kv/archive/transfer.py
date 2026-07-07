# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Span transfer backends for the KV archive.

The archive's byte movement is isolated behind ``SpanStore`` so the storage
tier is swappable: pinned host memory today (microsecond fetches on-node),
NIXL for remote/disaggregated tiers where hiding fetch latency actually
matters. The negative-cache index (centroids) always stays on the GPU — only
the exact span bytes move tiers.

Backends:
- ``PinnedCpuStore`` — pinned host tensors; async H2D on a caller stream.
  The default; fully validated on-node.
- ``NixlStore``      — NIXL-registered host memory, async reads via a NIXL
  agent. Loopback (self-agent) mode validates the full descriptor/xfer path
  on one node; point ``remote_agent_meta`` at another node's agent for a true
  remote tier. Hard-fails at construction if the ``nixl`` package is absent.
"""

from __future__ import annotations

import torch


class PinnedCpuStore:
    """Span bytes in pinned host memory (the on-node tier)."""

    def __init__(self) -> None:
        self._data: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def put(self, span_id: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """keys/values: (L, T, H, D) GPU tensors; stored pinned host-side."""
        self._data[span_id] = (
            keys.to("cpu", non_blocking=True).pin_memory(),
            values.to("cpu", non_blocking=True).pin_memory(),
        )

    def get(self, span_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self._data[span_id]
        return k.to("cuda", non_blocking=True), v.to("cuda", non_blocking=True)

    def get_async(self, span_id: int, stream: torch.cuda.Stream,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        k, v = self._data[span_id]
        with torch.cuda.stream(stream):
            gk = k.to("cuda", non_blocking=True)
            gv = v.to("cuda", non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        return gk, gv, event

    def drop(self, span_id: int) -> None:
        self._data.pop(span_id, None)


class NixlStore:
    """Span bytes behind a NIXL agent (the remote/disaggregated tier).

    Loopback mode (``remote_agent_meta=None``): spans live in NIXL-registered
    DRAM on this process and reads go through full NIXL descriptor/transfer
    machinery — validates the integration on one node and measures the
    protocol overhead that prefetch must hide. Remote mode: pass another
    agent's metadata blob; ``put`` then targets that agent's registered
    region (disaggregated archive host).
    """

    def __init__(self, agent_name: str = "kv-archive",
                 remote_agent_meta: bytes | None = None) -> None:
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "NixlStore needs the 'nixl' package (NIXL python bindings) — "
                "not present in this container (only UCX is). Use the "
                "nixl-enabled image or `pip install nixl`, then rerun. The "
                "pinned backend (--kv-compaction-archive-transfer pinned) is "
                "the on-node default and needs nothing.") from e
        self._agent = nixl_agent(agent_name, nixl_agent_config(backends=["UCX"]))
        self._remote = None
        if remote_agent_meta is not None:
            self._remote = self._agent.add_remote_agent(remote_agent_meta)
        # span_id -> (host tensors kept alive, registration handle)
        self._data: dict[int, tuple[torch.Tensor, torch.Tensor, object]] = {}

    def agent_metadata(self) -> bytes:
        """This agent's metadata blob — hand to a peer for remote mode."""
        return self._agent.get_agent_metadata()

    def put(self, span_id: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        k = keys.to("cpu", non_blocking=True).pin_memory()
        v = values.to("cpu", non_blocking=True).pin_memory()
        torch.cuda.synchronize()
        reg = self._agent.register_memory([k, v])
        if reg is None:
            raise RuntimeError(f"NIXL register_memory failed for span {span_id}")
        self._data[span_id] = (k, v, reg)

    def _read(self, span_id: int, stream: torch.cuda.Stream | None):
        k_host, v_host, _reg = self._data[span_id]
        gk = torch.empty_like(k_host, device="cuda")
        gv = torch.empty_like(v_host, device="cuda")
        vram_reg = self._agent.register_memory([gk, gv])
        if vram_reg is None:
            raise RuntimeError(f"NIXL register_memory (VRAM) failed for span {span_id}")
        local_descs = self._agent.get_xfer_descs([k_host, v_host])
        gpu_descs = self._agent.get_xfer_descs([gk, gv])
        peer = self._remote if self._remote is not None else self._agent.name
        handle = self._agent.initialize_xfer("READ", gpu_descs, local_descs, peer)
        if handle is None:
            raise RuntimeError(f"NIXL initialize_xfer failed for span {span_id}")
        state = self._agent.transfer(handle)
        while state != "DONE":
            if state == "ERR":
                raise RuntimeError(f"NIXL transfer failed for span {span_id}")
            state = self._agent.check_xfer_state(handle)
        self._agent.release_xfer_handle(handle)
        self._agent.deregister_memory(vram_reg)
        return gk, gv

    def get(self, span_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._read(span_id, None)

    def get_async(self, span_id: int, stream: torch.cuda.Stream,
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]:
        # NIXL transfers complete host-side; the event marks GPU visibility.
        gk, gv = self._read(span_id, stream)
        event = torch.cuda.Event()
        event.record(stream)
        return gk, gv, event

    def drop(self, span_id: int) -> None:
        entry = self._data.pop(span_id, None)
        if entry is not None:
            self._agent.deregister_memory(entry[2])


def build_span_store(transfer: str, **kwargs):
    """'pinned' (default, on-node) or 'nixl' (remote/disagg tier)."""
    if transfer == "pinned":
        return PinnedCpuStore()
    if transfer == "nixl":
        return NixlStore(**kwargs)
    raise ValueError(f"unknown archive transfer backend {transfer!r} "
                     "(expected 'pinned' or 'nixl')")
