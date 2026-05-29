# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native prefill->decode KV handoff for disaggregated inference.

Minimal coordinator/engine hook that ties together two existing pieces:

* ``DynamicInferenceContext.export_request_kv`` / ``import_request_kv``
  (the KV / Mamba staging hooks added for the Dynamo integration), and
* a pluggable :mod:`kv_transport_backend` (NCCL or NVSHMEM).

It performs a *non-blocking* request migration: the prefill worker
stages a finished request's KV, ships it to the decode worker, and is
free to keep generating; the decode worker receives the blobs and
imports them so its engine resumes the request without re-prefilling.

Two planes:

* **Control plane** — a small Python metadata dict (layout, shapes,
  dtypes, block hashes, presence flags) sent via
  ``torch.distributed`` object messaging. Works for every backend and
  keeps the data-plane backend a pure tensor mover.
* **Data plane** — the KV staging tensors, moved via the active
  :class:`KVTransportBackend` (``isend`` / ``irecv``).

The wire ordering of tensors is fixed and mirrored on both sides:
``[attn_staging, (mamba_conv, mamba_ssm), (snap_conv, snap_ssm)]``.

This is intentionally small. The full route-DAG dispatcher and the
bulk request-state migration on ``hetero-inference`` are future MRs;
this gets one prefill->decode KV handoff working natively end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.kv_transport_backend import (
    KVTransportBackend,
    TransferHandle,
    get_kv_transport_backend,
)

logger = logging.getLogger(__name__)


# Tensor wire order. Each entry: (top_key, sub_key_or_None, tensor_field).
# ``sub`` None means the tensor lives directly under the payload's
# ``mamba_payload``; otherwise under ``mamba_payload[sub]``.
def _tensor_slots(meta: dict) -> List[tuple]:
    slots = [("attn", None)]
    if meta.get("has_mamba"):
        slots.append(("mamba_conv", None))
        slots.append(("mamba_ssm", None))
        if meta.get("has_snapshots"):
            slots.append(("snap_conv", "snapshots"))
            slots.append(("snap_ssm", "snapshots"))
    return slots


def _build_metadata(payload: dict) -> dict:
    """Strip tensors out of an export payload into a picklable metadata
    dict carrying shapes/dtypes so the receiver can allocate buffers."""
    meta: dict = {
        "layout": payload["layout"],
        "block_count": payload["block_count"],
        "block_size_tokens": payload["block_size_tokens"],
        "num_layers": payload["num_layers"],
        "num_heads_per_partition": payload.get("num_heads_per_partition"),
        "hidden_per_head": payload.get("hidden_per_head"),
        "block_hashes": list(payload.get("block_hashes") or []),
        "attn_shape": tuple(payload["staging_tensor"].shape),
        "attn_dtype": payload["staging_tensor"].dtype,
        "has_mamba": False,
        "has_snapshots": False,
    }
    mp = payload.get("mamba_payload")
    if mp is not None:
        meta["has_mamba"] = True
        meta["mamba"] = {
            "num_mamba_layers": mp["num_mamba_layers"],
            "conv_shape": tuple(mp["conv_states_tensor"].shape),
            "conv_dtype": mp["conv_states_tensor"].dtype,
            "ssm_shape": tuple(mp["ssm_states_tensor"].shape),
            "ssm_dtype": mp["ssm_states_tensor"].dtype,
        }
        snaps = mp.get("snapshots")
        if snaps is not None:
            meta["has_snapshots"] = True
            meta["snapshots"] = {
                "block_hashes": list(snaps.get("block_hashes") or []),
                "conv_shape": tuple(snaps["conv_states_tensor"].shape),
                "conv_dtype": snaps["conv_states_tensor"].dtype,
                "ssm_shape": tuple(snaps["ssm_states_tensor"].shape),
                "ssm_dtype": snaps["ssm_states_tensor"].dtype,
            }
    return meta


def _ordered_tensors(payload: dict, meta: dict) -> List[torch.Tensor]:
    out = [payload["staging_tensor"]]
    if meta["has_mamba"]:
        mp = payload["mamba_payload"]
        out.append(mp["conv_states_tensor"])
        out.append(mp["ssm_states_tensor"])
        if meta["has_snapshots"]:
            out.append(mp["snapshots"]["conv_states_tensor"])
            out.append(mp["snapshots"]["ssm_states_tensor"])
    return out


def _slot_shape_dtype(meta: dict, idx: int):
    slots = _tensor_slots(meta)
    name, _ = slots[idx]
    if name == "attn":
        return meta["attn_shape"], meta["attn_dtype"]
    if name == "mamba_conv":
        return meta["mamba"]["conv_shape"], meta["mamba"]["conv_dtype"]
    if name == "mamba_ssm":
        return meta["mamba"]["ssm_shape"], meta["mamba"]["ssm_dtype"]
    if name == "snap_conv":
        return meta["snapshots"]["conv_shape"], meta["snapshots"]["conv_dtype"]
    if name == "snap_ssm":
        return meta["snapshots"]["ssm_shape"], meta["snapshots"]["ssm_dtype"]
    raise KeyError(name)


@dataclass
class PrefillHandoff:
    """Bookkeeping the prefill side holds until the transfer drains.

    Keeps the staged tensors alive (so the backend's in-flight sends
    don't reference freed memory) until :meth:`wait` completes.
    """

    handles: List[TransferHandle]
    keepalive: List[torch.Tensor] = field(default_factory=list)

    def wait(self) -> None:
        for h in self.handles:
            h.wait()
        self.keepalive.clear()


def _send_object(obj, dst: int, group) -> None:
    import torch.distributed as dist

    dist.send_object_list([obj], dst=dst, group=group)


def _recv_object(src: int, group):
    import torch.distributed as dist

    holder = [None]
    dist.recv_object_list(holder, src=src, group=group)
    return holder[0]


def send_request_kv(
    engine: Any,
    request_id: int,
    dst: int,
    *,
    backend: Optional[KVTransportBackend] = None,
    group: Optional[object] = None,
    base_tag: int = 0,
) -> Optional[PrefillHandoff]:
    """Prefill side: stage ``request_id``'s KV and ship it to ``dst``.

    Non-blocking on the data plane: returns a :class:`PrefillHandoff`
    whose ``wait()`` the caller can defer (e.g. until after the next
    engine step). Returns ``None`` if the request has no exportable KV
    (zero-length / unsupported layout), in which case the caller should
    fall back to having the decode side re-prefill.
    """
    backend = backend or get_kv_transport_backend()
    payload = engine.context.export_request_kv(request_id)
    if payload is None:
        # Tell the decode side there is nothing to receive.
        _send_object({"empty": True}, dst, group)
        return None

    meta = _build_metadata(payload)
    meta["empty"] = False
    _send_object(meta, dst, group)  # control plane (small, blocking)

    tensors = _ordered_tensors(payload, meta)
    handles = [
        backend.isend(t, dst=dst, tag=base_tag + i) for i, t in enumerate(tensors)
    ]
    return PrefillHandoff(handles=handles, keepalive=tensors)


def recv_request_kv(
    engine: Any,
    src: int,
    *,
    backend: Optional[KVTransportBackend] = None,
    group: Optional[object] = None,
    base_tag: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[dict]:
    """Decode side: receive a request's KV from ``src`` and import it.

    Returns the dict from :meth:`import_request_kv` (``block_ids`` etc.)
    on success, or ``None`` if the prefill side had nothing to send or
    the import was refused (caller then re-prefills).
    """
    backend = backend or get_kv_transport_backend()
    if device is None:
        mb = getattr(engine.context, "memory_buffer", None)
        device = mb.device if mb is not None else None

    meta = _recv_object(src, group)
    if meta is None or meta.get("empty", True):
        return None

    slots = _tensor_slots(meta)
    handles = []
    for i in range(len(slots)):
        shape, dtype = _slot_shape_dtype(meta, i)
        handles.append(
            backend.irecv(shape, dtype, src=src, tag=base_tag + i, device=device)
        )
    recvd = [h.wait() for h in handles]

    # Reassemble the payload import_request_kv expects.
    payload: dict = {
        "layout": meta["layout"],
        "block_count": meta["block_count"],
        "block_size_tokens": meta["block_size_tokens"],
        "num_layers": meta["num_layers"],
        "num_heads_per_partition": meta["num_heads_per_partition"],
        "hidden_per_head": meta["hidden_per_head"],
        "block_hashes": list(meta.get("block_hashes") or []),
        "staging_tensor": recvd[0],
    }
    if meta["has_mamba"]:
        mp = {
            "num_mamba_layers": meta["mamba"]["num_mamba_layers"],
            "conv_states_shape": list(meta["mamba"]["conv_shape"]),
            "ssm_states_shape": list(meta["mamba"]["ssm_shape"]),
            "conv_states_dtype": str(meta["mamba"]["conv_dtype"]),
            "ssm_states_dtype": str(meta["mamba"]["ssm_dtype"]),
            "conv_states_tensor": recvd[1],
            "ssm_states_tensor": recvd[2],
        }
        if meta["has_snapshots"]:
            mp["snapshots"] = {
                "block_hashes": list(meta["snapshots"].get("block_hashes") or []),
                "conv_states_shape": list(meta["snapshots"]["conv_shape"]),
                "ssm_states_shape": list(meta["snapshots"]["ssm_shape"]),
                "conv_states_dtype": str(meta["snapshots"]["conv_dtype"]),
                "ssm_states_dtype": str(meta["snapshots"]["ssm_dtype"]),
                "conv_states_tensor": recvd[3],
                "ssm_states_tensor": recvd[4],
            }
        payload["mamba_payload"] = mp

    return engine.context.import_request_kv(payload)
