# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native prefill->decode KV handoff: stage, transport, import."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation.kv_transport_backend import (
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


def derive_decode_schema(engine: Any, prompt_token_ids) -> Optional[dict]:
    """Reconstruct the KV schema on the decode side with no control message.

    Returns the same metadata dict :func:`_build_metadata` produces, but
    computed locally from the engine's static config + the prompt tokens
    the decode worker already holds. Returns ``None`` for the MLA latent
    cache (not header-free in this MR; use ``header_free=False``).

    Assumes a homogeneous, fresh prefill: ``block_count`` and the
    snapshot count follow directly from the prompt length and block size.
    """
    from megatron.core.inference.inference_request import compute_block_hashes_batched

    ctx = engine.context
    if getattr(ctx, "cache_mla_latent", False):
        return None

    bs = int(ctx.block_size_tokens)
    if isinstance(prompt_token_ids, torch.Tensor):
        toks = prompt_token_ids
        prompt_len = int(toks.numel())
    else:
        prompt_len = len(prompt_token_ids)
        toks = torch.tensor(list(prompt_token_ids), dtype=torch.int64)
    block_count = (prompt_len + bs - 1) // bs
    block_hashes = list(compute_block_hashes_batched(toks, bs))

    mb = ctx.memory_buffer  # (2, num_layers, total_blocks, block_size, heads, hidden)
    _, num_layers, _, _, heads, hidden = mb.shape
    meta: dict = {
        "layout": "std_attn_v1",
        "block_count": block_count,
        "block_size_tokens": bs,
        "num_layers": int(num_layers),
        "num_heads_per_partition": int(heads),
        "hidden_per_head": int(hidden),
        "block_hashes": block_hashes,
        "attn_shape": (block_count, 2, int(num_layers), bs, int(heads), int(hidden)),
        "attn_dtype": mb.dtype,
        "has_mamba": False,
        "has_snapshots": False,
        "empty": False,
    }
    if getattr(ctx, "is_hybrid_model", False):
        conv = ctx.mamba_conv_states  # (num_mamba_layers, max_requests, *conv_state)
        ssm = ctx.mamba_ssm_states
        nml = int(conv.shape[0])
        conv_state = tuple(int(x) for x in conv.shape[2:])
        ssm_state = tuple(int(x) for x in ssm.shape[2:])
        meta["has_mamba"] = True
        meta["mamba"] = {
            "num_mamba_layers": nml,
            "conv_shape": (nml, *conv_state),
            "conv_dtype": conv.dtype,
            "ssm_shape": (nml, *ssm_state),
            "ssm_dtype": ssm.dtype,
        }
        n_snap = len(block_hashes)  # one snapshot per complete block
        if getattr(ctx, "mamba_slot_allocator", None) is not None and n_snap > 0:
            meta["has_snapshots"] = True
            meta["snapshots"] = {
                "block_hashes": block_hashes,
                "conv_shape": (n_snap, nml, *conv_state),
                "conv_dtype": conv.dtype,
                "ssm_shape": (n_snap, nml, *ssm_state),
                "ssm_dtype": ssm.dtype,
            }
    return meta


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


def send_request_kv_resharded(
    engine: Any,
    request_id: int,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    *,
    backend: Optional[KVTransportBackend] = None,
    group: Optional[object] = None,
    base_tag: int = 0,
) -> Optional["PrefillHandoff"]:
    """Hetero-layout prefill send: reshard this rank's KV sub-blocks to
    the decode layout via global-coordinate range intersection.

    ``my_layout`` is this prefill rank's :class:`KVShardLayout`;
    ``src_layouts`` / ``dst_layouts`` are the full prefill / decode
    layout lists (known from a one-time config handshake). Attention KV
    only in this MR -- a hybrid request raises ``NotImplementedError``.
    Header-free: the decode side derives shapes from config + prompt.
    """
    from megatron.core.inference.disaggregation.kv_shard_layout import (
        plan_kv_reshard,
        transfers_for_src,
    )

    backend = backend or get_kv_transport_backend()
    payload = engine.context.export_request_kv(request_id)
    if payload is None:
        raise ValueError(
            f"send_request_kv_resharded: request {request_id} has no exportable KV"
        )
    if payload.get("mamba_payload") is not None:
        raise NotImplementedError(
            "hetero KV reshard supports attention KV only in this MR; "
            "hybrid/Mamba hetero handoff is a future MR"
        )

    attn = payload["staging_tensor"]  # [BC, 2, local_layers, BS, local_heads, HD]
    plan = plan_kv_reshard(src_layouts, dst_layouts)
    mine = transfers_for_src(plan, my_layout.global_rank)
    handles: List[TransferHandle] = []
    keep: List[torch.Tensor] = []
    for t in mine:
        sub = attn[
            :, :, t.src_layer_slice(my_layout), :, t.src_head_slice(my_layout), :
        ].contiguous()
        keep.append(sub)
        tag = t.tag(my_layout.num_layers, my_layout.num_heads, base_tag)
        handles.append(backend.isend(sub, dst=t.dst_rank, tag=tag))
    return PrefillHandoff(handles=handles, keepalive=keep)


def recv_request_kv_resharded(
    engine: Any,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    prompt_token_ids,
    *,
    backend: Optional[KVTransportBackend] = None,
    group: Optional[object] = None,
    base_tag: int = 0,
    device: Optional[torch.device] = None,
) -> Optional[dict]:
    """Hetero-layout decode receive: pull the KV sub-blocks covering this
    rank's (layer x head) rectangle and assemble the local staging
    tensor, then import. Header-free (schema derived from config+prompt)."""
    from megatron.core.inference.disaggregation.kv_shard_layout import (
        plan_kv_reshard,
        transfers_for_dst,
    )

    backend = backend or get_kv_transport_backend()
    meta = derive_decode_schema(engine, prompt_token_ids)
    if meta is None:
        return None
    if meta["has_mamba"]:
        raise NotImplementedError(
            "hetero KV reshard supports attention KV only in this MR"
        )

    bc = meta["block_count"]
    bs = meta["block_size_tokens"]
    hd = meta["hidden_per_head"]
    dtype = meta["attn_dtype"]
    if device is None:
        mb = getattr(engine.context, "memory_buffer", None)
        device = mb.device if mb is not None else None

    local_layers = my_layout.local_num_layers()
    local_heads = my_layout.local_num_heads()
    staging = torch.empty(
        bc, 2, local_layers, bs, local_heads, hd, dtype=dtype, device=device
    )

    plan = plan_kv_reshard(src_layouts, dst_layouts)
    mine = transfers_for_dst(plan, my_layout.global_rank)
    pending = []
    for t in mine:
        n_lay = t.g_layer1 - t.g_layer0
        n_head = t.g_head1 - t.g_head0
        tag = t.tag(my_layout.num_layers, my_layout.num_heads, base_tag)
        h = backend.irecv(
            (bc, 2, n_lay, bs, n_head, hd), dtype, src=t.src_rank, tag=tag, device=device
        )
        pending.append((t, h))
    for t, h in pending:
        sub = h.wait()
        staging[
            :, :, t.dst_layer_slice(my_layout), :, t.dst_head_slice(my_layout), :
        ] = sub

    payload = {
        "layout": "std_attn_v1",
        "block_count": bc,
        "block_size_tokens": bs,
        "num_layers": local_layers,
        "num_heads_per_partition": local_heads,
        "hidden_per_head": hd,
        "block_hashes": list(meta.get("block_hashes") or []),
        "staging_tensor": staging,
    }
    return engine.context.import_request_kv(payload)


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
    header_free: bool = True,
) -> Optional[PrefillHandoff]:
    """Prefill side: stage ``request_id``'s KV and ship it to ``dst``.

    Non-blocking on the data plane: returns a :class:`PrefillHandoff`
    whose ``wait()`` the caller can defer (e.g. until after the next
    engine step).

    With ``header_free=True`` (default) no metadata is sent -- the
    decode side reconstructs the schema from config + prompt; only the
    KV tensors go on the wire. With ``header_free=False`` a metadata
    object is sent first (the fallback for one-sided / hetero cases).

    Returns ``None`` if the request has no exportable KV (zero-length /
    MLA): in ``header_free`` mode this raises, since the contract is
    that a valid KV exists; in object mode it signals the decode side.
    """
    backend = backend or get_kv_transport_backend()
    payload = engine.context.export_request_kv(request_id)
    if payload is None:
        if header_free:
            raise ValueError(
                f"send_request_kv(header_free=True): request {request_id} has no "
                "exportable KV (MLA / zero-length). Use header_free=False or "
                "re-prefill on the decode side."
            )
        _send_object({"empty": True}, dst, group)  # signal nothing to receive
        return None

    meta = _build_metadata(payload)
    meta["empty"] = False
    if not header_free:
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
    header_free: bool = True,
    prompt_token_ids=None,
) -> Optional[dict]:
    """Decode side: receive a request's KV from ``src`` and import it.

    With ``header_free=True`` (default) the schema is derived locally
    from config + ``prompt_token_ids`` (required) -- no control message
    is received. With ``header_free=False`` the metadata object is
    received from the prefill side.

    Returns the dict from :meth:`import_request_kv` (``block_ids`` etc.)
    on success, or ``None`` if the prefill side had nothing to send or
    the import was refused (caller then re-prefills).
    """
    backend = backend or get_kv_transport_backend()
    if device is None:
        mb = getattr(engine.context, "memory_buffer", None)
        device = mb.device if mb is not None else None

    if header_free:
        if prompt_token_ids is None:
            raise ValueError(
                "recv_request_kv(header_free=True) requires prompt_token_ids "
                "to derive the KV schema."
            )
        meta = derive_decode_schema(engine, prompt_token_ids)
    else:
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
