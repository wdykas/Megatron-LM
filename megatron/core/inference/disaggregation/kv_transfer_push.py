# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill->decode KV transfer, push family: two-sided (NCCL) hand-off.

Both ranks post matched send/recv ops (the coordinator triggers both sides): the
prefill gathers its KV into a staging tensor, ships it, and the decode scatters
the received sub-blocks into its paged cache. The one-sided (pull) family lives
in ``kv_transfer_pull.py``; the header-free schema (:func:`derive_decode_schema`)
lives here because only the push receive path needs it.

Hybrid (Mamba) models hand off *block-boundary snapshots*, not the live
end-state (see the ``kv_transfer_pull`` docstring for why). The snapshot count
isn't derivable header-free, so PREFILL_DONE carries the snapshot hashes and
the decode sizes its receives from them. Snapshots move as whole per-slot
tensors between identical Mamba shards; a hetero remap skips them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation import kv_reshard, utils
from megatron.core.inference.disaggregation.transfer_backends.base import (
    KVTransportBackend,
    TransferHandle,
)
from megatron.core.inference.inference_request import compute_block_hashes_batched


def derive_decode_schema(engine: Any, prompt_token_ids) -> dict:
    """Reconstruct the KV schema (shapes/dtypes, block count) on the decode side
    with no control message -- computed locally from the engine's static config +
    the prompt tokens, so only KV tensors cross the wire. Raises
    ``NotImplementedError`` for the MLA latent cache (unsupported). Assumes the
    whole prompt is handed off (``block_count`` follows from the prompt length)
    into a uniform KV layout."""

    ctx = engine.context
    if ctx.cache_mla_latent:
        raise NotImplementedError(
            "disaggregated KV transfer does not support the MLA latent KV cache"
        )

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
    # The decode rebuilds its own staging tensor from its KVShardLayout
    # (local_num_layers/heads), so only the dtype + per-head width are needed here.
    return {
        "block_count": block_count,
        "block_size_tokens": bs,
        "hidden_per_head": int(mb.shape[5]),
        "block_hashes": block_hashes,
        "attn_dtype": mb.dtype,
    }


def matching_mamba_peer(my_mamba, peer_mamba_layouts) -> Optional[Any]:
    """The peer instance's rank holding the *identical* Mamba shard (same TP
    split, layer range, and structural dims), or ``None`` for a hetero remap.
    Snapshots move as whole per-slot tensors, so they ship only between
    identical shards; without a match the decode re-prefills past the last
    usable boundary instead. Unique when it exists (one rank per
    ``(tp_rank, layer_start)`` in an instance), and symmetric -- prefill and
    decode compute the same pairing from the same layout lists."""
    for peer in peer_mamba_layouts or []:
        if (
            peer.tp_size == my_mamba.tp_size
            and peer.tp_rank == my_mamba.tp_rank
            and peer.layer_start == my_mamba.layer_start
            and peer.num_layers == my_mamba.num_layers
            and peer.dims == my_mamba.dims
        ):
            return peer
    return None


@dataclass
class PrefillHandoff:
    """Prefill-side bookkeeping held until the transfer drains: keeps the staged
    tensors alive until :meth:`wait` completes."""

    handles: List[TransferHandle]
    keepalive: List[torch.Tensor] = field(default_factory=list)

    def wait(self) -> None:
        for h in self.handles:
            h.wait()
        self.keepalive.clear()


def send_request_kv_resharded(
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    *,
    backend: KVTransportBackend,
    payload: dict,
    my_mamba_layout,
    dst_mamba_layouts: list,
) -> "PrefillHandoff":
    """Hetero-layout prefill send: reshard this rank's pre-exported KV to the
    decode layout and ship it. ``my_layout`` is this rank's
    :class:`KVShardLayout`; ``src_layouts`` / ``dst_layouts`` are the full
    prefill / decode layout lists. Hybrid models also pass Mamba layouts, which
    gate the snapshot send (identical shard only). ``payload`` is the staging
    dict the context exported when the request was staged."""

    attn = payload["staging_tensor"]  # [BC, 2, local_layers, BS, local_heads, HD]
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    mine = utils.transfers_for_src(plan, my_layout.global_rank)
    # Collect every sub-block this request ships (attention, then Mamba
    # snapshots) and issue them as ONE coalesced batch. Posting dozens of
    # separate isends for a single request races on NCCL (un-grouped concurrent
    # P2P -> illegal memory access); batching wraps them in one ncclGroup so the
    # request's transfer is atomic. ``keep`` holds the staged slices alive until
    # the batch drains.
    sends: List[tuple] = []  # (tensor, dst)
    keep: List[torch.Tensor] = []
    for t in mine:
        sub = attn[
            :, :, t.src_layer_slice(my_layout), :, t.src_head_slice(my_layout), :
        ].contiguous()
        keep.append(sub)
        sends.append((sub, t.dst_rank))

    snapshots = payload["mamba_snapshots"]
    if snapshots is not None:
        peer = matching_mamba_peer(my_mamba_layout, dst_mamba_layouts)
        if peer is not None:
            # Whole-tensor snapshot send to the identical-shard peer, posted
            # after the attention sends (the recv side matches the post-order).
            for tensor in (snapshots["conv_states_tensor"], snapshots["ssm_states_tensor"]):
                keep.append(tensor)
                sends.append((tensor, peer.global_rank))
    handle, _ = backend.batch(sends, [])
    return PrefillHandoff(handles=[handle], keepalive=keep)


@dataclass
class DecodeRecv:
    """In-flight decode receive: the irecv handle + staging buffer it fills.
    :meth:`finish` waits the transfer, assembles the local KV tensor, and imports
    it -- letting the caller defer completion so it overlaps an engine step."""

    meta: dict
    staging: torch.Tensor
    pending: List[tuple]  # [(KVReshardTransfer, recv_buffer)]
    my_layout: Any
    # Single coalesced handle for the whole request's batched receives.
    handle: Optional[TransferHandle] = None
    # Mamba snapshots (hybrid, identical shard only): hashes + received tensors.
    snapshot_hashes: List[int] = field(default_factory=list)
    snapshot_conv: Optional[torch.Tensor] = None
    snapshot_ssm: Optional[torch.Tensor] = None

    def finish(self, engine: Any) -> Optional[dict]:
        """Wait the (single, coalesced) receive, assemble the staging
        tensor(s), and import them."""
        if self.handle is not None:
            self.handle.wait()
        for t, sub in self.pending:
            dst = self.staging[
                :, :, t.dst_layer_slice(self.my_layout), :, t.dst_head_slice(self.my_layout), :
            ]
            assert dst.shape == sub.shape, (
                f"DISAGG_RECV attn shape mismatch: dst={tuple(dst.shape)} recv={tuple(sub.shape)} "
                f"transfer=({t.global_layer_lo}:{t.global_layer_hi},{t.global_head_lo}:{t.global_head_hi}) "
                f"src={t.src_rank} dst_rank={t.dst_rank}"
            )
            dst.copy_(sub)
        mamba_snapshots = None
        if self.snapshot_hashes:
            mamba_snapshots = {
                "block_hashes": list(self.snapshot_hashes),
                "conv_states_tensor": self.snapshot_conv,
                "ssm_states_tensor": self.snapshot_ssm,
            }
        # import_request_kv writes the received KV into the cache + block
        # bookkeeping (inference tensors), but runs from the engine's message
        # loop (schedule_requests), outside the inference_mode the model step
        # uses -- re-enter it so the in-place writes are permitted.
        with torch.inference_mode():
            return engine.context.import_request_kv(
                self.staging,
                list(self.meta["block_hashes"]),
                mamba_snapshots=mamba_snapshots,
            )


def post_recv_request_kv_resharded(
    engine: Any,
    my_layout,
    src_layouts: list,
    dst_layouts: list,
    prompt_token_ids,
    *,
    backend: KVTransportBackend,
    handoff: Optional[dict],
    my_mamba_layout,
    src_mamba_layouts: list,
) -> "DecodeRecv":
    """Hetero-layout decode receive (non-blocking): post the irecv for every KV
    sub-block covering this rank's (layer x head) rectangle, plus the Mamba
    snapshot tensors for hybrid models (sized from the ``handoff``'s snapshot
    hashes; skipped for a hetero Mamba shard), and return a :class:`DecodeRecv`
    to complete later."""

    meta = derive_decode_schema(engine, prompt_token_ids)
    bc = meta["block_count"]
    bs = meta["block_size_tokens"]
    hd = meta["hidden_per_head"]
    dtype = meta["attn_dtype"]
    device = engine.context.memory_buffer.device

    staging = torch.empty(
        bc,
        2,
        my_layout.local_num_layers(),
        bs,
        my_layout.local_num_heads(),
        hd,
        dtype=dtype,
        device=device,
    )

    # Collect every sub-block this request receives (attention, then Mamba
    # snapshots) and post them as ONE coalesced batch -- mirrors the send side's
    # single batch so the request's transfer is atomic and ordered (un-grouped
    # concurrent irecvs race on NCCL -> illegal memory access).
    plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    attn_transfers = utils.transfers_for_dst(plan, my_layout.global_rank)
    recvs: List[tuple] = []  # (shape, dtype, src)
    for t in attn_transfers:
        n_lay = t.global_layer_hi - t.global_layer_lo
        n_head = t.global_head_hi - t.global_head_lo
        recvs.append(((bc, 2, n_lay, bs, n_head, hd), dtype, t.src_rank))

    recv = DecodeRecv(meta=meta, staging=staging, pending=[], my_layout=my_layout)
    snapshot_hashes = list((handoff or {}).get("snapshot_hashes") or [])
    if snapshot_hashes:
        peer = matching_mamba_peer(my_mamba_layout, src_mamba_layouts)
        if peer is not None:
            # Per-snapshot entry shapes come from this rank's own snapshot pools
            # (identical shard, so they equal the sender's staged shapes).
            sa = engine.context.mamba_slot_allocator
            n = len(snapshot_hashes)
            conv_entry = sa.conv_states.shape[:1] + sa.conv_states.shape[2:]
            ssm_entry = sa.ssm_states.shape[:1] + sa.ssm_states.shape[2:]
            recvs.append(((n, *conv_entry), sa.conv_states.dtype, peer.global_rank))
            recvs.append(((n, *ssm_entry), sa.ssm_states.dtype, peer.global_rank))
            recv.snapshot_hashes = snapshot_hashes

    handle, bufs = backend.batch([], recvs, device=device)
    recv.handle = handle
    n_attn = len(attn_transfers)
    recv.pending = list(zip(attn_transfers, bufs[:n_attn]))
    if recv.snapshot_hashes:
        recv.snapshot_conv, recv.snapshot_ssm = bufs[n_attn], bufs[n_attn + 1]
    return recv
