# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prefill->decode KV transfer, pull family: one-sided (RDMA, NIXL) hand-off.

The prefill registers its paged KV (and Mamba snapshot) buffers once; the
decode rank reads the prefill's blocks straight into its own freshly allocated
blocks, with no staging copy and no per-request registration. The prefill
publishes only references (block ids, snapshot refs, static region meta),
which the coordinator relays to the decode. The two-sided (push) family lives
in kv_transfer_push.py.

Hybrid (Mamba) models hand off block-boundary snapshots rather than the live
end-state: admission always re-runs at least the trailing tokens of the
prompt, and the recurrent Mamba state is only correct when restored at the
block boundary the re-run starts from, which is what the prefix-cache restore
path does with the imported snapshots. Snapshots move as whole per-slot
entries, so they ship only between identical Mamba shards; a hetero remap
skips them and the decode re-prefills past the last usable boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from megatron.core.inference.disaggregation import kv_reshard, utils


def pull_static_meta(backend, my_layout, kv_dims):
    """Build this prefill rank's request-invariant pull metadata.

    Gathered once at registration and merged with pull_request_meta per
    request, so the per-request hand-off carries only block references.

    Args:
        backend: this rank's one-sided (NIXL) transport, regions already
            registered.
        my_layout: this rank's KVShardLayout.
        kv_dims: KV buffer geometry {num_layers, total_blocks, block_size,
            heads, hidden, elem}.
    """
    return {
        "global_rank": int(my_layout.global_rank),
        "region_meta": backend.export_regions_meta(),
        "kv_dims": kv_dims,
    }


def pull_request_meta(ref_payload):
    """Build a request's block-level references, the per-request half of the
    hand-off.

    Keeps only the fields that vary per request; the static geometry goes in
    pull_static_meta. Block ids are replicated across the MP group (every rank
    schedules identically), so one rank's copy is authoritative for all.

    Args:
        ref_payload: the dict returned by context.export_request_kv_ref().

    Returns:
        {block_ids, block_hashes, block_count, snapshots}.
    """
    return {
        "block_ids": ref_payload["block_ids"],
        "block_hashes": ref_payload["block_hashes"],
        "block_count": ref_payload["block_count"],
        "snapshots": ref_payload["snapshots"],
    }


@dataclass
class NixlPullRecv:
    """Decode side: an in-flight one-sided read of a request's blocks.

    finish() waits the read, then commits by registering the block hashes (the
    KV already landed in the allocated blocks). If the prefill published Mamba
    snapshots, a second read pulls them after the KV commit. Only the KV
    blocks are ref-count pinned on the prefill; the snapshot slots stay valid
    for the second read because snapshot eviction is gated on that same
    ref count.

    Attributes:
        handles: pollable transport handles for the in-flight reads.
        block_ids: destination KV block ids (reused prefix + newly pulled).
        block_hashes: per-block hashes to register on commit.
        backend: the one-sided transport, for the snapshot read.
        peer_meta: the single source's region meta; None when snapshots are
            skipped (multi-source or hetero Mamba shard).
        snapshots: prefill's Mamba snapshot refs to pull, or empty.
    """

    handles: List[Any]
    block_ids: List[int]
    block_hashes: List[int]
    backend: Any
    peer_meta: Optional[dict] = None
    snapshots: List = field(default_factory=list)

    def poll(self) -> bool:
        """Return True if every in-flight read for this request has drained,
        without blocking. No handles (e.g. a full prefix hit) reads as done."""
        return all(h.poll() for h in self.handles)

    def finish(self, engine: Any) -> dict:
        """Wait the reads, commit the KV, and pull any Mamba snapshots.

        Returns:
            The import result dict from disagg_pull_commit.
        """
        for h in self.handles:
            h.wait()
        with torch.inference_mode():
            result = engine.context.disagg_pull_commit(self.block_ids, self.block_hashes)
            if self.snapshots:
                # Resolve hashes to the blocks the commit just registered,
                # allocate snapshot slots, read the peer's snapshots into them,
                # and register them for prefix-cache hits.
                plan = engine.context.disagg_snapshot_pull_plan(self.snapshots)
                if plan is not None:
                    self.backend.begin_pull(self.peer_meta, plan["transfers"]).wait()
                    engine.context.disagg_snapshot_commit(plan["block_ids"], plan["hashes"])
            return result


def _ctx_kv_dims(ctx) -> dict:
    """Return this rank's KV memory_buffer geometry: {num_layers,
    total_blocks, block_size, heads, hidden, elem}. The decode uses it as the
    destination side of the byte-offset math; the prefill publishes it as the
    hand-off's kv_dims."""
    mb = ctx.memory_buffer
    return {
        "num_layers": int(mb.shape[1]),
        "total_blocks": int(mb.shape[2]),
        "block_size": int(mb.shape[3]),
        "heads": int(mb.shape[4]),
        "hidden": int(mb.shape[5]),
        "elem": int(mb.element_size()),
    }


def _check_pull_dims(src_dims: dict, dst_dims: dict) -> None:
    """Raise if prefill and decode disagree on geometry the byte-offset math
    assumes is shared. Heads, layers, and total_blocks may differ across
    TP/PP; block size, per-head width, and element size must match."""
    for key in ("block_size", "hidden", "elem"):
        if src_dims[key] != dst_dims[key]:
            raise RuntimeError(
                f"disagg pull: prefill/decode KV geometry mismatch on {key!r}: "
                f"src={src_dims[key]} dst={dst_dims[key]}"
            )


def _kv_fragment_descriptors(src_dims, dst_dims, src_block, dst_block,
                             src_layer_slice, dst_layer_slice, src_head_slice, dst_head_slice):
    """Build (local_offset, remote_offset, nbytes) read descriptors for one
    block's head/layer fragment, coalesced to the contiguous minimum.

    In the row-major (2, num_layers, total_blocks, block_size, heads, hidden)
    buffer, a head range is contiguous across tokens only when it spans all
    heads. When the fragment takes the full head dim on both sides (same TP),
    the whole (block_size, heads, hidden) region is one contiguous run per
    (kv, layer): one descriptor. A partial head range (a TP remap) is
    contiguous only per (kv, layer, token): one descriptor each. The full-head
    form matches the whole-block stride math, so the same-TP case moves blocks
    at the minimal descriptor count without a separate code path.

    Args:
        src_dims: source (prefill) KV geometry, from the hand-off's kv_dims.
        dst_dims: destination (decode) KV geometry, from _ctx_kv_dims().
        src_block: source physical block id.
        dst_block: destination physical block id.
        src_layer_slice / dst_layer_slice: layer range on each side.
        src_head_slice / dst_head_slice: head range on each side.

    Returns:
        A list of (local_offset, remote_offset, nbytes) byte descriptors.
    """
    hidden = dst_dims["hidden"]
    elem = dst_dims["elem"]
    src_num_layers, src_total_blocks, src_block_size, src_heads = (
        src_dims["num_layers"], src_dims["total_blocks"], src_dims["block_size"], src_dims["heads"]
    )
    dst_num_layers, dst_total_blocks, dst_block_size, dst_heads = (
        dst_dims["num_layers"], dst_dims["total_blocks"], dst_dims["block_size"], dst_dims["heads"]
    )
    src_layer_ids = range(src_layer_slice.start, src_layer_slice.stop)
    dst_layer_ids = range(dst_layer_slice.start, dst_layer_slice.stop)
    full_head_span = (
        src_head_slice.start == 0 and src_head_slice.stop == src_heads
        and dst_head_slice.start == 0 and dst_head_slice.stop == dst_heads
        and src_heads == dst_heads
    )
    descriptors = []
    if full_head_span:
        nbytes = src_block_size * src_heads * hidden * elem
        for kv in (0, 1):
            for src_layer, dst_layer in zip(src_layer_ids, dst_layer_ids):
                src_offset = (((kv * src_num_layers + src_layer) * src_total_blocks + src_block) * src_block_size) * src_heads * hidden * elem
                dst_offset = (((kv * dst_num_layers + dst_layer) * dst_total_blocks + dst_block) * dst_block_size) * dst_heads * hidden * elem
                descriptors.append((dst_offset, src_offset, nbytes))  # (local=dst, remote=src, nbytes)
        return descriptors
    head_count = src_head_slice.stop - src_head_slice.start
    nbytes = head_count * hidden * elem
    for kv in (0, 1):
        for src_layer, dst_layer in zip(src_layer_ids, dst_layer_ids):
            for token in range(src_block_size):
                src_offset = ((((kv * src_num_layers + src_layer) * src_total_blocks + src_block) * src_block_size + token) * src_heads + src_head_slice.start) * hidden * elem
                dst_offset = ((((kv * dst_num_layers + dst_layer) * dst_total_blocks + dst_block) * dst_block_size + token) * dst_heads + dst_head_slice.start) * hidden * elem
                descriptors.append((dst_offset, src_offset, nbytes))
    return descriptors


def _snapshot_regions_match(backend, peer_meta: dict) -> bool:
    """Return whether the peer's Mamba snapshot pools have the same per-entry
    geometry as this rank's. Snapshots move as whole per-slot entries, so a
    hetero shard cannot use them; the caller skips the snapshot read and the
    decode re-prefills past the last usable boundary instead."""
    local = backend.export_regions_meta()["regions"]
    peer = peer_meta["regions"]
    return all(
        name in local and name in peer
        and local[name]["num_outer"] == peer[name]["num_outer"]
        and local[name]["inner_bytes"] == peer[name]["inner_bytes"]
        for name in ("snap_conv", "snap_ssm")
    )


def post_pull_request_kv(engine, backend, rank_handoffs, my_layout,
                         src_layouts, dst_layouts):
    """Allocate destination blocks and issue the one-sided reads pulling the
    request's KV into them (decode side).

    Driven by kv_reshard.plan_kv_reshard: each decode rank reads its
    (layer x head) shard as byte fragments from the prefill ranks that hold
    it. Mamba snapshots are carried only when a single prefill rank with an
    identical Mamba shard holds this request's data; a multi-source or hetero
    remap skips them.

    Args:
        engine: the decode engine.
        backend: this rank's one-sided (NIXL) transport.
        rank_handoffs: the per-prefill-rank hand-offs relayed from
            PREFILL_DONE, each {**pull_static_meta, **pull_request_meta}.
        my_layout: this decode rank's KVShardLayout.
        src_layouts / dst_layouts: full prefill / decode KV layout lists.

    Returns:
        A NixlPullRecv to complete later, or None if the decode KV cache is
        full.
    """
    if not rank_handoffs:
        raise RuntimeError("disagg pull: RECV_KV carried an empty hand-off list")
    block_count = int(rank_handoffs[0]["block_count"])
    # Reuse the longest block prefix the decode already has cached (hashes are
    # TP-independent) and pull only blocks [match_len, block_count).
    match = engine.context.disagg_pull_match_prefix(rank_handoffs[0]["block_hashes"])
    reused, match_len = match["reused_block_ids"], match["match_len"]
    alloc = engine.context.disagg_pull_alloc(block_count - match_len)
    if alloc is None:
        engine.context.disagg_pull_unmatch(reused)
        return None
    dst_block_ids = list(reused) + list(alloc["block_ids"])
    dst_dims = _ctx_kv_dims(engine.context)
    handoff_by_rank = {int(h["global_rank"]): h for h in rank_handoffs}
    src_layout_by_rank = {layout.global_rank: layout for layout in src_layouts}
    kv_plan = kv_reshard.plan_kv_reshard(src_layouts, dst_layouts)
    handles = []
    source_handoffs: dict = {}  # src_rank -> handoff (the ranks this dst reads from)
    # Attention KV: per source rank, read its head/layer fragments of the
    # missing suffix. Full-head fragments coalesce to whole blocks.
    for transfer in utils.transfers_for_dst(kv_plan, my_layout.global_rank):
        # Every source rank in the plan published static meta, so a miss is a
        # protocol violation; skipping it would commit never-filled blocks.
        src_handoff = handoff_by_rank[transfer.src_rank]
        src_layout = src_layout_by_rank[transfer.src_rank]
        _check_pull_dims(src_handoff["kv_dims"], dst_dims)
        source_handoffs[transfer.src_rank] = src_handoff
        src_layer_slice = transfer.src_layer_slice(src_layout)
        src_head_slice = transfer.src_head_slice(src_layout)
        dst_layer_slice = transfer.dst_layer_slice(my_layout)
        dst_head_slice = transfer.dst_head_slice(my_layout)
        descriptors = []
        for block_idx in range(match_len, block_count):
            descriptors += _kv_fragment_descriptors(
                src_handoff["kv_dims"], dst_dims,
                int(src_handoff["block_ids"][block_idx]), int(dst_block_ids[block_idx]),
                src_layer_slice, dst_layer_slice, src_head_slice, dst_head_slice,
            )
        if descriptors:
            handles.append(backend.begin_pull_raw(src_handoff["region_meta"], "kv", descriptors))

    # Snapshots ship only when a single prefill rank with an identical Mamba
    # shard holds this request's data, so the snapshot read in finish()
    # resolves against one peer as whole entries.
    peer_meta, snapshots = None, []
    if len(source_handoffs) == 1:
        single_source = next(iter(source_handoffs.values()))
        if single_source["snapshots"] and _snapshot_regions_match(
            backend, single_source["region_meta"]
        ):
            peer_meta = single_source["region_meta"]
            snapshots = list(single_source["snapshots"])
    return NixlPullRecv(
        handles=handles, block_ids=dst_block_ids,
        block_hashes=list(rank_handoffs[0]["block_hashes"]),
        backend=backend, peer_meta=peer_meta, snapshots=snapshots,
    )
