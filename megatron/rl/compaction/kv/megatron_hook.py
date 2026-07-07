# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Paged-KV cache access for Megatron's DynamicInferenceContext.

The low-level read/write primitives every compaction consumer builds on:
reads (``get_kv_matrices``, ``get_kv_for_request``, ``approx_attention_scores``)
poll the live paged cache; writes (``apply_mask``, ``apply_mask_for_request``,
``apply_belief_memory*``) prune or replace a request's cache in place while
keeping ALL of the engine's per-request bookkeeping consistent (block table,
counts, last-block id/offset, kv_length_offsets). See kv/README.md for the
offset semantics these maintain.

Usage
-----
    hook = MegatronInferenceHook.from_engine(inference_engine)
    # Live post-prefill compaction builds on this in kv/live.py; the pomdp
    # recorder polls it per step in shadow mode.

Tensor parallelism
------------------
Each GPU holds ``num_attention_heads_per_partition`` heads.  With TP=2 and
16 total heads, each rank holds 8 heads.  ``get_kv_matrices()`` returns the
local partition's KV (shape: (B, S, n_kv_heads_local * d_head)).  The
Belief-Still compactor can be trained on this partition-local view; the
compressor learns to compress the local KV independently on each rank.

If you need the full multi-head KV across TP ranks, call
``get_kv_matrices(all_gather=True)`` (requires torch.distributed to be
initialised with a TP process group).

approx_attention_scores
--------------------
Returns a per-position importance proxy: mean L2 norm of K vectors averaged
over layers.  This correlates with attention importance and is sufficient for
TopK / H2O / StreamingLLM selection without requiring per-step forward hooks.
For exact attention weights, register PyTorch forward hooks on Megatron's
CoreAttention module during decode and accumulate them externally.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist

from .types import KVMask


class NullHook:
    """No-op hook for tests and when compaction is disabled."""

    def approx_attention_scores(self) -> list[float]:
        return []

    def apply_mask(self, mask: KVMask) -> None:
        pass

    def get_kv_matrices(self) -> None:
        return None

    def apply_belief_memory(self, memory: Any) -> None:
        pass

    def apply_belief_memory_for_request(self, b_local: int, memory: Any) -> None:
        pass


class MegatronInferenceHook:
    """Reads KV matrices from a live DynamicInferenceContext.

    Parameters
    ----------
    context:
        The ``DynamicInferenceContext`` from the inference engine.
        Access via ``engine.controller.inference_wrapped_model.inference_context``.
    tp_group:
        Optional torch.distributed process group for tensor-parallel all-gather.
        If None, only the local TP partition's KV is returned.
    """

    def __init__(
        self,
        context: Any,
        tp_group: Any | None = None,
    ) -> None:
        self._ctx = context
        self._tp_group = tp_group

    @classmethod
    def from_engine(
        cls,
        engine: Any,  # DynamicInferenceEngine
        tp_group: Any | None = None,
    ) -> "MegatronInferenceHook":
        """Convenience constructor: extract context from a DynamicInferenceEngine."""
        context = engine.controller.inference_wrapped_model.inference_context
        return cls(context, tp_group=tp_group)

    # ------------------------------------------------------------------
    # Shared context access
    # ------------------------------------------------------------------

    def _context_kv(self) -> tuple[Any, torch.Tensor, int] | None:
        """Return (ctx, memory_buffer, n_active) for the live cache.

        Returns None when there is no KV to read yet — the cache is not
        allocated or no requests are active.  This is a normal transient state
        for callers that poll every step.

        Hard-fails (NotImplementedError) on MLA caches: their latent KV layout
        is unsupported, so silently skipping would hide a misconfiguration.
        """
        ctx = self._ctx
        if getattr(ctx, "cache_mla_latent", False):
            raise NotImplementedError(
                "MLA (multi-latent attention) KV caches use a different layout and "
                "are not supported by KV compaction."
            )
        buf = getattr(ctx, "memory_buffer", None)
        if buf is None:
            return None
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return None
        return ctx, buf, n_active

    # ------------------------------------------------------------------
    # InferenceEngineHook protocol
    # ------------------------------------------------------------------

    def approx_attention_scores(self) -> list[float]:
        """APPROXIMATE per-position importance: mean ||K||₂ across layers.

        This is NOT real attention mass — it is a key-norm *proxy*. Positions with
        large key norms tend to receive more attention, so it is a cheap heuristic
        for TopK / H2O / StreamingLLM selection without registering per-step forward
        hooks. For exact attention weights, hook Megatron's CoreAttention during
        decode and accumulate the softmax outputs.

        Returns a flat list of floats (one per KV position) for the FIRST
        active request.  Returns [] when no active request or KV unavailable.
        """
        got = self._context_kv()
        if got is None:
            return []
        ctx, buf, _ = got

        b_global = ctx.paused_request_count  # first active request
        BS = ctx.block_size_tokens
        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
        seq_len = (n_blocks - 1) * BS + last_offset
        if seq_len <= 0:
            return []

        block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks]
        n_layers = ctx.num_attention_layers

        # keys: (n_layers, n_blocks, BS, H, D) → (n_layers, seq_len, H*D)
        k_blocks = buf[0, :, block_ids]                         # (L, n_blocks, BS, H, D)
        H, D = buf.shape[-2], buf.shape[-1]
        k_flat = k_blocks.reshape(n_layers, n_blocks * BS, H * D)[:, :seq_len, :]  # (L, S, H*D)

        # Mean L2 norm across layers and head dims → (S,)
        scores = k_flat.norm(dim=-1).mean(dim=0)                # (S,)
        return scores.float().cpu().tolist()

    def apply_mask(self, mask: KVMask) -> None:
        """Compact paged KV blocks so only retained_positions are kept.

        Gathers retained token slots from scattered blocks and writes them back
        contiguously, starting at block 0, slot 0.  Excess blocks are returned
        to the block allocator.  Metadata tensors
        (request_kv_block_counts, request_last_kv_block_offset,
        request_to_kv_block_ids) are updated in-place.

        Operates on ALL active requests using the SAME retained_positions,
        which is correct when the entire batch follows the same compaction
        schedule.  For per-request selection, extend KVMask with a batch
        dimension.
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "apply_mask: no live KV cache to compact (engine not allocated "
                "or no active requests)."
            )
        ctx, buf, n_active = got

        retained = sorted(mask.retained_positions)
        if len(retained) == 0:
            raise RuntimeError("apply_mask: retained_positions is empty; nothing to keep.")

        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            self._prune_request(ctx, buf, b_global, retained)

    def apply_mask_for_request(self, b_local: int, retained_positions: list[int]) -> None:
        """Prune ONE active request's paged KV to ``retained_positions``.

        Per-request variant of apply_mask — the seam used by live post-prefill
        compaction, where every request gets its own retained set. Positions are
        token-level: the paged cache shares one block table across all layers
        and heads, so a dropped position is dropped everywhere.
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "apply_mask_for_request: no live KV cache (engine not allocated "
                "or no active requests)."
            )
        ctx, buf, n_active = got
        if b_local >= n_active:
            raise RuntimeError(
                f"apply_mask_for_request: b_local={b_local} >= n_active={n_active}"
            )
        retained = sorted(retained_positions)
        if len(retained) == 0:
            raise RuntimeError("apply_mask_for_request: retained_positions is empty.")
        self._prune_request(ctx, buf, ctx.paused_request_count + b_local, retained)

    @staticmethod
    def _prune_request(ctx: Any, buf: torch.Tensor, b_global: int, retained: list[int]) -> None:
        """Gather ``retained`` token slots of one request and write them back
        contiguously from block 0 slot 0; release the freed blocks and update
        the request's block-table metadata in place."""
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers
        H, D = buf.shape[-2], buf.shape[-1]

        n_retained = len(retained)
        retained_idx = torch.tensor(retained, dtype=torch.long, device=buf.device)
        # Offset semantics (matches the engine's post-update state): the COUNT of
        # tokens in the last block. A retained count on a block boundary is
        # represented the way the engine represents it — data blocks full, plus a
        # trailing EMPTY current block with offset 0.
        n_data_blocks = math.ceil(n_retained / BS)
        new_last_offset = n_retained % BS
        n_new_blocks = n_data_blocks + (1 if new_last_offset == 0 else 0)

        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks].to(buf.device)

        if n_new_blocks > n_blocks:
            raise RuntimeError(
                f"_prune_request: need {n_new_blocks} blocks for {n_retained} retained "
                f"tokens but the request only has {n_blocks} — prune cannot grow a request."
            )

        # Gather all blocks → flatten → select retained → (n_layers, n_retained, H, D)
        k_flat = buf[0, :, block_ids].reshape(n_layers, n_blocks * BS, H, D)
        v_flat = buf[1, :, block_ids].reshape(n_layers, n_blocks * BS, H, D)
        k_ret = k_flat[:, retained_idx]
        v_ret = v_flat[:, retained_idx]

        # Write retained tokens back into the first data blocks.
        for bi in range(n_data_blocks):
            start = bi * BS
            end = min(start + BS, n_retained)
            chunk_len = end - start
            buf[0, :, block_ids[bi], :chunk_len] = k_ret[:, start:end]
            buf[1, :, block_ids[bi], :chunk_len] = v_ret[:, start:end]

        # Free excess blocks.
        if n_new_blocks < n_blocks:
            excess = block_ids[n_new_blocks:].clone()
            ctx.block_allocator.release_memory_blocks(excess)
            ctx.request_to_kv_block_ids[b_global, n_new_blocks:n_blocks] = -1

        # Keep ALL of the engine's per-request bookkeeping consistent with the
        # pruned cache: the next decode token's position id AND its write slot
        # both derive from request_kv_length_offsets (token_to_pos_ids =
        # kv_length_offsets), and the block-boundary logic reads
        # request_last_kv_block_id/offset.
        ctx.request_kv_block_counts[b_global] = n_new_blocks
        ctx.request_last_kv_block_offset[b_global] = new_last_offset
        ctx.request_last_kv_block_id[b_global] = block_ids[n_new_blocks - 1]
        ctx.request_kv_length_offsets[b_global] = n_retained

    def append_kv_to_request(
        self, b_local: int, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        """Append T tokens of K/V to the END of one active request's cache.

        The retrieval primitive: restores archived (evicted) KV spans back into
        the paged cache. keys/values: (n_layers, T, H, D), written after the
        request's current last token; new blocks are allocated as needed and
        every bookkeeping field is advanced (the appended tokens simply extend
        the attention span — sound for position-embedding-free models, which is
        the only kind live compaction supports).
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "append_kv_to_request: no live KV cache (engine not allocated "
                "or no active requests)."
            )
        ctx, buf, n_active = got
        if b_local >= n_active:
            raise RuntimeError(
                f"append_kv_to_request: b_local={b_local} >= n_active={n_active}"
            )
        b_global = ctx.paused_request_count + b_local
        BS = ctx.block_size_tokens
        T = keys.shape[1]
        if T < 1:
            raise RuntimeError("append_kv_to_request: nothing to append.")

        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
        cur_len = (n_blocks - 1) * BS + last_offset
        new_len = cur_len + T
        n_total_blocks = math.ceil(new_len / BS) + (1 if new_len % BS == 0 else 0)

        if n_total_blocks > n_blocks:
            extra = ctx.block_allocator.allocate_memory_blocks(n_total_blocks - n_blocks)
            if extra is None:
                raise RuntimeError(
                    f"append_kv_to_request: cannot allocate "
                    f"{n_total_blocks - n_blocks} blocks (allocator exhausted)."
                )
            ctx.request_to_kv_block_ids[
                b_global, n_blocks:n_total_blocks
            ] = extra.to(ctx.request_to_kv_block_ids.dtype)

        block_ids = ctx.request_to_kv_block_ids[b_global, :n_total_blocks].to(buf.device)
        keys = keys.to(device=buf.device, dtype=buf.dtype)
        values = values.to(device=buf.device, dtype=buf.dtype)
        pos = torch.arange(cur_len, new_len, device=buf.device)
        buf[0, :, block_ids[pos // BS], pos % BS] = keys
        buf[1, :, block_ids[pos // BS], pos % BS] = values

        ctx.request_kv_block_counts[b_global] = n_total_blocks
        ctx.request_last_kv_block_offset[b_global] = new_len % BS
        ctx.request_last_kv_block_id[b_global] = block_ids[n_total_blocks - 1]
        ctx.request_kv_length_offsets[b_global] = new_len

    def overwrite_keys_for_request(self, b_local: int, keys: torch.Tensor) -> None:
        """Rewrite one active request's cached KEYS in place; values untouched.

        The RoPE-renumber primitive: after a prune moves retained K/V to
        compacted slots, the delta-rotated keys (see ``rope.py``) are written
        back over them. ``keys``: (n_layers, S, H, D) where S must equal the
        request's current KV length.
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "overwrite_keys_for_request: no live KV cache (engine not "
                "allocated or no active requests)."
            )
        ctx, buf, n_active = got
        if b_local >= n_active:
            raise RuntimeError(
                f"overwrite_keys_for_request: b_local={b_local} >= n_active={n_active}"
            )
        b_global = ctx.paused_request_count + b_local
        BS = ctx.block_size_tokens
        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
        seq_len = (n_blocks - 1) * BS + last_offset
        if keys.shape[1] != seq_len:
            raise RuntimeError(
                f"overwrite_keys_for_request: got {keys.shape[1]} keys for a "
                f"request with KV length {seq_len}."
            )
        block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks].to(buf.device)
        keys = keys.to(device=buf.device, dtype=buf.dtype)
        pos = torch.arange(seq_len, device=buf.device)
        buf[0, :, block_ids[pos // BS], pos % BS] = keys

    def get_kv_for_request(
        self, b_local: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one active request's KV as (K, V), each (n_layers, S, H, D).

        Unpadded and head-explicit — the read primitive for live per-request
        scoring (S = the request's current KV length).
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "get_kv_for_request: no live KV cache (engine not allocated "
                "or no active requests)."
            )
        ctx, buf, n_active = got
        if b_local >= n_active:
            raise RuntimeError(
                f"get_kv_for_request: b_local={b_local} >= n_active={n_active}"
            )
        b_global = ctx.paused_request_count + b_local
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers
        H, D = buf.shape[-2], buf.shape[-1]

        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
        seq_len = (n_blocks - 1) * BS + last_offset
        block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks].to(buf.device)

        k = buf[0, :, block_ids].reshape(n_layers, n_blocks * BS, H, D)[:, :seq_len]
        v = buf[1, :, block_ids].reshape(n_layers, n_blocks * BS, H, D)[:, :seq_len]
        return k, v

    def get_kv_matrices(
        self,
        all_gather: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]] | None:
        """Return (keys_per_layer, values_per_layer) from the current KV cache.

        Each returned tensor has shape (B, S, H_local * d_head) where:
            B       = number of active requests (batch size)
            S       = max sequence length among active requests (shorter
                      requests are zero-padded)
            H_local = num_attention_heads_per_partition (local TP slice)
            d_head  = hidden_size_per_attention_head

        With all_gather=True, H_local is replaced by the full num_attention_heads
        after an all-gather across the TP group.

        Returns None when no active requests are in the context or the
        memory_buffer is not yet allocated.  Hard-fails on MLA caches.
        """
        got = self._context_kv()
        if got is None:
            return None
        ctx, buf, n_active = got

        active_slice = slice(ctx.paused_request_count, ctx.total_request_count)
        block_counts = ctx.request_kv_block_counts[active_slice].cpu()          # (B,)
        last_offsets = ctx.request_last_kv_block_offset[active_slice].cpu()     # (B,)
        block_ids_all = ctx.request_to_kv_block_ids[active_slice].cpu()         # (B, max_blocks)

        # seq_len[b] = (block_counts[b] - 1) * block_size + last_offsets[b] + 1
        BS = ctx.block_size_tokens
        seq_lens = (block_counts - 1) * BS + last_offsets   # (B,)
        max_seq = int(seq_lens.max().item())
        B = n_active

        n_layers = ctx.num_attention_layers
        H = ctx.num_attention_heads_per_partition
        D = ctx.hidden_size_per_attention_head

        keys_per_layer: list[torch.Tensor] = []
        vals_per_layer: list[torch.Tensor] = []

        # Gather each request's blocks once across ALL layers — one block-id
        # upload and one gather per request, not n_layers × B of each.
        k_all = torch.zeros(n_layers, B, max_seq, H * D, dtype=buf.dtype, device=buf.device)
        v_all = torch.zeros(n_layers, B, max_seq, H * D, dtype=buf.dtype, device=buf.device)
        for b in range(B):
            n_blocks = int(block_counts[b].item())
            seq_len = int(seq_lens[b].item())
            ids = block_ids_all[b, :n_blocks].to(buf.device)
            # (L, n_blocks, BS, H, D) → (L, n_blocks*BS, H*D), trimmed to seq_len
            k_all[:, b, :seq_len] = buf[0, :, ids].reshape(n_layers, n_blocks * BS, H * D)[:, :seq_len]
            v_all[:, b, :seq_len] = buf[1, :, ids].reshape(n_layers, n_blocks * BS, H * D)[:, :seq_len]

        for layer in range(n_layers):
            k_out = k_all[layer]
            v_out = v_all[layer]

            if all_gather and self._tp_group is not None:
                # All-gather across TP ranks along the head dimension.
                # Each rank contributes (B, S, H_local * d_head);
                # result is (B, S, H_total * d_head) after cat.
                k_list = [torch.zeros_like(k_out) for _ in range(dist.get_world_size(self._tp_group))]
                v_list = [torch.zeros_like(v_out) for _ in range(dist.get_world_size(self._tp_group))]
                dist.all_gather(k_list, k_out, group=self._tp_group)
                dist.all_gather(v_list, v_out, group=self._tp_group)
                k_out = torch.cat(k_list, dim=-1)
                v_out = torch.cat(v_list, dim=-1)

            keys_per_layer.append(k_out)
            vals_per_layer.append(v_out)

        return keys_per_layer, vals_per_layer

    def _inject_compact_request(
        self,
        ctx: Any,
        buf: torch.Tensor,
        b_global: int,
        keys: torch.Tensor,   # (n_layers, C, d_model)
        values: torch.Tensor,  # (n_layers, C, d_model)
    ) -> None:
        """Replace one request's paged KV with the C compact synthetic tokens.

        Frees the request's old blocks, allocates ceil(C / block_size) fresh
        blocks, writes the compact KV into them, and updates the metadata.
        Raises RuntimeError when the allocator is exhausted.
        """
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers
        H, D = buf.shape[-2], buf.shape[-1]
        C = keys.shape[1]
        n_data_blocks = math.ceil(C / BS)
        new_last_offset = C % BS
        n_new_blocks = n_data_blocks + (1 if new_last_offset == 0 else 0)

        n_old_blocks = int(ctx.request_kv_block_counts[b_global].item())
        old_block_ids = ctx.request_to_kv_block_ids[b_global, :n_old_blocks].to(buf.device)

        new_block_ids = ctx.block_allocator.allocate_memory_blocks(n_new_blocks)
        if new_block_ids is None:
            raise RuntimeError(
                f"apply_belief_memory: cannot allocate {n_new_blocks} KV blocks "
                "(allocator exhausted). Reduce n_compress or increase kv-cache-size."
            )
        new_block_ids = new_block_ids.to(buf.device)

        ctx.block_allocator.release_memory_blocks(old_block_ids)
        ctx.request_to_kv_block_ids[b_global, :n_old_blocks] = -1

        for layer in range(n_layers):
            k_compact = keys[layer].to(buf.device).reshape(C, H, D)
            v_compact = values[layer].to(buf.device).reshape(C, H, D)
            for bi in range(n_data_blocks):
                start = bi * BS
                end = min(start + BS, C)
                chunk = end - start
                buf[0, layer, new_block_ids[bi], :chunk] = k_compact[start:end]
                buf[1, layer, new_block_ids[bi], :chunk] = v_compact[start:end]

        ctx.request_to_kv_block_ids[b_global, :n_new_blocks] = new_block_ids
        ctx.request_kv_block_counts[b_global] = n_new_blocks
        ctx.request_last_kv_block_offset[b_global] = new_last_offset
        ctx.request_last_kv_block_id[b_global] = new_block_ids[-1]
        ctx.request_kv_length_offsets[b_global] = C

    def apply_belief_memory(self, memory: Any) -> None:
        """Inject compact BeliefMemory into the live KV cache for every request.

        Replaces the current paged KV for each active request with the C
        synthetic tokens from ``memory`` (keys/values shape
        (n_layers, B, C, d_model), d_model = H_local * d_head). B must equal the
        number of active requests. Raises RuntimeError on allocator exhaustion.
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "apply_belief_memory: no live KV cache (engine not allocated "
                "or no active requests)."
            )
        ctx, buf, n_active = got

        B_mem = memory.keys.shape[1]
        if B_mem != n_active:
            raise RuntimeError(
                f"apply_belief_memory: memory batch size {B_mem} != "
                f"active request count {n_active}"
            )

        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            self._inject_compact_request(
                ctx, buf, b_global, memory.keys[:, b_local], memory.values[:, b_local]
            )

    def apply_belief_memory_for_request(self, b_local: int, memory: Any) -> None:
        """Inject compact BeliefMemory into the KV cache for a single request.

        Same as apply_belief_memory() but for one batch element; ``memory`` has
        batch size 1 (keys shape (n_layers, 1, C, d_model)). Used by
        LiveKVCompactor's belief_still strategy per prefilled request.
        """
        got = self._context_kv()
        if got is None:
            raise RuntimeError(
                "apply_belief_memory_for_request: no live KV cache (engine not "
                "allocated or no active requests)."
            )
        ctx, buf, n_active = got
        if b_local >= n_active:
            raise RuntimeError(
                f"apply_belief_memory_for_request: b_local={b_local} >= "
                f"n_active={n_active}"
            )

        b_global = ctx.paused_request_count + b_local
        self._inject_compact_request(
            ctx, buf, b_global, memory.keys[:, 0], memory.values[:, 0]
        )
