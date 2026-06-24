# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Concrete InferenceEngineHook backed by DynamicInferenceContext.

Reads per-layer KV matrices directly from Megatron's paged KV cache
after each inference step, making them available for BeliefUpdater training
and KV compaction benchmarking.  Also supports live compaction via
``apply_mask()`` and ``apply_belief_memory()``.

Usage
-----
    # At inference engine startup (inside MegatronLocal.launch):
    hook = MegatronInferenceHook.from_engine(inference_engine)

    # Pass to the recorder:
    recorder = PomdpRolloutRecorder(..., kv_hook=hook, belief_updater=updater)

    # That's it.  The recorder calls hook methods after each step.

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
        ctx = self._ctx
        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return []
        if getattr(ctx, "cache_mla_latent", False):
            return []
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return []

        b_global = ctx.paused_request_count  # first active request
        BS = ctx.block_size_tokens
        n_blocks = int(ctx.request_kv_block_counts[b_global].item())
        last_offset = int(ctx.request_last_kv_block_offset[b_global].item())
        seq_len = (n_blocks - 1) * BS + last_offset + 1
        if seq_len <= 0:
            return []

        block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks]
        buf = ctx.memory_buffer   # (2, n_layers, total_blocks, BS, H, D)
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
        ctx = self._ctx
        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return
        if getattr(ctx, "cache_mla_latent", False):
            return

        retained = sorted(mask.retained_positions)
        n_retained = len(retained)
        if n_retained == 0:
            return

        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return

        buf = ctx.memory_buffer   # (2, n_layers, total_blocks, BS, H, D)
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers

        retained_idx = torch.tensor(retained, dtype=torch.long, device=buf.device)
        n_new_blocks = math.ceil(n_retained / BS)
        new_last_offset = (n_retained - 1) % BS

        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            n_blocks = int(ctx.request_kv_block_counts[b_global].item())
            block_ids = ctx.request_to_kv_block_ids[b_global, :n_blocks].to(buf.device)

            # Gather all blocks → (n_layers, n_blocks, BS, H, D)
            k_all = buf[0, :, block_ids]
            v_all = buf[1, :, block_ids]

            H, D = buf.shape[-2], buf.shape[-1]
            # Flatten blocks → (n_layers, n_blocks*BS, H, D)
            k_flat = k_all.reshape(n_layers, n_blocks * BS, H, D)
            v_flat = v_all.reshape(n_layers, n_blocks * BS, H, D)

            # Gather retained positions → (n_layers, n_retained, H, D)
            k_ret = k_flat[:, retained_idx]
            v_ret = v_flat[:, retained_idx]

            # Write retained tokens back into first n_new_blocks blocks
            for bi in range(n_new_blocks):
                start = bi * BS
                end = min(start + BS, n_retained)
                chunk_len = end - start
                buf[0, :, block_ids[bi], :chunk_len] = k_ret[:, start:end]
                buf[1, :, block_ids[bi], :chunk_len] = v_ret[:, start:end]

            # Free excess blocks
            if n_new_blocks < n_blocks:
                excess = block_ids[n_new_blocks:].clone()
                ctx.block_allocator.release_memory_blocks(excess)
                ctx.request_to_kv_block_ids[b_global, n_new_blocks:n_blocks] = -1

            ctx.request_kv_block_counts[b_global] = n_new_blocks
            ctx.request_last_kv_block_offset[b_global] = new_last_offset

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

        Returns None when:
            • No active requests are currently in the context.
            • The model uses MLA (multi-latent attention) — not yet supported.
            • memory_buffer is not yet allocated.
        """
        ctx = self._ctx

        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return None

        if ctx.cache_mla_latent:
            # MLA format: (n_layers, total_blocks, block_size, kv_reduced_dim)
            # Different layout — not yet supported for compaction research.
            return None

        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return None

        active_slice = slice(ctx.paused_request_count, ctx.total_request_count)
        block_counts = ctx.request_kv_block_counts[active_slice].cpu()          # (B,)
        last_offsets = ctx.request_last_kv_block_offset[active_slice].cpu()     # (B,)
        block_ids_all = ctx.request_to_kv_block_ids[active_slice].cpu()         # (B, max_blocks)

        # seq_len[b] = (block_counts[b] - 1) * block_size + last_offsets[b] + 1
        BS = ctx.block_size_tokens
        seq_lens = (block_counts - 1) * BS + last_offsets + 1   # (B,)
        max_seq = int(seq_lens.max().item())
        B = n_active

        buf = ctx.memory_buffer   # (2, n_layers, total_blocks, BS, H_local, d_head)
        n_layers = ctx.num_attention_layers
        H = ctx.num_attention_heads_per_partition
        D = ctx.hidden_size_per_attention_head

        keys_per_layer: list[torch.Tensor] = []
        vals_per_layer: list[torch.Tensor] = []

        for layer in range(n_layers):
            # Allocate output buffers (padded to max_seq with zeros)
            k_out = torch.zeros(B, max_seq, H * D, dtype=buf.dtype, device=buf.device)
            v_out = torch.zeros(B, max_seq, H * D, dtype=buf.dtype, device=buf.device)

            for b in range(B):
                n_blocks = int(block_counts[b].item())
                seq_len = int(seq_lens[b].item())
                req_block_ids = block_ids_all[b, :n_blocks]  # (n_blocks,) on CPU

                # Gather blocks: (n_blocks, BS, H_local, d_head)
                k_blocks = buf[0, layer, req_block_ids.to(buf.device)]
                v_blocks = buf[1, layer, req_block_ids.to(buf.device)]

                # Flatten blocks → (n_blocks * BS, H_local, d_head), trim to seq_len
                k_flat = k_blocks.reshape(n_blocks * BS, H, D)[:seq_len]
                v_flat = v_blocks.reshape(n_blocks * BS, H, D)[:seq_len]

                k_out[b, :seq_len] = k_flat.reshape(seq_len, H * D)
                v_out[b, :seq_len] = v_flat.reshape(seq_len, H * D)

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

    def apply_belief_memory(self, memory: Any) -> None:
        """Inject compact BeliefMemory into the live KV cache.

        Replaces the current paged KV for every active request with the C
        synthetic tokens from ``memory``.  The old blocks are freed and new
        blocks (ceil(C / block_size)) are allocated and filled.

        ``memory.keys`` shape: (n_layers, B, C, d_model)
        ``memory.values`` shape: (n_layers, B, C, d_model)

        where d_model = H_local * d_head (TP-partition-local).
        B must equal the number of active requests.

        Raises RuntimeError when the block allocator cannot satisfy the
        allocation (out of KV memory).
        """
        ctx = self._ctx
        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return
        if getattr(ctx, "cache_mla_latent", False):
            return

        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return

        buf = ctx.memory_buffer   # (2, n_layers, total_blocks, BS, H, D)
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers
        H, D = buf.shape[-2], buf.shape[-1]

        # memory.keys: (n_layers, B, C, d_model)
        compact_keys = memory.keys    # stays on its device (possibly different from buf)
        compact_vals = memory.values
        B_mem, C = compact_keys.shape[1], compact_keys.shape[2]

        if B_mem != n_active:
            raise RuntimeError(
                f"apply_belief_memory: memory batch size {B_mem} != "
                f"active request count {n_active}"
            )

        n_new_blocks = math.ceil(C / BS)
        new_last_offset = (C - 1) % BS

        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            n_old_blocks = int(ctx.request_kv_block_counts[b_global].item())
            old_block_ids = ctx.request_to_kv_block_ids[b_global, :n_old_blocks].to(buf.device)

            # Allocate fresh blocks for compact memory
            new_block_ids = ctx.block_allocator.allocate_memory_blocks(n_new_blocks)
            if new_block_ids is None:
                raise RuntimeError(
                    f"apply_belief_memory: cannot allocate {n_new_blocks} KV blocks "
                    f"(allocator exhausted).  Reduce n_compress or increase kv-cache-size."
                )
            new_block_ids = new_block_ids.to(buf.device)

            # Free old blocks
            ctx.block_allocator.release_memory_blocks(old_block_ids)
            ctx.request_to_kv_block_ids[b_global, :n_old_blocks] = -1

            # Write compact KV into new blocks
            # compact_keys[layer, b_local] shape: (C, d_model) = (C, H*D)
            for layer in range(n_layers):
                k_compact = compact_keys[layer, b_local].to(buf.device)  # (C, H*D)
                v_compact = compact_vals[layer, b_local].to(buf.device)

                # Reshape: (C, H, D)
                k_compact = k_compact.reshape(C, H, D)
                v_compact = v_compact.reshape(C, H, D)

                for bi in range(n_new_blocks):
                    start = bi * BS
                    end = min(start + BS, C)
                    chunk = end - start
                    buf[0, layer, new_block_ids[bi], :chunk] = k_compact[start:end]
                    buf[1, layer, new_block_ids[bi], :chunk] = v_compact[start:end]

            # Update metadata
            ctx.request_to_kv_block_ids[b_global, :n_new_blocks] = new_block_ids
            ctx.request_kv_block_counts[b_global] = n_new_blocks
            ctx.request_last_kv_block_offset[b_global] = new_last_offset

    def apply_belief_memory_for_request(self, b_local: int, memory: Any) -> None:
        """Inject compact BeliefMemory into the KV cache for a single request.

        Same as apply_belief_memory() but operates on one batch element only.
        ``memory`` must have batch size 1 (memory.keys shape: (n_layers, 1, C, d_model)).

        This is the per-request variant used by BeliefServerCompactor when
        the engine has multiple active requests and only one needs updating.
        """
        ctx = self._ctx
        if not hasattr(ctx, "memory_buffer") or ctx.memory_buffer is None:
            return
        if getattr(ctx, "cache_mla_latent", False):
            return

        n_active = ctx.total_request_count - ctx.paused_request_count
        if b_local >= n_active:
            raise RuntimeError(
                f"apply_belief_memory_for_request: b_local={b_local} >= "
                f"n_active={n_active}"
            )

        buf = ctx.memory_buffer   # (2, n_layers, total_blocks, BS, H, D)
        BS = ctx.block_size_tokens
        n_layers = ctx.num_attention_layers
        H, D = buf.shape[-2], buf.shape[-1]

        # memory.keys: (n_layers, 1, C, d_model)
        C = memory.keys.shape[2]
        compact_keys = memory.keys   # (n_layers, 1, C, d_model)
        compact_vals = memory.values

        n_new_blocks = math.ceil(C / BS)
        new_last_offset = (C - 1) % BS

        b_global = ctx.paused_request_count + b_local
        n_old_blocks = int(ctx.request_kv_block_counts[b_global].item())
        old_block_ids = ctx.request_to_kv_block_ids[b_global, :n_old_blocks].to(buf.device)

        new_block_ids = ctx.block_allocator.allocate_memory_blocks(n_new_blocks)
        if new_block_ids is None:
            raise RuntimeError(
                f"apply_belief_memory_for_request: cannot allocate {n_new_blocks} "
                "KV blocks (allocator exhausted)."
            )
        new_block_ids = new_block_ids.to(buf.device)

        ctx.block_allocator.release_memory_blocks(old_block_ids)
        ctx.request_to_kv_block_ids[b_global, :n_old_blocks] = -1

        for layer in range(n_layers):
            k_compact = compact_keys[layer, 0].to(buf.device)   # (C, d_model)
            v_compact = compact_vals[layer, 0].to(buf.device)
            k_compact = k_compact.reshape(C, H, D)
            v_compact = v_compact.reshape(C, H, D)
            for bi in range(n_new_blocks):
                start = bi * BS
                end = min(start + BS, C)
                chunk = end - start
                buf[0, layer, new_block_ids[bi], :chunk] = k_compact[start:end]
                buf[1, layer, new_block_ids[bi], :chunk] = v_compact[start:end]

        ctx.request_to_kv_block_ids[b_global, :n_new_blocks] = new_block_ids
        ctx.request_kv_block_counts[b_global] = n_new_blocks
        ctx.request_last_kv_block_offset[b_global] = new_last_offset
