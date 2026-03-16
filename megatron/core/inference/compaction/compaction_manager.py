# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compaction manager for paged KV cache.

Orchestrates offline (Phase 1) and online (Phase 2) compaction of the
paged KV cache in Megatron-LM's inference engine.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .am_compaction import am_compact, am_compact_with_mass, AMCompactionResult
from .kv_utils import gather_kv, write_kv, write_kv_with_biases


@dataclass
class CompactionConfig:
    """Configuration for KV cache compaction."""

    # Budget: number of compacted memory tokens per layer
    memory_budget: int = 512

    # Hot window: number of recent tokens to keep uncompacted
    hot_window: int = 256

    # Compaction trigger: compact every N new tokens
    compact_every_n: int = 256

    # AM algorithm settings
    method: str = "omp"  # "omp" or "top_attention"
    use_mass_matching: bool = True
    nnls_iters: int = 100
    omp_keys_per_iter: int = 4
    omp_refit_every: int = 2

    # Reference query settings
    num_ref_queries: int = 64  # R: number of reference queries to use

    # Attention scale (None = 1/sqrt(d))
    attention_scale: Optional[float] = None

    # Whether to store per-token biases
    store_biases: bool = True

    # Per-head budget (None = uniform)
    per_head_budgets: Optional[Dict[int, int]] = None

    # Compute dtype for compaction
    compute_dtype: torch.dtype = torch.float32


@dataclass
class PerSequenceCompactionState:
    """Tracks compaction state for a single sequence."""

    # Number of tokens that have been compacted into memory
    compacted_token_count: int = 0

    # Logical position counter (monotonic, never rewinds for RoPE)
    logical_pos: int = 0

    # Block IDs for memory region (compacted KV)
    mem_block_ids: Optional[Tensor] = None

    # Number of valid tokens in memory blocks
    mem_token_count: int = 0

    # Block IDs for hot region (recent uncompacted KV)
    hot_block_ids: Optional[Tensor] = None

    # Number of tokens in hot region
    hot_token_count: int = 0

    # Number of compaction rounds performed
    compaction_count: int = 0

    # Last compaction metrics per layer
    last_mass_errors: Optional[List[float]] = None
    last_output_errors: Optional[List[float]] = None

    # Total tokens generated since creation
    tokens_generated: int = 0


class CompactionManager:
    """Manages KV cache compaction for the paged inference engine.

    Supports:
      - Offline compaction (Phase 1): single-shot compact of a built cache
      - Online compaction (Phase 2): periodic compact during decoding
      - Two-tier cache: [mem_pages] + [hot_pages] block table layout
    """

    def __init__(
        self,
        config: CompactionConfig,
        memory_buffer: Tensor,
        block_allocator: "BlockAllocator",
        block_size: int,
        num_layers: int,
        bias_buffer: Optional[Tensor] = None,
    ):
        """Initialize compaction manager.

        Args:
            config: Compaction configuration.
            memory_buffer: 6D KV cache tensor
                (2, num_layers, num_blocks, block_size, num_heads, head_dim).
            block_allocator: Block allocator for page management.
            block_size: Tokens per block.
            num_layers: Number of attention layers.
            bias_buffer: Optional bias storage
                (num_layers, num_blocks, block_size). Created if None and
                config.store_biases is True.
        """
        self.config = config
        self.memory_buffer = memory_buffer
        self.block_allocator = block_allocator
        self.block_size = block_size
        self.num_layers = num_layers

        if config.store_biases and bias_buffer is None:
            num_blocks_total = memory_buffer.shape[2]
            num_heads = memory_buffer.shape[4]
            self.bias_buffer = torch.zeros(
                (num_layers, num_blocks_total, block_size, num_heads),
                dtype=memory_buffer.dtype,
                device=memory_buffer.device,
            )
        else:
            self.bias_buffer = bias_buffer

        # Per-sequence state
        self.seq_states: Dict[int, PerSequenceCompactionState] = {}

        self.logger = logging.getLogger(__name__)

    def register_sequence(self, seq_id: int, initial_pos: int = 0) -> None:
        """Register a new sequence for compaction tracking."""
        self.seq_states[seq_id] = PerSequenceCompactionState(
            logical_pos=initial_pos,
        )

    def unregister_sequence(self, seq_id: int) -> None:
        """Unregister a sequence and free its memory blocks."""
        state = self.seq_states.pop(seq_id, None)
        if state is not None and state.mem_block_ids is not None:
            self.block_allocator.release_memory_blocks(state.mem_block_ids)

    # =========================================================================
    # Phase 1: Offline compaction
    # =========================================================================

    def compact_offline(
        self,
        block_ids: Tensor,
        token_count: int,
        Q_ref: Optional[Tensor] = None,
        Q_ref_per_layer: Optional[Dict[int, Tensor]] = None,
    ) -> Tuple[Tensor, int, Dict[str, float]]:
        """Perform offline (single-shot) compaction of a built KV cache.

        Takes the full KV cache stored in the given blocks, compacts it
        to config.memory_budget tokens per layer, and writes the result
        into newly allocated blocks.

        Args:
            block_ids: (num_blocks,) block IDs of the full cache.
            token_count: Total valid tokens in the cache.
            Q_ref: (R, H, D) reference queries (shared across layers).
                If None and Q_ref_per_layer is None, uses random queries.
            Q_ref_per_layer: Dict mapping layer -> (R, H, D) queries.
                Takes precedence over Q_ref.

        Returns:
            new_block_ids: (num_new_blocks,) block IDs for compacted cache.
            mem_token_count: Number of compacted tokens (= budget).
            metrics: Dict of compaction quality metrics.
        """
        M = self.config.memory_budget
        num_new_blocks = math.ceil(M / self.block_size)

        # Allocate new blocks for compacted cache
        new_block_ids = self.block_allocator.allocate_memory_blocks(num_new_blocks)
        if new_block_ids is None:
            raise RuntimeError(
                f"Cannot allocate {num_new_blocks} blocks for compaction"
            )

        mass_errors = []
        output_errors = []

        for layer in range(self.num_layers):
            # Gather full KV from pages
            K_full, V_full = gather_kv(
                self.memory_buffer, layer, block_ids, self.block_size, token_count
            )

            # Get reference queries
            if Q_ref_per_layer is not None and layer in Q_ref_per_layer:
                q = Q_ref_per_layer[layer]
            elif Q_ref is not None:
                q = Q_ref
            else:
                # Random fallback (not ideal but functional)
                H = K_full.shape[1]
                D = K_full.shape[2]
                R = self.config.num_ref_queries
                q = torch.randn(R, H, D, device=K_full.device, dtype=K_full.dtype)

            # Move to compute dtype
            K_f = K_full.to(self.config.compute_dtype)
            V_f = V_full.to(self.config.compute_dtype)
            q_f = q.to(self.config.compute_dtype)

            # Run AM compaction
            compact_fn = am_compact_with_mass if self.config.use_mass_matching else am_compact
            result: AMCompactionResult = compact_fn(
                K_f, V_f, q_f, M,
                scale=self.config.attention_scale,
                method=self.config.method,
                nnls_iters=self.config.nnls_iters,
                omp_keys_per_iter=self.config.omp_keys_per_iter,
                omp_refit_every=self.config.omp_refit_every,
            )

            # Write compacted KV back to pages
            K_mem = result.K_mem.to(self.memory_buffer.dtype)
            V_mem = result.V_mem.to(self.memory_buffer.dtype)

            if self.config.store_biases and self.bias_buffer is not None:
                write_kv_with_biases(
                    self.memory_buffer, self.bias_buffer, layer,
                    new_block_ids, self.block_size,
                    K_mem, V_mem, result.biases.to(self.memory_buffer.dtype),
                )
            else:
                write_kv(
                    self.memory_buffer, layer,
                    new_block_ids, self.block_size,
                    K_mem, V_mem,
                )

            mass_errors.append(result.mass_error)
            output_errors.append(result.output_error)

        metrics = {
            "mean_mass_error": sum(mass_errors) / len(mass_errors),
            "max_mass_error": max(mass_errors),
            "mean_output_error": sum(output_errors) / len(output_errors),
            "max_output_error": max(output_errors),
            "compression_ratio": token_count / M,
            "original_tokens": token_count,
            "compacted_tokens": M,
        }

        self.logger.info(
            f"Offline compaction: {token_count} -> {M} tokens "
            f"({token_count/M:.1f}x), mass_err={metrics['mean_mass_error']:.4f}, "
            f"output_err={metrics['mean_output_error']:.4f}"
        )

        return new_block_ids, M, metrics

    # =========================================================================
    # Phase 2: Online compaction
    # =========================================================================

    def should_compact(self, seq_id: int) -> bool:
        """Check if a sequence should be compacted based on hot window size."""
        state = self.seq_states.get(seq_id)
        if state is None:
            return False
        return state.hot_token_count >= (self.config.hot_window + self.config.compact_every_n)

    def compact_online(
        self,
        seq_id: int,
        all_block_ids: Tensor,
        total_token_count: int,
        Q_ref: Optional[Tensor] = None,
        Q_ref_per_layer: Optional[Dict[int, Tensor]] = None,
    ) -> Tuple[Tensor, int]:
        """Perform online compaction during decoding.

        Compacts the cold prefix (everything older than the hot window)
        into memory pages, keeps hot window pages intact, and rebuilds
        the block table as [mem_pages] + [hot_pages].

        Args:
            seq_id: Sequence identifier.
            all_block_ids: Current block IDs for this sequence.
            total_token_count: Total tokens in the sequence.
            Q_ref: Reference queries (shared across layers).
            Q_ref_per_layer: Per-layer reference queries.

        Returns:
            new_block_table: Updated block IDs [mem_blocks + hot_blocks].
            new_kv_len: Physical KV length (M + hot_token_count).
        """
        state = self.seq_states.get(seq_id)
        if state is None:
            raise ValueError(f"Sequence {seq_id} not registered")

        W = self.config.hot_window
        M = self.config.memory_budget

        # Determine cold/hot split
        hot_start = max(0, total_token_count - W)
        cold_token_count = hot_start

        if cold_token_count <= M:
            # Not enough cold tokens to justify compaction
            return all_block_ids, total_token_count

        # Determine block boundaries
        cold_num_blocks = math.ceil(cold_token_count / self.block_size)
        cold_block_ids = all_block_ids[:cold_num_blocks]

        hot_num_blocks = all_block_ids.shape[0] - cold_num_blocks
        hot_block_ids = all_block_ids[cold_num_blocks:]
        hot_token_count = total_token_count - cold_token_count

        # If we already have memory blocks, gather them + cold to re-compact
        if state.mem_block_ids is not None and state.mem_token_count > 0:
            # Merge existing memory + new cold for re-compaction
            gather_block_ids = torch.cat([state.mem_block_ids, cold_block_ids])
            gather_token_count = state.mem_token_count + cold_token_count
        else:
            gather_block_ids = cold_block_ids
            gather_token_count = cold_token_count

        # Allocate new memory blocks
        num_mem_blocks = math.ceil(M / self.block_size)
        new_mem_blocks = self.block_allocator.allocate_memory_blocks(num_mem_blocks)
        if new_mem_blocks is None:
            self.logger.warning("Cannot allocate memory blocks for compaction")
            return all_block_ids, total_token_count

        mass_errors = []
        output_errors = []

        for layer in range(self.num_layers):
            K_cold, V_cold = gather_kv(
                self.memory_buffer, layer, gather_block_ids,
                self.block_size, gather_token_count,
            )

            if Q_ref_per_layer is not None and layer in Q_ref_per_layer:
                q = Q_ref_per_layer[layer]
            elif Q_ref is not None:
                q = Q_ref
            else:
                H = K_cold.shape[1]
                D = K_cold.shape[2]
                R = self.config.num_ref_queries
                q = torch.randn(R, H, D, device=K_cold.device, dtype=K_cold.dtype)

            K_f = K_cold.to(self.config.compute_dtype)
            V_f = V_cold.to(self.config.compute_dtype)
            q_f = q.to(self.config.compute_dtype)

            compact_fn = am_compact_with_mass if self.config.use_mass_matching else am_compact
            result = compact_fn(
                K_f, V_f, q_f, M,
                scale=self.config.attention_scale,
                method=self.config.method,
                nnls_iters=self.config.nnls_iters,
                omp_keys_per_iter=self.config.omp_keys_per_iter,
                omp_refit_every=self.config.omp_refit_every,
            )

            K_mem = result.K_mem.to(self.memory_buffer.dtype)
            V_mem = result.V_mem.to(self.memory_buffer.dtype)

            if self.config.store_biases and self.bias_buffer is not None:
                write_kv_with_biases(
                    self.memory_buffer, self.bias_buffer, layer,
                    new_mem_blocks, self.block_size,
                    K_mem, V_mem, result.biases.to(self.memory_buffer.dtype),
                )
            else:
                write_kv(
                    self.memory_buffer, layer,
                    new_mem_blocks, self.block_size,
                    K_mem, V_mem,
                )

            mass_errors.append(result.mass_error)
            output_errors.append(result.output_error)

        # Free old memory blocks and cold blocks
        if state.mem_block_ids is not None:
            self.block_allocator.release_memory_blocks(state.mem_block_ids)
        self.block_allocator.release_memory_blocks(cold_block_ids)

        # Update state
        state.mem_block_ids = new_mem_blocks
        state.mem_token_count = M
        state.hot_block_ids = hot_block_ids
        state.hot_token_count = hot_token_count
        state.compacted_token_count += cold_token_count
        state.compaction_count += 1
        state.last_mass_errors = mass_errors
        state.last_output_errors = output_errors
        # Logical position NEVER rewinds
        state.logical_pos = max(state.logical_pos, total_token_count)

        # Build new block table: [mem_pages] + [hot_pages]
        new_block_table = torch.cat([new_mem_blocks, hot_block_ids])
        new_kv_len = M + hot_token_count

        self.logger.info(
            f"Online compaction seq={seq_id} round={state.compaction_count}: "
            f"{gather_token_count} -> {M}+{hot_token_count} tokens, "
            f"mass_err={sum(mass_errors)/len(mass_errors):.4f}"
        )

        return new_block_table, new_kv_len

    def get_rope_position(self, seq_id: int) -> int:
        """Get the current logical position for RoPE (never rewinds).

        Q positions must use this monotonic counter, NOT the physical
        KV length (which is M + hot_len after compaction).
        """
        state = self.seq_states.get(seq_id)
        if state is None:
            raise ValueError(f"Sequence {seq_id} not registered")
        return state.logical_pos

    def advance_position(self, seq_id: int, num_tokens: int = 1) -> int:
        """Advance the logical position counter and return new position."""
        state = self.seq_states.get(seq_id)
        if state is None:
            raise ValueError(f"Sequence {seq_id} not registered")
        state.logical_pos += num_tokens
        state.tokens_generated += num_tokens
        state.hot_token_count += num_tokens
        return state.logical_pos

    def get_block_table(self, seq_id: int) -> Optional[Tensor]:
        """Get the current two-tier block table [mem + hot] for a sequence."""
        state = self.seq_states.get(seq_id)
        if state is None:
            return None
        parts = []
        if state.mem_block_ids is not None:
            parts.append(state.mem_block_ids)
        if state.hot_block_ids is not None:
            parts.append(state.hot_block_ids)
        if not parts:
            return None
        return torch.cat(parts)

    def get_physical_kv_len(self, seq_id: int) -> int:
        """Get physical KV length (mem_tokens + hot_tokens) for attention."""
        state = self.seq_states.get(seq_id)
        if state is None:
            return 0
        return state.mem_token_count + state.hot_token_count

    def get_compaction_metrics(self, seq_id: int) -> Dict[str, float]:
        """Get compaction metrics for a sequence."""
        state = self.seq_states.get(seq_id)
        if state is None:
            return {}
        return {
            "compaction_count": state.compaction_count,
            "compacted_tokens": state.compacted_token_count,
            "mem_tokens": state.mem_token_count,
            "hot_tokens": state.hot_token_count,
            "logical_pos": state.logical_pos,
            "compression_ratio": (
                state.compacted_token_count / max(state.mem_token_count, 1)
                if state.mem_token_count > 0 else 0.0
            ),
            "mean_mass_error": (
                sum(state.last_mass_errors) / len(state.last_mass_errors)
                if state.last_mass_errors else 0.0
            ),
            "mean_output_error": (
                sum(state.last_output_errors) / len(state.last_output_errors)
                if state.last_output_errors else 0.0
            ),
        }
