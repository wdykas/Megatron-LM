# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fast streaming compactor (Phase 3).

Replaces the expensive AM solver with a cheap streaming compactor that
reads cold pages and writes compacted KV directly to memory pages.

Two designs:
  A) Streaming cluster bins (top-1/top-2 routing into M anchors)
  B) Learned projection compactor (trained via distillation)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class StreamingCompactorConfig:
    """Configuration for the streaming compactor."""

    # Number of anchor/bin slots per head-group
    num_anchors: int = 512

    # Routing: "top1" or "top2"
    routing: str = "top1"

    # Head grouping: number of KV heads per group (1 = per-head)
    heads_per_group: int = 1

    # Whether to learn anchor positions or initialize from data
    learnable_anchors: bool = False

    # Temperature for routing softmax
    routing_temperature: float = 1.0

    # Whether to normalize accumulated keys/values
    normalize_output: bool = True


class StreamingClusterCompactor(nn.Module):
    """Fast streaming compactor using cluster-bin routing.

    For each cold token key k:
      1. Route k to its best anchor(s) via dot-product similarity
      2. Accumulate weighted k into anchor's key accumulator
      3. Accumulate weighted v into anchor's value accumulator
    At the end, normalize accumulators to produce K_mem, V_mem.

    This avoids OMP, avoids storing cold KV contiguously, and runs in
    O(T * M) per head-group with small constant.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        config: StreamingCompactorConfig,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.config = config
        self.num_anchors = config.num_anchors
        self.heads_per_group = config.heads_per_group
        self.num_groups = math.ceil(num_heads / config.heads_per_group)

        # Anchor keys: (num_groups, num_anchors, head_dim)
        # Initialized via k-means or random; optionally learnable
        self.anchors = nn.Parameter(
            torch.randn(self.num_groups, self.num_anchors, head_dim) * 0.02,
            requires_grad=config.learnable_anchors,
        )

    def initialize_anchors_from_data(self, K_sample: Tensor) -> None:
        """Initialize anchors from a sample of keys using farthest-point sampling.

        Args:
            K_sample: (T, H, D) sample keys to initialize from.
        """
        T, H, D = K_sample.shape
        with torch.no_grad():
            for g in range(self.num_groups):
                h_start = g * self.heads_per_group
                h_end = min(h_start + self.heads_per_group, H)
                # Average across heads in group
                Kg = K_sample[:, h_start:h_end, :].mean(dim=1)  # (T, D)

                # Farthest-point sampling for diverse anchors
                M = min(self.num_anchors, T)
                indices = [0]
                min_dists = torch.full((T,), float('inf'), device=Kg.device)

                for _ in range(1, M):
                    last = Kg[indices[-1]]
                    dists = (Kg - last.unsqueeze(0)).pow(2).sum(dim=-1)
                    min_dists = torch.minimum(min_dists, dists)
                    indices.append(min_dists.argmax().item())

                self.anchors.data[g, :M] = Kg[indices]

    def compact(
        self,
        K_cold: Tensor,
        V_cold: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run streaming compaction on cold KV tensors.

        Args:
            K_cold: (T, H, D) cold keys.
            V_cold: (T, H, D) cold values.

        Returns:
            K_mem: (M, H, D) compacted keys.
            V_mem: (M, H, D) compacted values.
        """
        T, H, D = K_cold.shape
        M = self.num_anchors
        device = K_cold.device

        # Accumulators per group
        K_acc = torch.zeros(self.num_groups, M, D, device=device, dtype=torch.float32)
        V_acc = torch.zeros(self.num_groups, M, D, device=device, dtype=torch.float32)
        Z_acc = torch.zeros(self.num_groups, M, device=device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)

        for g in range(self.num_groups):
            h_start = g * self.heads_per_group
            h_end = min(h_start + self.heads_per_group, H)

            # Average across heads in this group for routing
            Kg = K_cold[:, h_start:h_end, :].float().mean(dim=1)  # (T, D)
            Vg = V_cold[:, h_start:h_end, :].float().mean(dim=1)  # (T, D)

            # Routing scores: (T, M)
            anchors_g = self.anchors[g].float()  # (M, D)
            scores = Kg @ anchors_g.T * scale / self.config.routing_temperature

            if self.config.routing == "top1":
                # Top-1 routing
                top_idx = scores.argmax(dim=-1)  # (T,)
                weights = torch.ones(T, device=device, dtype=torch.float32)

                # Scatter-add
                K_acc[g].index_add_(0, top_idx, Kg * weights.unsqueeze(1))
                V_acc[g].index_add_(0, top_idx, Vg * weights.unsqueeze(1))
                Z_acc[g].index_add_(0, top_idx, weights)

            elif self.config.routing == "top2":
                # Top-2 routing with softmax weights
                top2_vals, top2_idx = scores.topk(2, dim=-1)  # (T, 2)
                top2_weights = F.softmax(top2_vals, dim=-1)  # (T, 2)

                for r in range(2):
                    idx = top2_idx[:, r]
                    w = top2_weights[:, r]
                    K_acc[g].index_add_(0, idx, Kg * w.unsqueeze(1))
                    V_acc[g].index_add_(0, idx, Vg * w.unsqueeze(1))
                    Z_acc[g].index_add_(0, idx, w)

        # Normalize accumulators
        if self.config.normalize_output:
            Z_safe = Z_acc.clamp(min=1e-8).unsqueeze(-1)
            K_acc = K_acc / Z_safe
            V_acc = V_acc / Z_safe

        # Expand groups back to per-head: (M, H, D)
        K_mem = torch.zeros(M, H, D, device=device, dtype=K_cold.dtype)
        V_mem = torch.zeros(M, H, D, device=device, dtype=V_cold.dtype)

        for g in range(self.num_groups):
            h_start = g * self.heads_per_group
            h_end = min(h_start + self.heads_per_group, H)
            K_mem[:, h_start:h_end, :] = K_acc[g].unsqueeze(1).expand(-1, h_end - h_start, -1).to(K_cold.dtype)
            V_mem[:, h_start:h_end, :] = V_acc[g].unsqueeze(1).expand(-1, h_end - h_start, -1).to(V_cold.dtype)

        return K_mem, V_mem

    def compact_from_pages(
        self,
        memory_buffer: Tensor,
        layer: int,
        block_ids: Tensor,
        block_size: int,
        token_count: int,
    ) -> Tuple[Tensor, Tensor]:
        """Stream over paged blocks and compact directly (avoids full gather).

        Processes blocks one at a time to minimize peak memory.

        Args:
            memory_buffer: 6D KV cache tensor.
            layer: Layer index.
            block_ids: Block IDs of cold pages.
            block_size: Tokens per block.
            token_count: Total valid tokens.

        Returns:
            K_mem: (M, H, D) compacted keys.
            V_mem: (M, H, D) compacted values.
        """
        key_cache = memory_buffer[0, layer]  # (total_blocks, block_size, H, D)
        val_cache = memory_buffer[1, layer]

        H = key_cache.shape[2]
        D = key_cache.shape[3]
        M = self.num_anchors
        device = memory_buffer.device

        K_acc = torch.zeros(self.num_groups, M, D, device=device, dtype=torch.float32)
        V_acc = torch.zeros(self.num_groups, M, D, device=device, dtype=torch.float32)
        Z_acc = torch.zeros(self.num_groups, M, device=device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(D)
        tokens_processed = 0

        for i, bid in enumerate(block_ids):
            bid_int = bid.long().item()
            remaining = token_count - tokens_processed
            length = min(block_size, remaining)
            if length <= 0:
                break

            K_block = key_cache[bid_int, :length].float()  # (length, H, D)
            V_block = val_cache[bid_int, :length].float()

            for g in range(self.num_groups):
                h_start = g * self.heads_per_group
                h_end = min(h_start + self.heads_per_group, H)

                Kg = K_block[:, h_start:h_end, :].mean(dim=1)  # (length, D)
                Vg = V_block[:, h_start:h_end, :].mean(dim=1)

                anchors_g = self.anchors[g].float()
                scores = Kg @ anchors_g.T * scale / self.config.routing_temperature

                if self.config.routing == "top1":
                    top_idx = scores.argmax(dim=-1)
                    w = torch.ones(length, device=device, dtype=torch.float32)
                    K_acc[g].index_add_(0, top_idx, Kg * w.unsqueeze(1))
                    V_acc[g].index_add_(0, top_idx, Vg * w.unsqueeze(1))
                    Z_acc[g].index_add_(0, top_idx, w)
                elif self.config.routing == "top2":
                    top2_vals, top2_idx = scores.topk(2, dim=-1)
                    top2_w = F.softmax(top2_vals, dim=-1)
                    for r in range(2):
                        idx = top2_idx[:, r]
                        wr = top2_w[:, r]
                        K_acc[g].index_add_(0, idx, Kg * wr.unsqueeze(1))
                        V_acc[g].index_add_(0, idx, Vg * wr.unsqueeze(1))
                        Z_acc[g].index_add_(0, idx, wr)

            tokens_processed += length

        # Normalize
        if self.config.normalize_output:
            Z_safe = Z_acc.clamp(min=1e-8).unsqueeze(-1)
            K_acc = K_acc / Z_safe
            V_acc = V_acc / Z_safe

        # Expand to per-head
        K_mem = torch.zeros(M, H, D, device=device, dtype=memory_buffer.dtype)
        V_mem = torch.zeros(M, H, D, device=device, dtype=memory_buffer.dtype)

        for g in range(self.num_groups):
            h_start = g * self.heads_per_group
            h_end = min(h_start + self.heads_per_group, H)
            K_mem[:, h_start:h_end, :] = K_acc[g].unsqueeze(1).expand(-1, h_end - h_start, -1).to(memory_buffer.dtype)
            V_mem[:, h_start:h_end, :] = V_acc[g].unsqueeze(1).expand(-1, h_end - h_start, -1).to(memory_buffer.dtype)

        return K_mem, V_mem

    def compact_soft(
        self,
        K_cold: Tensor,
        V_cold: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Differentiable soft-routing compaction for training.

        Uses full softmax attention over all anchors instead of hard top-k
        routing, so gradients flow through the anchor parameters.

        Args:
            K_cold: (T, H, D) cold keys.
            V_cold: (T, H, D) cold values.

        Returns:
            K_mem: (M, H, D) compacted keys.
            V_mem: (M, H, D) compacted values.
        """
        T, H, D = K_cold.shape
        M = self.num_anchors
        device = K_cold.device

        scale = 1.0 / math.sqrt(D)
        K_mem_parts = []
        V_mem_parts = []

        for g in range(self.num_groups):
            h_start = g * self.heads_per_group
            h_end = min(h_start + self.heads_per_group, H)

            Kg = K_cold[:, h_start:h_end, :].float().mean(dim=1)  # (T, D)
            Vg = V_cold[:, h_start:h_end, :].float().mean(dim=1)  # (T, D)

            # Soft attention: anchors attend to all keys
            # scores: (M, T) = anchors @ keys^T
            anchors_g = self.anchors[g]  # (M, D) — has grad
            scores = anchors_g.float() @ Kg.T * scale / self.config.routing_temperature
            weights = F.softmax(scores, dim=-1)  # (M, T)

            # Weighted aggregation
            K_agg = weights @ Kg  # (M, D)
            V_agg = weights @ Vg  # (M, D)

            K_mem_parts.append(K_agg)
            V_mem_parts.append(V_agg)

        # Expand groups to per-head using cat (preserves grad)
        K_head_list = []
        V_head_list = []
        for g in range(self.num_groups):
            h_start = g * self.heads_per_group
            h_end = min(h_start + self.heads_per_group, H)
            n_heads = h_end - h_start
            K_head_list.append(K_mem_parts[g].unsqueeze(1).expand(-1, n_heads, -1))
            V_head_list.append(V_mem_parts[g].unsqueeze(1).expand(-1, n_heads, -1))

        K_mem = torch.cat(K_head_list, dim=1)  # (M, H, D)
        V_mem = torch.cat(V_head_list, dim=1)

        return K_mem.to(K_cold.dtype), V_mem.to(V_cold.dtype)

    def forward(self, K_cold: Tensor, V_cold: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass — uses soft routing during training, hard routing at eval."""
        if self.training:
            return self.compact_soft(K_cold, V_cold)
        return self.compact(K_cold, V_cold)
