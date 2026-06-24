# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Chunk feature extractor: a fixed-dim feature vector z_t for gate conditioning.

z_t summarizes a KV chunk (chunk index, key-norm mean/std, memory pressure) and is
optionally fed to the GatedRecurrentUpdater's gate (GatedUpdaterConfig.feature_dim).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Chunk feature extractor (moved from features.py)
# ---------------------------------------------------------------------------

# Public constant so GatedUpdaterConfig can import the expected dim.
FEATURE_DIM: int = 4


@dataclass
class ChunkFeatures:
    """Computed feature vector for a single chunk at step t.

    Attributes
    ----------
    chunk_index:       Zero-based index of this chunk in the trajectory.
    chunk_index_norm:  chunk_index / max_chunks, clipped to [0, 1].
    key_norm_mean:     Mean L2 norm of chunk keys (averaged over layers).
    key_norm_std:      Std of L2 norms of chunk keys (averaged over layers).
    memory_pressure:   current_memory_used / memory_budget ∈ [0, 1].
    vector:            (FEATURE_DIM,) float tensor — concatenation of the
                       normalised scalars, ready to be fed into the gate.
    """

    chunk_index: int
    chunk_index_norm: float
    key_norm_mean: float
    key_norm_std: float
    memory_pressure: float
    vector: torch.Tensor   # (FEATURE_DIM,)


class ChunkFeatureExtractor:
    """Extract a fixed-dim feature vector z_t from a KV chunk.

    Usage:
        extractor = ChunkFeatureExtractor(memory_budget=64, max_chunks=512)
        features = extractor.extract(chunk_keys, chunk_index, current_slots)
        # features.vector: (FEATURE_DIM,) tensor
    """

    def __init__(
        self,
        memory_budget: int,
        max_chunks: int = 512,
    ) -> None:
        """
        Parameters
        ----------
        memory_budget: C — total compact memory slots (from PerceiverConfig.n_compress).
        max_chunks:    Expected maximum number of chunks per trajectory.  Used
                       only for normalisation; safe to over-estimate.
        """
        self.memory_budget = memory_budget
        self.max_chunks = max(max_chunks, 1)

    def extract(
        self,
        chunk_keys: list[torch.Tensor],  # n_layers × (B, T, d)
        chunk_index: int,
        current_memory_slots: int = 0,
    ) -> ChunkFeatures:
        """Compute z_t from the chunk's key tensors and bookkeeping state."""
        # --- 1. chunk index (normalised) ------------------------------------
        chunk_idx_norm = float(min(chunk_index, self.max_chunks - 1)) / self.max_chunks

        # --- 2. key norms (mean and std) ------------------------------------
        norms_per_layer = []
        for k in chunk_keys:
            norms_per_layer.append(k.norm(dim=-1).reshape(-1))   # (B*T,)
        all_norms = torch.cat(norms_per_layer)                   # (n_layers * B * T,)

        key_norm_mean = float(all_norms.mean().item())
        key_norm_std  = float(all_norms.std(unbiased=False).item()) if all_norms.numel() > 1 else 0.0

        _norm_scale = 5.0
        key_norm_mean_n = key_norm_mean / _norm_scale
        key_norm_std_n  = key_norm_std  / _norm_scale

        # --- 3. memory pressure ---------------------------------------------
        pressure = float(min(current_memory_slots, self.memory_budget)) / self.memory_budget

        # --- 4. assemble tensor ---------------------------------------------
        device = chunk_keys[0].device
        vec = torch.tensor(
            [chunk_idx_norm, key_norm_mean_n, key_norm_std_n, pressure],
            dtype=torch.float32,
            device=device,
        )

        return ChunkFeatures(
            chunk_index=chunk_index,
            chunk_index_norm=chunk_idx_norm,
            key_norm_mean=key_norm_mean,
            key_norm_std=key_norm_std,
            memory_pressure=pressure,
            vector=vec,
        )

    def batch_extract(
        self,
        chunk_keys: list[torch.Tensor],  # n_layers × (B, T, d)
        chunk_index: int,
        current_memory_slots: int = 0,
    ) -> torch.Tensor:
        """Convenience wrapper that returns the (FEATURE_DIM,) vector directly."""
        return self.extract(chunk_keys, chunk_index, current_memory_slots).vector
