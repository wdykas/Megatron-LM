# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Value head V_ψ(M_t, z_t) for value-directed belief compression.

The value head estimates the cumulative future reward achievable from the
current belief state M_t.  It serves two purposes:

    1. Curriculum stage 4 (value-directed training):
       Minimise |V_ψ(M_t, z_t) - G_t|^2  where G_t is a return estimate
       from the task reward signal.  This back-propagates through the value
       head into the compactor, shaping which information is worth remembering.

    2. Slot ablation estimation:
       Remove one slot at a time and measure the predicted value drop to
       identify which slots matter most.  Used for interpretability and for
       informed eviction in memory-constrained deployment.

Architecture:

    pool    = mean_pool(M_t.keys, dim=slots)   → (B, n_layers, d_model)
    flat    = reshape(pool, (B, n_layers * d_model))
    z_cat   = cat([flat, z_t], dim=-1)  if z_t is not None  else flat
    value   = MLP(z_cat)                → (B, 1) → scalar
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.rl.compaction.learned.models.belief import BeliefMemory
from megatron.rl.compaction.learned.models.compactor import (
    column_linear,
    compactor_transformer_config,
)


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


class ValueHead(nn.Module):
    """MLP mapping belief state → scalar value estimate.

    Parameters
    ----------
    n_layers:    Number of transformer layers (= BeliefMemory.n_layers).
    d_model:     Per-layer key/value dimension (= BeliefMemory.d_model).
    hidden_dim:  Hidden dimension of the two-layer MLP.
    feature_dim: Dimension of optional z_t feature vector fed to the gate.
                 Pass 0 (default) to use only the pooled memory representation.
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        hidden_dim: int = 256,
        feature_dim: int = 0,
        params_dtype: torch.dtype = torch.float32,
        pg_collection=None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.feature_dim = feature_dim

        in_dim = n_layers * d_model + feature_dim
        config = compactor_transformer_config(hidden_dim, n_heads=1, ffn_hidden_size=hidden_dim,
                                              params_dtype=params_dtype)
        self.fc1 = column_linear(in_dim, hidden_dim, config, pg_collection, bias=True)
        self.fc2 = column_linear(hidden_dim, hidden_dim, config, pg_collection, bias=True)
        self.fc3 = column_linear(hidden_dim, 1, config, pg_collection, bias=True)

    def forward(
        self,
        memory: BeliefMemory,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate value from belief state.

        Parameters
        ----------
        memory:   BeliefMemory with keys of shape (n_layers, B, C, d_model).
        features: Optional z_t feature vector, shape (B, feature_dim) or
                  (feature_dim,) which is broadcast over the batch.

        Returns: (B,) float tensor — per-sample value estimate.
        """
        # Pool over the C memory slots: (n_layers, B, C, d) → (B, n_layers, d)
        keys_pooled = memory.keys.mean(dim=2)           # (n_layers, B, d)
        keys_pooled = keys_pooled.permute(1, 0, 2)     # (B, n_layers, d)
        B = keys_pooled.shape[0]
        flat = keys_pooled.reshape(B, -1)               # (B, n_layers * d)

        if features is not None:
            if features.dim() == 1:
                features = features.unsqueeze(0).expand(B, -1)
            flat = torch.cat([flat, features], dim=-1)  # (B, n_layers*d + feature_dim)

        h, _ = self.fc1(flat)
        h = F.gelu(h)
        h, _ = self.fc2(h)
        h = F.gelu(h)
        out, _ = self.fc3(h)
        return out.squeeze(-1)                          # (B,)

    def slot_ablation(
        self,
        memory: BeliefMemory,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Estimate value drop when each memory slot is removed.

        Returns a (B, C) tensor where entry [b, i] is the estimated value
        decrease when slot i is zeroed out.  Use to identify high-importance
        slots for interpretability or informed eviction.

        This is a first-order approximation: it re-runs the value head C+1
        times (once with full memory, C times with each slot zeroed).  For
        inference-time use, prefer the gradient-based shortcut:
            importance_i ≈ |∂V/∂M_t[:, :, i, :]|
        """
        baseline = self.forward(memory, features)  # (B,)

        C = memory.budget
        importances = []
        for i in range(C):
            ablated_keys = memory.keys.clone()
            ablated_vals = memory.values.clone()
            ablated_keys[:, :, i, :] = 0.0
            ablated_vals[:, :, i, :] = 0.0
            ablated_mem = BeliefMemory(ablated_keys, ablated_vals, memory.step)
            v_ablated = self.forward(ablated_mem, features)  # (B,)
            importances.append((baseline - v_ablated).unsqueeze(-1))  # (B, 1)

        return torch.cat(importances, dim=-1)  # (B, C)
