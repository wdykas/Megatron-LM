# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Belief-Still: recurrent POMDP belief state over compact KV memory.

This module implements the POMDP extension of Still described in the design doc.

The belief state M_t represents the compact KV memory at step t. The update rule:

    M_{t+1} = U_θ(M_t, R_t)

is implemented by concatenating the current belief with the new chunk's raw KV
cache and re-compressing using the same PerceiverCompactor:

    [M_t; R_t]  → PerceiverCompactor →  M_{t+1}  (same C tokens as M_t)

This mirrors the classical POMDP belief update:
    b_{t+1} = τ(b_t, a_t, o_{t+1})

but replaces the explicit Bayesian filter with a learned neural compressor
trained with prediction-directed losses.

The update is value-directed in the sense of Poupart & Boutilier (NeurIPS 2002):
M_t is trained to preserve future model behavior rather than to reconstruct
the full KV cache. The CompactorLosses module enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.rl.compaction.learned.models.compactor import (
    PerceiverCompactor,
    CrossAttention,
    FeedForward,
    compactor_transformer_config,
    _count_attention_layers,
)


# ---------------------------------------------------------------------------
# BeliefMemory dataclass
# ---------------------------------------------------------------------------

@dataclass
class BeliefMemory:
    """Compact POMDP belief state: C synthetic KV tokens across all layers.

    Maintained recurrently as M_0 → M_1 → … → M_t through BeliefUpdater calls.

    Shape convention:
        keys   : (n_layers, B, C, d_model)
        values : (n_layers, B, C, d_model)
    """

    keys:   torch.Tensor                   # (n_layers, B, C, d)
    values: torch.Tensor                   # (n_layers, B, C, d)
    step:   int = field(default=0)

    # --- shape accessors ---------------------------------------------------

    @property
    def n_layers(self) -> int:
        return self.keys.shape[0]

    @property
    def budget(self) -> int:
        return self.keys.shape[2]

    @property
    def batch_size(self) -> int:
        return self.keys.shape[1]

    @property
    def d_model(self) -> int:
        return self.keys.shape[3]

    # --- factory -----------------------------------------------------------

    @classmethod
    def zero(
        cls,
        n_layers: int,
        batch: int,
        budget: int,
        d_kv: int,
        device: torch.device | str | None = None,
    ) -> "BeliefMemory":
        """Create a zero-initialised belief state (empty memory)."""
        shape = (n_layers, batch, budget, d_kv)
        return cls(
            keys=torch.zeros(shape, device=device),
            values=torch.zeros(shape, device=device),
            step=0,
        )

    @classmethod
    def zero_from_compactor(
        cls,
        compactor,
        batch: int,
        device: torch.device | str | None = None,
    ) -> "BeliefMemory":
        """Create zero belief matching a compactor's dimensions."""
        cfg = compactor.cfg
        return cls.zero(cfg.n_attn_layers, batch, cfg.n_compress, cfg.d_kv, device=device)

    # --- utilities ---------------------------------------------------------

    def layer(self, l: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) for transformer layer l: each (B, C, d)."""
        return self.keys[l], self.values[l]

    def detach(self) -> "BeliefMemory":
        """Detach from computation graph for truncated BPTT."""
        return BeliefMemory(self.keys.detach(), self.values.detach(), self.step)

    def to(self, device) -> "BeliefMemory":
        return BeliefMemory(self.keys.to(device), self.values.to(device), self.step)

    def keys_list(self) -> list[torch.Tensor]:
        """Return keys as a list of (B, C, d) tensors, one per layer."""
        return [self.keys[l] for l in range(self.n_layers)]

    def values_list(self) -> list[torch.Tensor]:
        return [self.values[l] for l in range(self.n_layers)]


# ---------------------------------------------------------------------------
# BeliefUpdater  (M_{t+1} = U_θ(M_t, R_t))
# ---------------------------------------------------------------------------

class BeliefUpdater(nn.Module):
    """Recurrent POMDP belief update: M_{t+1} = U_θ(M_t, R_t).

    Implements the POMDP update rule using the Perceiver pattern:
      1. Concatenate current belief M_t with new chunk KV R_t along the
         sequence dimension: [M_t ; R_t] is (C + T_new) tokens.
      2. Re-compress back to C tokens using the PerceiverCompactor.

    The compactor weights are shared between the initial compression step
    (BeliefUpdater.initial_compress) and all subsequent update steps, so the
    total parameter count does not grow with the number of recurrent steps.

    For the initial step (step 0) where there is no prior memory, call
    initial_compress() directly on the first chunk's KV cache.
    """

    def __init__(self, compactor: PerceiverCompactor) -> None:
        super().__init__()
        self.compactor = compactor

    def forward(
        self,
        memory:     BeliefMemory,
        new_keys:   list[torch.Tensor],   # n_layers × (B, T_new, d)
        new_values: list[torch.Tensor],   # n_layers × (B, T_new, d)
    ) -> BeliefMemory:
        """Update belief memory with a new context chunk.

        Concatenates [M_t; R_t] along the sequence dimension and re-compresses
        to the same C-token budget. The compactor attends over all tokens in
        [M_t; R_t] and selectively updates what matters.

        Parameters
        ----------
        memory:     Current belief state M_t.
        new_keys:   Per-layer keys of the new chunk R_t.
        new_values: Per-layer values of the new chunk R_t.

        Returns the next belief state M_{t+1}.
        """
        updated_keys, updated_values = [], []

        for l in range(memory.n_layers):
            mem_k, mem_v = memory.layer(l)          # (B, C, d)
            r_k = new_keys[l]                        # (B, T_new, d)
            r_v = new_values[l]

            # Concatenate belief + new chunk
            combined_k = torch.cat([mem_k, r_k], dim=1)   # (B, C + T_new, d)
            combined_v = torch.cat([mem_v, r_v], dim=1)

            # Re-compress to C tokens
            ck, cv = self.compactor(combined_k, combined_v, layer_idx=l)
            updated_keys.append(ck)
            updated_values.append(cv)

        return BeliefMemory(
            keys=torch.stack(updated_keys),      # (n_layers, B, C, d)
            values=torch.stack(updated_values),
            step=memory.step + 1,
        )

    def initial_compress(
        self,
        keys_per_layer:   list[torch.Tensor],   # n_layers × (B, T, d)
        values_per_layer: list[torch.Tensor],
    ) -> BeliefMemory:
        """Bootstrap belief state from the first chunk (step 0).

        Equivalent to forward() with an empty prior memory. Compresses the
        first chunk directly without any concatenation.
        """
        ck_list, cv_list = self.compactor.compress_all_layers(
            keys_per_layer, values_per_layer
        )
        return BeliefMemory(
            keys=torch.stack(ck_list),     # (n_layers, B, C, d)
            values=torch.stack(cv_list),
            step=0,
        )


# ---------------------------------------------------------------------------
# GatedUpdaterConfig
# ---------------------------------------------------------------------------

@dataclass
class GatedUpdaterConfig:
    """Configuration for GatedRecurrentUpdater.

    Use ``GatedUpdaterConfig.from_transformer_config()`` to construct — do not
    set ``d_kv`` or ``n_attn_layers`` by hand.

    Attributes
    ----------
    n_compress:          Number of compact slots C (memory budget).
    n_heads:             Number of attention heads.
    d_kv:                Flattened KV dimension per token
                         (= kv_channels × num_query_groups from TransformerConfig).
                         Set automatically by ``from_transformer_config``.
    n_attn_layers:       Number of attention layers (excludes Mamba layers).
                         Set automatically by ``from_transformer_config``.
    d_ff:                Feed-forward hidden dim. Default: 4 × d_kv.
    dropout:             Dropout rate.
    feature_dim:         Dimension of optional z_t feature vector fed to the
                         gate.  0 disables this path.
    share_across_layers: If True, one set of updater weights for all layers.
    """

    n_compress:          int
    n_heads:             int
    d_kv:                int
    n_attn_layers:       int
    d_ff:                int | None = None
    dropout:             float = 0.0
    feature_dim:         int = 0
    share_across_layers: bool = False
    use_dynamics_head:   bool = False

    def __post_init__(self) -> None:
        if self.d_kv % self.n_heads != 0:
            raise ValueError(
                f"d_kv ({self.d_kv}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_compress < 1:
            raise ValueError(f"n_compress must be >= 1, got {self.n_compress}")
        if self.n_attn_layers < 1:
            raise ValueError(f"n_attn_layers must be >= 1, got {self.n_attn_layers}")

    @classmethod
    def from_transformer_config(
        cls,
        transformer_config,
        n_compress: int,
        n_heads: int = 8,
        **kwargs,
    ) -> "GatedUpdaterConfig":
        """Construct from a Megatron TransformerConfig."""
        d_kv = transformer_config.kv_channels * transformer_config.num_query_groups
        n_attn_layers = _count_attention_layers(transformer_config)
        return cls(
            n_compress=n_compress,
            n_heads=n_heads,
            d_kv=d_kv,
            n_attn_layers=n_attn_layers,
            **kwargs,
        )

    @property
    def ff_dim(self) -> int:
        return self.d_ff or self.d_kv * 4


# ---------------------------------------------------------------------------
# Per-layer gated updater block
# ---------------------------------------------------------------------------

class _GatedLayerUpdater(nn.Module):
    """Single-layer gated recurrent belief update."""

    def __init__(self, cfg: GatedUpdaterConfig, config, pg_collection=None) -> None:
        super().__init__()
        d = cfg.d_kv
        tp_group = pg_collection.tp if pg_collection is not None else None
        lin = dict(config=config, init_method=config.init_method, gather_output=False,
                   bias=False, skip_bias_add=False, is_expert=False, tp_group=tp_group)

        # Cross-attention: Q = memory keys, KV = [memory; chunk]
        self.cross_attn = CrossAttention(config, pg_collection)

        # Self-attention among the C updated slots (pre-norm is internal to the block)
        self.self_attn = CrossAttention(config, pg_collection)

        # FFN
        self.ffn = FeedForward(config, pg_collection)

        # Prediction head: each slot predicts the mean content of the next chunk
        self.predict_head = TEColumnParallelLinear(d, d, **lin)
        # Gate input now includes pred_err: d*2 + 1 + feature_dim
        gate_in_dim = d * 2 + 1 + cfg.feature_dim
        self.gate_proj = TEColumnParallelLinear(gate_in_dim, d, **{**lin, "bias": True})

        # Project gated slot representation → synthetic K and V
        self.key_proj = TEColumnParallelLinear(d, d, **lin)
        self.val_proj = TEColumnParallelLinear(d, d, **lin)

    def forward(
        self,
        mem_keys:    torch.Tensor,   # (B, C, d)
        mem_values:  torch.Tensor,   # (B, C, d)
        chunk_keys:  torch.Tensor,   # (B, T, d)
        chunk_values: torch.Tensor,  # (B, T, d)
        features:    torch.Tensor | None = None,  # (B, feature_dim)
        slot_pos:    torch.Tensor | None = None,  # (1, C, d) persistent slot identity
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (new_keys, new_values, gates, pred_Rt), each (B, C, d)."""
        # Add persistent slot identity to the cross-attention query.
        # Without this, all slot queries converge after a few update steps →
        # identical cross-attention patterns → slot collapse.
        q = mem_keys if slot_pos is None else mem_keys + slot_pos

        # POMDP predictive gate: predict R_t from M_{t-1}
        pred_Rt, _ = self.predict_head(q)                          # (B, C, d) — slot predictions
        chunk_mean = chunk_keys.mean(dim=1, keepdim=True)          # (B, 1, d)
        pred_err = (pred_Rt - chunk_mean).pow(2).mean(dim=-1, keepdim=True)  # (B, C, 1)

        combined_k = torch.cat([mem_keys, chunk_keys], dim=1)     # (B, C+T, d)
        combined_v = torch.cat([mem_values, chunk_values], dim=1)

        # Cross-attend: positioned query → distinct attention pattern per slot
        h = q + self.cross_attn(q, combined_k, combined_v)

        # Refine slot interactions via self-attention (block applies its own norm)
        h = h + self.self_attn(h, h, h)

        # FFN
        h = h + self.ffn(h)

        # Gate: use positioned query (q) so gate is identity-aware
        gate_input = torch.cat([q, h, pred_err], dim=-1)          # (B, C, 2d+1)
        if features is not None:
            feat = features.unsqueeze(1).expand(-1, h.shape[1], -1)
            gate_input = torch.cat([gate_input, feat], dim=-1)
        gate_logits, _ = self.gate_proj(gate_input)
        g = torch.sigmoid(gate_logits)                            # (B, C, d)

        # Blend old memory with candidate update (separate blend per modality so
        # mem_values is preserved when g≈0, not overwritten by a projection of mem_keys).
        slot_repr_k = (1.0 - g) * mem_keys + g * h
        slot_repr_v = (1.0 - g) * mem_values + g * h

        new_k, _ = self.key_proj(slot_repr_k)
        new_v, _ = self.val_proj(slot_repr_v)
        return new_k, new_v, g, pred_Rt


# ---------------------------------------------------------------------------
# GatedRecurrentUpdater  (M_{t+1} = U_θ(M_t, R_t))
# ---------------------------------------------------------------------------

class GatedRecurrentUpdater(MegatronModule):
    """Full Belief-Still gated recurrent updater.

    Drop-in replacement for BeliefUpdater with richer per-slot dynamics:
    each memory slot decides independently how much to update vs. retain.

    Usage — recurrent update:
        updater = GatedRecurrentUpdater(cfg)
        M0 = updater.initial_compress(keys_list, values_list)
        M1 = updater(M0, chunk_keys, chunk_values)  # returns BeliefMemory

    Usage — with gate and prediction access:
        M1, gates, preds = updater.update(M0, chunk_keys, chunk_values)
        gate_loss = gates.abs().mean()  # sparsity penalty
    """

    def __init__(self, cfg: GatedUpdaterConfig, params_dtype: torch.dtype = torch.float32,
                 pg_collection=None) -> None:
        config = compactor_transformer_config(cfg.d_kv, cfg.n_heads, cfg.ff_dim, params_dtype)
        super().__init__(config)
        self.cfg = cfg
        n_sets = 1 if cfg.share_across_layers else cfg.n_attn_layers
        self._layer_modules = nn.ModuleList([
            _GatedLayerUpdater(cfg, config, pg_collection) for _ in range(n_sets)
        ])
        # Persistent slot identity embeddings.
        # Serves two roles:
        #   1. Initialization: seeds initial memory with distinct slot keys so
        #      cross-attention queries differ from the very first step.
        #   2. Persistent query bias: added to mem_keys at EVERY update step
        #      so each slot retains a fixed "address" regardless of content.
        #      Without this, slot keys converge after a few updates and the
        #      self-attention homogenises them further (slot collapse).
        # Shape: (1, C, d) — broadcast across batch; shared across all layers.
        # randn init: gives slots different random magnitudes in addition to
        # different directions.  This is intentional — the magnitude variance
        # means self-attention produces unequal weights across slots, which
        # preserves slot diversity through the self-attention step.  Perfectly
        # orthogonal (equal-magnitude) init empirically collapses faster after
        # self-attention because LayerNorm erases the magnitude differences.
        self.slot_pos = nn.Parameter(
            torch.randn(1, cfg.n_compress, cfg.d_kv) * (cfg.d_kv ** -0.5)
        )
        n_head_sets = 1 if cfg.share_across_layers else cfg.n_attn_layers
        if cfg.use_dynamics_head:
            tp_group = pg_collection.tp if pg_collection is not None else None
            lin = dict(config=config, init_method=config.init_method, gather_output=False,
                       bias=False, skip_bias_add=False, is_expert=False, tp_group=tp_group)
            self.dynamics_key_heads = nn.ModuleList([
                TEColumnParallelLinear(cfg.d_kv, cfg.d_kv, **lin) for _ in range(n_head_sets)
            ])
            self.dynamics_val_heads = nn.ModuleList([
                TEColumnParallelLinear(cfg.d_kv, cfg.d_kv, **lin) for _ in range(n_head_sets)
            ])
        else:
            self.dynamics_key_heads = None
            self.dynamics_val_heads = None

    def _module(self, l: int) -> _GatedLayerUpdater:
        return self._layer_modules[0 if self.cfg.share_across_layers else l]

    def update(
        self,
        memory:      BeliefMemory,
        chunk_keys:  list[torch.Tensor],    # n_layers × (B, T, d)
        chunk_values: list[torch.Tensor],
        features:    torch.Tensor | None = None,
    ) -> tuple[BeliefMemory, torch.Tensor, torch.Tensor]:
        """Update belief, returning new memory, gate activations, and slot predictions.

        Returns
        -------
        new_memory: BeliefMemory  — M_{t+1}
        gates:      Tensor (n_layers, B, C, d) — gate values in [0, 1]
                    0 = protect old slot, 1 = overwrite with new information.
        preds:      Tensor (n_layers, B, C, d) — slot predictions of R_t keys
                    (for POMDP predictive coding loss).
        """
        new_keys, new_vals, all_gates, all_preds = [], [], [], []
        for l in range(memory.n_layers):
            mem_k, mem_v = memory.layer(l)
            new_k, new_v, g, pred = self._module(l)(
                mem_k, mem_v, chunk_keys[l], chunk_values[l], features,
                slot_pos=self.slot_pos,
            )
            new_keys.append(new_k)
            new_vals.append(new_v)
            all_gates.append(g)
            all_preds.append(pred)

        return (
            BeliefMemory(
                keys=torch.stack(new_keys),
                values=torch.stack(new_vals),
                step=memory.step + 1,
            ),
            torch.stack(all_gates),   # (n_layers, B, C, d)
            torch.stack(all_preds),   # (n_layers, B, C, d) — predictions for loss
        )

    def forward(
        self,
        memory:      BeliefMemory,
        chunk_keys:  list[torch.Tensor],
        chunk_values: list[torch.Tensor],
        features:    torch.Tensor | None = None,
    ) -> BeliefMemory:
        """BeliefUpdater-compatible interface — returns only the new memory."""
        new_memory, _, _ = self.update(memory, chunk_keys, chunk_values, features)
        return new_memory

    def predict_next_memory(self, memory):
        """Predict M_{t+1} keys and values from M_t. Returns (pred_keys, pred_values) or None."""
        if self.dynamics_key_heads is None:
            return None
        pred_keys, pred_values = [], []
        for l in range(memory.n_layers):
            head_idx = 0 if self.cfg.share_across_layers else l
            mk, mv = memory.layer(l)
            pk, _ = self.dynamics_key_heads[head_idx](mk)
            pv, _ = self.dynamics_val_heads[head_idx](mv)
            pred_keys.append(pk)
            pred_values.append(pv)
        return pred_keys, pred_values

    def initial_compress(
        self,
        keys_per_layer:   list[torch.Tensor],   # n_layers × (B, T, d)
        values_per_layer: list[torch.Tensor],
        features:         torch.Tensor | None = None,
        return_preds:     bool = False,
    ) -> "BeliefMemory | tuple[BeliefMemory, torch.Tensor]":
        """Bootstrap the belief state from the first chunk (step 0).

        Uses a zero-initialised prior memory so cross-attention attends
        entirely to the new chunk, making the first step equivalent to a
        vanilla Perceiver compression.

        Parameters
        ----------
        return_preds:  If True, return ``(memory, preds)`` where ``preds`` is
                       the prediction tensor ``(n_layers, B, C, d)`` from the
                       underlying update call.  Pass to ``predictive_coding_loss``
                       to train the prediction head on step 0 (predicting chunk 0
                       from a zero prior, which trains the head toward the data mean).
        """
        B = keys_per_layer[0].shape[0]
        # Use slot_pos to seed initial memory with distinct slot identities.
        # All-zero initialization collapses all C slots to identical values
        # because every slot produces the same cross-attention pattern.
        init_keys = self.slot_pos.expand(B, -1, -1).clone()    # (B, C, d) — clone avoids shared storage with slot_pos
        # Stack across layers: each layer gets the same initial slot queries
        init_keys_stacked = init_keys.unsqueeze(0).expand(
            self.cfg.n_attn_layers, -1, -1, -1
        ).clone()                                               # (L, B, C, d)
        init_vals_stacked = torch.zeros_like(init_keys_stacked)
        prior_mem = BeliefMemory(
            keys=init_keys_stacked,
            values=init_vals_stacked,
            step=0,
        )
        new_memory, _, preds = self.update(prior_mem, keys_per_layer, values_per_layer, features)
        memory = BeliefMemory(new_memory.keys, new_memory.values, step=0)
        if return_preds:
            return memory, preds
        return memory
