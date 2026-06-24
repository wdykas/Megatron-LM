# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Still: per-layer Perceiver-based KV cache compactor.

Paper: "Still: Amortized KV Cache Compaction in a Single Forward Pass"
       https://arxiv.org/abs/2606.07878

Core architecture:

    C learnable "compression queries" Q_c ∈ R^{C×d}
    cross-attend to the full KV cache (T tokens):

        h = CrossAttn(Q_c → K_full, V_full)    # (C, d)
        C_k = W_k h                              # synthetic compact keys
        C_v = W_v h                              # synthetic compact values

    The compactor is trained per-layer (separate weights per transformer layer)
    or can optionally share weights across layers for a cheaper model.

    A single forward pass compresses all layers simultaneously.

Megatron integration
--------------------
PerceiverCompactor is a MegatronModule when ``transformer_config`` is provided,
enabling:
  • ColumnParallelLinear / RowParallelLinear when tensor parallelism > 1.
  • Proper distributed checkpointing via ``sharded_state_dict``.

Standalone / unit-test usage (no Megatron parallel state required):
  cfg = PerceiverConfig(n_compress=32, n_heads=4, d_kv=64, n_attn_layers=12)
  model = PerceiverCompactor(cfg)           # plain nn.Module, nn.Linear

Megatron training:
  perceiver_cfg = PerceiverConfig.from_transformer_config(transformer_config, n_compress=32)
  model = PerceiverCompactor(perceiver_cfg, transformer_config=transformer_config)

Belief-Still extends this with a recurrent update:

    M_{t+1} = PerceiverCompactor([M_t ; R_t])

    where M_t is the current compact belief memory and R_t is the new chunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TELinear, TENorm


# ---------------------------------------------------------------------------
# Layer factory — replicated TransformerEngine layers (Megatron wrappers)
# ---------------------------------------------------------------------------
#
# The compactor is a small module REPLICATED across every tensor-parallel rank
# (not sharded). We use Megatron's TransformerEngine wrappers with
# parallel_mode='duplicated' (the same replicated mode MoE uses): fused /
# FP8-capable TE kernels, weights replicated across the TP group, and grads kept
# in sync by Megatron's machinery. TE is CUDA-only — the compactor is GPU-only.

def _minimal_transformer_config(d_model: int, n_heads: int) -> TransformerConfig:
    """Build a minimal TransformerConfig for the compactor.

    Used when no model TransformerConfig is supplied. Drives the TE wrapper
    layers (dims, init, dtype) and registers the module as a MegatronModule with
    sharded_state_dict / distributed-checkpointing support.
    """
    return TransformerConfig(
        num_layers=1,
        hidden_size=d_model,
        num_attention_heads=n_heads,
    )


def duplicated_linear(in_features: int, out_features: int, config, bias: bool = False) -> TELinear:
    """A replicated TELinear (``parallel_mode='duplicated'``), as MoE / MLA use.

    The weight is replicated across the TP group (not sharded) and Megatron's
    grad machinery keeps the replicas in sync. Returns a ``TELinear`` directly so
    it is stored as the attribute (clean ``<name>.weight`` keys); like every
    Megatron linear it returns an ``(output, bias)`` tuple — unwrap at the call
    site (``out, _ = layer(x)``).
    """
    return TELinear(
        in_features,
        out_features,
        parallel_mode='duplicated',
        config=config,
        init_method=config.init_method,
        bias=bias,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        is_expert=False,
    )


# ---------------------------------------------------------------------------
# Helper: count attention layers from a TransformerConfig
# ---------------------------------------------------------------------------

def _count_attention_layers(transformer_config) -> int:
    """Return the number of attention (non-Mamba) layers in the model."""
    pattern = getattr(transformer_config, 'hybrid_layer_pattern', None)
    if not pattern:
        return transformer_config.num_layers
    from megatron.core.ssm.mamba_hybrid_layer_allocation import get_hybrid_layer_counts
    counts = get_hybrid_layer_counts(pattern)
    return sum(v for k, v in counts.items() if k not in ('M', '*'))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PerceiverConfig:
    """Configuration for the PerceiverCompactor.

    Use ``PerceiverConfig.from_transformer_config()`` to construct — do not
    set ``d_kv`` or ``n_attn_layers`` by hand.

    Attributes
    ----------
    n_compress:          Target compressed sequence length C (budget).
    n_heads:             Number of attention heads in the compactor cross-attn.
    d_kv:                Flattened KV dimension per token
                         (= kv_channels × num_query_groups from TransformerConfig).
                         Set automatically by ``from_transformer_config``.
    n_attn_layers:       Number of attention layers in the model (excludes Mamba
                         layers for hybrid models).
                         Set automatically by ``from_transformer_config``.
    d_ff:                Feed-forward hidden dim. Default: 4 × d_kv.
    dropout:             Dropout rate.
    share_across_layers: If True, one set of compactor weights for all layers
                         (cheap). If False (default), per-layer weights
                         (higher quality).
    """

    n_compress: int
    n_heads: int
    d_kv: int
    n_attn_layers: int
    d_ff: int | None = None
    dropout: float = 0.0
    share_across_layers: bool = False

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
    ) -> "PerceiverConfig":
        """Construct from a Megatron TransformerConfig.

        Derives ``d_kv`` and ``n_attn_layers`` from the model config so they
        never need to be specified manually.
        """
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
    def d_head(self) -> int:
        return self.d_kv // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.d_ff or self.d_kv * 4


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CrossAttentionBlock(nn.Module):
    """Perceiver-style cross-attention: queries attend to keys and values.

    Queries come from the learnable compression state (C vectors); keys and
    values come from the full (or combined) KV cache (T vectors).

    Projections are replicated linears (Megatron ``TELinear`` duplicated). The
    attention is **raw** ``te.DotProductAttention`` (parameter-free), not the
    Megatron ``TEDotProductAttention`` wrapper: the wrapper shards heads across
    the TP group, but the compactor is REPLICATED (full heads on every rank), so
    we run the un-sharded attention. Being parameter-free, it needs no grad-sync
    or checkpointing. qkv_format='bshd', no causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, config) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm_q  = TENorm(config, d_model)
        self.norm_kv = TENorm(config, d_model)

        self.q_proj   = duplicated_linear(d_model, d_model, config)
        self.k_proj   = duplicated_linear(d_model, d_model, config)
        self.v_proj   = duplicated_linear(d_model, d_model, config)
        self.out_proj = duplicated_linear(d_model, d_model, config)

        self.attn = te.DotProductAttention(
            num_attention_heads=n_heads,
            kv_channels=self.d_head,
            attention_dropout=dropout,
            qkv_format='bshd',
            attn_mask_type='no_mask',
        )

    def forward(
        self,
        queries: torch.Tensor,   # (B, C, d)
        keys:    torch.Tensor,   # (B, T, d)
        values:  torch.Tensor,   # (B, T, d)
    ) -> torch.Tensor:           # (B, C, d)
        B, C, _ = queries.shape
        T = keys.shape[1]
        h, dh = self.n_heads, self.d_head

        q, _ = self.q_proj(self.norm_q(queries))
        k, _ = self.k_proj(self.norm_kv(keys))
        v, _ = self.v_proj(self.norm_kv(values))
        Q = q.view(B, C, h, dh)
        K = k.view(B, T, h, dh)
        V = v.view(B, T, h, dh)

        out = self.attn(Q, K, V)            # (B, C, h*dh) with qkv_format='bshd'
        out, _ = self.out_proj(out)
        return out


class _FFNBlock(nn.Module):
    """Pre-norm feed-forward block (Megatron TE wrappers, replicated)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, config) -> None:
        super().__init__()
        self.norm = TENorm(config, d_model)
        self.fc1  = duplicated_linear(d_model, d_ff, config)
        self.fc2  = duplicated_linear(d_ff, d_model, config)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h1, _ = self.fc1(h)
        h = self.drop(F.gelu(h1))
        h2, _ = self.fc2(h)
        return h2


class _LayerCompactor(nn.Module):
    """Single-layer compactor: cross-attend + FFN + project to K, V spaces."""

    def __init__(self, cfg: PerceiverConfig, config) -> None:
        super().__init__()
        self.cross_attn = _CrossAttentionBlock(cfg.d_kv, cfg.n_heads, cfg.dropout, config)
        self.ffn        = _FFNBlock(cfg.d_kv, cfg.ff_dim, cfg.dropout, config)
        self.key_proj   = duplicated_linear(cfg.d_kv, cfg.d_kv, config)
        self.val_proj   = duplicated_linear(cfg.d_kv, cfg.d_kv, config)

    def forward(
        self,
        compress_q: torch.Tensor,   # (B, C, d)
        keys:       torch.Tensor,   # (B, T, d)
        values:     torch.Tensor,   # (B, T, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = compress_q + self.cross_attn(compress_q, keys, values)
        h = h + self.ffn(h)
        ck, _ = self.key_proj(h)
        cv, _ = self.val_proj(h)
        return ck, cv


# ---------------------------------------------------------------------------
# PerceiverCompactor  (the Still model)
# ---------------------------------------------------------------------------

class PerceiverCompactor(MegatronModule):
    """Per-layer Perceiver KV compactor (Still architecture).

    For each transformer layer, takes the full KV cache of length T and
    produces a compact synthetic KV cache of length C using cross-attention.

    Inherits from MegatronModule when available and ``transformer_config`` is
    provided, enabling tensor-parallel linear layers and distributed
    checkpointing.  Falls back to plain ``nn.Module`` with ``nn.Linear``
    weights for standalone / unit-test usage.

    Parameters
    ----------
    cfg:                PerceiverConfig (dimensions, n_compress, n_heads, ...).
    transformer_config: Optional Megatron ModelParallelConfig / TransformerConfig.
                        When provided:
                          • Enables ColumnParallelLinear / RowParallelLinear
                            when tensor_model_parallel_size > 1.
                          • Registers this module as a MegatronModule with
                            proper sharded_state_dict support.

    Usage — standalone (tests, scripts):
        model = PerceiverCompactor(cfg)

    Usage — Megatron training:
        model = PerceiverCompactor(cfg, transformer_config=transformer_config)
    """

    def __init__(
        self,
        cfg: PerceiverConfig,
        transformer_config=None,
    ) -> None:
        config = transformer_config or _minimal_transformer_config(cfg.d_kv, cfg.n_heads)
        super().__init__(config)

        self.cfg = cfg

        # Learnable compression queries: one set per layer (or one shared set).
        n_sets = 1 if cfg.share_across_layers else cfg.n_attn_layers
        self.compress_queries = nn.Parameter(
            torch.randn(n_sets, cfg.n_compress, cfg.d_kv) * 0.02
        )

        n_modules = 1 if cfg.share_across_layers else cfg.n_attn_layers
        self._layer_modules = nn.ModuleList([
            _LayerCompactor(cfg, config) for _ in range(n_modules)
        ])

    # --- helpers --------------------------------------------------------

    def _queries(self, layer_idx: int) -> torch.Tensor:
        idx = 0 if self.cfg.share_across_layers else layer_idx
        return self.compress_queries[idx]                           # (C, d)

    def _module(self, layer_idx: int) -> _LayerCompactor:
        return self._layer_modules[0 if self.cfg.share_across_layers else layer_idx]

    # --- public API -----------------------------------------------------

    def forward(
        self,
        keys:      torch.Tensor,   # (B, T, d)
        values:    torch.Tensor,   # (B, T, d)
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a single layer's KV cache from T tokens to C tokens."""
        B = keys.shape[0]
        Q = self._queries(layer_idx).unsqueeze(0).expand(B, -1, -1)   # (B, C, d)
        return self._module(layer_idx)(Q, keys, values)

    def compress_all_layers(
        self,
        keys_per_layer:   list[torch.Tensor],   # n_layers × (B, T, d)
        values_per_layer: list[torch.Tensor],   # n_layers × (B, T, d)
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compress all layers in one call. Returns (compact_keys, compact_values)."""
        ck_list, cv_list = [], []
        for l, (k, v) in enumerate(zip(keys_per_layer, values_per_layer)):
            ck, cv = self.forward(k, v, l)
            ck_list.append(ck)
            cv_list.append(cv)
        return ck_list, cv_list
