# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Still: per-layer Perceiver-based KV cache compactor.

    C learnable "compression queries" cross-attend to the full KV cache (T tokens):

        h   = CrossAttn(queries -> K_full, V_full)   # (B, C, d)
        C_k = W_k h                                   # synthetic compact keys
        C_v = W_v h                                   # synthetic compact values

    A single forward pass compresses all layers (separate weights per layer, or
    shared across layers for a cheaper model).

Megatron integration
--------------------
The compactor is a ``MegatronModule`` built from standard Megatron building
blocks — ``MLP`` for the feed-forward, ``TEDotProductAttention`` for the attention
core, and TE column/row-parallel linears for the projections. It is **replicated**
on every rank (tensor_model_parallel_size = 1): each rank trains it on its own
local KV slice and gradients are averaged across the world (see
``compaction/learned/training/parallel.py``). Replication is achieved the
idiomatic way — the modules run with a tensor-parallel group of size 1 (the unit
tests use ``parallel_state`` tp=1; the online path passes a singleton tp group via
``pg_collection``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
    TEDotProductAttention,
    TENorm,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def compactor_transformer_config(
    d_kv: int,
    n_heads: int,
    ffn_hidden_size: int,
    params_dtype: torch.dtype = torch.float32,
) -> TransformerConfig:
    """TransformerConfig that drives the compactor's Megatron sub-modules.

    The compactor is replicated, so ``tensor_model_parallel_size = 1``; the linear
    layers and attention therefore never shard (they run on a tp-group of size 1).
    ``hidden_size = d_kv`` and ``num_query_groups = num_attention_heads`` (plain
    multi-head attention — no GQA inside the compactor).
    """
    return TransformerConfig(
        num_layers=1,
        hidden_size=d_kv,
        num_attention_heads=n_heads,
        num_query_groups=n_heads,
        kv_channels=d_kv // n_heads,
        ffn_hidden_size=ffn_hidden_size,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=F.gelu,
        normalization="LayerNorm",
        bf16=(params_dtype == torch.bfloat16),
        fp16=(params_dtype == torch.float16),
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
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
# Building blocks (standard Megatron modules, replicated)
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """Cross-attention: learned queries attend to a KV cache with SEPARATE K/V.

    This is *not* ``megatron.core.transformer.attention.CrossAttention`` — that
    module derives K and V from a single ``key_value_states`` via one projection.
    The compactor compresses a real KV cache, where keys and values are
    independent tensors (q from the latent queries, k from cache-K, v from
    cache-V). The internals are still standard Megatron: pre-norms (``TENorm``),
    TE column/row-parallel linears for q/k/v/out, and the Megatron
    ``TEDotProductAttention`` core (sequence-first ``sbhd`` layout, no mask).
    """

    def __init__(self, config: TransformerConfig, pg_collection=None, layer_number: int = 1) -> None:
        super().__init__()
        d = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = config.kv_channels
        proj = self.d_head * self.n_heads
        tp_group = pg_collection.tp if pg_collection is not None else None
        lin = dict(config=config, init_method=config.init_method, gather_output=False,
                   bias=False, skip_bias_add=False, is_expert=False, tp_group=tp_group)

        self.norm_q = TENorm(config, d)
        self.norm_kv = TENorm(config, d)
        self.linear_q = TEColumnParallelLinear(d, proj, **lin)
        self.linear_k = TEColumnParallelLinear(d, proj, **lin)
        self.linear_v = TEColumnParallelLinear(d, proj, **lin)
        self.core_attention = TEDotProductAttention(
            config=config,
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="cross",
            pg_collection=pg_collection,
        )
        self.linear_proj = TERowParallelLinear(
            proj, d, config=config, init_method=config.init_method, bias=False,
            input_is_parallel=True, skip_bias_add=False, is_expert=False, tp_group=tp_group,
        )

    def forward(
        self,
        queries: torch.Tensor,   # (B, Sq, d)
        keys: torch.Tensor,      # (B, Sk, d)
        values: torch.Tensor,    # (B, Sk, d)
    ) -> torch.Tensor:           # (B, Sq, d)
        q, _ = self.linear_q(self.norm_q(queries))
        k, _ = self.linear_k(self.norm_kv(keys))
        v, _ = self.linear_v(self.norm_kv(values))

        B, Sq, _ = q.shape
        Sk = k.shape[1]
        h, dh = self.n_heads, self.d_head
        # Megatron attention is sequence-first (sbhd): (S, B, np, hn).
        q = q.transpose(0, 1).contiguous().view(Sq, B, h, dh)
        k = k.transpose(0, 1).contiguous().view(Sk, B, h, dh)
        v = v.transpose(0, 1).contiguous().view(Sk, B, h, dh)

        ctx = self.core_attention(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.no_mask)
        ctx = ctx.transpose(0, 1)            # (B, Sq, proj)
        out, _ = self.linear_proj(ctx)
        return out


class FeedForward(nn.Module):
    """Pre-norm feed-forward block — ``TENorm`` + the Megatron ``MLP``."""

    def __init__(self, config: TransformerConfig, pg_collection=None) -> None:
        super().__init__()
        self.norm = TENorm(config, config.hidden_size)
        self.mlp = MLP(
            config,
            MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
            tp_group=(pg_collection.tp if pg_collection is not None else None),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mlp(self.norm(x))
        return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PerceiverConfig:
    """Configuration for the PerceiverCompactor.

    Use ``PerceiverConfig.from_transformer_config()`` to construct — do not
    set ``d_kv`` or ``n_attn_layers`` by hand.
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
            raise ValueError(f"d_kv ({self.d_kv}) must be divisible by n_heads ({self.n_heads})")
        if self.n_compress < 1:
            raise ValueError(f"n_compress must be >= 1, got {self.n_compress}")
        if self.n_attn_layers < 1:
            raise ValueError(f"n_attn_layers must be >= 1, got {self.n_attn_layers}")

    @classmethod
    def from_transformer_config(cls, transformer_config, n_compress: int, n_heads: int = 8, **kwargs) -> "PerceiverConfig":
        """Construct from a Megatron TransformerConfig (derives d_kv, n_attn_layers)."""
        d_kv = transformer_config.kv_channels * transformer_config.num_query_groups
        n_attn_layers = _count_attention_layers(transformer_config)
        return cls(n_compress=n_compress, n_heads=n_heads, d_kv=d_kv, n_attn_layers=n_attn_layers, **kwargs)

    @property
    def d_head(self) -> int:
        return self.d_kv // self.n_heads

    @property
    def ff_dim(self) -> int:
        return self.d_ff or self.d_kv * 4


# ---------------------------------------------------------------------------
# Per-layer compactor
# ---------------------------------------------------------------------------

class _LayerCompactor(nn.Module):
    """Single-layer compactor: cross-attend + FFN + project to K, V spaces."""

    def __init__(self, cfg: PerceiverConfig, config: TransformerConfig, pg_collection=None) -> None:
        super().__init__()
        tp_group = pg_collection.tp if pg_collection is not None else None
        lin = dict(config=config, init_method=config.init_method, gather_output=False,
                   bias=False, skip_bias_add=False, is_expert=False, tp_group=tp_group)
        self.cross_attn = CrossAttention(config, pg_collection)
        self.ffn = FeedForward(config, pg_collection)
        self.key_proj = TEColumnParallelLinear(cfg.d_kv, cfg.d_kv, **lin)
        self.val_proj = TEColumnParallelLinear(cfg.d_kv, cfg.d_kv, **lin)

    def forward(
        self,
        compress_q: torch.Tensor,   # (B, C, d)
        keys: torch.Tensor,         # (B, T, d)
        values: torch.Tensor,       # (B, T, d)
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

    For each transformer layer, takes the full KV cache of length T and produces
    a compact synthetic KV cache of length C using cross-attention.

    Parameters
    ----------
    cfg:                PerceiverConfig (dimensions, n_compress, n_heads, ...).
    params_dtype:       dtype of the compactor parameters (bf16 in training).
    pg_collection:      Optional ProcessGroupCollection. Online training passes a
                        collection whose tp group is a per-rank singleton so the
                        module is replicated across the real tensor-parallel group;
                        unit tests pass None and rely on ``parallel_state`` tp=1.
    """

    def __init__(
        self,
        cfg: PerceiverConfig,
        params_dtype: torch.dtype = torch.float32,
        pg_collection=None,
    ) -> None:
        config = compactor_transformer_config(cfg.d_kv, cfg.n_heads, cfg.ff_dim, params_dtype)
        super().__init__(config)
        self.cfg = cfg

        n_sets = 1 if cfg.share_across_layers else cfg.n_attn_layers
        self.compress_queries = nn.Parameter(torch.randn(n_sets, cfg.n_compress, cfg.d_kv) * 0.02)
        self._layer_modules = nn.ModuleList([
            _LayerCompactor(cfg, config, pg_collection) for _ in range(n_sets)
        ])

    def _queries(self, layer_idx: int) -> torch.Tensor:
        return self.compress_queries[0 if self.cfg.share_across_layers else layer_idx]

    def _module(self, layer_idx: int) -> _LayerCompactor:
        return self._layer_modules[0 if self.cfg.share_across_layers else layer_idx]

    def forward(
        self,
        keys: torch.Tensor,    # (B, T, d)
        values: torch.Tensor,  # (B, T, d)
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a single layer's KV cache from T tokens to C tokens."""
        B = keys.shape[0]
        Q = self._queries(layer_idx).unsqueeze(0).expand(B, -1, -1)   # (B, C, d)
        return self._module(layer_idx)(Q, keys, values)

    def compress_all_layers(
        self,
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Compress all layers in one call. Returns (compact_keys, compact_values)."""
        ck_list, cv_list = [], []
        for l, (k, v) in enumerate(zip(keys_per_layer, values_per_layer)):
            ck, cv = self.forward(k, v, l)
            ck_list.append(ck)
            cv_list.append(cv)
        return ck_list, cv_list
