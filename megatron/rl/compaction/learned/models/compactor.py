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

# ---------------------------------------------------------------------------
# Optional Megatron base class (falls back to nn.Module when unavailable)
# ---------------------------------------------------------------------------

try:
    from megatron.core.transformer.module import MegatronModule as _MegatronBase
    _MEGATRON_AVAILABLE = True
except ImportError:
    _MegatronBase = nn.Module
    _MEGATRON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tensor-parallel linear factory
# ---------------------------------------------------------------------------

class _Linear(nn.Module):
    """Uniform call interface over nn.Linear and Megatron parallel linears.

    ColumnParallelLinear / RowParallelLinear always return ``(output, bias)``;
    nn.Linear returns just ``output``.  This wrapper normalises to a single
    tensor return so callers don't need to branch.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self._inner = inner
        self._is_parallel = not isinstance(inner, nn.Linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._inner(x)
        return out[0] if self._is_parallel else out


def _make_linear(
    in_features: int,
    out_features: int,
    transformer_config=None,
    *,
    column: bool,
    gather_output: bool = True,
) -> _Linear:
    """Create a linear layer, using Megatron parallel variants when TP > 1.

    Parameters
    ----------
    in_features:        Input dimension.
    out_features:       Output dimension.
    transformer_config: Megatron ModelParallelConfig (or TransformerConfig).
                        When provided AND tensor parallelism > 1, creates
                        ColumnParallelLinear or RowParallelLinear.
                        When None, always uses nn.Linear.
    column:             True → ColumnParallelLinear; False → RowParallelLinear.
    gather_output:      Passed to ColumnParallelLinear.  Set False when the
                        output feeds another parallel layer (e.g. q_proj feeding
                        the attention kernel before out_proj).
    """
    if transformer_config is not None:
        try:
            from megatron.core.parallel_state import get_tensor_model_parallel_world_size
            from megatron.core import tensor_parallel
            from megatron.core.utils import init_method_normal
            tp = get_tensor_model_parallel_world_size()
        except Exception:
            tp = 1

        if tp > 1:
            # init_method: use config's method when available, else Xavier.
            init_fn = getattr(transformer_config, 'init_method', None)
            if init_fn is None:
                init_fn = init_method_normal(0.02)

            if column:
                inner = tensor_parallel.ColumnParallelLinear(
                    in_features,
                    out_features,
                    config=transformer_config,
                    init_method=init_fn,
                    bias=False,
                    gather_output=gather_output,
                    skip_bias_add=False,
                )
            else:
                inner = tensor_parallel.RowParallelLinear(
                    in_features,
                    out_features,
                    config=transformer_config,
                    init_method=init_fn,
                    bias=False,
                    input_is_parallel=True,
                    skip_bias_add=False,
                )
            return _Linear(inner)

    return _Linear(nn.Linear(in_features, out_features, bias=False))


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

    Queries come from the learnable compression state (C vectors).
    Keys and values come from the full (or combined) KV cache (T vectors).

    Uses ``F.scaled_dot_product_attention`` which dispatches to flash attention
    on CUDA when available, falling back to the math kernel otherwise.

    When ``transformer_config`` is provided and TP > 1, Q/K/V projections use
    Megatron's ColumnParallelLinear and the output projection uses
    RowParallelLinear so the attention computation is distributed across ranks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        transformer_config=None,
    ) -> None:
        super().__init__()
        d_head = d_model // n_heads
        self.n_heads = n_heads
        self.d_head = d_head
        self.dropout_p = dropout

        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Determine if projections will be tensor-parallel (column-parallel output split).
        # Store at init time so forward() knows which head count to use for .view().
        _tp_at_init = 1
        if transformer_config is not None:
            try:
                from megatron.core.parallel_state import get_tensor_model_parallel_world_size
                _tp_at_init = get_tensor_model_parallel_world_size()
            except Exception:
                pass
        self._is_parallel = (_tp_at_init > 1)

        # Column-parallel projections split the output dimension across TP ranks;
        # gather_output=False because the result feeds the attention kernel
        # (another parallel op) before the row-parallel out_proj.
        self.q_proj   = _make_linear(d_model, d_model, transformer_config,
                                     column=True,  gather_output=False)
        self.k_proj   = _make_linear(d_model, d_model, transformer_config,
                                     column=True,  gather_output=False)
        self.v_proj   = _make_linear(d_model, d_model, transformer_config,
                                     column=True,  gather_output=False)
        # Row-parallel: reduces partial sums from all TP ranks into a full output.
        self.out_proj = _make_linear(d_model, d_model, transformer_config,
                                     column=False)

    def forward(
        self,
        queries: torch.Tensor,   # (B, C, d)
        keys:    torch.Tensor,   # (B, T, d)
        values:  torch.Tensor,   # (B, T, d)
    ) -> torch.Tensor:           # (B, C, d)
        B, C, d = queries.shape
        T = keys.shape[1]

        # Only split heads if projections are actually ColumnParallelLinear
        # (i.e. transformer_config was provided at init AND TP > 1).
        if self._is_parallel:
            try:
                from megatron.core.parallel_state import get_tensor_model_parallel_world_size as _tp
                local_heads = self.n_heads // _tp()
            except Exception:
                local_heads = self.n_heads
        else:
            local_heads = self.n_heads
        dh = self.d_head

        Q = self.q_proj(self.norm_q(queries)).view(B, C, local_heads, dh).transpose(1, 2)
        K = self.k_proj(self.norm_kv(keys)).view(B, T, local_heads, dh).transpose(1, 2)
        V = self.v_proj(self.norm_kv(values)).view(B, T, local_heads, dh).transpose(1, 2)

        # Flash-attention aware: dispatches to efficient kernel when on CUDA.
        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)  # (B, h, C, dh)

        out = out.transpose(1, 2).contiguous().view(B, C, local_heads * dh)
        return self.out_proj(out)


class _FFNBlock(nn.Module):
    """Pre-norm feed-forward block.

    When ``transformer_config`` is provided and TP > 1, uses
    ColumnParallelLinear (fc1) + RowParallelLinear (fc2) for tensor parallelism.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        transformer_config=None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # fc1 splits d_ff across TP ranks; fc2 expects the sharded input.
        self.fc1  = _make_linear(d_model, d_ff,    transformer_config,
                                 column=True,  gather_output=False)
        self.fc2  = _make_linear(d_ff,    d_model, transformer_config,
                                 column=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.drop(F.gelu(self.fc1(h)))
        return self.fc2(h)


class _LayerCompactor(nn.Module):
    """Single-layer compactor: cross-attend + FFN + project to K, V spaces."""

    def __init__(self, cfg: PerceiverConfig, transformer_config=None) -> None:
        super().__init__()
        self.cross_attn = _CrossAttentionBlock(
            cfg.d_kv, cfg.n_heads, cfg.dropout, transformer_config)
        self.ffn        = _FFNBlock(
            cfg.d_kv, cfg.ff_dim, cfg.dropout, transformer_config)
        # Output heads: gather_output=True so each rank has the full compact KV.
        self.key_proj   = _make_linear(cfg.d_kv, cfg.d_kv, transformer_config,
                                       column=True, gather_output=True)
        self.val_proj   = _make_linear(cfg.d_kv, cfg.d_kv, transformer_config,
                                       column=True, gather_output=True)

    def forward(
        self,
        compress_q: torch.Tensor,   # (B, C, d)
        keys:       torch.Tensor,   # (B, T, d)
        values:     torch.Tensor,   # (B, T, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = compress_q + self.cross_attn(compress_q, keys, values)
        h = h + self.ffn(h)
        return self.key_proj(h), self.val_proj(h)


# ---------------------------------------------------------------------------
# PerceiverCompactor  (the Still model)
# ---------------------------------------------------------------------------

class PerceiverCompactor(_MegatronBase):
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
        if _MEGATRON_AVAILABLE and transformer_config is not None:
            super().__init__(config=transformer_config)
        else:
            nn.Module.__init__(self)

        self.cfg = cfg
        self._transformer_config = transformer_config

        # Learnable compression queries: one set per layer (or one shared set).
        n_sets = 1 if cfg.share_across_layers else cfg.n_attn_layers
        self.compress_queries = nn.Parameter(
            torch.randn(n_sets, cfg.n_compress, cfg.d_kv) * 0.02
        )

        if cfg.share_across_layers:
            self._layer_modules = nn.ModuleList([
                _LayerCompactor(cfg, transformer_config)
            ])
        else:
            self._layer_modules = nn.ModuleList([
                _LayerCompactor(cfg, transformer_config)
                for _ in range(cfg.n_attn_layers)
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
