# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Validation harness for KV cache compaction.

Provides metrics for evaluating compaction quality:
  - Attention output matching error (per layer/head)
  - End-to-end logit drift (KL divergence, logprob delta)
  - Memory savings measurement
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .kv_utils import gather_kv, gather_kv_with_biases


@dataclass
class AttentionMatchMetrics:
    """Per-layer attention output matching metrics."""

    layer: int
    relative_l2_per_head: List[float]  # per-head relative L2 error
    mean_relative_l2: float
    max_relative_l2: float
    mass_relative_error_per_head: List[float]
    mean_mass_error: float


@dataclass
class LogitDriftMetrics:
    """End-to-end logit drift metrics over continuation tokens."""

    mean_logprob_delta: float
    max_logprob_delta: float
    kl_divergence: float  # KL(full || compact)
    reverse_kl: float  # KL(compact || full)
    output_match_rate: float  # fraction of tokens where argmax matches
    num_eval_tokens: int


@dataclass
class CompactionValidationReport:
    """Complete validation report."""

    attention_metrics: List[AttentionMatchMetrics]
    logit_metrics: Optional[LogitDriftMetrics]
    compression_ratio: float
    original_tokens: int
    compacted_tokens: int
    memory_saved_bytes: int


def validate_attention_output(
    K_full: Tensor,
    V_full: Tensor,
    K_mem: Tensor,
    V_mem: Tensor,
    Q_eval: Tensor,
    biases: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> AttentionMatchMetrics:
    """Validate attention output matching for a single layer.

    Compares attention(Q_eval, K_full, V_full) vs attention(Q_eval, K_mem, V_mem).

    Args:
        K_full: (T, H, D) original keys.
        V_full: (T, H, D) original values.
        K_mem: (M, H, D) compacted keys.
        V_mem: (M, H, D) compacted values.
        Q_eval: (R_eval, H, D) held-out evaluation queries.
        biases: (M, H) or (M,) optional per-token biases for compacted cache.
        scale: Attention scale factor.

    Returns:
        AttentionMatchMetrics for this layer.
    """
    _, H, D = K_full.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    l2_errors = []
    mass_errors = []

    for h in range(H):
        Qh = Q_eval[:, h, :].float()
        Kh_full = K_full[:, h, :].float()
        Vh_full = V_full[:, h, :].float()
        Kh_mem = K_mem[:, h, :].float()
        Vh_mem = V_mem[:, h, :].float()

        # Full attention output
        scores_full = Qh @ Kh_full.T * scale
        weights_full = F.softmax(scores_full, dim=-1)
        O_full = weights_full @ Vh_full

        # Compact attention output
        scores_mem = Qh @ Kh_mem.T * scale
        if biases is not None:
            if biases.dim() == 2:
                # Per-head biases: (M, H) -> select head h -> (M,)
                scores_mem = scores_mem + biases[:, h].float().unsqueeze(0)
            else:
                # Shared biases: (M,)
                scores_mem = scores_mem + biases.float().unsqueeze(0)
        weights_mem = F.softmax(scores_mem, dim=-1)
        O_mem = weights_mem @ Vh_mem

        # Relative L2 error
        diff = (O_full - O_mem).pow(2).sum()
        norm = O_full.pow(2).sum() + 1e-12
        l2_errors.append((diff / norm).sqrt().item())

        # Mass error
        mass_full = scores_full.exp().sum(dim=-1)
        mass_mem = scores_mem.exp().sum(dim=-1)
        mass_err = ((mass_full - mass_mem).pow(2).mean() / (mass_full.pow(2).mean() + 1e-12)).sqrt()
        mass_errors.append(mass_err.item())

    return AttentionMatchMetrics(
        layer=-1,  # Caller should set this
        relative_l2_per_head=l2_errors,
        mean_relative_l2=sum(l2_errors) / H,
        max_relative_l2=max(l2_errors),
        mass_relative_error_per_head=mass_errors,
        mean_mass_error=sum(mass_errors) / H,
    )


def validate_attention_from_pages(
    memory_buffer: Tensor,
    layer: int,
    full_block_ids: Tensor,
    full_token_count: int,
    compact_block_ids: Tensor,
    compact_token_count: int,
    Q_eval: Tensor,
    block_size: int,
    bias_buffer: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> AttentionMatchMetrics:
    """Validate attention output by gathering from paged cache.

    Convenience wrapper that gathers from pages then calls validate_attention_output.

    Args:
        memory_buffer: 6D KV cache tensor.
        layer: Layer index.
        full_block_ids: Block IDs of the full (uncompacted) cache.
        full_token_count: Token count in full cache.
        compact_block_ids: Block IDs of compacted cache.
        compact_token_count: Token count in compacted cache.
        Q_eval: (R, H, D) evaluation queries.
        block_size: Tokens per block.
        bias_buffer: Optional bias buffer.
        scale: Attention scale.

    Returns:
        AttentionMatchMetrics.
    """
    K_full, V_full = gather_kv(memory_buffer, layer, full_block_ids, block_size, full_token_count)

    if bias_buffer is not None:
        K_mem, V_mem, biases = gather_kv_with_biases(
            memory_buffer, bias_buffer, layer,
            compact_block_ids, block_size, compact_token_count,
        )
    else:
        K_mem, V_mem = gather_kv(
            memory_buffer, layer, compact_block_ids, block_size, compact_token_count,
        )
        biases = None

    metrics = validate_attention_output(K_full, V_full, K_mem, V_mem, Q_eval, biases, scale)
    metrics.layer = layer
    return metrics


def validate_logit_drift(
    logits_full: Tensor,
    logits_compact: Tensor,
    temperature: float = 1.0,
) -> LogitDriftMetrics:
    """Compute logit drift metrics between full and compacted cache runs.

    Args:
        logits_full: (num_tokens, vocab_size) logits from full cache run.
        logits_compact: (num_tokens, vocab_size) logits from compacted cache run.
        temperature: Temperature for softmax.

    Returns:
        LogitDriftMetrics.
    """
    num_tokens = logits_full.shape[0]
    assert logits_compact.shape[0] == num_tokens

    logits_f = logits_full.float() / temperature
    logits_c = logits_compact.float() / temperature

    # Log probabilities
    log_p_full = F.log_softmax(logits_f, dim=-1)
    log_p_compact = F.log_softmax(logits_c, dim=-1)

    p_full = log_p_full.exp()
    p_compact = log_p_compact.exp()

    # KL(full || compact) = sum p_full * (log_p_full - log_p_compact)
    kl_fc = (p_full * (log_p_full - log_p_compact)).sum(dim=-1)
    # KL(compact || full)
    kl_cf = (p_compact * (log_p_compact - log_p_full)).sum(dim=-1)

    # Logprob delta at argmax positions
    argmax_full = logits_f.argmax(dim=-1)
    lp_full = log_p_full.gather(1, argmax_full.unsqueeze(1)).squeeze(1)
    lp_compact = log_p_compact.gather(1, argmax_full.unsqueeze(1)).squeeze(1)
    logprob_delta = (lp_full - lp_compact).abs()

    # Output match rate
    argmax_compact = logits_c.argmax(dim=-1)
    match_rate = (argmax_full == argmax_compact).float().mean().item()

    return LogitDriftMetrics(
        mean_logprob_delta=logprob_delta.mean().item(),
        max_logprob_delta=logprob_delta.max().item(),
        kl_divergence=kl_fc.mean().item(),
        reverse_kl=kl_cf.mean().item(),
        output_match_rate=match_rate,
        num_eval_tokens=num_tokens,
    )


def compute_memory_savings(
    original_tokens: int,
    compacted_tokens: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    dtype_bytes: int = 2,  # bf16
    include_biases: bool = True,
) -> Dict[str, float]:
    """Compute memory savings from compaction.

    Args:
        original_tokens: T, original sequence length.
        compacted_tokens: M, compacted size.
        num_heads: Number of KV heads.
        head_dim: Dimension per head.
        num_layers: Number of layers.
        dtype_bytes: Bytes per element.
        include_biases: Include bias storage overhead.

    Returns:
        Dict with memory statistics.
    """
    # Per-layer KV: 2 * T * H * D * dtype_bytes
    bytes_per_layer_full = 2 * original_tokens * num_heads * head_dim * dtype_bytes
    bytes_per_layer_compact = 2 * compacted_tokens * num_heads * head_dim * dtype_bytes

    if include_biases:
        bytes_per_layer_compact += compacted_tokens * dtype_bytes

    total_full = bytes_per_layer_full * num_layers
    total_compact = bytes_per_layer_compact * num_layers
    saved = total_full - total_compact

    return {
        "original_bytes": total_full,
        "compacted_bytes": total_compact,
        "saved_bytes": saved,
        "savings_pct": 100.0 * saved / total_full if total_full > 0 else 0,
        "compression_ratio": original_tokens / compacted_tokens if compacted_tokens > 0 else float('inf'),
    }


def run_full_validation(
    memory_buffer: Tensor,
    full_block_ids: Tensor,
    full_token_count: int,
    compact_block_ids: Tensor,
    compact_token_count: int,
    Q_eval: Tensor,
    block_size: int,
    num_layers: int,
    bias_buffer: Optional[Tensor] = None,
    scale: Optional[float] = None,
    logits_full: Optional[Tensor] = None,
    logits_compact: Optional[Tensor] = None,
) -> CompactionValidationReport:
    """Run complete validation across all layers with optional logit drift.

    Args:
        memory_buffer: 6D KV cache.
        full_block_ids: Block IDs for full cache.
        full_token_count: Token count.
        compact_block_ids: Block IDs for compacted cache.
        compact_token_count: Compacted token count.
        Q_eval: Evaluation queries.
        block_size: Tokens per block.
        num_layers: Number of attention layers.
        bias_buffer: Optional bias buffer.
        scale: Attention scale.
        logits_full: Optional full-cache logits for drift measurement.
        logits_compact: Optional compact-cache logits for drift measurement.

    Returns:
        CompactionValidationReport.
    """
    attention_metrics = []
    for layer in range(num_layers):
        m = validate_attention_from_pages(
            memory_buffer, layer,
            full_block_ids, full_token_count,
            compact_block_ids, compact_token_count,
            Q_eval, block_size, bias_buffer, scale,
        )
        attention_metrics.append(m)

    logit_metrics = None
    if logits_full is not None and logits_compact is not None:
        logit_metrics = validate_logit_drift(logits_full, logits_compact)

    # Estimate memory savings
    H = memory_buffer.shape[4] if memory_buffer.dim() == 6 else 1
    D = memory_buffer.shape[5] if memory_buffer.dim() == 6 else memory_buffer.shape[3]
    dtype_bytes = memory_buffer.element_size()
    savings = compute_memory_savings(
        full_token_count, compact_token_count, H, D, num_layers, dtype_bytes,
    )

    return CompactionValidationReport(
        attention_metrics=attention_metrics,
        logit_metrics=logit_metrics,
        compression_ratio=savings["compression_ratio"],
        original_tokens=full_token_count,
        compacted_tokens=compact_token_count,
        memory_saved_bytes=int(savings["saved_bytes"]),
    )
