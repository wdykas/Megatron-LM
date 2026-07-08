# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Online KV cache position selectors implementing KVCompressor.

All selectors implement the unified compress(keys, values, budget,
ref_queries=None, run_id="", step_id=0) -> CompactionResult interface.
They operate directly on K/V tensors (shape T×d) rather than token_id lists.
"""

from __future__ import annotations

import time

import torch

from ..compressors import (
    CompactionResult,
    _select_recent_plus_heavy,
    _softmax_attention,
    _validate_budget,
)


class AttentionSumScorer:
    """Keep the top-k positions by attention received, protecting a recent window.

    The last ``min(min_recent, budget)`` positions are always retained; the rest
    of the budget goes to the highest scorers. When ref_queries is provided,
    scores are the mean softmax attention weight across ref_queries rows;
    otherwise key L2-norm is used as a proxy score.
    """

    def __init__(self, min_recent: int = 32) -> None:
        if min_recent < 0:
            raise ValueError(f"min_recent must be >= 0, got {min_recent}")
        self.min_recent = min_recent

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
        ref_query_end: int | None = None,
    ) -> CompactionResult:
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)

        if ref_queries is not None:
            scores = _softmax_attention(ref_queries, keys).mean(dim=0)   # (T,)
        else:
            scores = keys.norm(dim=-1)                                   # (T,)

        positions = _select_recent_plus_heavy(
            scores, T, budget, n_recent=min(self.min_recent, budget)
        )
        return CompactionResult(
            run_id=run_id,
            step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy="attention_sum",
            original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )


class UniformScorer:
    """Keep every Nth token (uniform subsampling). Does not use ref_queries."""

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
        ref_query_end: int | None = None,
    ) -> CompactionResult:
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)
        if budget >= T:
            positions = list(range(T))
        else:
            step = T / budget
            positions = sorted(set(int(i * step) for i in range(budget)))

        return CompactionResult(
            run_id=run_id,
            step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy="uniform",
            original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )
