# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Online KV cache position selectors implementing KVCompressor.

All selectors implement the unified compress(keys, values, budget,
ref_queries=None, run_id="", step_id=0) -> CompactionResult interface.
They operate directly on K/V tensors (shape T×d) rather than token_id lists.
"""

from __future__ import annotations

import torch

from .compressors import CompactionResult


# ---------------------------------------------------------------------------
# AttentionSumScorer
# ---------------------------------------------------------------------------

class AttentionSumScorer:
    """Keep the top-k positions by cumulative attention received.

    Always protects the last ``min_recent`` positions regardless of score,
    mirroring the H2O (Heavy-Hitter Oracle) heuristic.

    When ref_queries is provided, scores are computed as the mean softmax
    attention weight across ref_queries rows. When ref_queries is None,
    key L2-norm is used as a proxy score.
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
    ) -> CompactionResult:
        n = keys.shape[0]
        budget = max(1, min(budget, n))

        if ref_queries is not None:
            import math
            d = keys.shape[1]
            logits = ref_queries @ keys.T / math.sqrt(d)    # (n_q, T)
            logits = logits - logits.max(dim=1, keepdim=True).values
            weights = torch.softmax(logits, dim=1)           # (n_q, T)
            scores = weights.mean(dim=0)                     # (T,)
        else:
            scores = keys.norm(dim=-1)                       # (T,)

        recent_start = max(0, n - self.min_recent)
        protected = set(range(recent_start, n))

        remaining_budget = max(0, budget - len(protected))
        # Score positions not in protected set
        candidate_indices = list(range(recent_start))
        if candidate_indices and remaining_budget > 0:
            candidate_scores = scores[:recent_start]
            k = min(remaining_budget, len(candidate_indices))
            top_indices = candidate_scores.topk(k).indices.tolist()
            top = set(top_indices)
        else:
            top = set()

        retained = sorted(top | protected)
        positions = retained

        return CompactionResult(
            run_id=run_id,
            step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy="attention_sum",
            original_length=n,
            wall_time_s=0.0,
        )


# ---------------------------------------------------------------------------
# UniformScorer
# ---------------------------------------------------------------------------

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
    ) -> CompactionResult:
        n = keys.shape[0]
        budget = max(1, min(budget, n))
        if budget >= n:
            positions = list(range(n))
        else:
            step = n / budget
            positions = sorted(set(int(i * step) for i in range(budget)))

        return CompactionResult(
            run_id=run_id,
            step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy="uniform",
            original_length=n,
            wall_time_s=0.0,
        )
