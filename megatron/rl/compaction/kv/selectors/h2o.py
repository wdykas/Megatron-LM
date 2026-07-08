# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H2O KV compressor (Heavy-Hitter Oracle).

Source: Zhang et al., 2023 (arXiv:2306.14048).

Paper-exact policy: the retained KV budget is split between the most-recent
tokens (recent/local window) and the heavy hitters, where a token's
heavy-hitter score is the *accumulated* attention probability it has received
(F_score(j) = Σ_i softmax(q_i·K)_j summed over queries i). Retained keys and
values are kept unchanged — H2O selects, it does not refit K or V.

Shared types/helpers live in `compressors.py`.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from ..compressors import (
    CompactionResult,
    _select_recent_plus_heavy,
    _softmax_attention,
    _validate_budget,
)


class H2OAccumulator:
    """Paper-exact H2O: recent window + heavy hitters by accumulated attention.

    The score of a key is the total softmax attention probability it received,
    accumulated over queries (Σ_i softmax(q_i·K)_j). The budget is split into a
    recent window (the last ``recent_ratio`` of the budget) and heavy hitters
    (the rest, by accumulated score). Retained K/V are the originals — no fitting.

    Heavy-hitter scores come from, in priority order: ``accumulated_scores``, the
    online state built by ``update()`` (call once per decode step), or — offline —
    the softmax attention over ``ref_queries`` passed to ``compress()``.

    Parameters
    ----------
    recent_ratio: Fraction of the budget reserved for the recent window. The
                  paper uses an even split (0.5): half recent, half heavy.
    """

    def __init__(self, recent_ratio: float = 0.5) -> None:
        if not 0.0 <= recent_ratio <= 1.0:
            raise ValueError(f"recent_ratio must be in [0, 1], got {recent_ratio}")
        self.recent_ratio = recent_ratio
        self._accumulated: torch.Tensor | None = None

    @property
    def strategy(self) -> str:
        return f"h2o_recent{self.recent_ratio:g}"

    def update(self, attn_weights: torch.Tensor) -> None:
        """Accumulate softmax attention weights from one decode step.

        attn_weights: (T,) or (H, T) — averaged over heads if 2-D.
        """
        w = attn_weights.mean(dim=0) if attn_weights.dim() == 2 else attn_weights
        if self._accumulated is None:
            self._accumulated = w.clone()
        else:
            T_prev = self._accumulated.shape[0]
            T_new = w.shape[0]
            if T_new > T_prev:
                self._accumulated = F.pad(self._accumulated, (0, T_new - T_prev))
            self._accumulated[:T_new] += w

    def reset(self) -> None:
        """Clear accumulated state between requests."""
        self._accumulated = None

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
        accumulated_scores: torch.Tensor | None = None,
        ref_query_end: int | None = None,
    ) -> CompactionResult:
        """Retain the recent window plus the top heavy hitters within ``budget``."""
        scores = accumulated_scores if accumulated_scores is not None else self._accumulated
        if scores is None:
            if ref_queries is None:
                raise RuntimeError(
                    "H2OAccumulator needs heavy-hitter scores: call update() after each "
                    "decode step (online), pass accumulated_scores, or pass ref_queries "
                    "to score offline."
                )
            # Offline H2O score: total CAUSAL softmax attention each key received
            # over the queries (the paper's accumulated-attention F_score; the
            # official kernel is causally masked — non-causal scoring leaks mass
            # to future keys and deflates sink/early-token scores).
            scores = _softmax_attention(ref_queries, keys, causal_tail=True,
                                        query_end=ref_query_end).sum(dim=0)

        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)

        positions = _select_recent_plus_heavy(
            scores, T, budget, n_recent=round(budget * self.recent_ratio)
        )
        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy=self.strategy, original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )
