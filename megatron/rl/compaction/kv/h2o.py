# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H2O KV compressors (heavy-hitter oracle).

Source: Zhang et al., 2023 (arXiv:2306.14048).

Shared types/helpers (CompactionResult, KVCompressor protocol, attention-math
primitives) live in `compressors.py`; this module holds only the algorithm(s)
from the cited paper.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from .compressors import (
    CompactionResult,
    _fit_bias,
    _fit_values,
)


# ---------------------------------------------------------------------------
# H2OAccumulator  (paper-faithful — accumulates real softmax weights per step)
# ---------------------------------------------------------------------------

class H2OAccumulator:
    """Paper-faithful H2O: heavy-hitter eviction from real softmax attention mass.

    Online: call ``update(attn_weights)`` after every decode step with the actual
    softmax attention weights, then ``compress()`` to evict lowest-mass tokens.
    Offline (benchmark/eval): call ``compress(..., ref_queries=...)`` and it scores
    keys by the total softmax attention they receive over those queries.

    Parameters
    ----------
    n_sink:     Number of initial tokens to always preserve (default 4).
    fit_bias:   NNLS bias fitting after selection (requires ref_queries; default False).
    fit_values: OLS value fitting after selection (requires ref_queries; default False).
    """

    def __init__(self, n_sink: int = 4, fit_bias: bool = False, fit_values: bool = False) -> None:
        if n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {n_sink}")
        self.n_sink = n_sink
        self.fit_bias = fit_bias
        self.fit_values = fit_values
        self._accumulated: torch.Tensor | None = None

    @property
    def strategy(self) -> str:
        suffix = "+bias+values" if (self.fit_bias and self.fit_values) else \
                 "+bias" if self.fit_bias else \
                 "+values" if self.fit_values else ""
        return f"h2o_paper_sink{self.n_sink}{suffix}"

    def update(self, attn_weights: torch.Tensor) -> None:
        """Accumulate attention weights from one decode step.

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
    ) -> CompactionResult:
        """Compress by evicting the lowest-attention-mass tokens.

        Heavy-hitter scores come from (in priority order): ``accumulated_scores``,
        then the online state built by ``update()``, then — for offline benchmarking
        — the real softmax attention mass each key receives over ``ref_queries``.
        The offline path is paper-faithful (normalized softmax).
        """
        scores = accumulated_scores if accumulated_scores is not None else self._accumulated
        if scores is None:
            if ref_queries is None:
                raise RuntimeError(
                    "H2OAccumulator needs heavy-hitter scores: call update() after each "
                    "decode step (online), pass accumulated_scores, or pass ref_queries "
                    "to score offline."
                )
            # Offline H2O: total softmax attention each key received over the queries.
            d = keys.shape[1]
            scores = torch.softmax(ref_queries @ keys.T / math.sqrt(d), dim=-1).sum(dim=0)
        if (self.fit_bias or self.fit_values) and ref_queries is None:
            raise ValueError("ref_queries required when fit_bias or fit_values is True.")

        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = max(1, min(budget, T))
        n_sink = min(self.n_sink, budget)

        sink_positions = list(range(n_sink))

        n_heavy = budget - n_sink
        if n_heavy > 0 and T > n_sink:
            heavy_scores = scores[:T].clone()
            heavy_scores[:n_sink] = -torch.inf
            n_select = min(n_heavy, T - n_sink)
            heavy_positions = sorted(heavy_scores.topk(n_select).indices.tolist())
        else:
            heavy_positions = []

        positions = sorted(set(sink_positions + heavy_positions))
        C_k = keys[positions]

        if self.fit_bias and ref_queries is not None:
            beta, _ = _fit_bias(keys, C_k, ref_queries)
        else:
            beta = torch.zeros(len(positions), device=keys.device, dtype=keys.dtype)

        if self.fit_values and ref_queries is not None:
            C_v = _fit_values(keys, values, C_k, beta, ref_queries)
        else:
            C_v = values[positions]

        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=positions,
            compacted_keys=C_k, compacted_values=C_v, bias=beta,
            strategy=self.strategy, original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )


