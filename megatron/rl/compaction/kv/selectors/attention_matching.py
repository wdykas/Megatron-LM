# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention-Matching KV compressors (TopK, OMP).

Source: Zweiger et al., 2026 (arXiv:2602.16284).

Shared types/helpers (CompactionResult, KVCompressor protocol, attention-math
primitives) live in `compressors.py`; this module holds only the algorithm(s)
from the cited paper.
"""
from __future__ import annotations

import math
import time

import torch

from ..compressors import (
    CompactionResult,
    _mass_features,
    _nnls_box,
    _fit_bias,
    _fit_values,
    _validate_budget,
)


# ---------------------------------------------------------------------------
# TopKCompressor
# ---------------------------------------------------------------------------

class TopKCompressor:
    """Select the top-k keys by RMS attention score over reference queries.

    Fast heuristic baseline from the paper. No iterative fitting.
    """

    def __init__(self, fit_bias: bool = True, fit_values: bool = True) -> None:
        self.fit_bias = fit_bias
        self.fit_values = fit_values

    @property
    def strategy(self) -> str:
        suffix = "+bias+values" if (self.fit_bias and self.fit_values) else \
                 "+bias" if self.fit_bias else \
                 "+values" if self.fit_values else ""
        return f"topk{suffix}"

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
    ) -> CompactionResult:
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)

        if ref_queries is None:
            raise ValueError("TopKCompressor requires ref_queries for scoring.")

        Phi = _mass_features(ref_queries, keys)                     # (n, T)
        norm = Phi / Phi.sum(dim=1, keepdim=True)                   # softmax (n, T)
        rms_scores = (norm ** 2).mean(dim=0).sqrt()                 # (T,)
        positions = sorted(rms_scores.topk(budget).indices.tolist())

        C_k = keys[positions]

        if self.fit_bias:
            beta = _fit_bias(keys, C_k, ref_queries)
        else:
            beta = torch.zeros(len(positions), device=keys.device, dtype=keys.dtype)

        C_v = (_fit_values(keys, values, C_k, beta, ref_queries,
                           values_init=values[positions])
               if self.fit_values else values[positions])

        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=positions,
            compacted_keys=C_k, compacted_values=C_v, bias=beta,
            strategy=self.strategy, original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# OMPCompressor
# ---------------------------------------------------------------------------

class OMPCompressor:
    """Orthogonal Matching Pursuit key selection (Algorithm 1 from paper).

    Greedily selects keys that best reconstruct the attention mass, then fits
    values via OLS.

    Parameters
    ----------
    keys_per_iter:  k in the paper — keys added per OMP iteration. Default 1
                    = the official paper config (their fast variant uses 4).
    nnls_every:     τ in the paper — how often to refit weights (default 1).
                    Values > 1 cause stale residuals: keys added between refits
                    are scored against an outdated residual that doesn't account
                    for the most recently added keys, violating the OMP invariant.
    fit_values:     Whether to apply value fitting (default True).
    value_fit:      'lsq' = official fp32 gels lstsq (assumes MANY ref queries;
                    min-norm zeroes weakly-attended values when
                    underdetermined) | 'residual' (default; our deviation for
                    the small-n regime: ridge around the original values).
    drop_key_beta_cutoff: official refinement, default OFF as in their
                    constructor (paper configs use -7.0): drop keys with
                    fitted log-weight below cutoff, exclude, REFILL to
                    budget, refit — up to 3 rounds.
    """

    def __init__(
        self,
        keys_per_iter: int = 1,
        nnls_every: int = 1,
        fit_bias: bool = True,
        fit_values: bool = True,
        drop_key_beta_cutoff: float | None = None,
        value_fit: str = "residual",
    ) -> None:
        if keys_per_iter < 1:
            raise ValueError(f"keys_per_iter must be >= 1, got {keys_per_iter}")
        if nnls_every < 1:
            raise ValueError(f"nnls_every must be >= 1, got {nnls_every}")
        if value_fit not in ("lsq", "residual"):
            raise ValueError(f"value_fit must be 'lsq' or 'residual', got {value_fit!r}")
        self.keys_per_iter = keys_per_iter
        self.nnls_every = nnls_every
        self.fit_bias = fit_bias
        self.fit_values = fit_values
        self.drop_key_beta_cutoff = drop_key_beta_cutoff
        self.value_fit = value_fit

    @property
    def strategy(self) -> str:
        return (f"omp_k{self.keys_per_iter}"
                + ("+bias" if self.fit_bias else "")
                + ("+values" if self.fit_values else ""))

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
    ) -> CompactionResult:
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)

        if ref_queries is None:
            raise ValueError("OMPCompressor requires ref_queries for scoring.")

        positions, w = self._omp(keys, ref_queries, budget)
        w = w[:len(positions)]

        sort_idx = torch.argsort(torch.tensor(positions, dtype=torch.long))
        positions = [positions[i] for i in sort_idx.tolist()]
        w = w[sort_idx]

        beta = torch.log(w).to(keys.dtype) if self.fit_bias else None
        C_k = keys[positions]

        C_v = (_fit_values(keys, values, C_k, beta, ref_queries,
                           values_init=(values[positions]
                                        if self.value_fit == "residual" else None))
               if self.fit_values else values[positions])

        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=positions,
            compacted_keys=C_k, compacted_values=C_v, bias=beta,
            strategy=self.strategy, original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )

    def _omp(
        self, keys: torch.Tensor, ref_queries: torch.Tensor, budget: int
    ) -> tuple[list[int], torch.Tensor]:
        Phi = _mass_features(ref_queries, keys)      # (n, T) fp32
        m = Phi.sum(dim=1)                           # (n,) target mass
        UPPER = math.exp(7.0)                        # official paper weight cap

        excluded = torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device)

        def select(S, w, r, target):
            iter_idx = 0
            while len(S) < target:
                scores = r @ Phi                     # (T,)
                scores[excluded] = -torch.inf
                if not bool(torch.isfinite(scores).any()):
                    break                            # candidate pool exhausted
                k = min(self.keys_per_iter, target - len(S))
                new_idxs = scores.topk(k).indices.tolist()
                S.extend(new_idxs)
                excluded[new_idxs] = True
                iter_idx += 1
                Phi_S = Phi[:, S]
                if iter_idx % self.nnls_every == 0 or len(S) >= target:
                    w = _nnls_box(Phi_S, m, lower=1e-12, upper=UPPER)
                else:
                    # official: pad stale weights with min_val so the residual
                    # still moves between refits
                    w = torch.cat([w, torch.full((len(new_idxs),), 1e-12,
                                                 device=w.device, dtype=w.dtype)])
                r = m - Phi_S @ w
            return S, w, r

        S: list[int] = []
        w = torch.zeros(0, device=keys.device, dtype=torch.float32)
        S, w, r = select(S, w, m.clone(), budget)

        # Official refinement (opt-in): drop keys with beta below the cutoff,
        # permanently exclude, refill to budget, refit — at most 3 rounds.
        if self.drop_key_beta_cutoff is not None:
            for _ in range(3):
                w = _nnls_box(Phi[:, S], m, lower=1e-12, upper=UPPER)
                live = torch.log(w) >= self.drop_key_beta_cutoff
                if bool(live.all()):
                    break
                S = [s for s, keep in zip(S, live.tolist()) if keep]
                w = w[live]
                r = (m - Phi[:, S] @ w) if S else m.clone()
                S, w, r = select(S, w, r, budget)
            w = _nnls_box(Phi[:, S], m, lower=1e-12, upper=UPPER)

        return S, w
