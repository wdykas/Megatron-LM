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

from .compressors import (
    CompactionResult,
    _mass_features,
    _nnls_pgd,
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
    keys_per_iter:  k in the paper — keys added per OMP iteration (default 4).
    nnls_every:     τ in the paper — how often to refit weights (default 1).
                    Values > 1 cause stale residuals: keys added between refits
                    are scored against an outdated residual that doesn't account
                    for the most recently added keys, violating the OMP invariant.
    fit_values:     Whether to apply OLS value fitting (default True).
    """

    def __init__(
        self,
        keys_per_iter: int = 4,
        nnls_every: int = 1,
        fit_bias: bool = True,
        fit_values: bool = True,
        drop_key_beta_cutoff: float = -10.0,
    ) -> None:
        if keys_per_iter < 1:
            raise ValueError(f"keys_per_iter must be >= 1, got {keys_per_iter}")
        if nnls_every < 1:
            raise ValueError(f"nnls_every must be >= 1, got {nnls_every}")
        self.keys_per_iter = keys_per_iter
        self.nnls_every = nnls_every
        self.fit_bias = fit_bias
        self.fit_values = fit_values
        # Official refinement (drop_key_beta_cutoff): a key whose fitted
        # log-weight falls below this is effectively deleted from attention —
        # drop it from the retained set instead of carrying a dead slot.
        self.drop_key_beta_cutoff = drop_key_beta_cutoff

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
        w = w[:len(positions)].clamp(min=1e-12)

        sort_idx = torch.argsort(torch.tensor(positions, dtype=torch.long))
        positions = [positions[i] for i in sort_idx.tolist()]
        w = w[sort_idx]

        beta = None
        if self.fit_bias:
            beta = torch.log(w)
            live = beta >= self.drop_key_beta_cutoff
            if not bool(live.all()) and bool(live.any()):
                positions = [p for p, keep in zip(positions, live.tolist()) if keep]
                beta = beta[live]
        C_k = keys[positions]

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

    def _omp(
        self, keys: torch.Tensor, ref_queries: torch.Tensor, budget: int
    ) -> tuple[list[int], torch.Tensor]:
        Phi = _mass_features(ref_queries, keys)      # (n, T)
        m = Phi.sum(dim=1)                           # (n,) target mass
        # Mild ridge toward w=1 (bias 0); the old mean(m)*0.1 ridge toward 0
        # crushed weights on real captures (beta -> very negative = keys
        # deleted from attention at eval).
        lam = float(Phi.pow(2).sum(dim=0).mean().item()) * 1e-3 + 1e-12

        r = m.clone()
        S: list[int] = []
        w = torch.zeros(0, device=keys.device, dtype=keys.dtype)
        excluded = torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device)
        iter_idx = 0

        while len(S) < budget:
            scores = r @ Phi                         # (T,)
            scores[excluded] = -torch.inf

            k = min(self.keys_per_iter, budget - len(S))
            new_idxs = scores.topk(k).indices.tolist()
            S.extend(new_idxs)
            excluded[new_idxs] = True
            iter_idx += 1

            if iter_idx % self.nnls_every == 0 or len(S) >= budget:
                Phi_S = Phi[:, S]
                t = len(S)
                if len(S) >= budget:
                    reg = math.sqrt(lam) * torch.eye(t, device=Phi.device, dtype=Phi.dtype)
                    A_aug = torch.cat([Phi_S, reg], dim=0)
                    b_aug = torch.cat([m, math.sqrt(lam) * torch.ones(t, device=m.device, dtype=m.dtype)])
                    w = _nnls_pgd(A_aug, b_aug)
                else:
                    w = _nnls_pgd(Phi_S, m)
                r = m - Phi_S @ w

        return S, w


