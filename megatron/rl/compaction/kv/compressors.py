# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared KV-compression types and attention-math primitives.

Holds the pieces every compressor builds on: the ``CompactionResult`` output
type, the ``KVCompressor`` protocol, and the attention/fitting helpers
(softmax attention, mass features, NNLS bias fit, OLS value fit, recent+heavy
selection). The algorithms themselves live one file per paper:
``attention_matching.py`` (TopK/OMP), ``h2o.py``, ``snapkv.py``,
``streaming_llm.py`` — see ``kv/__init__.py`` for the full index and the
``build_kv_compressor`` factory.

All math runs on GPU via PyTorch. Input tensors must share a device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from .types import KVMask


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """Output of an offline compaction run.

    Keys are always a subset of the original (``compacted_keys = keys[retained_positions]``).
    Values are either the same subset or a fitted matrix depending on whether
    value fitting was applied. Bias is either zero or fitted to match attention mass.
    """

    run_id: str
    step_id: int

    retained_positions: list[int]
    compacted_keys: torch.Tensor    # (t, d)
    compacted_values: torch.Tensor  # (t, d)
    bias: torch.Tensor              # (t,) — zeros if not fitted

    strategy: str
    original_length: int
    wall_time_s: float = 0.0

    def retention_ratio(self) -> float:
        if self.original_length == 0:
            return 1.0
        return len(self.retained_positions) / self.original_length

    def to_kv_mask(self) -> KVMask:
        return KVMask(
            run_id=self.run_id,
            step_id=self.step_id,
            retained_positions=self.retained_positions,
            total_positions=self.original_length,
            strategy=self.strategy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_id": self.step_id,
            "retained_positions": self.retained_positions,
            "compacted_keys": self.compacted_keys.tolist(),
            "compacted_values": self.compacted_values.tolist(),
            "bias": self.bias.tolist(),
            "strategy": self.strategy,
            "original_length": self.original_length,
            "retention_ratio": self.retention_ratio(),
        }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class KVCompressor(Protocol):
    """Unified compressor protocol for KV cache compaction.

    All selectors and compressors implement this interface.
    ref_queries is optional — positional selectors ignore it,
    attention-based ones use it for scoring.
    """

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
    ) -> CompactionResult: ...



# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------

def _validate_budget(budget: int, T: int) -> int:
    """Validate a retention budget against sequence length T.

    budget < 1 is a caller bug and hard-fails; budget > T clamps to T
    (retaining everything is well-defined).
    """
    if budget < 1:
        raise ValueError(f"KV retention budget must be >= 1, got {budget}")
    return min(budget, T)


def _softmax_attention(queries: torch.Tensor, keys: torch.Tensor,
                       causal_tail: bool = False,
                       query_end: int | None = None) -> torch.Tensor:
    """Normalised softmax attention softmax(q·Kᵀ/√d) of each query row. (n, T).

    ``causal_tail=True`` treats the n query rows as the LAST n positions of the
    sequence (how window queries are captured) and masks each row's future
    keys before the softmax — the official SnapKV/H2O kernels are causal;
    without the mask, earlier window queries leak probability mass to keys
    they could never attend to, systematically deflating prefix scores.
    Softmax in fp32 (official SnapKV): bf16 loses tail-key mass over long T.
    """
    d = keys.shape[1]
    logits = (queries @ keys.T / math.sqrt(d)).float()
    if causal_tail:
        n, T = logits.shape
        end = T if query_end is None else query_end
        col = torch.arange(T, device=logits.device)
        row = torch.arange(n, device=logits.device)
        future = col[None, :] > (end - n + row)[:, None]
        logits = logits.masked_fill(future, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _select_recent_plus_heavy(
    scores: torch.Tensor, T: int, budget: int, n_recent: int
) -> list[int]:
    """Retain the last ``n_recent`` positions plus the top-``budget - n_recent``
    scorers among the rest. Shared by the H2O and SnapKV selection policies."""
    n_recent = min(n_recent, budget)
    recent_positions = list(range(T - n_recent, T))

    n_heavy = budget - n_recent
    if n_heavy > 0 and T > n_recent:
        heavy_scores = scores[:T].clone()
        heavy_scores[T - n_recent:] = -torch.inf   # don't double-count the recent window
        n_select = min(n_heavy, T - n_recent)
        heavy_positions = heavy_scores.topk(n_select).indices.tolist()
    else:
        heavy_positions = []

    return sorted(set(recent_positions + heavy_positions))


def _mass_features(queries: torch.Tensor, keys: torch.Tensor,
                   query_end: int | None = None) -> torch.Tensor:
    """Unnormalised attention mass Φ_ij = exp(q_i · K_j^T / sqrt(d)). (n, T).
    fp32 throughout (official policy: QK matmul in model dtype, exp/solve fp32).
    ``query_end`` marks the queries as the last n positions BEFORE that key
    index and causally masks later keys — for benchmarking with an early
    query window (the official pipeline's generated queries see everything,
    so the default is unmasked)."""
    d = keys.shape[1]
    logits = (queries @ keys.T).float() / math.sqrt(d)
    if query_end is not None:
        n, T = logits.shape
        col = torch.arange(T, device=logits.device)
        row = torch.arange(n, device=logits.device)
        logits = logits.masked_fill(
            col[None, :] > (query_end - n + row)[:, None], float("-inf"))
    logits = logits - logits.max(dim=1, keepdim=True).values
    return torch.exp(logits)


def _attention_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax attention output and unnormalised mass.  Returns (output, mass)."""
    d = keys.shape[1]
    logits = queries @ keys.T / math.sqrt(d)
    if bias is not None:
        logits = logits + bias
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits)
    mass = exp_logits.sum(dim=1)
    weights = exp_logits / mass.unsqueeze(1)
    return weights @ values, mass


def _nnls_box(A: torch.Tensor, b: torch.Tensor, lower: float = 1e-12,
              upper: float | None = None, iters: int = 0) -> torch.Tensor:
    """Official AM-OMP weight solve (Zweiger et al. base.py _nnls_pg).

    iters=0 (the paper default) is NOT iterative NNLS: it is a plain
    unregularized least-squares solve followed by clamping into
    [lower, upper]. iters>0 runs that many projected-gradient steps from the
    clamped-lstsq init with a spectral step size. All math in fp32.
    """
    A32, b32 = A.float(), b.float()
    try:
        w = torch.linalg.lstsq(A32, b32.unsqueeze(1), driver="gels").solution.squeeze(1)
    except Exception:
        # official fallback: tiny-ridge Cholesky
        AtA = A32.T @ A32
        lam = 1e-6 * AtA.diagonal().mean().clamp(min=1e-12)
        w = torch.linalg.solve(AtA + lam * torch.eye(A32.shape[1], device=A32.device), A32.T @ b32)
    w = w.clamp(min=lower)
    if upper is not None:
        w = w.clamp(max=upper)
    if iters > 0:
        AtA = A32.T @ A32
        Atb = A32.T @ b32
        z = torch.randn(A32.shape[1], device=A32.device)
        for _ in range(3):
            z = AtA @ z
            z = z / (z.norm() + 1e-30)
        step = 1.0 / (float(z @ (AtA @ z)) + 1e-8)
        for _ in range(iters):
            w = (w - step * (AtA @ w - Atb)).clamp(min=lower)
            if upper is not None:
                w = w.clamp(max=upper)
    return w


def _fit_bias(
    keys_orig: torch.Tensor,
    keys_compact: torch.Tensor,
    ref_queries: torch.Tensor,
) -> torch.Tensor:
    """Fit bias β to match original attention mass via L2-regularised NNLS (Section 3.2).

    Solves:  min_{w ≥ 0}  ||Φ_compact w - m_orig||²  + λ||w||²
    Returns β = log(w) (w clamped for numerical safety).
    """
    d = keys_orig.shape[1]
    logits_orig = ref_queries @ keys_orig.T / math.sqrt(d)    # (n, T)
    logits_c    = ref_queries @ keys_compact.T / math.sqrt(d) # (n, t)

    # Use max over both sets so Phi_c is not artificially crushed when
    # compact logits are lower than the full-key max (which causes Phi_c ≈ 0
    # and makes the NNLS unsolvable, driving bias → -inf).
    row_max = torch.cat([logits_orig, logits_c], dim=1).max(dim=1, keepdim=True).values
    Phi_orig = torch.exp(logits_orig - row_max)               # (n, T)
    Phi_c    = torch.exp(logits_c    - row_max)               # (n, t)

    m = Phi_orig.sum(dim=1)                                   # (n,)
    # Official AM-HighestAttnKeys paper config: 2 projected-gradient steps
    # from the clamped-lstsq init, weights boxed into [e^-3, e^3] (beta in
    # [-3, 3]) — regularization is the box clamp, not a ridge.
    w = _nnls_box(Phi_c, m, lower=math.exp(-3.0), upper=math.exp(3.0), iters=2)
    return torch.log(w).to(keys_orig.dtype)


def _fit_values(
    keys_orig: torch.Tensor,
    values_orig: torch.Tensor,
    keys_compact: torch.Tensor,
    bias: torch.Tensor | None,
    ref_queries: torch.Tensor,
    values_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit compacted values via closed-form ridge OLS (Section 3.2, Eq. 3–4).

    Solved in RESIDUAL form around ``values_init`` (the original values at the
    retained positions): C_v = V_init + D with the ridge shrinking D toward 0.
    With n ref queries < t retained keys the plain system is underdetermined —
    solving for C_v directly zeroes the null space and destroys the values of
    weakly-attended keys; the residual form degrades to the ORIGINAL values
    instead. Returns C_v (t, d).
    """
    Y, _ = _attention_output(ref_queries, keys_orig, values_orig)   # (n, d)
    d_orig = keys_orig.shape[1]

    logits_c = ref_queries @ keys_compact.T / math.sqrt(d_orig)
    if bias is not None:
        logits_c = logits_c + bias
    logits_c = logits_c - logits_c.max(dim=1, keepdim=True).values
    exp_c = torch.exp(logits_c)
    X = exp_c / exp_c.sum(dim=1, keepdim=True)                      # (n, t)

    if values_init is None:
        # Official AM-OMP 'lsq': plain unregularized least squares in fp32
        # (gels => minimum-norm when underdetermined). NOTE the official
        # pipeline fits against ~10k GENERATED queries per KV head; with far
        # fewer queries this is underdetermined and the min-norm solution
        # zeroes weakly-attended values — pass values_init for the residual
        # form, which degrades to the original values instead.
        C_v = torch.linalg.lstsq(X.float(), Y.float(), driver="gels").solution
        return C_v.to(values_orig.dtype)
    t = X.shape[1]
    X32, Y32 = X.float(), Y.float()
    XtX = X32.T @ X32                                   # (t, t)
    lam = XtX.diagonal().mean().clamp(min=1e-8) * 1e-2
    A = XtX + lam * torch.eye(t, device=X.device, dtype=torch.float32)
    D = torch.linalg.solve(A, X32.T @ (Y32 - X32 @ values_init.float()))
    return (values_init.float() + D).to(values_orig.dtype)  # (t, d)
