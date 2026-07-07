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


def _softmax_attention(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Normalised softmax attention softmax(q·Kᵀ/√d) of each query row.  Shape (n, T)."""
    d = keys.shape[1]
    return torch.softmax(queries @ keys.T / math.sqrt(d), dim=-1)


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


def _mass_features(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Unnormalised attention mass Φ_ij = exp(q_i · K_j^T / sqrt(d)).  Shape (n, T)."""
    d = keys.shape[1]
    logits = queries @ keys.T / math.sqrt(d)
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


def _nnls_pgd(A: torch.Tensor, b: torch.Tensor, max_iter: int = 500) -> torch.Tensor:
    """Non-negative least squares: FISTA-accelerated projected gradient.

    Warm-started from the clamped least-squares solution; step size from the
    spectral norm of A^T A (power iteration) rather than the Frobenius bound —
    on ill-conditioned real attention features the Frobenius step is orders of
    magnitude too small and plain PGD from zero never leaves the origin.
    """
    AtA = A.T @ A
    Atb = A.T @ b
    # spectral norm via power iteration (tight step bound)
    z = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
    for _ in range(20):
        z = AtA @ z
        z = z / (z.norm() + 1e-30)
    L = float(z @ (AtA @ z)) + 1e-8
    step = 1.0 / L
    try:
        w = torch.linalg.lstsq(A, b).solution.clamp(min=0)
    except Exception:
        w = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype)
    y, t_acc = w.clone(), 1.0
    for _ in range(max_iter):
        w_next = (y - step * (AtA @ y - Atb)).clamp(min=0)
        t_next = (1 + (1 + 4 * t_acc * t_acc) ** 0.5) / 2
        y = w_next + ((t_acc - 1) / t_next) * (w_next - w)
        w, t_acc = w_next, t_next
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
    n, t = Phi_c.shape
    # Mild ridge REGULARIZED TOWARD w=1 (bias 0): over-regularization then
    # degrades to "no bias" instead of w→0 = bias→-inf, which silently
    # DELETES retained keys from attention at eval (the old mean(m)*0.1
    # ridge toward zero did exactly that on real captures).
    lam = float(Phi_c.pow(2).sum(dim=0).mean().item()) * 1e-3 + 1e-12
    reg = math.sqrt(lam) * torch.eye(t, device=Phi_c.device, dtype=Phi_c.dtype)
    A_aug = torch.cat([Phi_c, reg], dim=0)
    b_aug = torch.cat([m, math.sqrt(lam) * torch.ones(t, device=m.device, dtype=m.dtype)])

    w = _nnls_pgd(A_aug, b_aug)
    return torch.log(w.clamp(min=1e-12))


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

    t = X.shape[1]
    if values_init is None:
        values_init = torch.zeros(t, values_orig.shape[1],
                                  device=values_orig.device, dtype=values_orig.dtype)
    XtX = X.T @ X                                       # (t, t)
    lam = XtX.diagonal().mean().clamp(min=1e-8) * 1e-2
    A = XtX + lam * torch.eye(t, device=X.device, dtype=X.dtype)
    D = torch.linalg.solve(A, X.T @ (Y - X @ values_init))
    return values_init + D                              # (t, d)
