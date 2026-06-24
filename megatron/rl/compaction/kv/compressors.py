# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline KV cache compressors.

These algorithms take raw K/V tensors, optionally reference queries, and
return a CompactionResult with selected positions, compacted
key/value arrays, and optional bias.

All math runs on GPU via PyTorch. Input tensors should be on the same device.

Algorithms
----------
TopKCompressor         — top-k by RMS attention weight; fast heuristic baseline.
OMPCompressor          — greedy key selection via Orthogonal Matching Pursuit.
H2OProxyCompressor     — H2O with ref_queries as proxy for accumulated mass.
H2OAccumulator         — paper-faithful H2O; call update() after each decode step.
StreamingLLMCompressor — attention sinks + recent window; no query-based scoring.

References
----------
Attention Matching: Zweiger et al., 2026 (arXiv:2602.16284)
H2O:               Zhang et al., 2023 (arXiv:2306.14048)
StreamingLLM:      Xiao et al., 2023  (arXiv:2309.17453)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

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


# TODO(Peter): there is likely a better library for this rather than writing it out manually.
def _nnls_pgd(A: torch.Tensor, b: torch.Tensor, max_iter: int = 500) -> torch.Tensor:
    """Non-negative least squares via projected gradient descent.

    Solves min_{w >= 0} ||A @ w - b||^2 using a fixed step size derived from
    the Frobenius norm of A^T A (a safe upper bound on the spectral radius).
    """
    AtA = A.T @ A
    Atb = A.T @ b
    step = 1.0 / (torch.linalg.norm(AtA).item() + 1e-8)
    w = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype)
    for _ in range(max_iter):
        w = (w - step * (AtA @ w - Atb)).clamp(min=0)
    return w


def _fit_bias(
    keys_orig: torch.Tensor,
    keys_compact: torch.Tensor,
    ref_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit bias β to match original attention mass via L2-regularised NNLS (Section 3.2).

    Solves:  min_{w ≥ 0}  ||Φ_compact w - m_orig||²  + λ||w||²
    Returns β = log(w), w (clamped for numerical safety).
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
    lam = float(m.mean().item()) * 0.1

    n, t = Phi_c.shape
    reg = math.sqrt(lam) * torch.eye(t, device=Phi_c.device, dtype=Phi_c.dtype)
    A_aug = torch.cat([Phi_c, reg], dim=0)
    b_aug = torch.cat([m, torch.zeros(t, device=m.device, dtype=m.dtype)])

    w = _nnls_pgd(A_aug, b_aug)
    w = w.clamp(min=1e-30)
    return torch.log(w), w


def _fit_values(
    keys_orig: torch.Tensor,
    values_orig: torch.Tensor,
    keys_compact: torch.Tensor,
    bias: torch.Tensor,
    ref_queries: torch.Tensor,
) -> torch.Tensor:
    """Fit compacted values via closed-form OLS (Section 3.2, Eq. 3–4).

    min_{C_v}  ||X C_v - Y||_F²  where Y_i = Attn(q_i; K_orig, V_orig).
    Returns C_v (t, d).
    """
    Y, _ = _attention_output(ref_queries, keys_orig, values_orig)   # (n, d)
    d_orig = keys_orig.shape[1]

    logits_c = ref_queries @ keys_compact.T / math.sqrt(d_orig) + bias
    logits_c = logits_c - logits_c.max(dim=1, keepdim=True).values
    exp_c = torch.exp(logits_c)
    X = exp_c / exp_c.sum(dim=1, keepdim=True)                      # (n, t)

    # Ridge regression: (X^T X + λI) C_v = X^T Y.  Regularization prevents
    # extreme solutions when X is near-rank-deficient (e.g. one-hot attention
    # from peaked logits), without significantly perturbing well-constrained cols.
    t = X.shape[1]
    XtX = X.T @ X                                       # (t, t)
    lam = XtX.diagonal().mean().clamp(min=1e-8) * 1e-4
    A = XtX + lam * torch.eye(t, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)               # (t, d)
