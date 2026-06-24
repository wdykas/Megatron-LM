# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KL-divergence sufficiency probe (§1.1 in research directions).

Measures D_KL(π(·|full) || π(·|compact)) to quantify how much information
the compaction loses from the policy's perspective.

Works at both levels:
  - Text compaction: compare full-context tokens vs compact-context tokens.
  - KV compaction: compare full KV cache vs masked KV cache (the policy_fn
    handles the forward pass; this module only does the math).

Usage
-----
    probe = KLSufficiencyProbe()

    # Policy callable: list[int] -> list[float] (log-probs over vocab)
    result = probe.measure(policy_fn, full_tokens, compact_tokens)
    print(result.kl_full_to_compact)   # main sufficiency metric
    print(result.js_divergence)        # symmetric alternative
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# Policy callable: takes a token sequence, returns log-probs over the vocab.
PolicyFn = Callable[[list[int]], list[float]]


@dataclass
class KLResult:
    """Divergence measurements between full-context and compact-context policies."""

    # D_KL(π(·|full) || π(·|compact)) — primary sufficiency metric.
    # Low means compact belief preserves the policy's action distribution.
    kl_full_to_compact: float

    # D_KL(π(·|compact) || π(·|full)) — useful for asymmetry analysis.
    kl_compact_to_full: float

    # Jensen-Shannon divergence: symmetric, bounded in [0, log(2)].
    js_divergence: float

    full_token_count: int
    compact_token_count: int

    def compression_ratio(self) -> float | None:
        if self.compact_token_count == 0:
            return None
        return self.full_token_count / self.compact_token_count


class KLSufficiencyProbe:
    """Measures KL divergence between full-context and compact-context policies.

    The probe is stateless — all state lives in the caller's policy_fn.
    Instantiate once and call ``measure`` for each (full, compact) pair.
    """

    @staticmethod
    def kl(log_p: list[float], log_q: list[float]) -> float:
        """D_KL(P || Q) where P and Q are given as log-probabilities.

        Numerically stable: skips entries where p ≈ 0 to avoid 0 * log(0).
        """
        if len(log_p) != len(log_q):
            raise ValueError(
                f"log_p and log_q must have the same length, "
                f"got {len(log_p)} and {len(log_q)}"
            )
        total = 0.0
        for lp, lq in zip(log_p, log_q):
            p = math.exp(lp)
            if p > 1e-10:
                total += p * (lp - lq)
        return max(0.0, total)  # clamp float noise to non-negative

    @staticmethod
    def js(log_p: list[float], log_q: list[float]) -> float:
        """Jensen-Shannon divergence: (KL(P||M) + KL(Q||M)) / 2 where M = (P+Q)/2."""
        if len(log_p) != len(log_q):
            raise ValueError(
                f"log_p and log_q must have the same length, "
                f"got {len(log_p)} and {len(log_q)}"
            )
        # log M[i] = log((exp(lp) + exp(lq)) / 2) = log(exp(lp) + exp(lq)) - log(2)
        log_m = [math.log((math.exp(lp) + math.exp(lq)) / 2.0) for lp, lq in zip(log_p, log_q)]
        kl_p_m = KLSufficiencyProbe.kl(log_p, log_m)
        kl_q_m = KLSufficiencyProbe.kl(log_q, log_m)
        return (kl_p_m + kl_q_m) / 2.0

    def measure(
        self,
        policy_fn: PolicyFn,
        full_tokens: list[int],
        compact_tokens: list[int],
    ) -> KLResult:
        """Run ``policy_fn`` on both token sequences and compute divergences.

        ``policy_fn`` should return log-probabilities over the next-token
        vocabulary for the last position of the given token sequence.
        """
        log_p_full = policy_fn(full_tokens)
        log_p_compact = policy_fn(compact_tokens)

        return KLResult(
            kl_full_to_compact=self.kl(log_p_full, log_p_compact),
            kl_compact_to_full=self.kl(log_p_compact, log_p_full),
            js_divergence=self.js(log_p_full, log_p_compact),
            full_token_count=len(full_tokens),
            compact_token_count=len(compact_tokens),
        )
