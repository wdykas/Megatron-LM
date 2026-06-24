# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShadowStepMetrics:
    step_id: int
    run_id: str
    full_context_tokens: int | None
    compact_context_tokens: int | None
    compression_ratio: float | None  # full / compact

    belief_json_valid: bool
    belief_token_estimate: int | None
    compactor_error: bool
    fallback_used: bool
    uncertainty_level: str | None

    # Set externally when KLSufficiencyProbe is run on this step.
    kl_divergence: float | None = None         # D_KL(π(·|full) || π(·|compact))

    # Set by recorder when kv_algorithm is active.
    kv_retention_ratio: float | None = None    # retained / total KV positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "run_id": self.run_id,
            "full_context_tokens": self.full_context_tokens,
            "compact_context_tokens": self.compact_context_tokens,
            "compression_ratio": self.compression_ratio,
            "belief_json_valid": self.belief_json_valid,
            "belief_token_estimate": self.belief_token_estimate,
            "compactor_error": self.compactor_error,
            "fallback_used": self.fallback_used,
            "uncertainty_level": self.uncertainty_level,
            "kl_divergence": self.kl_divergence,
            "kv_retention_ratio": self.kv_retention_ratio,
        }


@dataclass
class ShadowRunMetrics:
    run_id: str
    steps: list[ShadowStepMetrics] = field(default_factory=list)
    total_compactor_errors: int = 0
    total_fallbacks: int = 0

    def mean_compression_ratio(self) -> float | None:
        ratios = [s.compression_ratio for s in self.steps if s.compression_ratio is not None]
        return sum(ratios) / len(ratios) if ratios else None

    def mean_kl_divergence(self) -> float | None:
        kls = [s.kl_divergence for s in self.steps if s.kl_divergence is not None]
        return sum(kls) / len(kls) if kls else None

    def mean_kv_retention_ratio(self) -> float | None:
        ratios = [s.kv_retention_ratio for s in self.steps if s.kv_retention_ratio is not None]
        return sum(ratios) / len(ratios) if ratios else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps": [s.to_dict() for s in self.steps],
            "total_compactor_errors": self.total_compactor_errors,
            "total_fallbacks": self.total_fallbacks,
            "mean_compression_ratio": self.mean_compression_ratio(),
            "mean_kl_divergence": self.mean_kl_divergence(),
            "mean_kv_retention_ratio": self.mean_kv_retention_ratio(),
        }
