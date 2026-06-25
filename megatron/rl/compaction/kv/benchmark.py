# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Algorithm comparison benchmark for KV cache compaction.

Usage
-----
    import torch
    from megatron.rl.compaction.kv.benchmark import KVCompactionBenchmark
    from megatron.rl.compaction.kv import OMPCompressor, TopKCompressor
    from megatron.rl.compaction.kv.selectors import AttentionSumScorer, UniformScorer

    device = torch.device("cuda")
    K = torch.randn(512, 64, device=device)
    V = torch.randn(512, 64, device=device)
    Q_ref  = torch.randn(16, 64, device=device)
    Q_eval = torch.randn(32, 64, device=device)

    bench = KVCompactionBenchmark()
    results = bench.run(
        compressors={
            "omp_full":    OMPCompressor(fit_values=True),
            "topk_full":   TopKCompressor(fit_bias=True, fit_values=True),
            "attn_sum":    AttentionSumScorer(),
            "uniform":     UniformScorer(),
        },
        keys=K, values=V, ref_queries=Q_ref, eval_queries=Q_eval, budget=128,
    )
    for r in results:
        print(f"{r.algorithm:20s}  MSE={r.output_mse:.4f}  t={r.wall_time_s*1000:.1f}ms")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .compressors import (
    CompactionResult,
    KVCompressor,
    _attention_output,
)


@dataclass
class CompactionBenchmarkResult:
    """Per-algorithm result from KVCompactionBenchmark.run()."""

    algorithm: str
    strategy: str
    retention_ratio: float
    output_mse: float
    mass_error: float
    wall_time_s: float
    original_length: int
    compacted_length: int

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "strategy": self.strategy,
            "retention_ratio": self.retention_ratio,
            "output_mse": self.output_mse,
            "mass_error": self.mass_error,
            "wall_time_s": self.wall_time_s,
            "original_length": self.original_length,
            "compacted_length": self.compacted_length,
        }


class KVCompactionBenchmark:
    """Compare multiple KVCompressor implementations on the same data.

    All compressors receive identical (keys, values, ref_queries, budget).
    Quality is evaluated on a held-out eval_queries set not used during compression.
    """

    def run(
        self,
        compressors: dict[str, KVCompressor],
        keys: torch.Tensor,
        values: torch.Tensor,
        ref_queries: torch.Tensor,
        eval_queries: torch.Tensor,
        budget: int,
        run_id: str = "bench",
        step_id: int = 0,
    ) -> list[CompactionBenchmarkResult]:
        """Run all compressors and return results sorted by output_mse (ascending)."""
        Y_full, mass_full = _attention_output(eval_queries, keys, values)

        results = []
        for name, compressor in compressors.items():
            result = compressor.compress(keys, values, budget, ref_queries=ref_queries,
                                         run_id=run_id, step_id=step_id)
            mse = self._output_mse(Y_full, eval_queries, result)
            merr = self._mass_error(mass_full, eval_queries, keys, result)
            results.append(CompactionBenchmarkResult(
                algorithm=name,
                strategy=result.strategy,
                retention_ratio=result.retention_ratio(),
                output_mse=mse,
                mass_error=merr,
                wall_time_s=result.wall_time_s,
                original_length=result.original_length,
                compacted_length=len(result.retained_positions),
            ))

        results.sort(key=lambda r: r.output_mse)
        return results

    @staticmethod
    def _output_mse(
        Y_full: torch.Tensor,
        eval_queries: torch.Tensor,
        result: CompactionResult,
    ) -> float:
        Y_compact, _ = _attention_output(
            eval_queries, result.compacted_keys, result.compacted_values, result.bias,
        )
        return float(((Y_full - Y_compact) ** 2).mean().item())

    @staticmethod
    def _mass_error(
        mass_full: torch.Tensor,
        eval_queries: torch.Tensor,
        keys_orig: torch.Tensor,
        result: CompactionResult,
    ) -> float:
        d = keys_orig.shape[1]
        logits_full = eval_queries @ keys_orig.T / math.sqrt(d)
        row_max = logits_full.max(dim=1, keepdim=True).values
        logits_c = eval_queries @ result.compacted_keys.T / math.sqrt(d) + result.bias
        mass_compact = torch.exp(logits_c - row_max).sum(dim=1)
        rel_err = (mass_full - mass_compact).abs() / (mass_full + 1e-12)
        return float(rel_err.mean().item())
