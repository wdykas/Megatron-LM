# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""StreamingLLM KV compressor (attention sinks + recent window).

Source: Xiao et al., 2023 (arXiv:2309.17453).

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
    _attention_output,
    _nnls_pgd,
    _fit_bias,
    _fit_values,
)


# ---------------------------------------------------------------------------
# StreamingLLMCompressor
# ---------------------------------------------------------------------------

class StreamingLLMCompressor:
    """Attention sinks + recent window; no query-based scoring.

    Keeps a fixed prefix of ``n_sink`` tokens (attention sinks) and the most
    recent ``budget - n_sink`` tokens. This is a positional strategy — it does
    not use attention matching or reference queries for selection.

    ref_queries is accepted to match the KVCompressor protocol;
    it is only used if fit_bias or fit_values is True.

    Reference: "Efficient Streaming Language Models with Attention Sinks",
               Xiao et al. (2023), arXiv:2309.17453.

    Parameters
    ----------
    n_sink:     Number of initial tokens to always preserve (default 4).
    fit_bias:   NNLS bias fitting after selection (default False).
    fit_values: OLS value fitting after selection (default False).
    """

    def __init__(self, n_sink: int = 4, fit_bias: bool = False, fit_values: bool = False) -> None:
        if n_sink < 0:
            raise ValueError(f"n_sink must be >= 0, got {n_sink}")
        self.n_sink = n_sink
        self.fit_bias = fit_bias
        self.fit_values = fit_values

    @property
    def strategy(self) -> str:
        suffix = "+bias+values" if (self.fit_bias and self.fit_values) else \
                 "+bias" if self.fit_bias else \
                 "+values" if self.fit_values else ""
        return f"streaming_sink{self.n_sink}{suffix}"

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
        budget = max(1, min(budget, T))
        n_sink = min(self.n_sink, budget)

        sink_positions = list(range(n_sink))
        n_recent = budget - n_sink
        recent_start = max(n_sink, T - n_recent)
        recent_positions = list(range(recent_start, T))

        positions = sorted(set(sink_positions + recent_positions))
        C_k = keys[positions]

        if self.fit_bias:
            if ref_queries is None:
                raise ValueError("StreamingLLMCompressor requires ref_queries when fit_bias=True.")
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
