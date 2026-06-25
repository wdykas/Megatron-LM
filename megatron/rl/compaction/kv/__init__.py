# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .types import KVMask
from .selectors import (
    AttentionSumScorer,
    UniformScorer,
)
from .megatron_hook import MegatronInferenceHook, NullHook
from .compressors import CompactionResult, KVCompressor
from .attention_matching import TopKCompressor, OMPCompressor   # Zweiger et al. 2026
from .h2o import H2OAccumulator                                 # Zhang et al. 2023
from .streaming_llm import StreamingLLMCompressor               # Xiao et al. 2023
from .benchmark import KVCompactionBenchmark, CompactionBenchmarkResult


def build_kv_compressor(strategy: str, recent_ratio: float = 0.5) -> KVCompressor:
    """Map a strategy name to its paper-exact KV compressor.

    Single seam used by both the offline benchmark and the live inference server
    so deployment and eval share one definition of each method.

    h2o            -- Zhang et al. 2023: recent window + heavy hitters, no fitting.
    omp            -- greedy attention-mass matching + OLS value fit.
    topk           -- top-k by RMS attention weight (+ bias/value fit).
    streaming_llm  -- attention sinks + recent window.
    """
    s = strategy.lower()
    if s == "h2o":
        return H2OAccumulator(recent_ratio=recent_ratio)
    if s == "omp":
        return OMPCompressor()
    if s == "topk":
        return TopKCompressor()
    if s in ("streaming_llm", "streaming"):
        return StreamingLLMCompressor()
    raise ValueError(
        f"unknown KV compaction strategy {strategy!r}; "
        "expected one of: h2o, omp, topk, streaming_llm"
    )


__all__ = [
    "KVMask",
    "AttentionSumScorer",
    "UniformScorer",
    "MegatronInferenceHook",
    "NullHook",
    "CompactionResult",
    "KVCompressor",
    "TopKCompressor",
    "OMPCompressor",
    "H2OAccumulator",
    "StreamingLLMCompressor",
    "KVCompactionBenchmark",
    "CompactionBenchmarkResult",
    "build_kv_compressor",
]
