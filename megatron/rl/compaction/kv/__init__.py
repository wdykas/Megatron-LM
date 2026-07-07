# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from .types import KVMask
from .selectors.baselines import (
    AttentionSumScorer,
    UniformScorer,
)
from .serving.megatron_hook import MegatronInferenceHook, NullHook
from .compressors import CompactionResult, KVCompressor
from .selectors.attention_matching import TopKCompressor, OMPCompressor  # Zweiger et al. 2026
from .selectors.h2o import H2OAccumulator                       # Zhang et al. 2023
from .selectors.snapkv import SnapKVCompressor                  # Li et al. 2024
from .selectors.streaming_llm import StreamingLLMCompressor     # Xiao et al. 2023
from .benchmark import KVCompactionBenchmark, CompactionBenchmarkResult
from .selectors.eviction_policy import (                                  # eviction-policy RL
    EvictionGRPOConfig,
    EvictionPolicy,
    make_sufficiency_reward,
    train_eviction_policy_grpo,
)
from .selectors.oracle import (                                           # learned oracle
    LearnedOracleScorer,
    OracleCompressor,
    OracleScorerConfig,
    fit_oracle_scorer,
    fit_scorer_on_flywheel,
    load_oracle_scorer,
    save_oracle_scorer,
    token_level_oracle,
)


# Live H2O is intentionally NOT wired: H2O's heavy-hitter score is the attention
# accumulated over all queries, but the inference server runs flash attention,
# which never materialises attention weights — so the paper-exact score is
# unavailable online. The deployable flash-compatible heavy-hitter method is
# SnapKV (scores from a small observation window). H2OAccumulator stays usable
# OFFLINE (the benchmark passes explicit queries).
_LIVE_H2O_UNFINISHED = (
    "Live H2O is not finished: H2O scores heavy hitters by attention accumulated "
    "over all queries, but the server uses flash attention which does not expose "
    "attention weights online. Use 'snapkv' (observation-window scoring, "
    "flash-compatible) for live eval; H2OAccumulator remains available offline."
)


def build_kv_compressor(
    strategy: str, recent_ratio: float = 0.5, inference: bool = False
) -> KVCompressor:
    """Map a strategy name to its paper-exact KV compressor.

    Single seam used by both the offline benchmark and the live inference server
    so deployment and eval share one definition of each method.

    h2o            -- Zhang et al. 2023: recent window + heavy hitters (OFFLINE only).
    snapkv         -- Li et al. 2024: observation-window heavy hitters (flash-compatible).
    omp            -- greedy attention-mass matching + OLS value fit.
    topk           -- top-k by RMS attention weight (+ bias/value fit).
    streaming_llm  -- attention sinks + recent window.

    ``inference=True`` selects the live deployment path and hard-fails on
    strategies that cannot run online (currently h2o — see _LIVE_H2O_UNFINISHED).
    """
    s = strategy.lower()
    if s == "h2o":
        if inference:
            raise NotImplementedError(_LIVE_H2O_UNFINISHED)
        return H2OAccumulator(recent_ratio=recent_ratio)
    if s == "snapkv":
        return SnapKVCompressor()
    if s == "omp":
        return OMPCompressor()
    if s == "topk":
        return TopKCompressor()
    if s in ("streaming_llm", "streaming"):
        return StreamingLLMCompressor()
    if s == "learned_oracle":
        raise ValueError(
            "learned_oracle needs a trained scorer: construct "
            "OracleCompressor(load_oracle_scorer(path)) directly for offline "
            "use, or serve with --kv-compaction-strategy learned_oracle "
            "--kv-compaction-oracle-checkpoint <path>."
        )
    raise ValueError(
        f"unknown KV compaction strategy {strategy!r}; "
        "expected one of: h2o, snapkv, omp, topk, streaming_llm, learned_oracle"
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
    "SnapKVCompressor",
    "StreamingLLMCompressor",
    "EvictionGRPOConfig",
    "EvictionPolicy",
    "make_sufficiency_reward",
    "train_eviction_policy_grpo",
    "LearnedOracleScorer",
    "OracleCompressor",
    "OracleScorerConfig",
    "fit_oracle_scorer",
    "fit_scorer_on_flywheel",
    "load_oracle_scorer",
    "save_oracle_scorer",
    "token_level_oracle",
    "KVCompactionBenchmark",
    "CompactionBenchmarkResult",
    "build_kv_compressor",
]
