# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KV selection strategies: which token positions survive compaction.

Paper baselines (snapkv, streaming_llm, h2o, attention_matching/OMP-TopK),
simple scorers (baselines), and the learned selection policies (oracle,
eviction_policy).
"""

from .snapkv import SnapKVCompressor
from .streaming_llm import StreamingLLMCompressor
from .h2o import H2OAccumulator
from .attention_matching import TopKCompressor, OMPCompressor
from .baselines import AttentionSumScorer, UniformScorer
from .oracle import (
    LearnedOracleScorer, OracleScorerConfig, OracleCompressor,
    fit_oracle_scorer, save_oracle_scorer, load_oracle_scorer,
    token_level_oracle, fit_scorer_on_flywheel,
)
from .eviction_policy import (
    EvictionPolicy, EvictionGRPOConfig, make_sufficiency_reward,
    train_eviction_policy_grpo,
)
