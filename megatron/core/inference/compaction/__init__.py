# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KV cache compaction via Attention Matching.

Implements the method from "Fast KV Compaction via Attention Matching"
(Zweiger et al., 2026) adapted for Megatron-LM's paged KV cache.

Phases:
  1. Offline compaction: gather paged KV -> AM compact -> write back to pages
  2. Online compaction: hot window + compact memory with periodic compaction
  3. Fast streaming compactor + post-training hooks
"""

from .am_compaction import AMCompactionResult, am_compact, am_compact_with_mass
from .compaction_manager import CompactionConfig, CompactionManager
from .kv_utils import gather_kv, gather_kv_with_biases, write_kv, write_kv_with_biases
from .streaming_compactor import StreamingClusterCompactor, StreamingCompactorConfig
from .training_hooks import (
    CompactionConsistencyLoss,
    CompactionInTheLoopTrainer,
    CompactionRLCost,
    CompactionTrainingConfig,
    TeacherDistillationLoss,
)
from .validation import (
    run_full_validation,
    validate_attention_output,
    validate_logit_drift,
)

__all__ = [
    # Phase 1: KV utilities
    "gather_kv",
    "gather_kv_with_biases",
    "write_kv",
    "write_kv_with_biases",
    # Phase 1: AM compaction
    "am_compact",
    "am_compact_with_mass",
    "AMCompactionResult",
    # Phase 1: Validation
    "validate_attention_output",
    "validate_logit_drift",
    "run_full_validation",
    # Phase 2: Compaction manager
    "CompactionConfig",
    "CompactionManager",
    # Phase 3: Streaming compactor
    "StreamingClusterCompactor",
    "StreamingCompactorConfig",
    # Phase 3: Training hooks
    "CompactionConsistencyLoss",
    "CompactionTrainingConfig",
    "CompactionInTheLoopTrainer",
    "CompactionRLCost",
    "TeacherDistillationLoss",
]
