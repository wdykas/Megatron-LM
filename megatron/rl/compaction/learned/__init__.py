# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor
from megatron.rl.compaction.learned.models.belief import BeliefMemory, BeliefUpdater, GatedUpdaterConfig, GatedRecurrentUpdater
from megatron.rl.compaction.learned.training.losses import (
    CompactorLossWeights,
    CompactorLossTerms,
    CompactorLosses,
    teacher_kl_loss,
    future_kl_loss,
    consistency_loss,
    path_consistency_loss,
    task_loss,
    retrieval_loss,
    weighted_kl_loss,
    predictive_coding_loss,
    kv_reconstruction_loss,
    future_kv_reconstruction_loss,
    dynamics_prediction_loss,
    future_horizon_kl_loss,
)
from megatron.rl.compaction.learned.training.data import (
    CompactKV,
    StudentFn,
    TrainingProbe,
    Trajectory,
    TrajectoryDataset,
    trajectory_collate_fn,
    CompactorTrainerConfig,
    PipelineConfig,
)
from megatron.rl.compaction.learned.capture.hook_collector import HookTrajectoryCollector
from megatron.rl.compaction.learned.serving.belief_compactor import BeliefSession, BeliefSessionStore, BeliefServerCompactor
from megatron.rl.compaction.learned.training.checkpoint import (
    CheckpointMeta,
    save_checkpoint,
    load_checkpoint,
    load_optimizer_state,
)
from megatron.rl.compaction.learned.training.losses import advantage_weighted_kl_loss
from megatron.rl.compaction.learned.training.value_directed import (
    ValueDirectedConfig,
    attach_grpo_advantages,
)

__all__ = [
    "PerceiverConfig", "PerceiverCompactor",
    "BeliefMemory", "BeliefUpdater",
    "GatedUpdaterConfig", "GatedRecurrentUpdater",
    "CompactorLossWeights", "CompactorLossTerms", "CompactorLosses",
    "teacher_kl_loss", "future_kl_loss", "consistency_loss",
    "path_consistency_loss", "task_loss", "retrieval_loss", "weighted_kl_loss",
    "predictive_coding_loss", "kv_reconstruction_loss",
    "future_kv_reconstruction_loss", "dynamics_prediction_loss", "future_horizon_kl_loss",
    "CompactKV", "StudentFn",
    "TrainingProbe", "Trajectory", "TrajectoryDataset", "trajectory_collate_fn",
    "CompactorTrainerConfig",
    "PipelineConfig",
    "HookTrajectoryCollector",
    "BeliefSession", "BeliefSessionStore", "BeliefServerCompactor",
    "CheckpointMeta", "save_checkpoint", "load_checkpoint",
    "load_optimizer_state",
    "advantage_weighted_kl_loss",
    "ValueDirectedConfig", "attach_grpo_advantages",
]
