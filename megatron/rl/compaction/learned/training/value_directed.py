# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Value-directed compaction: advantage-weighted STILL training for RL integration.

Standard STILL trains with uniform teacher_kl loss — every context position is
treated equally.  This module provides advantage-weighted training via:

    L_value(t) = w(advantage_t) · L_still(t)

where w(a) maps the GRPO advantage to a loss weight.

This module provides:
    - ``_probe_weight``: compute the loss weight for a single probe
    - ``ValueDirectedConfig``: configuration for advantage weighting
    - ``attach_grpo_advantages``: attach GRPO advantages to trajectory probes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe, CompactorTrainerConfig, CompactKV, StudentFn
from megatron.rl.compaction.learned.training.losses import CompactorLosses, CompactorLossWeights


@dataclass
class ValueDirectedConfig:
    """Configuration for value-directed compaction training.

    Attributes
    ----------
    advantage_clip: Clip extreme advantages to this magnitude.
    min_weight:     Minimum loss weight for very-negative-advantage probes.
    """
    advantage_clip: float = 5.0
    min_weight:     float = 0.1


def _probe_weight(
    probe: TrainingProbe,
    cfg: ValueDirectedConfig,
    rollout_return: float | None = None,
) -> float:
    """Compute the loss weight for a single probe.

    Uses probe.advantage if set; falls back to rollout_return; falls back to 1.0.
    """
    adv = probe.advantage if probe.advantage is not None else rollout_return
    if adv is None:
        return 1.0
    adv = float(adv)
    if adv != adv:  # NaN guard — corrupted reward; fall back to neutral weight
        return 1.0
    adv = max(-cfg.advantage_clip, min(cfg.advantage_clip, adv))
    # Piecewise linear: [-clip, 0] → [min_weight, 1.0], [0, +clip] → [1.0, 2.0]
    if adv >= 0.0:
        w_adv = 1.0 + adv / cfg.advantage_clip
    else:
        w_adv = cfg.min_weight + (1.0 - cfg.min_weight) * (adv + cfg.advantage_clip) / cfg.advantage_clip
    return w_adv


def attach_grpo_advantages(
    trajectories:  list[Trajectory],
    advantages:    list[float],
    rollout_returns: list[float] | None = None,
) -> list[Trajectory]:
    """Attach GRPO advantages to trajectory probes for value-directed training.

    Call this after calculate_grpo_advantages() in the GRPO pipeline, before
    passing trajectories to train_compactor_trajectory().

    Parameters
    ----------
    trajectories:    Trajectory objects (one per GRPO rollout).
    advantages:      Per-rollout advantage scalars (same length as trajectories).
                     From calculate_grpo_advantages(): one value per turn/rollout.
    rollout_returns: Optional per-rollout total returns for rollout-level weighting.
                     When set on Trajectory.rollout_return, probes without explicit
                     .advantage fall back to this value.

    Returns: same trajectories with .rollout_return and probe.advantage set in place.
    """
    if len(advantages) != len(trajectories):
        raise ValueError(
            f"advantages length ({len(advantages)}) must match trajectories ({len(trajectories)})"
        )
    for i, (traj, adv) in enumerate(zip(trajectories, advantages)):
        if rollout_returns is not None:
            traj.rollout_return = rollout_returns[i]
        # Set advantage on every probe in this trajectory
        for probes in traj.probes_by_chunk.values():
            for probe in probes:
                probe.advantage = adv
    return trajectories
