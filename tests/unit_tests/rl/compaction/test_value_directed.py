# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for value-directed compaction (advantage-weighted STILL training)."""

import pytest
import torch
import torch.nn as nn

from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe, CompactorTrainerConfig
from megatron.rl.compaction.learned.training.losses import (
    advantage_weighted_kl_loss,
    teacher_kl_loss,
    CompactorLossWeights,
)
from megatron.rl.compaction.learned.models.belief import GatedRecurrentUpdater, GatedUpdaterConfig
from megatron.rl.compaction.learned.training.value_directed import (
    ValueDirectedConfig,
    _probe_weight,
    attach_grpo_advantages,
)


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

B, T, C, D, L, V = 2, 8, 4, 8, 2, 16  # batch, seq, compress, d_kv, layers, vocab


def _random_kv(T_len=T):
    keys   = [torch.randn(B, T_len, D) for _ in range(L)]
    values = [torch.randn(B, T_len, D) for _ in range(L)]
    return keys, values


def _updater_cfg(**kw):
    defaults = dict(d_kv=D, n_heads=2, n_compress=C, n_attn_layers=L)
    defaults.update(kw)
    return GatedUpdaterConfig(**defaults)


def _probe(advantage=None):
    q = torch.randint(0, V, (B, 4))
    tl = torch.randn(B, 4, V)
    return TrainingProbe(query_tokens=q, teacher_logits=tl, advantage=advantage)


def _dummy_student_fn(query_tokens, compact_kv):
    """Student fn that shares graph with compact_kv to allow gradient flow."""
    B_q, S_q = query_tokens.shape
    k0 = compact_kv[0][0]           # (B, C, D)
    flat = k0.reshape(B_q, -1)
    need = S_q * V
    reps = (need + flat.shape[1] - 1) // flat.shape[1]
    return flat.repeat(1, reps)[:, :need].reshape(B_q, S_q, V)


def _trajectory(n_chunks=4, probe_chunk=2, advantage=None):
    chunks = [_random_kv() for _ in range(n_chunks)]
    probes = {probe_chunk: [_probe(advantage=advantage)]}
    return Trajectory(chunks=chunks, probes_by_chunk=probes)


# ---------------------------------------------------------------------------
# Test 1: positive advantage increases weight
# ---------------------------------------------------------------------------

def test_advantage_weighted_kl_loss_positive_advantage_increases_weight():
    full_logits    = torch.randn(B, 4, V)
    compact_logits = torch.randn(B, 4, V)

    loss_zero = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=0.0)
    loss_pos  = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=3.0)

    # Positive advantage → weight > 1.0 → higher loss
    assert loss_pos.item() > loss_zero.item(), (
        f"Expected positive advantage to give higher loss: {loss_pos.item()} vs {loss_zero.item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: negative advantage decreases weight
# ---------------------------------------------------------------------------

def test_advantage_weighted_kl_loss_negative_advantage_decreases_weight():
    full_logits    = torch.randn(B, 4, V)
    compact_logits = torch.randn(B, 4, V)

    loss_zero = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=0.0)
    loss_neg  = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=-3.0)

    # Negative advantage → weight < 1.0 → lower loss
    assert loss_neg.item() < loss_zero.item(), (
        f"Expected negative advantage to give lower loss: {loss_neg.item()} vs {loss_zero.item()}"
    )


# ---------------------------------------------------------------------------
# Test 3: advantage clipping prevents extreme weights
# ---------------------------------------------------------------------------

def test_advantage_weighted_kl_loss_clip():
    full_logits    = torch.randn(B, 4, V)
    compact_logits = torch.randn(B, 4, V)
    clip = 5.0

    loss_at_clip     = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=clip,      advantage_clip=clip)
    loss_beyond_clip = advantage_weighted_kl_loss(full_logits, compact_logits, advantage=clip * 2,  advantage_clip=clip)

    # Clipping should make both equal (both clamped to +clip)
    assert torch.isclose(loss_at_clip, loss_beyond_clip, rtol=1e-5), (
        f"Clipping failed: {loss_at_clip.item()} vs {loss_beyond_clip.item()}"
    )


# ---------------------------------------------------------------------------
# Test 4: TrainingProbe has advantage field defaulting to None
# ---------------------------------------------------------------------------

def test_training_probe_has_advantage_field():
    probe = TrainingProbe(
        query_tokens=torch.randint(0, V, (B, 4)),
        teacher_logits=torch.randn(B, 4, V),
    )
    assert hasattr(probe, "advantage")
    assert probe.advantage is None


# ---------------------------------------------------------------------------
# Test 5: Trajectory has rollout_return field defaulting to None
# ---------------------------------------------------------------------------

def test_trajectory_has_rollout_return_field():
    traj = Trajectory(chunks=[_random_kv()])
    assert hasattr(traj, "rollout_return")
    assert traj.rollout_return is None

    traj2 = Trajectory(chunks=[_random_kv()], rollout_return=1.5)
    assert traj2.rollout_return == 1.5


# ---------------------------------------------------------------------------
# Test 6: attach_grpo_advantages populates all probes
# ---------------------------------------------------------------------------

def test_attach_grpo_advantages_sets_probe_advantage():
    traj1 = _trajectory(advantage=None)
    traj2 = _trajectory(advantage=None)

    result = attach_grpo_advantages([traj1, traj2], advantages=[1.5, -0.5])

    assert result is not None
    # traj1 probes should have advantage=1.5
    for probes in traj1.probes_by_chunk.values():
        for p in probes:
            assert p.advantage == 1.5, f"Expected 1.5, got {p.advantage}"
    # traj2 probes should have advantage=-0.5
    for probes in traj2.probes_by_chunk.values():
        for p in probes:
            assert p.advantage == -0.5, f"Expected -0.5, got {p.advantage}"


def test_attach_grpo_advantages_sets_rollout_return():
    traj = _trajectory()
    attach_grpo_advantages([traj], advantages=[0.5], rollout_returns=[2.0])
    assert traj.rollout_return == 2.0


def test_attach_grpo_advantages_length_mismatch_raises():
    traj = _trajectory()
    with pytest.raises(ValueError, match="advantages length"):
        attach_grpo_advantages([traj], advantages=[1.0, 2.0])


# ---------------------------------------------------------------------------
# Test 7: train_compactor_trajectory with vd_cfg runs without error and logs
#         advantage_weight when probes have advantages
# ---------------------------------------------------------------------------

def test_still_train_trajectory_with_vd_cfg_runs():
    from megatron.rl.compaction.learned.training.training import train_compactor_trajectory

    updater = GatedRecurrentUpdater(_updater_cfg())
    optimizer = torch.optim.Adam(updater.parameters(), lr=1e-3)
    traj = _trajectory(n_chunks=3, probe_chunk=2, advantage=1.0)
    vd_cfg = ValueDirectedConfig(advantage_clip=5.0, min_weight=0.1)
    cfg = CompactorTrainerConfig(
        loss_weights=CompactorLossWeights(consistency=0.0, predictive=0.0, future_kl=0.0),
        clip_grad_norm=1.0,
        vd_cfg=vd_cfg,
    )
    log = train_compactor_trajectory(
        model=updater,
        optimizer=optimizer,
        trajectory=traj,
        student_fn=_dummy_student_fn,
        cfg=cfg,
    )
    assert isinstance(log, dict)
    assert "teacher_kl" in log
    assert "advantage_weight" in log


def test_still_train_trajectory_with_vd_cfg_returns_advantage_weight_in_log():
    from megatron.rl.compaction.learned.training.training import train_compactor_trajectory

    updater = GatedRecurrentUpdater(_updater_cfg())
    optimizer = torch.optim.Adam(updater.parameters(), lr=1e-3)
    traj = _trajectory(n_chunks=3, probe_chunk=2, advantage=2.0)
    vd_cfg = ValueDirectedConfig(advantage_clip=5.0, min_weight=0.1)
    cfg = CompactorTrainerConfig(
        loss_weights=CompactorLossWeights(consistency=0.0, predictive=0.0, future_kl=0.0),
        vd_cfg=vd_cfg,
    )
    log = train_compactor_trajectory(updater, optimizer, traj, _dummy_student_fn, cfg)
    # advantage_weight should be > 1.0 for positive advantage
    assert log.get("advantage_weight", 1.0) > 1.0


# ---------------------------------------------------------------------------
# Test 8: no advantages → train_compactor_trajectory without vd_cfg works normally
# ---------------------------------------------------------------------------

def test_probe_weight_nan_advantage_returns_neutral():
    """NaN advantage must not propagate into the loss weight."""
    cfg = ValueDirectedConfig(advantage_clip=5.0, min_weight=0.1)
    probe = TrainingProbe(
        query_tokens=torch.zeros(1, dtype=torch.long),
        advantage=float("nan"),
    )
    w = _probe_weight(probe, cfg)
    assert w == 1.0, f"Expected 1.0 for NaN advantage, got {w}"
    assert w == w, "Weight must not be NaN"


def test_runtime_state_still_step_offset_default_zero():
    """compactor_step_offset must default to 0 so fresh runs start at step 1."""
    from megatron.rl.rl_utils import RLRuntimeState
    state = RLRuntimeState()
    assert state.compactor_step_offset == 0


def test_still_train_trajectory_no_advantages_no_vd_cfg():
    """When no advantages are set and no vd_cfg, train_compactor_trajectory behaves as standard."""
    from megatron.rl.compaction.learned.training.training import train_compactor_trajectory

    torch.manual_seed(42)
    updater = GatedRecurrentUpdater(_updater_cfg())
    opt = torch.optim.SGD(updater.parameters(), lr=0.0)

    torch.manual_seed(99)
    traj = _trajectory(advantage=None)

    cfg = CompactorTrainerConfig(
        loss_weights=CompactorLossWeights(consistency=0.0, predictive=0.0, future_kl=0.0),
        clip_grad_norm=None,
    )

    log = train_compactor_trajectory(updater, opt, traj, _dummy_student_fn, cfg)

    assert isinstance(log, dict)
    assert "teacher_kl" in log
    assert "advantage_weight" not in log
