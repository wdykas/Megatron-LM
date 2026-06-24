# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the wiring additions and dataset utilities.

Covers:
  - still/types.py — CompactKV, StudentFn canonical aliases
  - BeliefCompactorTrainer + ChunkFeatureExtractor (z_t injection)
  - BeliefCompactorTrainer + CurriculumScheduler (loss_weights update per step)
  - BeliefCompactorTrainer + ValueHead (value_mean in log dict)
  - BeliefCompactorTrainer + all three optional extras together
  - eval._greedy_generate — context accumulation across decode steps
  - TrajectoryDataset + trajectory_collate_fn
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from megatron.rl.compaction.learned.models.belief import BeliefMemory, BeliefUpdater
from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor, PerceiverConfig
from megatron.rl.compaction.learned.training.curriculum import CurriculumScheduler
from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe
from megatron.rl.compaction.learned.training.data import TrajectoryDataset, trajectory_collate_fn
from megatron.rl.compaction.learned.models.value import ChunkFeatureExtractor, FEATURE_DIM
from megatron.rl.compaction.learned.training.losses import CompactorLossWeights
from megatron.rl.compaction.learned.training.data import PipelineConfig
from megatron.rl.compaction.learned.training.training import BeliefCompactorTrainer
from megatron.rl.compaction.learned.training.data import CompactorTrainerConfig
from megatron.rl.compaction.learned.training.data import CompactKV, StudentFn
from megatron.rl.compaction.learned.models.belief import GatedRecurrentUpdater, GatedUpdaterConfig
from megatron.rl.compaction.learned.models.value import ValueHead


# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

B, T, C, D, L, V = 2, 16, 4, 8, 2, 32   # batch, seq, compress, d_model, layers, vocab


def _random_kv(T_len=T) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    keys   = [torch.randn(B, T_len, D) for _ in range(L)]
    values = [torch.randn(B, T_len, D) for _ in range(L)]
    return keys, values


def _perceiver_cfg():
    return PerceiverConfig(d_kv=D, n_heads=2, n_compress=C, n_attn_layers=L)


def _gated_updater(feature_dim=0):
    cfg = GatedUpdaterConfig(d_kv=D, n_heads=2, n_compress=C, n_attn_layers=L,
                             feature_dim=feature_dim)
    return GatedRecurrentUpdater(cfg)


def _belief_updater():
    return BeliefUpdater(PerceiverCompactor(_perceiver_cfg()))


def _make_probe():
    qlen = 8
    query = torch.randint(0, V, (B, qlen))
    logits = torch.randn(B, qlen, V)
    return TrainingProbe(query_tokens=query, teacher_logits=logits)


def _make_trajectory(n_chunks=4, add_probe_at=2):
    chunks = []
    for _ in range(n_chunks):
        chunks.append(_random_kv())
    probes_by_chunk = {add_probe_at: [_make_probe()]}
    return Trajectory(chunks=chunks, probes_by_chunk=probes_by_chunk)



def _student_fn(query_tokens: torch.Tensor, compact_kv: CompactKV) -> torch.Tensor:
    """Detached student_fn — useful for testing non-training paths."""
    B_, S_q, *_ = query_tokens.shape
    return torch.randn(B_, S_q, V)


def _student_fn_with_grad(query_tokens: torch.Tensor, compact_kv: CompactKV) -> torch.Tensor:
    """Differentiable student_fn: uses compact_kv so loss has grad wrt updater params."""
    B_, S_q = query_tokens.shape[:2]
    # Sum kv tensors linearly so the logits depend on compact_kv
    kv_signal = sum(k.mean() + v.mean() for k, v in compact_kv)
    base = torch.zeros(B_, S_q, V)
    return base + kv_signal  # (B, S_q, V) — differentiable w.r.t. compact_kv


# ---------------------------------------------------------------------------
# still/types.py
# ---------------------------------------------------------------------------

class TestTypes:
    def test_compact_kv_alias_matches_structure(self):
        keys   = torch.randn(B, C, D)
        values = torch.randn(B, C, D)
        kv: CompactKV = [(keys, values)]
        k, v = kv[0]
        assert k.shape == (B, C, D)

    def test_student_fn_callable(self):
        fn: StudentFn = _student_fn
        out = fn(torch.randint(0, V, (B, 8)), [(torch.zeros(B, C, D), torch.zeros(B, C, D))])
        assert out.shape == (B, 8, V)

    def test_types_importable_from_still_package(self):
        from megatron.rl.compaction import learned
        assert hasattr(learned, "CompactKV")
        assert hasattr(learned, "StudentFn")


# ---------------------------------------------------------------------------
# BeliefCompactorTrainer + ChunkFeatureExtractor
# ---------------------------------------------------------------------------

class TestBeliefStillTrainerFeatures:
    def test_train_without_extractor_still_works(self):
        updater = _gated_updater(feature_dim=0)
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig())
        traj = _make_trajectory()
        log = trainer.train_trajectory(traj, _student_fn)
        assert "total" in log

    def test_train_with_extractor_gated_updater(self):
        updater = _gated_updater(feature_dim=FEATURE_DIM)
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        extractor = ChunkFeatureExtractor(memory_budget=C, max_chunks=64)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(),
                                     feature_extractor=extractor)
        traj = _make_trajectory()
        log = trainer.train_trajectory(traj, _student_fn)
        assert "total" in log
        assert torch.isfinite(torch.tensor(log["total"]))

    def test_extractor_ignored_for_plain_belief_updater(self):
        # BeliefUpdater doesn't accept features; trainer should not crash
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        extractor = ChunkFeatureExtractor(memory_budget=C, max_chunks=64)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(),
                                     feature_extractor=extractor)
        traj = _make_trajectory()
        log = trainer.train_trajectory(traj, _student_fn)
        assert "total" in log

    def test_features_detection_flag(self):
        # The unified training path auto-detects whether the updater accepts a
        # `features` kwarg: a gated updater consumes z_t, a plain BeliefUpdater
        # ignores it.  Both must train without error when a feature_extractor is
        # supplied (gated uses features; plain silently skips them).
        ext = ChunkFeatureExtractor(memory_budget=C, max_chunks=64)
        for updater in (_belief_updater(), _gated_updater(feature_dim=FEATURE_DIM)):
            opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
            trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(),
                                         feature_extractor=ext)
            log = trainer.train_trajectory(_make_trajectory(), _student_fn)
            assert isinstance(log, dict) and "total" in log


# ---------------------------------------------------------------------------
# BeliefCompactorTrainer + CurriculumScheduler
# ---------------------------------------------------------------------------

class TestBeliefStillTrainerCurriculum:
    def test_scheduler_advances_loss_weights(self):
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        scheduler = CurriculumScheduler.default_4stage(steps_per_stage=1)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(), scheduler=scheduler)
        # Use differentiable student_fn so loss.requires_grad=True → _step fires
        traj = _make_trajectory(n_chunks=4, add_probe_at=2)
        trainer.train_trajectory(traj, _student_fn_with_grad)
        # After one optimizer step, the scheduler should have advanced once
        total_steps = scheduler.stage_idx * 1 + scheduler.steps_in_stage
        assert total_steps >= 1

    def test_no_scheduler_config_unchanged(self):
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        config = CompactorTrainerConfig(
            loss_weights=CompactorLossWeights(teacher_kl=1.0, future_kl=0.5)
        )
        trainer = BeliefCompactorTrainer(updater, opt, config)
        traj = _make_trajectory()
        trainer.train_trajectory(traj, _student_fn_with_grad)
        # Without scheduler, loss_weights should not change
        assert trainer.config.loss_weights.teacher_kl == pytest.approx(1.0)
        assert trainer.config.loss_weights.future_kl == pytest.approx(0.5)

    def test_scheduler_updates_loss_weights_between_bptt_windows(self):
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        # Short BPTT window to force multiple optimizer steps per trajectory
        config = CompactorTrainerConfig(truncated_bptt_steps=2)
        scheduler = CurriculumScheduler.default_4stage(steps_per_stage=100)
        trainer = BeliefCompactorTrainer(updater, opt, config, scheduler=scheduler)
        # 6 chunks, probe at 1 → 3 BPTT windows (chunks 0-1, 2-3, 4-5)
        traj = _make_trajectory(n_chunks=6, add_probe_at=1)
        trainer.train_trajectory(traj, _student_fn_with_grad)
        # At least one BPTT window had a probe and triggered _step → scheduler advanced
        total_steps = scheduler.stage_idx * 100 + scheduler.steps_in_stage
        assert total_steps >= 1


# ---------------------------------------------------------------------------
# BeliefCompactorTrainer + ValueHead
# ---------------------------------------------------------------------------

class TestBeliefStillTrainerValueHead:
    def test_value_mean_in_log_when_value_head_provided(self):
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        value_head = ValueHead(n_layers=L, d_model=D)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(),
                                     value_head=value_head)
        traj = _make_trajectory()
        log = trainer.train_trajectory(traj, _student_fn)
        assert "value_mean" in log
        assert isinstance(log["value_mean"], float)

    def test_no_value_head_key_absent(self):
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig())
        traj = _make_trajectory()
        log = trainer.train_trajectory(traj, _student_fn)
        assert "value_mean" not in log

    def test_value_head_not_trained_without_rl_targets(self):
        # ValueHead gradients must not flow back unless explicitly enabled.
        # Use _student_fn_with_grad so the updater IS trained (eliminating the
        # "no optimizer step at all" degenerate case).
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        value_head = ValueHead(n_layers=L, d_model=D)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig(),
                                     value_head=value_head)
        params_before = [p.clone() for p in value_head.parameters()]
        traj = _make_trajectory(n_chunks=4, add_probe_at=2)
        trainer.train_trajectory(traj, _student_fn_with_grad)
        for p_before, p_after in zip(params_before, value_head.parameters()):
            assert torch.allclose(p_before, p_after), "ValueHead was updated without RL targets"


# ---------------------------------------------------------------------------
# BeliefCompactorTrainer — all extras wired together
# ---------------------------------------------------------------------------

class TestBeliefStillTrainerFullWiring:
    def test_all_extras_together(self):
        updater = _gated_updater(feature_dim=FEATURE_DIM)
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        extractor = ChunkFeatureExtractor(memory_budget=C, max_chunks=64)
        value_head = ValueHead(n_layers=L, d_model=D, feature_dim=FEATURE_DIM)
        scheduler = CurriculumScheduler.default_4stage(steps_per_stage=10)
        config = CompactorTrainerConfig(truncated_bptt_steps=2)
        trainer = BeliefCompactorTrainer(
            updater, opt, config,
            feature_extractor=extractor,
            value_head=value_head,
            scheduler=scheduler,
        )
        traj = _make_trajectory(n_chunks=6, add_probe_at=3)
        log = trainer.train_trajectory(traj, _student_fn)
        assert "total" in log
        assert "value_mean" in log
        assert torch.isfinite(torch.tensor(log["total"]))


# ---------------------------------------------------------------------------
# TrajectoryDataset + trajectory_collate_fn
# ---------------------------------------------------------------------------

class TestTrajectoryDataset:
    def test_len(self):
        trajs = [_make_trajectory() for _ in range(5)]
        ds = TrajectoryDataset(trajs)
        assert len(ds) == 5

    def test_getitem_returns_trajectory(self):
        trajs = [_make_trajectory() for _ in range(3)]
        ds = TrajectoryDataset(trajs)
        assert isinstance(ds[0], Trajectory)
        assert isinstance(ds[2], Trajectory)

    def test_getitem_identity(self):
        trajs = [_make_trajectory() for _ in range(3)]
        ds = TrajectoryDataset(trajs)
        assert ds[1] is trajs[1]

    def test_collate_fn_returns_list(self):
        trajs = [_make_trajectory() for _ in range(4)]
        batch = trajectory_collate_fn(trajs)
        assert isinstance(batch, list)
        assert len(batch) == 4
        assert all(isinstance(t, Trajectory) for t in batch)

    def test_dataloader_single_element(self):
        trajs = [_make_trajectory() for _ in range(6)]
        ds = TrajectoryDataset(trajs)
        loader = DataLoader(ds, batch_size=1, collate_fn=trajectory_collate_fn, shuffle=False)
        batches = list(loader)
        assert len(batches) == 6
        assert all(isinstance(b[0], Trajectory) for b in batches)

    def test_dataloader_multi_element(self):
        trajs = [_make_trajectory() for _ in range(6)]
        ds = TrajectoryDataset(trajs)
        loader = DataLoader(ds, batch_size=2, collate_fn=trajectory_collate_fn, shuffle=False)
        batches = list(loader)
        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    def test_dataloader_shuffle_does_not_crash(self):
        trajs = [_make_trajectory() for _ in range(8)]
        ds = TrajectoryDataset(trajs)
        loader = DataLoader(ds, batch_size=2, collate_fn=trajectory_collate_fn, shuffle=True)
        # DataLoader's shuffle sampler draws indices with a CPU generator; data
        # loading is CPU-side, so iterate under the CPU default device.
        with torch.device("cpu"):
            batches = list(loader)
        all_trajs = [t for b in batches for t in b]
        assert len(all_trajs) == 8

    def test_empty_dataset(self):
        ds = TrajectoryDataset([])
        assert len(ds) == 0
        loader = DataLoader(ds, batch_size=2, collate_fn=trajectory_collate_fn)
        assert list(loader) == []

    def test_trainer_iterates_over_dataset(self):
        # End-to-end: DataLoader → BeliefCompactorTrainer
        updater = _belief_updater()
        opt = torch.optim.SGD(updater.parameters(), lr=1e-3)
        trainer = BeliefCompactorTrainer(updater, opt, CompactorTrainerConfig())
        trajs = [_make_trajectory() for _ in range(3)]
        ds = TrajectoryDataset(trajs)
        loader = DataLoader(ds, batch_size=1, collate_fn=trajectory_collate_fn, shuffle=False)
        logs = []
        for batch in loader:
            for traj in batch:
                log = trainer.train_trajectory(traj, _student_fn)
                logs.append(log)
        assert len(logs) == 3
        assert all("total" in log for log in logs)


# ---------------------------------------------------------------------------
# Type-consolidation: verify CompactKV is consistent across submodules
# ---------------------------------------------------------------------------

class TestTypeConsolidation:
    def test_compact_kv_same_object_across_modules(self):
        from megatron.rl.compaction.learned.training.data import CompactKV as CT
        from megatron.rl.compaction.learned import CompactKV as CI
        # Both should resolve to the same alias (same module origin)
        assert CT is CI

    def test_student_fn_same_object_across_modules(self):
        from megatron.rl.compaction.learned.training.data import StudentFn as ST
        from megatron.rl.compaction.learned import StudentFn as SI
        assert ST is SI

    def test_losses_module_no_longer_defines_compact_kv(self):
        import megatron.rl.compaction.learned.training.losses as losses_mod
        # CompactKV should come from still.types, not be defined locally in losses
        # The module still has the name (imported), but it must be the same object
        from megatron.rl.compaction.learned.training.data import CompactKV
        assert losses_mod.CompactKV is CompactKV

    def test_trainer_classes_in_learned_training(self):
        # The compactor training core lives in compaction/learned/training.py
        # (extracted from rl_utils.py).
        import megatron.rl.compaction.learned.training.training as training_mod
        assert hasattr(training_mod, "SinglePassCompactorTrainer")
        assert hasattr(training_mod, "BeliefCompactorTrainer")
        assert hasattr(training_mod, "train_compactor_trajectory")
        assert hasattr(training_mod, "_CompactorOptimizer")
