# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Still training infrastructure.

Covers:
  - retrieval_loss / weighted_kl_loss (new loss functions)
  - GatedRecurrentUpdater (full Belief-Still architecture)
  - TrainingProbe / Trajectory (data types)
  - SinglePassCompactorTrainer (single-pass baseline)
  - BeliefCompactorTrainer (recurrent, truncated BPTT)
"""

import math

import pytest
import torch
import torch.nn as nn

from megatron.rl.compaction.learned.models.belief import BeliefMemory, BeliefUpdater
from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor, PerceiverConfig
from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe
from megatron.rl.compaction.learned.training.losses import (
    CompactorLosses,
    CompactorLossWeights,
    retrieval_loss,
    teacher_kl_loss,
    weighted_kl_loss,
)
from megatron.rl.compaction.learned.training.training import (
    BeliefCompactorTrainer,
    SinglePassCompactorTrainer,
)
from megatron.rl.compaction.learned.training.data import CompactorTrainerConfig
from megatron.rl.compaction.learned.models.belief import (
    GatedRecurrentUpdater,
    GatedUpdaterConfig,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

B, T, C, D, L, V = 2, 16, 4, 8, 2, 32   # batch, seq, compress, d_kv, layers, vocab


def _random_kv(T_len=T) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    keys   = [torch.randn(B, T_len, D) for _ in range(L)]
    values = [torch.randn(B, T_len, D) for _ in range(L)]
    return keys, values


def _perceiver_cfg():
    return PerceiverConfig(d_kv=D, n_heads=2, n_compress=C, n_attn_layers=L)


def _updater_cfg(**kw):
    defaults = dict(d_kv=D, n_heads=2, n_compress=C, n_attn_layers=L)
    defaults.update(kw)
    return GatedUpdaterConfig(**defaults)


def _probe(with_answer=False, retrieval=False) -> TrainingProbe:
    q = torch.randint(0, V, (B, 4))
    tl = torch.randn(B, 4, V)
    ans = torch.randint(0, V, (B, 4)) if with_answer else None
    return TrainingProbe(query_tokens=q, teacher_logits=tl, answer_tokens=ans,
                         is_exact_retrieval=retrieval)


def _dummy_student_fn(query_tokens, compact_kv) -> torch.Tensor:
    """Logits that share the computational graph with compact_kv.

    Uses compact_kv[0][0] (first-layer compressed keys) as the raw logit
    tensor, tiled to (B, S_q, V).  This gives a direct, non-cancelling
    gradient path back to the compactor/updater parameters.  A scalar
    kv_sum would cancel because ∑ ∂KL/∂logit_v = 0; per-position signal avoids that.
    """
    B, S_q = query_tokens.shape
    k0 = compact_kv[0][0]           # (B, C, D)
    flat = k0.reshape(B, -1)        # (B, C*D)
    need = S_q * V
    reps = (need + flat.shape[1] - 1) // flat.shape[1]
    return flat.repeat(1, reps)[:, :need].reshape(B, S_q, V)


def _trajectory(n_chunks=4, probe_chunk=3) -> Trajectory:
    chunks = [_random_kv() for _ in range(n_chunks)]
    probes = {probe_chunk: [_probe(with_answer=True, retrieval=True)]}
    return Trajectory(chunks=chunks, probes_by_chunk=probes)


# ---------------------------------------------------------------------------
# retrieval_loss tests
# ---------------------------------------------------------------------------

class TestRetrievalLoss:
    def test_returns_scalar(self):
        logits = torch.randn(B, 4, V)
        ids = torch.randint(0, V, (B, 4))
        loss = retrieval_loss(logits, ids)
        assert loss.shape == ()

    def test_ignore_index_ignored(self):
        logits = torch.randn(B, 4, V)
        ids_full = torch.randint(0, V, (B, 4))
        ids_half = ids_full.clone()
        ids_half[:, 2:] = -100
        loss_full = retrieval_loss(logits, ids_full)
        loss_half = retrieval_loss(logits, ids_half)
        # Half-masked loss uses fewer positions — values differ
        assert not torch.isclose(loss_full, loss_half)

    def test_perfect_prediction_gives_low_loss(self):
        V_small = 4
        logits = torch.zeros(1, 2, V_small)
        ids = torch.zeros(1, 2, dtype=torch.long)
        # Set logit for correct class very high
        logits[..., 0] = 100.0
        loss = retrieval_loss(logits, ids)
        assert loss.item() < 0.01

    def test_grad_flows(self):
        logits = torch.randn(B, 4, V, requires_grad=True)
        ids = torch.randint(0, V, (B, 4))
        loss = retrieval_loss(logits, ids)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# weighted_kl_loss tests
# ---------------------------------------------------------------------------

class TestWeightedKLLoss:
    def test_returns_scalar(self):
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        assert weighted_kl_loss(fl, cl).shape == ()

    def test_rho_zero_equals_teacher_kl(self):
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        wkl = weighted_kl_loss(fl, cl, temperature=1.0, rho=0.0)
        tkl = teacher_kl_loss(fl, cl, temperature=1.0)
        assert torch.isclose(wkl, tkl, rtol=1e-4)

    def test_positive_rho_raises_loss_on_confident_teacher(self):
        # Teacher very confident at position 0 (low entropy) → weight > 1
        fl = torch.zeros(1, 2, V)
        fl[0, 0, 0] = 50.0   # near-deterministic at pos 0
        # pos 1 is uniform → H ≈ H_max
        cl = torch.randn(1, 2, V)
        wkl = weighted_kl_loss(fl, cl, rho=10.0)
        tkl = teacher_kl_loss(fl, cl)
        # With high rho and confident teacher, weighted should differ from plain
        assert not torch.isclose(wkl, tkl)

    def test_temperature_squared_scaling(self):
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        T = 2.0
        wkl_T = weighted_kl_loss(fl, cl, temperature=T, rho=0.0)
        tkl_T = teacher_kl_loss(fl, cl, temperature=T)
        assert torch.isclose(wkl_T, tkl_T, rtol=1e-4)

    def test_grad_flows(self):
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V, requires_grad=True)
        loss = weighted_kl_loss(fl, cl)
        loss.backward()
        assert cl.grad is not None

    def test_nonnegative(self):
        fl = torch.randn(B, 8, V)
        cl = torch.randn(B, 8, V)
        assert weighted_kl_loss(fl, cl, rho=1.0).item() >= 0.0


# ---------------------------------------------------------------------------
# CompactorLossWeights / CompactorLossTerms with new fields
# ---------------------------------------------------------------------------

class TestUpdatedLossTypes:
    def test_weights_have_retrieval_and_weighted_kl(self):
        w = CompactorLossWeights()
        assert hasattr(w, "retrieval")
        assert hasattr(w, "weighted_kl")

    def test_compute_includes_retrieval_term(self):
        w = CompactorLossWeights(retrieval=1.0, task=0.0, consistency=0.0,
                             future_kl=0.0, weighted_kl=0.0)
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        ids = torch.randint(0, V, (B, 4))
        terms = CompactorLosses.compute(
            weights=w, full_logits=fl, compact_logits=cl, retrieval_ids=ids
        )
        assert terms.retrieval is not None
        assert terms.retrieval.item() >= 0.0

    def test_compute_includes_weighted_kl_term(self):
        w = CompactorLossWeights(weighted_kl=1.0, teacher_kl=0.0, consistency=0.0,
                             future_kl=0.0, task=0.0, retrieval=0.0)
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        terms = CompactorLosses.compute(weights=w, full_logits=fl, compact_logits=cl)
        assert terms.weighted_kl is not None

    def test_as_dict_includes_all_terms(self):
        w = CompactorLossWeights(retrieval=1.0, weighted_kl=1.0)
        fl = torch.randn(B, 4, V)
        cl = torch.randn(B, 4, V)
        ids = torch.randint(0, V, (B, 4))
        terms = CompactorLosses.compute(
            weights=w, full_logits=fl, compact_logits=cl, retrieval_ids=ids
        )
        d = terms.as_dict()
        assert "retrieval" in d
        assert "weighted_kl" in d


# ---------------------------------------------------------------------------
# GatedRecurrentUpdater tests
# ---------------------------------------------------------------------------

class TestGatedRecurrentUpdater:
    def _make(self, **kw):
        return GatedRecurrentUpdater(_updater_cfg(**kw))

    def test_initial_compress_shape(self):
        updater = self._make()
        keys, values = _random_kv()
        m = updater.initial_compress(keys, values)
        assert m.keys.shape == (L, B, C, D)
        assert m.values.shape == (L, B, C, D)
        assert m.step == 0

    def test_forward_returns_belief_memory(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        m1 = updater(m0, *_random_kv())
        assert isinstance(m1, BeliefMemory)
        assert m1.step == 1

    def test_update_returns_gates(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        m1, gates, preds = updater.update(m0, *_random_kv())
        assert gates.shape == (L, B, C, D)
        assert preds.shape == (L, B, C, D)

    def test_gates_in_unit_interval(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        _, gates, _ = updater.update(m0, *_random_kv())
        assert gates.min().item() >= 0.0 - 1e-6
        assert gates.max().item() <= 1.0 + 1e-6

    def test_memory_changes_after_update(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        m1 = updater(m0, *_random_kv())
        assert not torch.allclose(m0.keys, m1.keys)

    def test_step_increments(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        assert m0.step == 0
        m1 = updater(m0, *_random_kv())
        assert m1.step == 1
        m2 = updater(m1, *_random_kv())
        assert m2.step == 2

    def test_grad_flows_through_update(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        m1, _, _ = updater.update(m0, *_random_kv())
        loss = m1.keys.sum() + m1.values.sum()
        loss.backward()
        grads = [p.grad for p in updater.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_shared_weights_option(self):
        updater = self._make(share_across_layers=True)
        assert len(updater._layer_modules) == 1
        m0 = updater.initial_compress(*_random_kv())
        assert m0.keys.shape == (L, B, C, D)

    def test_feature_dim_accepted(self):
        updater = self._make(feature_dim=3)
        features = torch.randn(B, 3)
        m0 = updater.initial_compress(*_random_kv(), features=features)
        m1 = updater(m0, *_random_kv(), features=features)
        assert m1.keys.shape == (L, B, C, D)

    def test_variable_chunk_size(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv(T_len=8))
        m1 = updater(m0, *_random_kv(T_len=32))
        assert m1.keys.shape == (L, B, C, D)

    def test_detach_breaks_graph(self):
        updater = self._make()
        m0 = updater.initial_compress(*_random_kv())
        m1 = updater(m0, *_random_kv())
        m1_det = m1.detach()
        assert not m1_det.keys.requires_grad

    def test_compatible_with_belief_updater_interface(self):
        """GatedRecurrentUpdater exposes the same API as BeliefUpdater."""
        updater = self._make()
        keys, values = _random_kv()
        m0 = updater.initial_compress(keys, values)
        m1 = updater(m0, *_random_kv())
        assert isinstance(m0, BeliefMemory)
        assert isinstance(m1, BeliefMemory)


# ---------------------------------------------------------------------------
# TrainingProbe / Trajectory tests
# ---------------------------------------------------------------------------

class TestDataTypes:
    def test_probe_stores_fields(self):
        probe = _probe(with_answer=True, retrieval=True)
        assert probe.query_tokens.shape[0] == B
        assert probe.teacher_logits.shape[-1] == V
        assert probe.answer_tokens is not None
        assert probe.is_exact_retrieval is True

    def test_trajectory_probes_at(self):
        traj = _trajectory(n_chunks=4, probe_chunk=2)
        assert len(traj.probes_at(2)) == 1
        assert len(traj.probes_at(0)) == 0

    def test_trajectory_n_chunks(self):
        traj = _trajectory(n_chunks=6)
        assert traj.n_chunks == 6

    def test_trajectory_device(self):
        traj = _trajectory()
        assert traj.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# SinglePassCompactorTrainer tests
# ---------------------------------------------------------------------------

class TestStillTrainer:
    def _make(self, **kw):
        cfg = _perceiver_cfg()
        compactor = PerceiverCompactor(cfg)
        optimizer = torch.optim.Adam(compactor.parameters(), lr=1e-3)
        return SinglePassCompactorTrainer(compactor, optimizer, CompactorTrainerConfig(**kw))

    def test_train_step_returns_loss_terms(self):
        trainer = self._make()
        keys, values = _random_kv()
        probes = [_probe()]
        terms = trainer.train_step(keys, values, probes, _dummy_student_fn)
        assert terms.total is not None
        assert terms.teacher_kl is not None

    def test_train_step_loss_nonnegative(self):
        trainer = self._make()
        keys, values = _random_kv()
        terms = trainer.train_step(keys, values, [_probe()], _dummy_student_fn)
        assert terms.total.item() >= 0.0

    def test_train_step_multiple_probes(self):
        trainer = self._make()
        keys, values = _random_kv()
        probes = [_probe(), _probe(with_answer=True)]
        terms = trainer.train_step(keys, values, probes, _dummy_student_fn)
        assert terms.total.item() >= 0.0

    def test_train_step_updates_parameters(self):
        trainer = self._make()
        keys, values = _random_kv()
        params_before = [p.clone().detach() for p in trainer.compactor.parameters()]
        trainer.train_step(keys, values, [_probe()], _dummy_student_fn)
        params_after = list(trainer.compactor.parameters())
        changed = any(not torch.allclose(pb, pa) for pb, pa in zip(params_before, params_after))
        assert changed

    def test_eval_step_no_grad(self):
        trainer = self._make()
        keys, values = _random_kv()
        terms = trainer.eval_step(keys, values, [_probe()], _dummy_student_fn)
        assert terms.total.grad_fn is None

    def test_eval_step_does_not_update_params(self):
        trainer = self._make()
        keys, values = _random_kv()
        params_before = [p.clone().detach() for p in trainer.compactor.parameters()]
        trainer.eval_step(keys, values, [_probe()], _dummy_student_fn)
        for pb, p in zip(params_before, trainer.compactor.parameters()):
            assert torch.allclose(pb, p)

    def test_retrieval_loss_included_when_weight_positive(self):
        w = CompactorLossWeights(retrieval=1.0)
        trainer = self._make(loss_weights=w)
        probes = [_probe(with_answer=True, retrieval=True)]
        terms = trainer.train_step(*_random_kv(), probes, _dummy_student_fn)
        assert terms.retrieval is not None

    def test_no_probes_falls_back_to_kv_recon(self):
        # Empty probes with a student_fn falls back to KV reconstruction loss.
        trainer = self._make(loss_weights=CompactorLossWeights(kv_reconstruction=1.0))
        terms = trainer.train_step(*_random_kv(), [], _dummy_student_fn)
        assert terms.total is not None
        assert terms.total.item() >= 0.0


# ---------------------------------------------------------------------------
# BeliefCompactorTrainer tests
# ---------------------------------------------------------------------------

class TestBeliefStillTrainer:
    def _make_trainer(self, use_gated=False, **kw):
        if use_gated:
            updater = GatedRecurrentUpdater(_updater_cfg())
        else:
            cfg = _perceiver_cfg()
            compactor = PerceiverCompactor(cfg)
            updater = BeliefUpdater(compactor)
        optimizer = torch.optim.Adam(updater.parameters(), lr=1e-3)
        return BeliefCompactorTrainer(updater, optimizer, CompactorTrainerConfig(**kw))

    def test_train_trajectory_returns_dict(self):
        trainer = self._make_trainer()
        log = trainer.train_trajectory(_trajectory(), _dummy_student_fn)
        assert isinstance(log, dict)
        assert "total" in log

    def test_train_trajectory_empty_probes_no_distillation_signal(self):
        # With no probes, probe-based losses (teacher_kl, total) must be absent.
        # Per-chunk losses (consistency, predictive) may still appear if their
        # weights are non-zero.
        trainer = self._make_trainer(
            loss_weights=CompactorLossWeights(consistency=0.0, predictive=0.0)
        )
        traj = Trajectory(chunks=[_random_kv() for _ in range(3)], probes_by_chunk={})
        log = trainer.train_trajectory(traj, _dummy_student_fn)
        assert "teacher_kl" not in log
        assert "total" not in log

    def test_updates_parameters(self):
        trainer = self._make_trainer()
        params_before = [p.clone().detach() for p in trainer.updater.parameters()]
        trainer.train_trajectory(_trajectory(), _dummy_student_fn)
        changed = any(
            not torch.allclose(pb, p)
            for pb, p in zip(params_before, trainer.updater.parameters())
        )
        assert changed

    def test_truncated_bptt_steps_honored(self):
        trainer = self._make_trainer(truncated_bptt_steps=2)
        traj = _trajectory(n_chunks=8, probe_chunk=7)
        log = trainer.train_trajectory(traj, _dummy_student_fn)
        assert "total" in log

    def test_works_with_gated_updater(self):
        trainer = self._make_trainer(use_gated=True)
        log = trainer.train_trajectory(_trajectory(), _dummy_student_fn)
        assert "total" in log

    def test_multiple_probe_chunks(self):
        chunks = [_random_kv() for _ in range(6)]
        probes = {
            1: [_probe(with_answer=True)],
            3: [_probe()],
            5: [_probe(with_answer=True, retrieval=True)],
        }
        traj = Trajectory(chunks=chunks, probes_by_chunk=probes)
        trainer = self._make_trainer(loss_weights=CompactorLossWeights(retrieval=1.0))
        log = trainer.train_trajectory(traj, _dummy_student_fn)
        assert log["total"] >= 0.0

    def test_loss_terms_in_log(self):
        trainer = self._make_trainer()
        log = trainer.train_trajectory(_trajectory(), _dummy_student_fn)
        assert "teacher_kl" in log

    def test_detach_memory_after_bptt_window(self):
        trainer = self._make_trainer(truncated_bptt_steps=2)
        # 6 chunks with probes at every chunk to ensure loss is non-zero each step
        chunks = [_random_kv() for _ in range(6)]
        probes = {i: [_probe()] for i in range(6)}
        traj = Trajectory(chunks=chunks, probes_by_chunk=probes)
        # Should complete without OOM or graph issues
        log = trainer.train_trajectory(traj, _dummy_student_fn)
        assert "total" in log
