# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for NextLat/future-horizon extensions to the Belief-Still module.

Covers:
  - future_kv_reconstruction_loss: belief-state predictive test
  - dynamics_prediction_loss: NextLat latent-dynamics loss
  - future_horizon_kl_loss: position-weighted KL distillation
  - head-free NextLat dynamics: train_compactor_trajectory(dynamics>0) trains the
    compactor (no dedicated dynamics head)
  - CompactorLossWeights: new fields default to 0.0, appear in as_dict()
  - CompactorTrainerConfig: new fields and their defaults
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from megatron.rl.compaction.learned.models.belief import (
    BeliefMemory,
    GatedRecurrentUpdater,
    GatedUpdaterConfig,
)
from megatron.rl.compaction.learned.training.losses import (
    CompactorLossWeights,
    dynamics_prediction_loss,
    future_horizon_kl_loss,
    future_kv_reconstruction_loss,
    teacher_kl_loss,
)
from megatron.rl.compaction.learned.training.data import (
    CompactorTrainerConfig,
    Trajectory,
    TrainingProbe,
)
from megatron.rl.compaction.learned.training.training import train_compactor_trajectory


# ---------------------------------------------------------------------------
# Shared test dimensions
# ---------------------------------------------------------------------------

N_LAYERS = 2
B = 1
C = 8
D = 16
T = 32
VOCAB = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv_list(n_layers=N_LAYERS, B=B, seq=T, d=D, *, seed=0, requires_grad=False):
    """Create a list of (B, seq, d) key/value tensors, one per layer."""
    torch.manual_seed(seed)
    keys = [
        torch.randn(B, seq, d, requires_grad=requires_grad) for _ in range(n_layers)
    ]
    values = [
        torch.randn(B, seq, d, requires_grad=requires_grad) for _ in range(n_layers)
    ]
    return keys, values


def _make_compact_kv_list(n_layers=N_LAYERS, B=B, C=C, d=D, *, seed=0, requires_grad=False):
    """Create compact (B, C, d) key/value tensors, one per layer."""
    return _make_kv_list(n_layers=n_layers, B=B, seq=C, d=d, seed=seed,
                         requires_grad=requires_grad)


def _make_logits(B=B, seq=T, vocab=VOCAB, *, seed=0, requires_grad=False):
    torch.manual_seed(seed)
    return torch.randn(B, seq, vocab, requires_grad=requires_grad)


def _make_gru_cfg():
    return GatedUpdaterConfig(n_compress=C, n_heads=2, d_kv=D, n_attn_layers=N_LAYERS)


def _make_belief_memory(n_layers=N_LAYERS, B=B, C=C, d=D, *, seed=0):
    torch.manual_seed(seed)
    keys   = torch.randn(n_layers, B, C, d)
    values = torch.randn(n_layers, B, C, d)
    return BeliefMemory(keys=keys, values=values, step=0)


# ---------------------------------------------------------------------------
# TestFutureKvReconstructionLoss
# ---------------------------------------------------------------------------

class TestFutureKvReconstructionLoss(unittest.TestCase):
    """Tests for future_kv_reconstruction_loss."""

    def _perfect_compact_kv(self, full_keys, full_values):
        """Return compact KV == copies of full KV (perfect memory)."""
        compact_keys = [k.clone() for k in full_keys]
        compact_values = [v.clone() for v in full_values]
        return compact_keys, compact_values

    def test_perfect_memory_lower_than_random(self):
        """Perfect compact KV (copy of full) should give lower loss than random."""
        full_keys, full_values = _make_kv_list(seed=0)

        # Perfect memory: compact == full KV
        perf_k, perf_v = self._perfect_compact_kv(full_keys, full_values)
        loss_perfect = future_kv_reconstruction_loss(
            perf_k, perf_v, full_keys, full_values, n_queries=8
        )

        # Random memory: compact is independent random noise
        rand_k, rand_v = _make_compact_kv_list(seed=999)
        loss_random = future_kv_reconstruction_loss(
            rand_k, rand_v, full_keys, full_values, n_queries=8
        )

        self.assertLess(
            loss_perfect.item(),
            loss_random.item(),
            msg=(
                f"Perfect memory loss ({loss_perfect.item():.6f}) should be lower "
                f"than random memory loss ({loss_random.item():.6f})"
            ),
        )

    def test_returns_scalar(self):
        """Output should be a 0-dimensional scalar tensor."""
        full_keys, full_values = _make_kv_list(seed=1)
        compact_keys, compact_values = _make_compact_kv_list(seed=2)
        loss = future_kv_reconstruction_loss(
            compact_keys, compact_values, full_keys, full_values, n_queries=8
        )
        self.assertEqual(loss.shape, torch.Size([]),
                         msg=f"Expected scalar shape [], got {loss.shape}")

    def test_requires_grad(self):
        """Loss should have requires_grad when compact_keys require grad."""
        full_keys, full_values = _make_kv_list(seed=3)
        compact_keys, compact_values = _make_compact_kv_list(seed=4, requires_grad=True)
        loss = future_kv_reconstruction_loss(
            compact_keys, compact_values, full_keys, full_values, n_queries=8
        )
        self.assertTrue(
            loss.requires_grad,
            msg="Loss should require grad when compact_keys require grad",
        )


# ---------------------------------------------------------------------------
# TestDynamicsPredictionLoss
# ---------------------------------------------------------------------------

class TestDynamicsPredictionLoss(unittest.TestCase):
    """Tests for dynamics_prediction_loss."""

    def test_zero_for_identical_prediction(self):
        """If pred == target, loss should be exactly 0."""
        target_keys, target_values = _make_compact_kv_list(seed=10)
        # Predictions identical to targets
        pred_keys   = [t.clone() for t in target_keys]
        pred_values = [t.clone() for t in target_values]
        loss = dynamics_prediction_loss(pred_keys, pred_values, target_keys, target_values)
        self.assertAlmostEqual(
            loss.item(), 0.0, places=6,
            msg=f"Expected 0 for identical pred/target, got {loss.item()}"
        )

    def test_stop_gradient_on_target(self):
        """Target tensors should not accumulate gradient (detach is applied internally)."""
        target_keys, target_values = _make_compact_kv_list(seed=11)
        pred_keys   = [t.clone().requires_grad_(True) for t in target_keys]
        pred_values = [t.clone().requires_grad_(True) for t in target_values]

        # Give targets requires_grad=True so we can check if grad flows back
        target_keys_grad   = [t.clone().requires_grad_(True) for t in target_keys]
        target_values_grad = [t.clone().requires_grad_(True) for t in target_values]

        loss = dynamics_prediction_loss(
            pred_keys, pred_values, target_keys_grad, target_values_grad
        )
        loss.backward()

        # Targets must NOT have gradients (stop-gradient enforced by .detach() inside)
        for i, tk in enumerate(target_keys_grad):
            self.assertIsNone(
                tk.grad,
                msg=f"target_keys[{i}] should have no grad (stop-gradient), "
                    f"but got grad={tk.grad}",
            )
        for i, tv in enumerate(target_values_grad):
            self.assertIsNone(
                tv.grad,
                msg=f"target_values[{i}] should have no grad (stop-gradient), "
                    f"but got grad={tv.grad}",
            )

        # Predictions SHOULD have gradients
        for i, pk in enumerate(pred_keys):
            self.assertIsNotNone(pk.grad, msg=f"pred_keys[{i}] should have grad")
        for i, pv in enumerate(pred_values):
            self.assertIsNotNone(pv.grad, msg=f"pred_values[{i}] should have grad")

    def test_returns_scalar(self):
        """Output should be a 0-dimensional scalar tensor."""
        target_keys, target_values = _make_compact_kv_list(seed=12)
        pred_keys   = [t + 0.1 for t in target_keys]
        pred_values = [t + 0.1 for t in target_values]
        loss = dynamics_prediction_loss(pred_keys, pred_values, target_keys, target_values)
        self.assertEqual(loss.shape, torch.Size([]),
                         msg=f"Expected scalar shape [], got {loss.shape}")


# ---------------------------------------------------------------------------
# TestFutureHorizonKlLoss
# ---------------------------------------------------------------------------

class TestFutureHorizonKlLoss(unittest.TestCase):
    """Tests for future_horizon_kl_loss."""

    def test_gamma_1_matches_teacher_kl(self):
        """gamma=1.0 should give the same result as teacher_kl_loss (uniform weights)."""
        torch.manual_seed(20)
        full_logits    = _make_logits(seed=20)
        compact_logits = _make_logits(seed=21)

        loss_fhkl = future_horizon_kl_loss(full_logits, compact_logits, gamma=1.0)
        loss_tkl  = teacher_kl_loss(full_logits, compact_logits)

        self.assertAlmostEqual(
            loss_fhkl.item(), loss_tkl.item(), places=5,
            msg=(
                f"gamma=1.0 future_horizon_kl ({loss_fhkl.item():.8f}) should match "
                f"teacher_kl ({loss_tkl.item():.8f})"
            ),
        )

    def test_gamma_less_1_differs(self):
        """gamma=0.8 should give a different result than gamma=1.0 when logits vary by position."""
        # Use logits with deliberate position variation so the weighting matters
        torch.manual_seed(30)
        # Make logits that strongly differ across the sequence dimension
        full_logits    = torch.zeros(B, T, VOCAB)
        compact_logits = torch.zeros(B, T, VOCAB)
        for i in range(T):
            full_logits[0, i, i % VOCAB]    = float(i + 1)   # peak shifts with position
            compact_logits[0, i, (i + 1) % VOCAB] = float(i + 1)

        loss_gamma_1  = future_horizon_kl_loss(full_logits, compact_logits, gamma=1.0)
        loss_gamma_08 = future_horizon_kl_loss(full_logits, compact_logits, gamma=0.8)

        self.assertNotAlmostEqual(
            loss_gamma_1.item(), loss_gamma_08.item(), places=4,
            msg=(
                f"gamma=0.8 ({loss_gamma_08.item():.8f}) should differ from "
                f"gamma=1.0 ({loss_gamma_1.item():.8f}) when logits vary by position"
            ),
        )

    def test_returns_scalar(self):
        """Output should be a 0-dimensional scalar tensor."""
        full_logits    = _make_logits(seed=31)
        compact_logits = _make_logits(seed=32)
        loss = future_horizon_kl_loss(full_logits, compact_logits, gamma=0.9)
        self.assertEqual(loss.shape, torch.Size([]),
                         msg=f"Expected scalar shape [], got {loss.shape}")


# ---------------------------------------------------------------------------
# TestGatedRecurrentUpdaterDynamicsHead
# ---------------------------------------------------------------------------

class TestHeadFreeDynamics(unittest.TestCase):
    """The NextLat dynamics loss trains the compactor with NO dedicated head.

    train_compactor_trajectory rolls the updater forward on its own predicted next
    chunk (M_pred = U(M_t, pred_R_t)) and matches the real M_{t+1}.
    """

    def _traj(self, n_chunks=3, seed=0):
        chunks = []
        for c in range(n_chunks):
            k, v = _make_kv_list(seq=T, seed=seed + c)
            chunks.append((k, v))
        return Trajectory(chunks=chunks, probes_by_chunk={}, rollout_return=0.0)

    def test_no_dynamics_head_attribute(self):
        updater = GatedRecurrentUpdater(_make_gru_cfg())
        self.assertFalse(hasattr(updater, "dynamics_key_heads"))
        self.assertFalse(hasattr(updater, "predict_next_memory"))

    def test_dynamics_loss_reported_and_trains_compactor(self):
        updater = GatedRecurrentUpdater(_make_gru_cfg())
        opt = torch.optim.SGD(updater.parameters(), lr=1e-2)
        cfg = CompactorTrainerConfig(
            loss_weights=CompactorLossWeights(kv_reconstruction=0.0, dynamics=1.0),
        )
        before = [p.detach().clone() for p in updater.parameters()]
        log = train_compactor_trajectory(updater, opt, self._traj(), student_fn=None, cfg=cfg)

        # The dynamics term is computed and reported...
        self.assertIn("dynamics", log)
        # ...and it actually moves the compactor weights (representation is trained).
        changed = any(not torch.equal(b, p) for b, p in zip(before, updater.parameters()))
        self.assertTrue(changed, msg="dynamics-only training did not update any compactor params")


# ---------------------------------------------------------------------------
# TestStillLossWeightsNewFields
# ---------------------------------------------------------------------------

class TestStillLossWeightsNewFields(unittest.TestCase):
    """Tests that new CompactorLossWeights fields behave correctly."""

    NEW_FIELDS = ("future_kv_reconstruction", "dynamics", "future_horizon_kl")

    def test_default_zero(self):
        """New fields should default to 0.0."""
        weights = CompactorLossWeights()
        for field in self.NEW_FIELDS:
            val = getattr(weights, field)
            self.assertEqual(
                val, 0.0,
                msg=f"CompactorLossWeights.{field} should default to 0.0, got {val}",
            )

    def test_as_dict_has_new_keys(self):
        """as_dict() should include all new field keys."""
        weights = CompactorLossWeights()
        d = weights.as_dict()
        for field in self.NEW_FIELDS:
            self.assertIn(
                field, d,
                msg=f"CompactorLossWeights.as_dict() missing key '{field}'",
            )

    def test_as_dict_values_match_fields(self):
        """as_dict() values should match the instance field values for new fields."""
        weights = CompactorLossWeights(
            future_kv_reconstruction=0.5,
            dynamics=0.3,
            future_horizon_kl=0.7,
        )
        d = weights.as_dict()
        self.assertAlmostEqual(d["future_kv_reconstruction"], 0.5, places=7)
        self.assertAlmostEqual(d["dynamics"], 0.3, places=7)
        self.assertAlmostEqual(d["future_horizon_kl"], 0.7, places=7)


# ---------------------------------------------------------------------------
# TestStillTrainerConfigNewFields
# ---------------------------------------------------------------------------

class TestStillTrainerConfigNewFields(unittest.TestCase):
    """Tests that CompactorTrainerConfig has the new NextLat/horizon fields with correct defaults."""

    def test_defaults(self):
        """future_horizon_gamma=1.0, use_future_accuracy_weight=False, merged_chunk_prob=0.0."""
        cfg = CompactorTrainerConfig()
        self.assertEqual(
            cfg.future_horizon_gamma, 1.0,
            msg=f"future_horizon_gamma should default to 1.0, got {cfg.future_horizon_gamma}",
        )
        self.assertFalse(
            cfg.use_future_accuracy_weight,
            msg="use_future_accuracy_weight should default to False",
        )
        self.assertEqual(
            cfg.merged_chunk_prob, 0.0,
            msg=f"merged_chunk_prob should default to 0.0, got {cfg.merged_chunk_prob}",
        )

    def test_custom_values_accepted(self):
        """Custom values for new fields should be stored correctly."""
        cfg = CompactorTrainerConfig(
            future_horizon_gamma=0.9,
            use_future_accuracy_weight=True,
            merged_chunk_prob=0.25,
        )
        self.assertAlmostEqual(cfg.future_horizon_gamma, 0.9, places=7)
        self.assertTrue(cfg.use_future_accuracy_weight)
        self.assertAlmostEqual(cfg.merged_chunk_prob, 0.25, places=7)

    def test_loss_weights_default_populated(self):
        """loss_weights should be auto-populated with a CompactorLossWeights instance."""
        cfg = CompactorTrainerConfig()
        self.assertIsNotNone(cfg.loss_weights)
        self.assertIsInstance(cfg.loss_weights, CompactorLossWeights)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
