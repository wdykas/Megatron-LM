# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest

from megatron.rl.compaction.pomdp.probes.kl_probe import KLSufficiencyProbe
from megatron.rl.compaction.pomdp.probes.diffing import BeliefDiffer
from megatron.rl.compaction.pomdp.probes.sufficiency import FieldAblationProbe
from megatron.rl.compaction.pomdp.types import BeliefState, new_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_log_probs(n: int) -> list[float]:
    return [math.log(1.0 / n)] * n


def _peaked_log_probs(n: int, peak: int) -> list[float]:
    """Log-prob distribution concentrated on position ``peak``."""
    eps = 1e-8
    probs = [eps / (n - 1)] * n
    probs[peak] = 1.0 - eps
    return [math.log(p) for p in probs]


def _make_belief(step_id: int = 0, constraints: list[str] | None = None) -> BeliefState:
    return BeliefState(
        belief_id=f"b_{step_id}",
        run_id="run1",
        step_id=step_id,
        task={
            "objective": "test task",
            "hard_constraints": constraints or ["no api changes"],
        },
        environment_state={"files_changed": [{"path": "a.py"}]},
        open_questions=[{"question": "why?"}],
        uncertainty={"level": "medium"},
    )


# ---------------------------------------------------------------------------
# KLSufficiencyProbe
# ---------------------------------------------------------------------------

class TestKLSufficiencyProbeStats:
    def test_identical_distributions_give_zero_kl(self):
        log_p = _uniform_log_probs(10)
        assert KLSufficiencyProbe.kl(log_p, log_p) == pytest.approx(0.0, abs=1e-9)

    def test_kl_is_non_negative(self):
        log_p = _uniform_log_probs(4)
        log_q = _peaked_log_probs(4, peak=0)
        assert KLSufficiencyProbe.kl(log_p, log_q) >= 0.0

    def test_kl_is_asymmetric(self):
        log_p = _uniform_log_probs(4)
        log_q = _peaked_log_probs(4, peak=0)
        assert KLSufficiencyProbe.kl(log_p, log_q) != pytest.approx(
            KLSufficiencyProbe.kl(log_q, log_p), abs=1e-4
        )

    def test_js_is_symmetric(self):
        log_p = _uniform_log_probs(4)
        log_q = _peaked_log_probs(4, peak=0)
        assert KLSufficiencyProbe.js(log_p, log_q) == pytest.approx(
            KLSufficiencyProbe.js(log_q, log_p), abs=1e-9
        )

    def test_js_bounded_by_log2(self):
        log_p = _uniform_log_probs(4)
        log_q = _peaked_log_probs(4, peak=0)
        assert KLSufficiencyProbe.js(log_p, log_q) <= math.log(2) + 1e-9

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            KLSufficiencyProbe.kl([0.0, 0.0], [0.0])


class TestKLSufficiencyProbeMeasure:
    def test_identical_token_sequences_give_zero_kl(self):
        probe = KLSufficiencyProbe()
        tokens = [1, 2, 3]
        log_p = _uniform_log_probs(10)
        result = probe.measure(lambda _: log_p, tokens, tokens)
        assert result.kl_full_to_compact == pytest.approx(0.0, abs=1e-9)

    def test_result_carries_token_counts(self):
        probe = KLSufficiencyProbe()
        log_p = _uniform_log_probs(8)
        result = probe.measure(lambda _: log_p, [1, 2, 3, 4, 5], [1, 2])
        assert result.full_token_count == 5
        assert result.compact_token_count == 2

    def test_compression_ratio(self):
        probe = KLSufficiencyProbe()
        log_p = _uniform_log_probs(8)
        result = probe.measure(lambda _: log_p, [1] * 10, [1] * 5)
        assert result.compression_ratio() == pytest.approx(2.0)

    def test_different_distributions_give_nonzero_kl(self):
        probe = KLSufficiencyProbe()
        log_p_full = _uniform_log_probs(8)
        log_p_compact = _peaked_log_probs(8, peak=0)

        def policy(tokens):
            return log_p_full if len(tokens) == 10 else log_p_compact

        result = probe.measure(policy, [1] * 10, [1] * 5)
        assert result.kl_full_to_compact > 0.0


# ---------------------------------------------------------------------------
# BeliefDiffer
# ---------------------------------------------------------------------------

class TestBeliefDiffer:
    def test_no_diff_on_identical_beliefs(self):
        differ = BeliefDiffer()
        b = _make_belief(step_id=0)
        diff = differ.diff(b, b)
        assert diff.added_keys == []
        assert diff.removed_keys == []
        assert diff.modified_keys == []
        assert diff.constraint_violations == []

    def test_detects_modified_field(self):
        differ = BeliefDiffer()
        prev = _make_belief(step_id=0)
        curr = _make_belief(step_id=1)
        curr.uncertainty = {"level": "high"}
        diff = differ.diff(prev, curr)
        assert "uncertainty" in diff.modified_keys or "step_id" in diff.modified_keys

    def test_detects_constraint_violation(self):
        differ = BeliefDiffer()
        prev = _make_belief(constraints=["no api changes", "no deletes"])
        curr = _make_belief(constraints=["no api changes"])  # "no deletes" was dropped
        diff = differ.diff(prev, curr)
        assert "no deletes" in diff.constraint_violations

    def test_no_violation_when_constraints_preserved(self):
        differ = BeliefDiffer()
        prev = _make_belief(constraints=["no api changes"])
        curr = _make_belief(constraints=["no api changes", "extra constraint"])
        diff = differ.diff(prev, curr)
        assert diff.constraint_violations == []

    def test_is_monotone_true_when_no_violations(self):
        differ = BeliefDiffer()
        prev = _make_belief()
        curr = _make_belief()
        diff = differ.diff(prev, curr)
        assert differ.is_monotone(diff) is True

    def test_is_monotone_false_when_violation(self):
        differ = BeliefDiffer()
        prev = _make_belief(constraints=["no api changes", "no deletes"])
        curr = _make_belief(constraints=["no api changes"])
        diff = differ.diff(prev, curr)
        assert differ.is_monotone(diff) is False

    def test_step_ids_recorded_in_diff(self):
        differ = BeliefDiffer()
        prev = _make_belief(step_id=3)
        curr = _make_belief(step_id=4)
        diff = differ.diff(prev, curr)
        assert diff.step_from == 3
        assert diff.step_to == 4

    def test_to_dict_serializable(self):
        differ = BeliefDiffer()
        diff = differ.diff(_make_belief(), _make_belief())
        d = diff.to_dict()
        assert "constraint_violations" in d
        assert isinstance(d["added_keys"], list)


# ---------------------------------------------------------------------------
# FieldAblationProbe
# ---------------------------------------------------------------------------

class _FakeContextBuilder:
    """Minimal context builder for testing: encode returns list of ints from text."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


class TestFieldAblationProbe:
    def test_ablate_dict_field_returns_empty_dict(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        ablated = probe.ablate(belief, "environment_state")
        assert ablated.environment_state == {}

    def test_ablate_list_field_returns_empty_list(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        ablated = probe.ablate(belief, "open_questions")
        assert ablated.open_questions == []

    def test_ablate_does_not_mutate_original(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        original_env = dict(belief.environment_state)
        probe.ablate(belief, "environment_state")
        assert belief.environment_state == original_env

    def test_ablate_unknown_field_is_noop(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        ablated = probe.ablate(belief, "nonexistent_field")
        assert ablated.step_id == belief.step_id

    def test_measure_returns_one_result_per_field(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        builder = _FakeContextBuilder()
        base_tokens = builder.encode(belief.to_context_str())
        log_p = _uniform_log_probs(20)
        results = probe.measure(
            policy_fn=lambda _: log_p,
            belief=belief,
            context_builder=builder,
            base_tokens=base_tokens,
            fields=["environment_state", "open_questions"],
        )
        assert len(results) == 2
        assert {r.field for r in results} == {"environment_state", "open_questions"}

    def test_measure_kl_is_non_negative(self):
        probe = FieldAblationProbe()
        belief = _make_belief()
        builder = _FakeContextBuilder()
        base_tokens = builder.encode(belief.to_context_str())
        log_p = _uniform_log_probs(20)
        results = probe.measure(
            policy_fn=lambda _: log_p,
            belief=belief,
            context_builder=builder,
            base_tokens=base_tokens,
        )
        for r in results:
            assert r.kl_when_ablated >= 0.0
