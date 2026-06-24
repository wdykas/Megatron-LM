# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.compaction.pomdp.metrics import ShadowRunMetrics, ShadowStepMetrics
from megatron.rl.compaction.pomdp.config import PomdpConfig
from megatron.rl.compaction.pomdp.context_builder import ContextBuilder
from megatron.rl.compaction.pomdp.recorder import PomdpRolloutRecorder
from megatron.rl.compaction.pomdp.store import JsonlPomdpTraceStore
from megatron.rl.compaction.pomdp.types import Action, Observation, new_id


def _make_step(step_id, run_id, full=None, compact=None, ratio=None, fallback=False, error=False, uncertainty=None):
    return ShadowStepMetrics(
        step_id=step_id,
        run_id=run_id,
        full_context_tokens=full,
        compact_context_tokens=compact,
        compression_ratio=ratio,
        belief_json_valid=True,
        belief_token_estimate=None,
        compactor_error=error,
        fallback_used=fallback,
        uncertainty_level=uncertainty,
    )


class TestShadowRunMetrics:
    def test_mean_compression_ratio_basic(self):
        run = ShadowRunMetrics(run_id="r1")
        run.steps = [
            _make_step(0, "r1", ratio=2.0),
            _make_step(1, "r1", ratio=4.0),
        ]
        assert run.mean_compression_ratio() == pytest.approx(3.0)

    def test_mean_compression_ratio_none_when_no_steps(self):
        run = ShadowRunMetrics(run_id="r1")
        assert run.mean_compression_ratio() is None

    def test_mean_compression_ratio_skips_none_values(self):
        run = ShadowRunMetrics(run_id="r1")
        run.steps = [
            _make_step(0, "r1", ratio=None),
            _make_step(1, "r1", ratio=3.0),
        ]
        assert run.mean_compression_ratio() == pytest.approx(3.0)

    def test_mean_compression_ratio_all_none(self):
        run = ShadowRunMetrics(run_id="r1")
        run.steps = [_make_step(0, "r1", ratio=None)]
        assert run.mean_compression_ratio() is None

    def test_to_dict_includes_all_fields(self):
        run = ShadowRunMetrics(run_id="r1")
        run.steps = [_make_step(0, "r1", ratio=2.0)]
        run.total_compactor_errors = 1
        run.total_fallbacks = 2
        d = run.to_dict()
        assert d["run_id"] == "r1"
        assert d["total_compactor_errors"] == 1
        assert d["total_fallbacks"] == 2
        assert d["mean_compression_ratio"] == pytest.approx(2.0)
        assert len(d["steps"]) == 1

    def test_fallback_counting_in_recorder(self, tmp_path):
        """Test that fallbacks are tracked in shadow_metrics when using record_step."""
        config = PomdpConfig(enabled=True, mode="shadow", trace_dir=str(tmp_path))
        store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
        builder = ContextBuilder(config)
        recorder = PomdpRolloutRecorder(config=config, store=store, context_builder=builder)

        run_id, belief = recorder.begin_run("Test task")
        obs = Observation(observation_id=new_id(), step_id=0, kind="tool_result", text="obs")
        action = Action(action_id=new_id(), step_id=0, kind="llm_message", text="act")
        next_obs = Observation(observation_id=new_id(), step_id=1, kind="tool_result", text="obs2")

        shadow = ShadowRunMetrics(run_id=run_id)
        recorder.record_step(
            run_id=run_id,
            step_id=0,
            belief=belief,
            current_observation=obs,
            action=action,
            next_observation=next_obs,
            reward=None,
            done=False,
            actor_context="ctx",
            actor_context_tokens=None,
            raw_history_text=None,
            shadow_metrics_ref=shadow,
        )
        assert len(shadow.steps) == 1
        # No fallback triggered in normal deterministic mode.
        assert shadow.total_fallbacks == 0

    def test_step_metrics_to_dict(self):
        step = _make_step(0, "r1", full=100, compact=50, ratio=2.0, uncertainty="medium")
        d = step.to_dict()
        assert d["step_id"] == 0
        assert d["full_context_tokens"] == 100
        assert d["compact_context_tokens"] == 50
        assert d["compression_ratio"] == pytest.approx(2.0)
        assert d["uncertainty_level"] == "medium"
