# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import pytest

from megatron.rl.compaction.pomdp.algorithm import LLMAlgorithm, validate_belief_dict
from megatron.rl.compaction.pomdp.types import Action, BeliefState, Observation, new_id


def _make_belief(run_id="run1", step_id=0, raw_trace_refs=None) -> BeliefState:
    return BeliefState(
        belief_id=f"belief_{run_id}_step{step_id}",
        run_id=run_id,
        step_id=step_id,
        task={"objective": "Test task"},
        uncertainty={"level": "medium"},
        raw_trace_refs=raw_trace_refs or [],
    )


def _make_action(step_id=0) -> Action:
    return Action(action_id=new_id(), step_id=step_id, kind="llm_message", text="test action")


def _make_obs(step_id=1) -> Observation:
    return Observation(observation_id=new_id(), step_id=step_id, kind="tool_result", text="test obs")


class _GoodLLMClient:
    def __init__(self, belief: BeliefState):
        self._belief = belief

    async def complete(self, prompt: str) -> str:
        return json.dumps(self._belief.to_dict())


class _BadJSONClient:
    async def complete(self, prompt: str) -> str:
        return "this is not json {{"


class _InvalidSchemaClient:
    async def complete(self, prompt: str) -> str:
        return json.dumps({"some_key": "missing required fields"})


class TestValidateBeliefDict:
    def _valid(self) -> dict:
        return {"belief_id": "b1", "run_id": "r1", "step_id": 0, "task": {"objective": "x"}}

    def test_valid_dict_returns_true(self):
        assert validate_belief_dict(self._valid()) is True

    def test_missing_belief_id_returns_false(self):
        d = self._valid(); del d["belief_id"]
        assert validate_belief_dict(d) is False

    def test_missing_run_id_returns_false(self):
        d = self._valid(); del d["run_id"]
        assert validate_belief_dict(d) is False

    def test_missing_step_id_returns_false(self):
        d = self._valid(); del d["step_id"]
        assert validate_belief_dict(d) is False

    def test_missing_task_returns_false(self):
        d = self._valid(); del d["task"]
        assert validate_belief_dict(d) is False

    def test_non_dict_returns_false(self):
        assert validate_belief_dict("not a dict") is False  # type: ignore[arg-type]
        assert validate_belief_dict(None) is False  # type: ignore[arg-type]
        assert validate_belief_dict([]) is False  # type: ignore[arg-type]

    def test_empty_belief_id_returns_false(self):
        d = self._valid(); d["belief_id"] = ""
        assert validate_belief_dict(d) is False

    def test_wrong_step_id_type_returns_false(self):
        d = self._valid(); d["step_id"] = "not_int"
        assert validate_belief_dict(d) is False

    def test_task_not_dict_returns_false(self):
        d = self._valid(); d["task"] = "not a dict"
        assert validate_belief_dict(d) is False


class TestLLMAlgorithm:
    @pytest.mark.asyncio
    async def test_valid_llm_response_returns_updated_belief(self):
        llm_belief = _make_belief(run_id="run1", step_id=1)
        algo = LLMAlgorithm(llm_client=_GoodLLMClient(llm_belief))
        prev = _make_belief(run_id="run1", step_id=0)
        result = await algo.async_update(prev, _make_action(), _make_obs(step_id=1))
        assert result.step_id == 1
        assert result.run_id == "run1"
        assert algo.fallback_count == 0

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_deterministic(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        prev = _make_belief()
        result = await algo.async_update(prev, _make_action(), _make_obs())
        assert algo.fallback_count == 1
        assert result.uncertainty.get("level") == "high"

    @pytest.mark.asyncio
    async def test_invalid_schema_falls_back_to_deterministic(self):
        algo = LLMAlgorithm(llm_client=_InvalidSchemaClient())
        prev = _make_belief()
        result = await algo.async_update(prev, _make_action(), _make_obs())
        assert algo.fallback_count == 1
        assert result.uncertainty.get("level") == "high"

    @pytest.mark.asyncio
    async def test_fallback_preserves_previous_raw_trace_refs(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        prev = _make_belief(raw_trace_refs=["ref-old"])
        result = await algo.async_update(prev, _make_action(), _make_obs())
        assert "ref-old" in result.raw_trace_refs

    @pytest.mark.asyncio
    async def test_fallback_sets_uncertainty_high_with_reason(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        result = await algo.async_update(_make_belief(), _make_action(), _make_obs())
        assert result.uncertainty.get("level") == "high"
        assert "LLM compaction failed" in str(result.uncertainty)

    @pytest.mark.asyncio
    async def test_fallback_count_increments_on_each_failure(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        prev = _make_belief()
        await algo.async_update(prev, _make_action(), _make_obs())
        await algo.async_update(prev, _make_action(), _make_obs())
        assert algo.fallback_count == 2

    def test_sync_update_is_deterministic(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        prev = _make_belief()
        result = algo.update(prev, _make_action(), _make_obs())
        assert result.step_id == _make_obs().step_id
        assert algo.fallback_count == 0
