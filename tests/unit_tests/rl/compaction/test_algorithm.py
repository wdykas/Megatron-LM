# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import pytest

from megatron.rl.compaction.pomdp.algorithm import (
    CompactionAlgorithm,
    DeterministicAlgorithm,
    LLMAlgorithm,
    PassthroughAlgorithm,
    TextBelief,
    WindowAlgorithm,
    apply_deterministic_reducers,
    validate_belief_dict,
)
from megatron.rl.compaction.pomdp.types import Action, BeliefState, Observation, new_id


def _obs(step_id: int = 1, kind: str = "tool_result", **kwargs) -> Observation:
    return Observation(observation_id=new_id(), step_id=step_id, kind=kind, **kwargs)


def _act(step_id: int = 0, kind: str = "tool_call") -> Action:
    return Action(action_id=new_id(), step_id=step_id, kind=kind)


# ---------------------------------------------------------------------------
# PassthroughAlgorithm
# ---------------------------------------------------------------------------

class TestPassthroughAlgorithm:
    def test_initialize_returns_belief_state(self):
        algo = PassthroughAlgorithm()
        belief = algo.initialize("run1", "Fix the bug.")
        assert isinstance(belief, BeliefState)
        assert belief.run_id == "run1"
        assert "Fix the bug." in belief.task["objective"]

    def test_update_bumps_step_id(self):
        algo = PassthroughAlgorithm()
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=3)
        updated = algo.update(belief, _act(), obs)
        assert updated.step_id == 3

    def test_update_does_not_change_task(self):
        algo = PassthroughAlgorithm()
        belief = algo.initialize("run1", "Important task.")
        obs = _obs(step_id=1)
        updated = algo.update(belief, _act(), obs)
        assert updated.task["objective"] == "Important task."

    @pytest.mark.asyncio
    async def test_async_update_uses_sync_path(self):
        algo = PassthroughAlgorithm()
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=2)
        updated = await algo.async_update(belief, _act(), obs)
        assert updated.step_id == 2

    def test_satisfies_protocol(self):
        assert isinstance(PassthroughAlgorithm(), CompactionAlgorithm)


# ---------------------------------------------------------------------------
# WindowAlgorithm
# ---------------------------------------------------------------------------

class TestWindowAlgorithm:
    def test_initialize_returns_text_belief(self):
        algo = WindowAlgorithm(window_size=5)
        belief = algo.initialize("run1", "Fix the bug.")
        assert isinstance(belief, TextBelief)
        assert "Fix the bug." in belief.to_context_str()

    def test_update_adds_events_to_window(self):
        algo = WindowAlgorithm(window_size=5)
        belief = algo.initialize("run1", "Task.")
        action = Action(action_id=new_id(), step_id=0, kind="tool_call", text="do something")
        obs = _obs(step_id=1, text="result of doing something")
        updated = algo.update(belief, action, obs)
        ctx = updated.to_context_str()
        assert "do something" in ctx
        assert "result of doing something" in ctx

    def test_window_capped_at_window_size(self):
        algo = WindowAlgorithm(window_size=2)
        belief = algo.initialize("run1", "Task.")
        for i in range(5):
            action = Action(action_id=new_id(), step_id=i, kind="tool_call", text=f"action_{i}")
            obs = _obs(step_id=i + 1, text=f"obs_{i}")
            belief = algo.update(belief, action, obs)
        ctx = belief.to_context_str()
        assert "action_3" in ctx
        assert "action_4" in ctx
        assert "action_0" not in ctx
        assert "action_1" not in ctx
        assert "action_2" not in ctx

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            WindowAlgorithm(window_size=0)

    def test_satisfies_protocol(self):
        assert isinstance(WindowAlgorithm(), CompactionAlgorithm)


# ---------------------------------------------------------------------------
# DeterministicAlgorithm
# ---------------------------------------------------------------------------

class TestDeterministicAlgorithm:
    def test_initialize_returns_belief_state(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Fix the parser bug.", task_metadata={
            "hard_constraints": ["no API changes"],
        })
        assert isinstance(belief, BeliefState)
        assert belief.task["hard_constraints"] == ["no API changes"]

    def test_update_increments_step_id(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=5)
        updated = algo.update(belief, _act(), obs)
        assert updated.step_id == 5

    def test_tool_failure_recorded(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.")
        action = Action(action_id=new_id(), step_id=0, kind="tool_call",
                        structured={"tool_name": "run_tests"})
        obs = _obs(step_id=1, text="failed", structured={"status": "error"})
        updated = algo.update(belief, action, obs)
        assert len(updated.action_history_digest.get("tried_and_failed", [])) == 1

    def test_file_edit_recorded(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.")
        action = Action(action_id=new_id(), step_id=0, kind="tool_call",
                        structured={"tool_name": "edit_file", "args": {"path": "foo.py"}})
        obs = _obs(step_id=1, structured={"status": "success"})
        updated = algo.update(belief, action, obs)
        files = updated.environment_state.get("files_changed", [])
        assert any(f["path"] == "foo.py" for f in files)

    @pytest.mark.asyncio
    async def test_async_update_uses_sync_path(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=1)
        updated = await algo.async_update(belief, _act(), obs)
        assert updated.step_id == 1

    def test_satisfies_protocol(self):
        assert isinstance(DeterministicAlgorithm(), CompactionAlgorithm)


# ---------------------------------------------------------------------------
# LLMAlgorithm
# ---------------------------------------------------------------------------

class _GoodLLMClient:
    def __init__(self, run_id: str, step_id: int):
        self._run_id = run_id
        self._step_id = step_id

    async def complete(self, prompt: str) -> str:
        d = {
            "belief_id": f"belief_{self._run_id}_step{self._step_id}",
            "run_id": self._run_id,
            "step_id": self._step_id,
            "schema_version": "v1",
            "compactor_version": "llm_v1",
            "task": {"objective": "Task."},
        }
        return json.dumps(d)


class _BadJSONClient:
    async def complete(self, prompt: str) -> str:
        return "not json {{"


class TestLLMAlgorithm:
    def test_initialize_returns_belief_state(self):
        algo = LLMAlgorithm(llm_client=_GoodLLMClient("run1", 0))
        belief = algo.initialize("run1", "Task.")
        assert isinstance(belief, BeliefState)

    def test_sync_update_is_deterministic(self):
        algo = LLMAlgorithm(llm_client=_GoodLLMClient("run1", 1))
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=1)
        updated = algo.update(belief, _act(), obs)
        assert updated.step_id == 1
        assert algo.fallback_count == 0

    @pytest.mark.asyncio
    async def test_async_update_uses_llm(self):
        algo = LLMAlgorithm(llm_client=_GoodLLMClient("run1", 1))
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=1)
        updated = await algo.async_update(belief, _act(), obs)
        assert updated.step_id == 1
        assert algo.fallback_count == 0

    @pytest.mark.asyncio
    async def test_async_update_fallback_on_bad_json(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=1)
        updated = await algo.async_update(belief, _act(), obs)
        assert algo.fallback_count == 1
        assert updated.uncertainty.get("level") == "high"

    @pytest.mark.asyncio
    async def test_fallback_count_accumulates(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        belief = algo.initialize("run1", "Task.")
        obs = _obs(step_id=1)
        await algo.async_update(belief, _act(), obs)
        await algo.async_update(belief, _act(), obs)
        assert algo.fallback_count == 2

    @pytest.mark.asyncio
    async def test_fallback_preserves_raw_trace_refs(self):
        algo = LLMAlgorithm(llm_client=_BadJSONClient())
        belief = algo.initialize("run1", "Task.")
        belief.raw_trace_refs = ["ref-old"]
        obs = _obs(step_id=1)
        updated = await algo.async_update(belief, _act(), obs)
        assert "ref-old" in updated.raw_trace_refs

    def test_satisfies_protocol(self):
        assert isinstance(LLMAlgorithm(llm_client=_BadJSONClient()), CompactionAlgorithm)
