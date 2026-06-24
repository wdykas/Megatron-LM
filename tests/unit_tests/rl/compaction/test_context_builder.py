# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

from megatron.rl.compaction.pomdp.config import PomdpConfig
from megatron.rl.compaction.pomdp.context_builder import ContextBuilder
from megatron.rl.compaction.pomdp.types import Action, BeliefState, Observation, new_id


def _make_config(**kwargs) -> PomdpConfig:
    return PomdpConfig(**kwargs)


def _make_belief(task_text: str = "Fix the bug.") -> BeliefState:
    return BeliefState(
        belief_id="b_test_0",
        run_id="run_test",
        step_id=0,
        task={
            "objective": task_text,
            "success_criteria": ["All tests pass"],
            "hard_constraints": ["Do not break public API"],
            "user_preferences": [],
        },
    )


def _make_obs(step_id: int = 0, text: str = "file content here") -> Observation:
    return Observation(
        observation_id=new_id(),
        step_id=step_id,
        kind="tool_result",
        text=text,
    )


def _make_action(step_id: int = 0) -> Action:
    return Action(
        action_id=new_id(),
        step_id=step_id,
        kind="tool_call",
        text="edit_file src/parser.py",
    )


class TestContextBuilder:
    def test_includes_task(self):
        config = _make_config()
        builder = ContextBuilder(config)
        belief = _make_belief("Fix the parser bug.")
        context = builder.build_actor_context(
            system_prompt="You are a helpful assistant.",
            task_text="Fix the parser bug.",
            belief=belief,
            recent_tail=[],
            current_observation=None,
            available_tools=[],
        )
        assert "Fix the parser bug." in context

    def test_includes_belief_json(self):
        config = _make_config()
        builder = ContextBuilder(config)
        belief = _make_belief()
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[],
            current_observation=None,
            available_tools=[],
        )
        assert "COMPACT_BELIEF_STATE" in context
        assert "b_test_0" in context  # belief_id should appear in JSON

    def test_includes_recent_tail(self):
        config = _make_config(recent_tail_steps=2)
        builder = ContextBuilder(config)
        belief = _make_belief()
        obs = _make_obs(step_id=0, text="output of step 0")
        act = _make_action(step_id=0)
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[obs, act],
            current_observation=None,
            available_tools=[],
        )
        assert "output of step 0" in context
        assert "edit_file src/parser.py" in context

    def test_includes_current_observation(self):
        config = _make_config()
        builder = ContextBuilder(config)
        belief = _make_belief()
        current_obs = _make_obs(step_id=1, text="current file state")
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[],
            current_observation=current_obs,
            available_tools=[],
        )
        assert "current file state" in context

    def test_includes_tool_schema(self):
        config = _make_config()
        builder = ContextBuilder(config)
        belief = _make_belief()
        tools = [{"name": "edit_file", "description": "Edit a file"}]
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[],
            current_observation=None,
            available_tools=tools,
        )
        assert "edit_file" in context

    def test_truncates_large_observation(self):
        config = _make_config(max_observation_chars=50)
        builder = ContextBuilder(config)
        belief = _make_belief()
        long_obs = _make_obs(text="x" * 200)
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[],
            current_observation=long_obs,
            available_tools=[],
        )
        assert "truncated" in context

    def test_tail_only_uses_last_k_steps(self):
        config = _make_config(recent_tail_steps=1)
        builder = ContextBuilder(config)
        belief = _make_belief()
        obs0 = _make_obs(step_id=0, text="old observation")
        obs1 = _make_obs(step_id=1, text="new observation")
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[obs0, obs1],
            current_observation=None,
            available_tools=[],
        )
        assert "new observation" in context
        # The old observation should not appear since recent_tail_steps=1
        assert "old observation" not in context

    def test_belief_not_truncated_when_tail_is_large(self):
        """Belief JSON must remain intact even when observation is large."""
        config = _make_config(max_observation_chars=100)
        builder = ContextBuilder(config)
        belief = _make_belief("Important objective that must always be present.")
        big_obs = _make_obs(text="y" * 500)
        context = builder.build_actor_context(
            system_prompt="sys",
            task_text="task",
            belief=belief,
            recent_tail=[],
            current_observation=big_obs,
            available_tools=[],
        )
        assert "Important objective that must always be present." in context
