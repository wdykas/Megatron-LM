# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json

import pytest

from megatron.rl.compaction.pomdp.types import (
    Action,
    ArtifactRef,
    BeliefState,
    Observation,
    PomdpTransition,
    RolloutTrace,
    new_id,
)


def _make_belief(run_id: str = "run1", step_id: int = 0) -> BeliefState:
    return BeliefState(
        belief_id=f"belief_{run_id}_{step_id}",
        run_id=run_id,
        step_id=step_id,
        task={
            "objective": "Fix the parser bug.",
            "success_criteria": ["Tests pass"],
            "hard_constraints": ["Do not modify public API"],
            "user_preferences": [],
        },
    )


def _make_transition(run_id: str = "run1", step_id: int = 0) -> PomdpTransition:
    return PomdpTransition(
        transition_id=new_id(),
        run_id=run_id,
        step_id=step_id,
        belief_state_id=f"belief_{run_id}_{step_id}",
        observation_id="obs_0",
        action_id="act_0",
        next_observation_id="obs_1",
        next_belief_state_id=f"belief_{run_id}_{step_id + 1}",
    )


class TestBeliefStateSerialization:
    def test_round_trip(self):
        belief = _make_belief()
        d = belief.to_dict()
        restored = BeliefState.from_dict(d)
        assert restored.belief_id == belief.belief_id
        assert restored.run_id == belief.run_id
        assert restored.task["objective"] == belief.task["objective"]
        assert restored.task["hard_constraints"] == ["Do not modify public API"]

    def test_json_round_trip(self):
        belief = _make_belief()
        j = belief.to_json()
        parsed = json.loads(j)
        assert parsed["task"]["objective"] == "Fix the parser bug."
        restored = BeliefState.from_json(j)
        assert restored.task["objective"] == "Fix the parser bug."

    def test_preserves_hard_constraints(self):
        belief = _make_belief()
        d = belief.to_dict()
        restored = BeliefState.from_dict(d)
        assert restored.task["hard_constraints"] == ["Do not modify public API"]


class TestPomdpTransitionSerialization:
    def test_round_trip(self):
        t = _make_transition()
        d = t.to_dict()
        restored = PomdpTransition.from_dict(d)
        assert restored.transition_id == t.transition_id
        assert restored.run_id == t.run_id
        assert restored.step_id == t.step_id
        assert restored.belief_state_id == t.belief_state_id

    def test_with_artifact_refs(self):
        t = _make_transition()
        t.actor_context_ref = ArtifactRef(artifact_id="art_1", kind="raw_actor_context")
        d = t.to_dict()
        restored = PomdpTransition.from_dict(d)
        assert restored.actor_context_ref is not None
        assert restored.actor_context_ref.artifact_id == "art_1"
        assert restored.actor_context_ref.kind == "raw_actor_context"

    def test_defaults(self):
        t = _make_transition()
        assert t.done is False
        assert t.reward is None


class TestRolloutTrace:
    def test_serialization(self):
        trace = RolloutTrace(
            run_id="run1",
            task_id="task_abc",
            initial_task_text="Solve the problem.",
        )
        d = trace.to_dict()
        restored = RolloutTrace.from_dict(d)
        assert restored.run_id == "run1"
        assert restored.initial_task_text == "Solve the problem."
        assert restored.final_status == "unknown"


class TestArtifactRef:
    def test_round_trip(self):
        ref = ArtifactRef(artifact_id="art_1", kind="tool_output", size_bytes=42)
        d = ref.to_dict()
        restored = ArtifactRef.from_dict(d)
        assert restored.artifact_id == "art_1"
        assert restored.size_bytes == 42
