# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import tempfile

import pytest

from megatron.rl.compaction.pomdp.types import (
    Action,
    BeliefState,
    Observation,
    PomdpTransition,
    RolloutTrace,
    new_id,
)
from megatron.rl.compaction.pomdp.store import JsonlPomdpTraceStore


def _make_store(tmp_path: str) -> JsonlPomdpTraceStore:
    return JsonlPomdpTraceStore(trace_dir=tmp_path)


def _make_belief(run_id: str, step_id: int = 0) -> BeliefState:
    return BeliefState(
        belief_id=f"b_{run_id}_{step_id}",
        run_id=run_id,
        step_id=step_id,
    )


class TestJsonlPomdpTraceStore:
    def test_create_run_and_retrieve(self, tmp_path):
        store = _make_store(str(tmp_path))
        trace = RolloutTrace(run_id="run1", task_id=None, initial_task_text="task")
        store.create_run(trace)
        assert "run1" in store._runs
        run_dir = tmp_path / "run1"
        assert run_dir.exists()
        assert (run_dir / "run.jsonl").exists()

    def test_put_and_get_belief(self, tmp_path):
        store = _make_store(str(tmp_path))
        run_id = "run_belief"
        trace = RolloutTrace(run_id=run_id, task_id=None, initial_task_text="t")
        store.create_run(trace)
        belief = _make_belief(run_id)
        store.put_belief_state(belief)
        fetched = store.get_belief_state(belief.belief_id)
        assert fetched.belief_id == belief.belief_id

    def test_put_and_get_artifact_str(self, tmp_path):
        store = _make_store(str(tmp_path))
        ref = store.put_artifact(kind="raw_observation", payload="hello world")
        fetched = store.get_artifact(ref.artifact_id)
        assert fetched == "hello world"

    def test_put_and_get_artifact_dict(self, tmp_path):
        store = _make_store(str(tmp_path))
        ref = store.put_artifact(kind="tool_output", payload={"key": "value"})
        fetched = store.get_artifact(ref.artifact_id)
        assert isinstance(fetched, dict)
        assert fetched["key"] == "value"

    def test_put_and_get_artifact_bytes(self, tmp_path):
        store = _make_store(str(tmp_path))
        ref = store.put_artifact(kind="other", payload=b"raw bytes")
        fetched = store.get_artifact(ref.artifact_id)
        assert "raw bytes" in fetched

    def test_append_transition_to_run(self, tmp_path):
        store = _make_store(str(tmp_path))
        trace = RolloutTrace(run_id="run2", task_id=None, initial_task_text="t")
        store.create_run(trace)
        store.append_transition_to_run("run2", "tid_1")
        store.append_transition_to_run("run2", "tid_2")
        assert store._runs["run2"].transitions == ["tid_1", "tid_2"]

    def test_finish_run(self, tmp_path):
        store = _make_store(str(tmp_path))
        trace = RolloutTrace(run_id="run3", task_id=None, initial_task_text="t")
        store.create_run(trace)
        store.finish_run("run3", final_reward=1.0, final_status="success")
        assert store._runs["run3"].final_reward == 1.0
        assert store._runs["run3"].final_status == "success"

    def test_put_transition(self, tmp_path):
        store = _make_store(str(tmp_path))
        run_id = "run4"
        trace = RolloutTrace(run_id=run_id, task_id=None, initial_task_text="t")
        store.create_run(trace)
        t = PomdpTransition(
            transition_id=new_id(),
            run_id=run_id,
            step_id=0,
            belief_state_id="b_0",
            observation_id=None,
            action_id="act_0",
            next_observation_id=None,
            next_belief_state_id="b_1",
        )
        store.put_transition(t)
        fetched = store.get_transition(t.transition_id)
        assert fetched.transition_id == t.transition_id

    def test_artifact_sha256_computed(self, tmp_path):
        store = _make_store(str(tmp_path))
        ref = store.put_artifact(kind="other", payload="test")
        assert ref.sha256 is not None and len(ref.sha256) == 64
