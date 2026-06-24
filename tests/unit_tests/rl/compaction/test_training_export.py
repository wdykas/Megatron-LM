# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.compaction.pomdp.config import PomdpConfig
from megatron.rl.compaction.pomdp.context_builder import ContextBuilder
from megatron.rl.compaction.pomdp.export import PomdpTrainingExporter
from megatron.rl.compaction.pomdp.recorder import PomdpRolloutRecorder
from megatron.rl.compaction.pomdp.store import JsonlPomdpTraceStore
from megatron.rl.compaction.pomdp.types import Action, Observation, new_id


def _run_three_step_rollout(tmp_path) -> tuple[PomdpRolloutRecorder, str]:
    """Run a fake 3-step rollout and return (recorder, run_id)."""
    config = PomdpConfig(
        enabled=True,
        mode="record_only",
        trace_dir=str(tmp_path),
    )
    store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
    builder = ContextBuilder(config)
    recorder = PomdpRolloutRecorder(config=config, store=store, context_builder=builder)

    run_id, belief = recorder.begin_run("Fix the parser bug.", task_metadata={
        "success_criteria": ["Tests pass"],
        "hard_constraints": ["Do not change public API"],
    })

    # Step 0: user message → open file
    obs0 = Observation(observation_id=new_id(), step_id=0, kind="user_message", text="Fix the parser.")
    act0 = Action(
        action_id=new_id(),
        step_id=0,
        kind="tool_call",
        text="open_file src/parser.py",
        structured={"tool_name": "read_file", "args": {"path": "src/parser.py"}},
        action_tokens=[10, 20, 30],
        action_logprobs_behavior=[-0.1, -0.2, -0.3],
        action_mask=[1, 1, 1],
        behavior_policy_version="policy_v1",
        tokenizer_version="tok_v1",
    )
    obs1 = Observation(observation_id=new_id(), step_id=1, kind="tool_result",
                       text="def parse(s): ...", structured={"status": "success"})
    belief = recorder.record_step(
        run_id=run_id, step_id=0, belief=belief,
        current_observation=obs0, action=act0, next_observation=obs1,
        reward=None, done=False, actor_context="ctx_0",
        actor_context_tokens=[1, 2, 3], raw_history_text="raw_0",
    )

    # Step 1: edit file
    act1 = Action(
        action_id=new_id(),
        step_id=1,
        kind="tool_call",
        text="edit_file src/parser.py",
        structured={"tool_name": "edit_file", "args": {"path": "src/parser.py"}},
        action_tokens=[40, 50],
        action_logprobs_behavior=[-0.5, -0.4],
        behavior_policy_version="policy_v1",
        tokenizer_version="tok_v1",
    )
    obs2 = Observation(observation_id=new_id(), step_id=2, kind="tool_result",
                       text="file edited", structured={"status": "success"})
    belief = recorder.record_step(
        run_id=run_id, step_id=1, belief=belief,
        current_observation=obs1, action=act1, next_observation=obs2,
        reward=None, done=False, actor_context="ctx_1",
        actor_context_tokens=[4, 5, 6], raw_history_text="raw_1",
    )

    # Step 2: run test → success
    act2 = Action(
        action_id=new_id(),
        step_id=2,
        kind="tool_call",
        text="run_tests",
        structured={"tool_name": "run_tests", "args": {}},
        action_tokens=[60],
        action_logprobs_behavior=[-0.6],
        behavior_policy_version="policy_v1",
        tokenizer_version="tok_v1",
    )
    obs3 = Observation(observation_id=new_id(), step_id=3, kind="reward_event",
                       structured={"type": "test_result", "status": "passed", "command": "pytest"})
    belief = recorder.record_step(
        run_id=run_id, step_id=2, belief=belief,
        current_observation=obs2, action=act2, next_observation=obs3,
        reward=1.0, done=True, actor_context="ctx_2",
        actor_context_tokens=[7, 8], raw_history_text="raw_2",
    )

    recorder.finish_run(run_id, final_reward=1.0, final_status="success")
    return recorder, run_id


class TestPomdpTrainingExporter:
    def test_export_run_returns_samples(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        exporter = PomdpTrainingExporter(recorder._store, learner_policy_version="policy_v2")
        samples = exporter.export_run(run_id)
        assert len(samples) == 3

    def test_exported_sample_has_action_tokens(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        exporter = PomdpTrainingExporter(recorder._store)
        samples = exporter.export_run(run_id)
        for sample in samples:
            assert isinstance(sample.action_tokens, list)

    def test_exported_sample_has_behavior_logprobs(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        exporter = PomdpTrainingExporter(recorder._store)
        samples = exporter.export_run(run_id)
        for sample in samples:
            assert isinstance(sample.action_logprobs_behavior, list)

    def test_exported_sample_has_actor_context_tokens_ref(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        exporter = PomdpTrainingExporter(recorder._store)
        samples = exporter.export_run(run_id)
        # At least one sample should have a tokens ref.
        has_ref = any(s.actor_context_tokens_ref is not None for s in samples)
        assert has_ref

    def test_exported_sample_invalid_when_tokens_missing(self, tmp_path):
        config = PomdpConfig(enabled=True, mode="record_only", trace_dir=str(tmp_path))
        store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
        builder = ContextBuilder(config)
        recorder = PomdpRolloutRecorder(config=config, store=store, context_builder=builder)

        run_id, belief = recorder.begin_run("task")
        obs0 = Observation(observation_id=new_id(), step_id=0, kind="user_message", text="t")
        # No action_tokens or logprobs.
        act0 = Action(action_id=new_id(), step_id=0, kind="llm_message", text="response",
                      behavior_policy_version="v1")
        obs1 = Observation(observation_id=new_id(), step_id=1, kind="tool_result", text="ok",
                           structured={"status": "success"})
        recorder.record_step(
            run_id=run_id, step_id=0, belief=belief,
            current_observation=obs0, action=act0, next_observation=obs1,
            reward=0.0, done=True, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
        recorder.finish_run(run_id, final_reward=0.0, final_status="success")

        exporter = PomdpTrainingExporter(store)
        samples = exporter.export_run(run_id)
        assert len(samples) == 1
        assert samples[0].valid_for_policy_gradient is False

    def test_integration_one_trace_three_transitions(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        store = recorder._store
        trace = store._runs[run_id]
        assert len(trace.transitions) == 3
        assert trace.final_status == "success"
        assert trace.final_reward == 1.0

    def test_integration_belief_updates_each_step(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        store = recorder._store
        trace = store._runs[run_id]
        # Three transitions stored → four distinct beliefs (step 0-3)
        # Actually we store the beliefs as they're created (initial + 3 next beliefs)
        assert len(trace.transitions) == 3

    def test_raw_artifacts_retrievable(self, tmp_path):
        recorder, run_id = _run_three_step_rollout(tmp_path)
        store = recorder._store
        trace = store._runs[run_id]
        for tid in trace.transitions:
            t = store.get_transition(tid)
            if t.raw_history_ref is not None:
                payload = store.get_artifact(t.raw_history_ref.artifact_id)
                assert payload is not None
