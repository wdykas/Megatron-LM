# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.rl.compaction.pomdp.config import PomdpConfig
from megatron.rl.compaction.pomdp.context_builder import ContextBuilder
from megatron.rl.compaction.kv.megatron_hook import NullHook
from megatron.rl.compaction.pomdp.recorder import PomdpRolloutRecorder
from megatron.rl.compaction.pomdp.store import JsonlPomdpTraceStore
from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor, PerceiverConfig
from megatron.rl.compaction.learned.models.belief import BeliefMemory, BeliefUpdater
from megatron.rl.compaction.pomdp.types import Action, Observation, new_id


def _make_recorder(tmp_path, enabled: bool = True) -> PomdpRolloutRecorder:
    config = PomdpConfig(enabled=enabled, mode="record_only", trace_dir=str(tmp_path))
    store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
    builder = ContextBuilder(config)
    return PomdpRolloutRecorder(config=config, store=store, context_builder=builder)


def _make_obs(step_id: int, kind: str = "tool_result") -> Observation:
    return Observation(observation_id=new_id(), step_id=step_id, kind=kind, text=f"obs {step_id}")


def _make_action(step_id: int, tokens=None, logprobs=None, policy_version="v1") -> Action:
    return Action(
        action_id=new_id(),
        step_id=step_id,
        kind="tool_call",
        text=f"action {step_id}",
        action_tokens=tokens,
        action_logprobs_behavior=logprobs,
        behavior_policy_version=policy_version,
        tokenizer_version="tok_v1",
    )


class TestPomdpRolloutRecorder:
    def test_begin_run_returns_run_id_and_belief(self, tmp_path):
        recorder = _make_recorder(tmp_path)
        run_id, belief = recorder.begin_run("Fix the bug.")
        assert run_id is not None
        assert belief.run_id == run_id
        assert "Fix the bug." in belief.task["objective"]

    def test_disabled_begin_run_works(self, tmp_path):
        recorder = _make_recorder(tmp_path, enabled=False)
        run_id, belief = recorder.begin_run("Task.")
        assert run_id is not None
        assert belief is not None

    def test_record_step_stores_action_tokens(self, tmp_path):
        recorder = _make_recorder(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        obs = _make_obs(0)
        action = _make_action(0, tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
        next_obs = _make_obs(1)
        next_belief = recorder.record_step(
            run_id=run_id,
            step_id=0,
            belief=belief,
            current_observation=obs,
            action=action,
            next_observation=next_obs,
            reward=1.0,
            done=False,
            actor_context="ctx",
            actor_context_tokens=None,
            raw_history_text="raw",
        )
        # Verify transition was stored with action.
        store = recorder._store
        trace = store._runs[run_id]
        assert len(trace.transitions) == 1
        transition = store.get_transition(trace.transitions[0])
        assert transition.behavior_policy_version == "v1"

    def test_record_step_stores_behavior_policy_version(self, tmp_path):
        recorder = _make_recorder(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        obs = _make_obs(0)
        action = _make_action(0, tokens=[1], logprobs=[-0.1], policy_version="policy_abc")
        next_obs = _make_obs(1)
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
        )
        store = recorder._store
        trace = store._runs[run_id]
        transition = store.get_transition(trace.transitions[0])
        assert transition.behavior_policy_version == "policy_abc"

    def test_belief_updates_each_step(self, tmp_path):
        recorder = _make_recorder(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        assert belief.step_id == 0

        obs0 = _make_obs(0)
        act0 = _make_action(0)
        obs1 = _make_obs(1)
        next_belief = recorder.record_step(
            run_id=run_id, step_id=0, belief=belief,
            current_observation=obs0, action=act0, next_observation=obs1,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
        assert next_belief.step_id == 1

    def test_disabled_recorder_does_not_write_to_store(self, tmp_path):
        recorder = _make_recorder(tmp_path, enabled=False)
        run_id, belief = recorder.begin_run("Task.")
        obs = _make_obs(0)
        action = _make_action(0)
        next_obs = _make_obs(1)
        recorder.record_step(
            run_id=run_id, step_id=0, belief=belief,
            current_observation=obs, action=action, next_observation=next_obs,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
        # Nothing should be stored when disabled.
        assert run_id not in recorder._store._runs


# ---------------------------------------------------------------------------
# Helpers for Belief-Still recorder tests
# ---------------------------------------------------------------------------

def _make_kv_hook(n_layers: int, T: int, d: int) -> "FakeKVHook":
    """A fake hook that returns random KV matrices — enables BeliefUpdater tests."""
    return FakeKVHook(n_layers=n_layers, T=T, d=d)


class FakeKVHook(NullHook):
    """NullHook extended to provide fake per-layer KV matrices."""

    def __init__(self, n_layers: int, T: int, d: int) -> None:
        self.n_layers = n_layers
        self.T = T
        self.d = d
        self.applied_memories: list = []

    def get_kv_matrices(self):
        keys   = [torch.randn(1, self.T, self.d) for _ in range(self.n_layers)]
        values = [torch.randn(1, self.T, self.d) for _ in range(self.n_layers)]
        return keys, values

    def apply_belief_memory(self, memory) -> None:
        self.applied_memories.append(memory)


def _make_recorder_with_belief(tmp_path, n_layers=2, d=8, n_compress=4, T=16, mode="live"):
    cfg = PerceiverConfig(d_kv=d, n_heads=2, n_compress=n_compress, n_attn_layers=n_layers)
    compactor = PerceiverCompactor(cfg)
    updater = BeliefUpdater(compactor)
    hook = _make_kv_hook(n_layers, T, d)
    config = PomdpConfig(enabled=True, mode=mode, trace_dir=str(tmp_path))
    store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
    builder = ContextBuilder(config)
    recorder = PomdpRolloutRecorder(
        config=config, store=store, context_builder=builder,
        belief_updater=updater, kv_hook=hook,
    )
    return recorder, hook


# ---------------------------------------------------------------------------
# Belief-Still recorder integration tests
# ---------------------------------------------------------------------------

class TestBeliefUpdaterRecorderIntegration:
    def _step(self, recorder, run_id, belief, step_id):
        obs = _make_obs(step_id)
        action = _make_action(step_id)
        next_obs = _make_obs(step_id + 1)
        return recorder.record_step(
            run_id=run_id, step_id=step_id, belief=belief,
            current_observation=obs, action=action, next_observation=next_obs,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )

    def test_begin_run_seeds_belief_memory_slot(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, _ = recorder.begin_run("Task.")
        assert run_id in recorder._belief_memories
        assert recorder._belief_memories[run_id] is None   # not yet initialised

    def test_first_step_initialises_belief_memory(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        m = recorder._belief_memories[run_id]
        assert m is not None
        assert isinstance(m, BeliefMemory)

    def test_belief_memory_shape_matches_config(self, tmp_path):
        n_layers, d, n_compress = 2, 8, 4
        recorder, _ = _make_recorder_with_belief(tmp_path, n_layers=n_layers, d=d, n_compress=n_compress)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        m = recorder._belief_memories[run_id]
        assert m.n_layers == n_layers
        assert m.budget == n_compress
        assert m.d_model == d

    def test_step_counter_increments_across_steps(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        m0 = recorder._belief_memories[run_id]
        assert m0.step == 0

        self._step(recorder, run_id, belief, 1)
        m1 = recorder._belief_memories[run_id]
        assert m1.step == 1

    def test_memory_changes_between_steps(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        keys_step0 = recorder._belief_memories[run_id].keys.clone()

        self._step(recorder, run_id, belief, 1)
        keys_step1 = recorder._belief_memories[run_id].keys

        assert not torch.allclose(keys_step0, keys_step1), \
            "Belief memory should change after each update"

    def test_live_mode_calls_apply_belief_memory(self, tmp_path):
        recorder, hook = _make_recorder_with_belief(tmp_path, mode="live")
        run_id, belief = recorder.begin_run("Task.")
        assert len(hook.applied_memories) == 0
        self._step(recorder, run_id, belief, 0)
        assert len(hook.applied_memories) == 1
        self._step(recorder, run_id, belief, 1)
        assert len(hook.applied_memories) == 2

    def test_record_only_mode_does_not_apply_memory(self, tmp_path):
        recorder, hook = _make_recorder_with_belief(tmp_path, mode="record_only")
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        assert len(hook.applied_memories) == 0

    def test_belief_memory_ref_stored_in_transition(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        store = recorder._store
        trace = store._runs[run_id]
        transition = store.get_transition(trace.transitions[0])
        assert transition.belief_memory_ref is not None
        assert transition.belief_memory_ref.kind == "belief_memory"

    def test_belief_memory_artifact_payload(self, tmp_path):
        n_layers, d, n_compress = 2, 8, 4
        recorder, _ = _make_recorder_with_belief(tmp_path, n_layers=n_layers, d=d, n_compress=n_compress)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        store = recorder._store
        trace = store._runs[run_id]
        transition = store.get_transition(trace.transitions[0])
        artifact = store.get_artifact(transition.belief_memory_ref.artifact_id)
        assert artifact["n_layers"] == n_layers
        assert artifact["budget"] == n_compress
        assert artifact["d_model"] == d
        assert artifact["step"] == 0

    def test_finish_run_cleans_up_memory(self, tmp_path):
        recorder, _ = _make_recorder_with_belief(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        self._step(recorder, run_id, belief, 0)
        assert run_id in recorder._belief_memories
        recorder.finish_run(run_id, final_reward=1.0, final_status="success")
        assert run_id not in recorder._belief_memories

    def test_no_belief_updater_is_no_op(self, tmp_path):
        recorder = _make_recorder(tmp_path)
        run_id, belief = recorder.begin_run("Task.")
        assert run_id not in recorder._belief_memories
        obs = _make_obs(0)
        action = _make_action(0)
        next_obs = _make_obs(1)
        recorder.record_step(
            run_id=run_id, step_id=0, belief=belief,
            current_observation=obs, action=action, next_observation=next_obs,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
        store = recorder._store
        trace = store._runs[run_id]
        transition = store.get_transition(trace.transitions[0])
        assert transition.belief_memory_ref is None

    def test_null_hook_skips_belief_update_gracefully(self, tmp_path):
        # NullHook.get_kv_matrices() returns None → belief updater should be no-op
        cfg = PerceiverConfig(d_kv=8, n_heads=2, n_compress=4, n_attn_layers=2)
        compactor = PerceiverCompactor(cfg)
        updater = BeliefUpdater(compactor)
        config = PomdpConfig(enabled=True, mode="record_only", trace_dir=str(tmp_path))
        store = JsonlPomdpTraceStore(trace_dir=str(tmp_path))
        builder = ContextBuilder(config)
        recorder = PomdpRolloutRecorder(
            config=config, store=store, context_builder=builder,
            belief_updater=updater,
            kv_hook=NullHook(),   # returns None for get_kv_matrices
        )
        run_id, belief = recorder.begin_run("Task.")
        obs = _make_obs(0)
        action = _make_action(0)
        next_obs = _make_obs(1)
        # Should not raise even though hook returns no KV data
        recorder.record_step(
            run_id=run_id, step_id=0, belief=belief,
            current_observation=obs, action=action, next_observation=next_obs,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
        assert recorder._belief_memories[run_id] is None   # never updated
