# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Infrastructure tests for text-level compaction through the POMDP recorder.

These exercise the generic enabling pieces (no paper-specific algorithm):
  - DigestAlgorithm           — simple model-free text baseline
  - async trigger hook        — model-driven "when to compact" on the async path
  - InferenceInterfaceClient  — adapter from the RL InferenceInterface to .complete()
  - "all angles" matrix smoke — one recorder drives every trigger x algorithm combo
"""

import asyncio

from megatron.rl.compaction.pomdp import (
    Action,
    Observation,
    PomdpConfig,
    ContextBuilder,
    PomdpRolloutRecorder,
    InferenceInterfaceClient,
    # triggers
    NeverTrigger,
    AlwaysTrigger,
    EveryNStepsTrigger,
    # algorithms
    PassthroughAlgorithm,
    WindowAlgorithm,
    DigestAlgorithm,
    DeterministicAlgorithm,
    TextBelief,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _act(step, text="do_thing"):
    return Action(action_id=f"a{step}", step_id=step, kind="tool_call", text=text)


def _obs(step, text="result"):
    return Observation(observation_id=f"o{step}", step_id=step, kind="tool_result", text=text)


class _DummyStore:
    """Never touched: recorder runs with config.enabled=False."""


def _recorder(trigger, algorithm):
    cfg = PomdpConfig(enabled=False)
    return PomdpRolloutRecorder(
        config=cfg,
        store=_DummyStore(),
        context_builder=ContextBuilder(cfg),
        trigger=trigger,
        algorithm=algorithm,
    )


# ---------------------------------------------------------------------------
# DigestAlgorithm
# ---------------------------------------------------------------------------

class TestDigestAlgorithm:
    def test_keeps_all_steps_breadth(self):
        algo = DigestAlgorithm(max_chars_per_step=200)
        b = algo.initialize("run", "solve the task")
        for i in range(1, 6):
            b = algo.update(b, _act(i, f"action_{i}"), _obs(i, f"obs_{i}"))
        text = b.to_context_str()
        # every step is represented (breadth), unlike a sliding window
        for i in range(1, 6):
            assert f"action_{i}" in text and f"obs_{i}" in text
        assert "solve the task" in text  # task header preserved

    def test_per_step_truncation(self):
        algo = DigestAlgorithm(max_chars_per_step=20)
        b = algo.initialize("run", "task")
        b = algo.update(b, _act(1, "x" * 100), _obs(1, "y" * 100))
        line = [l for l in b.text.splitlines() if l.startswith("[1]")][0]
        assert len(line) <= 20
        assert line.endswith("…")

    def test_returns_textbelief_and_bumps_step(self):
        algo = DigestAlgorithm()
        b = algo.initialize("run", "task")
        b2 = algo.update(b, _act(3), _obs(3))
        assert isinstance(b2, TextBelief)
        assert b2.step_id == 3

    def test_rejects_bad_budget(self):
        try:
            DigestAlgorithm(max_chars_per_step=0)
        except ValueError:
            return
        assert False, "expected ValueError"


# ---------------------------------------------------------------------------
# async trigger hook
# ---------------------------------------------------------------------------

class _RecordingAsyncTrigger:
    """Model-driven trigger: decision supplied by an (awaited) callback."""

    def __init__(self, decision: bool):
        self.decision = decision
        self.calls = 0

    def should_compact(self, step_id, belief, action, observation):  # sync fallback
        raise AssertionError("async path must use async_should_compact")

    async def async_should_compact(self, step_id, belief, action, observation):
        self.calls += 1
        return self.decision


def _run_step_async(recorder, belief, action, obs):
    return asyncio.run(
        recorder.record_step_async(
            run_id="run", step_id=obs.step_id, belief=belief,
            current_observation=None, action=action, next_observation=obs,
            reward=None, done=False, actor_context="ctx",
            actor_context_tokens=None, raw_history_text=None,
        )
    )


class TestAsyncTriggerHook:
    def test_async_trigger_fires_compaction(self):
        trig = _RecordingAsyncTrigger(decision=True)
        algo = DigestAlgorithm()
        rec = _recorder(trig, algo)
        _, belief = rec.begin_run("task")
        new_belief = _run_step_async(rec, belief, _act(1), _obs(1))
        assert trig.calls == 1
        assert new_belief.step_id == 1                 # algorithm ran
        assert new_belief is not belief

    def test_async_trigger_suppresses_compaction(self):
        trig = _RecordingAsyncTrigger(decision=False)
        rec = _recorder(trig, DigestAlgorithm())
        _, belief = rec.begin_run("task")
        new_belief = _run_step_async(rec, belief, _act(1), _obs(1))
        assert trig.calls == 1
        assert new_belief is belief                    # unchanged: no compaction

    def test_sync_trigger_still_works_on_async_path(self):
        # A plain sync trigger (no async_should_compact) must be awaited via fallback.
        rec = _recorder(AlwaysTrigger(), DigestAlgorithm())
        _, belief = rec.begin_run("task")
        new_belief = _run_step_async(rec, belief, _act(1), _obs(1))
        assert new_belief.step_id == 1


# ---------------------------------------------------------------------------
# InferenceInterfaceClient
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content):
        self.response = type("M", (), {"content": content})()


class _FakeInterface:
    """Minimal stand-in for InferenceInterface: echoes the prompt it received."""

    def __init__(self):
        self.last_prompt = None

    def prepare_request(self, prompt, generation_args):
        self.last_prompt = prompt
        return {"prompt": prompt, "gen": generation_args}

    async def agenerate(self, request):
        return _FakeResponse(f"echo:{request['prompt']}")


class TestInferenceInterfaceClient:
    def test_complete_roundtrip(self):
        iface = _FakeInterface()
        client = InferenceInterfaceClient(iface, generation_args={"max_tokens": 8})
        out = asyncio.run(client.complete("hello"))
        assert out == "echo:hello"
        assert iface.last_prompt == "hello"

    def test_system_prompt_prepended(self):
        iface = _FakeInterface()
        client = InferenceInterfaceClient(iface, generation_args=None, system_prompt="SYS")
        asyncio.run(client.complete("hi"))
        assert iface.last_prompt.startswith("SYS")
        assert "hi" in iface.last_prompt


# ---------------------------------------------------------------------------
# "all angles" matrix smoke — one recorder, every trigger x algorithm
# ---------------------------------------------------------------------------

class TestAllAnglesMatrix:
    def test_trigger_x_algorithm_combinations_run(self):
        triggers = [NeverTrigger(), AlwaysTrigger(), EveryNStepsTrigger(2)]
        algorithms = [
            PassthroughAlgorithm(),
            WindowAlgorithm(window_size=3),
            DigestAlgorithm(),
            DeterministicAlgorithm(),
        ]
        for trig in triggers:
            for algo in algorithms:
                rec = _recorder(trig, algo)
                _, belief = rec.begin_run("task")
                for i in range(1, 5):
                    belief = rec.record_step(
                        run_id="run", step_id=i, belief=belief,
                        current_observation=None, action=_act(i), next_observation=_obs(i),
                        reward=None, done=False, actor_context="ctx",
                        actor_context_tokens=None, raw_history_text=None,
                    )
                    # belief must always render to a context string
                    ctx, _, _ = rec.build_context_for_step(
                        run_id="run", system_prompt="sys", task_text="task",
                        belief=belief, recent_tail=[], current_observation=_obs(i),
                        available_tools=[],
                    )
                    assert isinstance(ctx, str) and len(ctx) > 0
