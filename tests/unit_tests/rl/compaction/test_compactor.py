# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.rl.compaction.pomdp.algorithm import DeterministicAlgorithm, apply_deterministic_reducers
from megatron.rl.compaction.pomdp.types import Action, BeliefState, Observation, new_id


def _make_belief(run_id: str = "run1", step_id: int = 0) -> BeliefState:
    return BeliefState(
        belief_id=f"b_{run_id}_{step_id}",
        run_id=run_id,
        step_id=step_id,
        task={
            "objective": "Fix the parser bug.",
            "success_criteria": ["All tests pass"],
            "hard_constraints": ["Do not modify public API"],
        },
    )


def _tool_call(tool_name: str, path: str | None = None) -> Action:
    args = {"path": path} if path else {}
    return Action(action_id=new_id(), step_id=0, kind="tool_call",
                  structured={"tool_name": tool_name, "args": args})


def _tool_result(status: str, text: str = "error msg") -> Observation:
    return Observation(observation_id=new_id(), step_id=1, kind="tool_result",
                       text=text, structured={"status": status})


class TestDeterministicAlgorithmInitialize:
    def test_preserves_task_text(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Fix the parser bug.", task_metadata={
            "hard_constraints": ["no public API changes"],
        })
        assert "Fix the parser bug." in belief.task["objective"]

    def test_preserves_hard_constraints(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.", task_metadata={"hard_constraints": ["no API changes"]})
        assert "no API changes" in belief.task["hard_constraints"]

    def test_initial_uncertainty_is_set(self):
        algo = DeterministicAlgorithm()
        belief = algo.initialize("run1", "Task.")
        assert belief.uncertainty.get("level") in ("low", "medium", "high")


class TestDeterministicReducers:
    def test_tool_failure_recorded_in_tried_and_failed(self):
        belief = _make_belief()
        action = _tool_call("run_tests")
        obs = _tool_result(status="error", text="test failed")
        updated = apply_deterministic_reducers(belief, action, obs)
        assert len(updated.action_history_digest.get("tried_and_failed", [])) == 1
        entry = updated.action_history_digest["tried_and_failed"][0]
        assert entry["source_ref"] == obs.observation_id

    def test_file_edit_recorded_in_files_changed(self):
        belief = _make_belief()
        action = _tool_call("edit_file", path="src/parser.py")
        obs = _tool_result(status="success")
        updated = apply_deterministic_reducers(belief, action, obs)
        files_changed = updated.environment_state.get("files_changed", [])
        assert len(files_changed) == 1
        assert files_changed[0]["path"] == "src/parser.py"
        assert files_changed[0]["action_id"] == action.action_id

    def test_write_file_also_recorded(self):
        belief = _make_belief()
        action = _tool_call("write_file", path="src/new_file.py")
        obs = _tool_result(status="success")
        updated = apply_deterministic_reducers(belief, action, obs)
        assert any(f["path"] == "src/new_file.py" for f in updated.environment_state.get("files_changed", []))

    def test_test_result_failure_creates_open_question(self):
        belief = _make_belief()
        action = Action(action_id=new_id(), step_id=0, kind="tool_call")
        obs = Observation(observation_id=new_id(), step_id=1, kind="tool_result",
                          structured={"type": "test_result", "status": "failed", "command": "pytest tests/"})
        updated = apply_deterministic_reducers(belief, action, obs)
        assert len(updated.open_questions) == 1
        assert "test" in updated.open_questions[0]["question"].lower()

    def test_test_result_success_does_not_create_open_question(self):
        belief = _make_belief()
        action = Action(action_id=new_id(), step_id=0, kind="tool_call")
        obs = Observation(observation_id=new_id(), step_id=1, kind="tool_result",
                          structured={"type": "test_result", "status": "passed", "command": "pytest tests/"})
        updated = apply_deterministic_reducers(belief, action, obs)
        assert len(updated.open_questions) == 0

    def test_hard_constraints_preserved_after_update(self):
        belief = _make_belief()
        action = _tool_call("run_tests")
        obs = _tool_result(status="error")
        updated = apply_deterministic_reducers(belief, action, obs)
        assert updated.task["hard_constraints"] == ["Do not modify public API"]


class TestDeterministicAlgorithmUpdate:
    def test_update_increments_step(self):
        algo = DeterministicAlgorithm()
        belief = _make_belief(step_id=0)
        action = _tool_call("run_tests")
        obs = Observation(observation_id=new_id(), step_id=1, kind="tool_result",
                          text="ok", structured={"status": "success"})
        updated = algo.update(belief, action, obs)
        assert updated.step_id == 1

    def test_sync_update_is_always_deterministic(self):
        algo = DeterministicAlgorithm()
        belief = _make_belief()
        action = Action(action_id=new_id(), step_id=0, kind="other")
        obs = Observation(observation_id=new_id(), step_id=1, kind="other")
        updated = algo.update(belief, action, obs)
        assert updated.step_id == 1
