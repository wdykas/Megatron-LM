# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pluggable compaction algorithms.

Each algorithm controls how a belief is initialised at the start of a rollout
and how it is updated after each (action, observation) pair.

Baselines
---------
PassthroughAlgorithm   – copy-forward; no compression.
WindowAlgorithm        – sliding window of raw events as plain text.
DeterministicAlgorithm – rule-based structured BeliefState updates.
LLMAlgorithm           – prompted LLM update with deterministic fallback.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .prompts import BELIEF_UPDATE_PROMPT_TEMPLATE
from .types import Action, BeliefState, Observation, new_id

logger = logging.getLogger(__name__)

_REQUIRED_BELIEF_KEYS = {"belief_id", "run_id", "step_id", "task"}


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CompactionAlgorithm(Protocol):
    """Pluggable compaction algorithm.

    ``initialize`` is called once per rollout to create the initial belief.
    ``update`` / ``async_update`` are called after each step where the trigger fires.
    """

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> Any: ...
    def update(self, belief: Any, action: Action, observation: Observation) -> Any: ...
    async def async_update(self, belief: Any, action: Action, observation: Observation) -> Any: ...


class _BaseAlgorithm:
    """Provides a default async_update that delegates to sync update."""

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def update(self, belief: Any, action: Action, observation: Observation) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def async_update(self, belief: Any, action: Action, observation: Observation) -> Any:
        return self.update(belief, action, observation)


# ---------------------------------------------------------------------------
# TextBelief — flat text representation for window-based algorithms
# ---------------------------------------------------------------------------

@dataclass
class TextBelief:
    """Belief as a plain text string (used by WindowAlgorithm)."""

    belief_id: str
    run_id: str
    step_id: int
    text: str

    def to_context_str(self) -> str:
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {"belief_id": self.belief_id, "run_id": self.run_id,
                "step_id": self.step_id, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TextBelief":
        return cls(belief_id=d["belief_id"], run_id=d["run_id"],
                   step_id=d["step_id"], text=d["text"])


# ---------------------------------------------------------------------------
# Shared helpers for text (TextBelief) algorithms
# ---------------------------------------------------------------------------

def _field_text(text: str | None, structured: dict[str, Any]) -> str:
    """Render an Action/Observation payload: explicit text, else its JSON."""
    return text or json.dumps(structured)


def _preserve_header(belief: Any, marker: str) -> str:
    """Return the task-header portion of a TextBelief: the lines before the
    first one starting with ``marker`` (e.g. ``[HISTORY`` / ``[DIGEST``)."""
    head = ""
    for line in getattr(belief, "text", "").split("\n"):
        if line.startswith(marker):
            break
        head += line + "\n"
    return head.strip()


# ---------------------------------------------------------------------------
# PassthroughAlgorithm
# ---------------------------------------------------------------------------

class PassthroughAlgorithm(_BaseAlgorithm):
    """No compaction: carry the belief forward with only the step_id bumped.

    Useful as a baseline or when paired with NeverTrigger in record-only mode.
    """

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> BeliefState:
        meta = kwargs.get("task_metadata") or {}
        return BeliefState(
            belief_id=f"belief_{run_id}_step0",
            run_id=run_id,
            step_id=0,
            task={
                "objective": task_text,
                "success_criteria": meta.get("success_criteria", []),
                "hard_constraints": meta.get("hard_constraints", []),
            },
            uncertainty={"level": "medium"},
            created_at_ms=time.time_ns() // 1_000_000,
        )

    def update(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        updated = copy.deepcopy(belief)
        updated.step_id = observation.step_id
        updated.belief_id = f"belief_{belief.run_id}_step{observation.step_id}"
        updated.created_at_ms = time.time_ns() // 1_000_000
        return updated


# ---------------------------------------------------------------------------
# WindowAlgorithm
# ---------------------------------------------------------------------------

class WindowAlgorithm(_BaseAlgorithm):
    """Sliding-window compaction: keep the last k (action, observation) pairs as text.

    Returns a TextBelief — the simplest baseline that actually reduces context length.
    No model call required.
    """

    def __init__(self, window_size: int = 10) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._windows: dict[str, list[dict[str, str]]] = {}

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> TextBelief:
        self._windows[run_id] = []
        return TextBelief(
            belief_id=f"belief_{run_id}_step0",
            run_id=run_id,
            step_id=0,
            text=f"[TASK]\n{task_text}\n\n[HISTORY]\n(none yet)",
        )

    def update(self, belief: TextBelief, action: Action, observation: Observation) -> TextBelief:
        run_id = belief.run_id
        window = self._windows.setdefault(run_id, [])
        window.append({
            "action": _field_text(action.text, action.structured),
            "obs": _field_text(observation.text, observation.structured),
        })
        if len(window) > self._window_size:
            window[:] = window[-self._window_size:]

        task_text = _preserve_header(belief, "[HISTORY")

        parts = [
            f"Step {i}:\n  Action: {e['action']}\n  Obs: {e['obs']}"
            for i, e in enumerate(window)
        ]
        history = "\n\n".join(parts) if parts else "(none)"
        return TextBelief(
            belief_id=f"belief_{run_id}_step{observation.step_id}",
            run_id=run_id,
            step_id=observation.step_id,
            text=f"{task_text}\n\n[HISTORY (last {self._window_size})]\n{history}",
        )


# ---------------------------------------------------------------------------
# DigestAlgorithm
# ---------------------------------------------------------------------------

class DigestAlgorithm(_BaseAlgorithm):
    """Model-free text baseline: task header + a one-line digest of every step.

    Each step is compressed to a single line ``[step] action -> obs`` truncated
    to ``max_chars_per_step``. Complements ``WindowAlgorithm``:

    * ``WindowAlgorithm`` keeps the last *k* steps in full  → favours *recency*.
    * ``DigestAlgorithm`` keeps *all* steps, one short line each → favours *breadth*.

    Both reduce context with no model call — useful trivial lower bounds when
    studying text-level compaction. Returns a ``TextBelief``.
    """

    def __init__(self, max_chars_per_step: int = 120) -> None:
        if max_chars_per_step < 1:
            raise ValueError(f"max_chars_per_step must be >= 1, got {max_chars_per_step}")
        self._max = max_chars_per_step
        self._digests: dict[str, list[str]] = {}

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> TextBelief:
        self._digests[run_id] = []
        return TextBelief(
            belief_id=f"belief_{run_id}_step0",
            run_id=run_id,
            step_id=0,
            text=f"[TASK]\n{task_text}\n\n[DIGEST]\n(none yet)",
        )

    def update(self, belief: TextBelief, action: Action, observation: Observation) -> TextBelief:
        run_id = belief.run_id
        digests = self._digests.setdefault(run_id, [])
        act = _field_text(action.text, action.structured).replace("\n", " ")
        obs = _field_text(observation.text, observation.structured).replace("\n", " ")
        line = f"[{observation.step_id}] {act} -> {obs}"
        if len(line) > self._max:
            line = line[: self._max - 1] + "…"
        digests.append(line)

        task_text = _preserve_header(belief, "[DIGEST")

        body = "\n".join(digests) if digests else "(none)"
        return TextBelief(
            belief_id=f"belief_{run_id}_step{observation.step_id}",
            run_id=run_id,
            step_id=observation.step_id,
            text=f"{task_text}\n\n[DIGEST]\n{body}",
        )


# ---------------------------------------------------------------------------
# DeterministicAlgorithm
# ---------------------------------------------------------------------------

class DeterministicAlgorithm(_BaseAlgorithm):
    """Rule-based structured belief updates — no model call, no randomness.

    Pattern-matches on action/observation kind to populate BeliefState fields
    (files_changed, tried_and_failed, open_questions, etc.).
    """

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> BeliefState:
        meta = kwargs.get("task_metadata") or {}
        return BeliefState(
            belief_id=f"belief_{run_id}_step0",
            run_id=run_id,
            step_id=0,
            task={
                "objective": task_text,
                "success_criteria": meta.get("success_criteria", []),
                "hard_constraints": meta.get("hard_constraints", []),
            },
            uncertainty={"level": "medium"},
            created_at_ms=time.time_ns() // 1_000_000,
        )

    def update(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        updated = copy.deepcopy(belief)
        updated.belief_id = f"belief_{belief.run_id}_step{observation.step_id}"
        updated.step_id = observation.step_id
        updated.created_at_ms = time.time_ns() // 1_000_000
        try:
            updated = apply_deterministic_reducers(updated, action, observation)
        except Exception as exc:
            logger.warning("Deterministic reducer failed: %s", exc)
            updated.uncertainty = {"level": "high", "reason": str(exc)}
        return updated


# ---------------------------------------------------------------------------
# LLMAlgorithm
# ---------------------------------------------------------------------------

class LLMAlgorithm(_BaseAlgorithm):
    """LLM-prompted belief update with automatic deterministic fallback.

    async_update calls the LLM; sync update always uses the deterministic path
    so existing synchronous code works without change.
    """

    def __init__(self, llm_client: Any, belief_token_budget: int = 3000) -> None:
        self._llm_client = llm_client
        self._belief_token_budget = belief_token_budget
        self._fallback = DeterministicAlgorithm()
        self.fallback_count: int = 0

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> BeliefState:
        return self._fallback.initialize(run_id, task_text, **kwargs)

    def update(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        return self._fallback.update(belief, action, observation)

    async def async_update(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        prompt = BELIEF_UPDATE_PROMPT_TEMPLATE.format(
            previous_belief_json=belief.to_json(),
            last_action_json=json.dumps(action.to_dict(), indent=2),
            new_observation_json=json.dumps(observation.to_dict(), indent=2),
            belief_token_budget=self._belief_token_budget,
        )
        try:
            raw = await self._llm_client.complete(prompt)
            text = raw.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(text)
        except Exception as exc:
            logger.warning("LLM compactor failed (parse/network): %s", exc)
            return self._do_fallback(belief, action, observation)

        if not validate_belief_dict(result):
            logger.warning("LLM compactor returned invalid belief dict; falling back.")
            return self._do_fallback(belief, action, observation)

        try:
            new_belief = BeliefState.from_dict(result)
            new_belief.belief_id = f"belief_{belief.run_id}_step{observation.step_id}"
            new_belief.run_id = belief.run_id
            new_belief.step_id = observation.step_id
            new_belief.created_at_ms = time.time_ns() // 1_000_000
            for ref in belief.raw_trace_refs:
                if ref not in new_belief.raw_trace_refs:
                    new_belief.raw_trace_refs.append(ref)
            return new_belief
        except Exception as exc:
            logger.warning("LLM compactor failed to build BeliefState: %s", exc)
            return self._do_fallback(belief, action, observation)

    def _do_fallback(self, belief: BeliefState, action: Action, observation: Observation) -> BeliefState:
        self.fallback_count += 1
        updated = self._fallback.update(belief, action, observation)
        updated.uncertainty = {"level": "high", "reason": "LLM compaction failed; fell back to deterministic"}
        for ref in belief.raw_trace_refs:
            if ref not in updated.raw_trace_refs:
                updated.raw_trace_refs.append(ref)
        return updated


# ---------------------------------------------------------------------------
# Validation and deterministic reducers
# ---------------------------------------------------------------------------

def validate_belief_dict(d: dict) -> bool:
    """Return True if d contains all required BeliefState keys with valid types."""
    if not isinstance(d, dict):
        return False
    if _REQUIRED_BELIEF_KEYS - d.keys():
        return False
    if not isinstance(d.get("belief_id"), str) or not d["belief_id"]:
        return False
    if not isinstance(d.get("run_id"), str) or not d["run_id"]:
        return False
    if not isinstance(d.get("step_id"), int):
        return False
    if not isinstance(d.get("task"), dict):
        return False
    return True


def apply_deterministic_reducers(belief: BeliefState, action: Action, obs: Observation) -> BeliefState:
    """Apply rule-based belief updates based on action/observation content."""

    if obs.kind == "tool_result" and obs.structured.get("status") == "error":
        belief.action_history_digest.setdefault("tried_and_failed", []).append({
            "action": action.structured or action.text,
            "error": obs.text or obs.structured,
            "source_ref": obs.observation_id,
        })

    if obs.kind == "tool_result" and obs.structured.get("status") == "success":
        belief.action_history_digest.setdefault("tried_and_worked", []).append({
            "action": action.structured or action.text,
            "source_ref": obs.observation_id,
        })

    if action.kind == "tool_call" and action.structured.get("tool_name") in {"edit_file", "write_file"}:
        path = action.structured.get("args", {}).get("path")
        if path:
            belief.environment_state.setdefault("files_changed", []).append({
                "path": path,
                "action_id": action.action_id,
            })

    if obs.structured.get("type") == "test_result":
        belief.environment_state.setdefault("tests", []).append({
            "command": obs.structured.get("command"),
            "status": obs.structured.get("status"),
            "source_ref": obs.observation_id,
        })
        if obs.structured.get("status") == "failed":
            belief.open_questions.append({
                "question": "Why did the latest test fail?",
                "possible_resolution_action": "Inspect failure output and relevant code.",
                "source_ref": obs.observation_id,
            })

    if obs.kind == "reward_event":
        reward_val = obs.structured.get("reward")
        if reward_val is not None:
            belief.progress.setdefault("reward_events", []).append({
                "reward": reward_val,
                "step_id": obs.step_id,
                "source_ref": obs.observation_id,
            })

    return belief
