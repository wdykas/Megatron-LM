# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import hashlib
import json
import os
import time
from typing import Any, Protocol, runtime_checkable

from .types import (
    Action,
    ArtifactRef,
    BeliefState,
    Observation,
    PomdpTransition,
    RolloutTrace,
    new_id,
)


@runtime_checkable
class PomdpTraceStore(Protocol):
    def create_run(self, trace: RolloutTrace) -> None: ...

    def put_artifact(
        self,
        kind: str,
        payload: bytes | str | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef: ...

    def put_observation(self, obs: Observation) -> None: ...
    def put_action(self, action: Action) -> None: ...
    def put_belief_state(self, belief: BeliefState) -> None: ...
    def put_transition(self, transition: PomdpTransition) -> None: ...

    def get_belief_state(self, belief_id: str) -> BeliefState: ...
    def get_transition(self, transition_id: str) -> PomdpTransition: ...
    def get_artifact(self, artifact_id: str) -> bytes | str | dict[str, Any]: ...

    def append_transition_to_run(self, run_id: str, transition_id: str) -> None: ...
    def finish_run(
        self,
        run_id: str,
        final_reward: float | None,
        final_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class JsonlPomdpTraceStore:
    """Append-only JSONL-backed store. Each entity type has its own .jsonl file per run."""

    def __init__(self, trace_dir: str) -> None:
        self._trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)
        # In-memory indices for fast reads within a session.
        self._beliefs: dict[str, BeliefState] = {}
        self._transitions: dict[str, PomdpTransition] = {}
        self._artifacts: dict[str, bytes | str | dict[str, Any]] = {}
        self._runs: dict[str, RolloutTrace] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> str:
        d = os.path.join(self._trace_dir, run_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _append_jsonl(self, path: str, obj: dict[str, Any]) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    # ------------------------------------------------------------------
    # PomdpTraceStore interface
    # ------------------------------------------------------------------

    def create_run(self, trace: RolloutTrace) -> None:
        self._runs[trace.run_id] = trace
        path = os.path.join(self._run_dir(trace.run_id), "run.jsonl")
        self._append_jsonl(path, trace.to_dict())

    def put_artifact(
        self,
        kind: str,
        payload: bytes | str | dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        artifact_id = new_id()
        now_ms = time.time_ns() // 1_000_000

        if isinstance(payload, bytes):
            sha256 = hashlib.sha256(payload).hexdigest()
            size_bytes = len(payload)
            serialized: Any = payload.decode("utf-8", errors="replace")
        elif isinstance(payload, str):
            sha256 = hashlib.sha256(payload.encode()).hexdigest()
            size_bytes = len(payload.encode())
            serialized = payload
        else:
            raw = json.dumps(payload)
            sha256 = hashlib.sha256(raw.encode()).hexdigest()
            size_bytes = len(raw.encode())
            serialized = payload

        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            sha256=sha256,
            size_bytes=size_bytes,
            created_at_ms=now_ms,
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = serialized

        # Persist to a global artifacts file; run_id is stored in metadata when available.
        artifacts_path = os.path.join(self._trace_dir, "artifacts.jsonl")
        self._append_jsonl(artifacts_path, {"ref": ref.to_dict(), "payload": serialized if isinstance(serialized, (str, dict)) else str(serialized)})

        return ref

    def put_observation(self, obs: Observation) -> None:
        # Observations are keyed by observation_id; store in a flat per-trace file.
        # We rely on callers to pass observations before transitions reference them.
        self._append_jsonl(os.path.join(self._trace_dir, "observations.jsonl"), obs.to_dict())

    def put_action(self, action: Action) -> None:
        self._append_jsonl(os.path.join(self._trace_dir, "actions.jsonl"), action.to_dict())

    def put_belief_state(self, belief: BeliefState) -> None:
        self._beliefs[belief.belief_id] = belief
        run_path = os.path.join(self._run_dir(belief.run_id), "beliefs.jsonl")
        self._append_jsonl(run_path, belief.to_dict())

    def put_transition(self, transition: PomdpTransition) -> None:
        self._transitions[transition.transition_id] = transition
        run_path = os.path.join(self._run_dir(transition.run_id), "transitions.jsonl")
        self._append_jsonl(run_path, transition.to_dict())

    def get_belief_state(self, belief_id: str) -> BeliefState:
        if belief_id in self._beliefs:
            return self._beliefs[belief_id]
        raise KeyError(f"BeliefState not found in session cache: {belief_id}")

    def get_transition(self, transition_id: str) -> PomdpTransition:
        if transition_id in self._transitions:
            return self._transitions[transition_id]
        raise KeyError(f"PomdpTransition not found in session cache: {transition_id}")

    def get_artifact(self, artifact_id: str) -> bytes | str | dict[str, Any]:
        if artifact_id in self._artifacts:
            return self._artifacts[artifact_id]
        raise KeyError(f"Artifact not found in session cache: {artifact_id}")

    def append_transition_to_run(self, run_id: str, transition_id: str) -> None:
        if run_id in self._runs:
            self._runs[run_id].transitions.append(transition_id)
        run_path = os.path.join(self._run_dir(run_id), "run_transition_ids.jsonl")
        self._append_jsonl(run_path, {"transition_id": transition_id})

    def finish_run(
        self,
        run_id: str,
        final_reward: float | None,
        final_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now_ms = time.time_ns() // 1_000_000
        if run_id in self._runs:
            trace = self._runs[run_id]
            trace.final_reward = final_reward
            trace.final_status = final_status
            trace.finished_at_ms = now_ms
            if metadata:
                trace.metadata.update(metadata)

        finish_record = {
            "run_id": run_id,
            "final_reward": final_reward,
            "final_status": final_status,
            "finished_at_ms": now_ms,
            "metadata": metadata or {},
        }
        run_path = os.path.join(self._run_dir(run_id), "run.jsonl")
        self._append_jsonl(run_path, finish_record)
