# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal  # Literal used by RolloutTrace.final_status


def new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# ArtifactRef
# ---------------------------------------------------------------------------

@dataclass
class ArtifactRef:
    artifact_id: str
    kind: str
    sha256: str | None = None
    size_bytes: int | None = None
    created_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArtifactRef":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    observation_id: str
    step_id: int
    kind: str
    text: str | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[ArtifactRef] = field(default_factory=list)
    created_at_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        artifact_refs = [ArtifactRef.from_dict(r) for r in d.get("artifact_refs", [])]
        return cls(
            observation_id=d["observation_id"],
            step_id=d["step_id"],
            kind=d["kind"],
            text=d.get("text"),
            structured=d.get("structured", {}),
            artifact_refs=artifact_refs,
            created_at_ms=d.get("created_at_ms"),
        )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

@dataclass
class Action:
    action_id: str
    step_id: int
    kind: str
    text: str | None = None
    structured: dict[str, Any] = field(default_factory=dict)

    # Training-critical fields.
    action_tokens: list[int] | None = None
    action_logprobs_behavior: list[float] | None = None
    action_mask: list[int] | None = None

    # Async forced-lag fields.
    behavior_policy_version: str | None = None
    tokenizer_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Action":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    belief_id: str
    run_id: str
    step_id: int

    task: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    open_questions: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    action_history_digest: dict[str, Any] = field(default_factory=dict)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    raw_trace_refs: list[str] = field(default_factory=list)

    token_estimate: int | None = None
    created_at_ms: int | None = None

    def to_context_str(self) -> str:
        return self.to_json()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeliefState":
        return cls(
            belief_id=d["belief_id"],
            run_id=d["run_id"],
            step_id=d["step_id"],
            task=d.get("task", {}),
            environment_state=d.get("environment_state", {}),
            open_questions=d.get("open_questions", []),
            progress=d.get("progress", {}),
            action_history_digest=d.get("action_history_digest", {}),
            uncertainty=d.get("uncertainty", {}),
            raw_trace_refs=d.get("raw_trace_refs", []),
            token_estimate=d.get("token_estimate"),
            created_at_ms=d.get("created_at_ms"),
        )

    @classmethod
    def from_json(cls, s: str) -> "BeliefState":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# PomdpTransition
# ---------------------------------------------------------------------------

@dataclass
class PomdpTransition:
    transition_id: str
    run_id: str
    step_id: int

    belief_state_id: str
    observation_id: str | None
    action_id: str
    next_observation_id: str | None
    next_belief_state_id: str | None

    reward: float | None = None
    done: bool = False

    actor_context_ref: ArtifactRef | None = None
    actor_context_tokens_ref: ArtifactRef | None = None
    actor_context_token_count: int | None = None

    raw_history_ref: ArtifactRef | None = None

    memory_snapshot_ref: ArtifactRef | None = None
    environment_snapshot_ref: ArtifactRef | None = None

    behavior_policy_version: str | None = None
    policy_lag_updates: int | None = None
    policy_lag_seconds: float | None = None

    parent_transition_id: str | None = None
    branch_group_id: str | None = None
    branch_index: int | None = None

    # KV compaction — ref to the KVMask applied at this step, if any.
    kv_mask_ref: ArtifactRef | None = None

    # Belief-Still — ref to the BeliefMemory M_t produced at this step, if any.
    belief_memory_ref: ArtifactRef | None = None

    tokenizer_version: str | None = None

    created_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PomdpTransition":
        def _ref(x):
            return ArtifactRef.from_dict(x) if x is not None else None

        return cls(
            transition_id=d["transition_id"],
            run_id=d["run_id"],
            step_id=d["step_id"],
            belief_state_id=d["belief_state_id"],
            observation_id=d.get("observation_id"),
            action_id=d["action_id"],
            next_observation_id=d.get("next_observation_id"),
            next_belief_state_id=d.get("next_belief_state_id"),
            reward=d.get("reward"),
            done=d.get("done", False),
            actor_context_ref=_ref(d.get("actor_context_ref")),
            actor_context_tokens_ref=_ref(d.get("actor_context_tokens_ref")),
            actor_context_token_count=d.get("actor_context_token_count"),
            raw_history_ref=_ref(d.get("raw_history_ref")),
            memory_snapshot_ref=_ref(d.get("memory_snapshot_ref")),
            environment_snapshot_ref=_ref(d.get("environment_snapshot_ref")),
            behavior_policy_version=d.get("behavior_policy_version"),
            policy_lag_updates=d.get("policy_lag_updates"),
            policy_lag_seconds=d.get("policy_lag_seconds"),
            parent_transition_id=d.get("parent_transition_id"),
            branch_group_id=d.get("branch_group_id"),
            branch_index=d.get("branch_index"),
            kv_mask_ref=_ref(d.get("kv_mask_ref")),
            belief_memory_ref=_ref(d.get("belief_memory_ref")),
            tokenizer_version=d.get("tokenizer_version"),
            created_at_ms=d.get("created_at_ms"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# RolloutTrace
# ---------------------------------------------------------------------------

@dataclass
class RolloutTrace:
    run_id: str
    task_id: str | None
    initial_task_text: str

    transitions: list[str] = field(default_factory=list)
    belief_states: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)

    final_reward: float | None = None
    final_status: Literal["success", "failure", "timeout", "cancelled", "unknown"] = "unknown"

    created_at_ms: int | None = None
    finished_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RolloutTrace":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
