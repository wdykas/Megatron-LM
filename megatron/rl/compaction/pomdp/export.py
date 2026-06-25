# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import asdict, dataclass, field
from typing import Any

from .store import PomdpTraceStore
from .types import new_id


@dataclass
class CompactTrainingSample:
    sample_id: str
    run_id: str
    transition_id: str
    step_id: int

    # Exact compact actor input used during rollout.
    actor_context_tokens: list[int] | None
    actor_context_ref: str | None
    actor_context_tokens_ref: str | None

    action_tokens: list[int]
    action_logprobs_behavior: list[float]
    action_mask: list[int] | None

    reward: float | None
    return_to_go: float | None
    advantage: float | None

    behavior_policy_version: str
    learner_policy_version_at_export: str | None
    tokenizer_version: str

    belief_state_id: str
    raw_history_ref: str | None
    environment_snapshot_ref: str | None
    memory_snapshot_ref: str | None

    # True only if action_tokens and action_logprobs_behavior are present and the
    # actor_context_tokens can be verified to match the context used during rollout.
    valid_for_policy_gradient: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PomdpTrainingExporter:
    """Exports CompactTrainingSample objects from recorded transitions."""

    def __init__(
        self,
        store: PomdpTraceStore,
        learner_policy_version: str | None = None,
    ) -> None:
        self._store = store
        self._learner_policy_version = learner_policy_version

    def export_transition(self, transition_id: str) -> CompactTrainingSample:
        transition = self._store.get_transition(transition_id)

        # Retrieve action.
        try:
            action_dict = self._store.get_artifact(transition.action_id)
        except KeyError:
            action_dict = {}

        action_tokens = action_dict.get("action_tokens") if isinstance(action_dict, dict) else None
        action_logprobs = action_dict.get("action_logprobs_behavior") if isinstance(action_dict, dict) else None
        action_mask = action_dict.get("action_mask") if isinstance(action_dict, dict) else None

        # Retrieve actor context tokens.
        actor_context_tokens: list[int] | None = None
        if transition.actor_context_tokens_ref is not None:
            try:
                tokens_payload = self._store.get_artifact(transition.actor_context_tokens_ref.artifact_id)
                if isinstance(tokens_payload, dict):
                    actor_context_tokens = tokens_payload.get("tokens")
            except KeyError:
                pass

        raw_history_ref_id: str | None = (
            transition.raw_history_ref.artifact_id if transition.raw_history_ref else None
        )
        env_snap_ref_id: str | None = (
            transition.environment_snapshot_ref.artifact_id
            if transition.environment_snapshot_ref
            else None
        )
        mem_snap_ref_id: str | None = (
            transition.memory_snapshot_ref.artifact_id
            if transition.memory_snapshot_ref
            else None
        )

        # A sample is valid for policy-gradient training when tokens and logprobs are present.
        valid = (
            action_tokens is not None
            and action_logprobs is not None
            and len(action_tokens) > 0
            and len(action_logprobs) > 0
            and (actor_context_tokens is not None or transition.actor_context_tokens_ref is not None)
        )

        return CompactTrainingSample(
            sample_id=new_id(),
            run_id=transition.run_id,
            transition_id=transition_id,
            step_id=transition.step_id,
            actor_context_tokens=actor_context_tokens,
            actor_context_ref=(
                transition.actor_context_ref.artifact_id if transition.actor_context_ref else None
            ),
            actor_context_tokens_ref=(
                transition.actor_context_tokens_ref.artifact_id
                if transition.actor_context_tokens_ref
                else None
            ),
            action_tokens=action_tokens or [],
            action_logprobs_behavior=action_logprobs or [],
            action_mask=action_mask,
            reward=transition.reward,
            return_to_go=None,
            advantage=None,
            behavior_policy_version=transition.behavior_policy_version or "unknown",
            learner_policy_version_at_export=self._learner_policy_version,
            tokenizer_version=transition.tokenizer_version or "unknown",
            belief_state_id=transition.belief_state_id,
            raw_history_ref=raw_history_ref_id,
            environment_snapshot_ref=env_snap_ref_id,
            memory_snapshot_ref=mem_snap_ref_id,
            valid_for_policy_gradient=valid,
        )

    def export_run(self, run_id: str) -> list[CompactTrainingSample]:
        trace = self._store._runs.get(run_id)
        if trace is None:
            raise KeyError(f"Run not found in session cache: {run_id}")
        return [self.export_transition(tid) for tid in trace.transitions]
