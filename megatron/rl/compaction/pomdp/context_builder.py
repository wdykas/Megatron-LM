# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from typing import Any

from .config import PomdpConfig
from .prompts import ACTING_MODE_INSTRUCTION, ACTOR_CONTEXT_TEMPLATE
from .types import Action, Observation


class ContextBuilder:
    """Builds compact actor context from a belief + recent tail + current observation.

    The belief argument can be any object that implements ``to_context_str()``
    (e.g. BeliefState, TextBelief, or a custom type returned by a user-defined
    CompactionAlgorithm).
    """

    def __init__(self, config: PomdpConfig, tokenizer: Any | None = None) -> None:
        self._config = config
        self._tokenizer = tokenizer

    def build_actor_context(
        self,
        system_prompt: str,
        task_text: str,
        belief: Any,
        recent_tail: list[Any],
        current_observation: Observation | None,
        available_tools: list[dict[str, Any]],
    ) -> str:
        """Build the compact actor context string.

        Truncation priority (highest → always kept):
        1. Belief context string.
        2. Recent raw tail (last N steps).
        3. Current observation (truncated to max_observation_chars).
        """
        belief_text = self._render_belief(belief)
        recent_tail_text = self._render_recent_tail(recent_tail)
        current_obs_text = self._render_observation(current_observation)
        tools_text = json.dumps(available_tools, indent=2) if available_tools else "[]"

        return ACTOR_CONTEXT_TEMPLATE.format(
            system_prompt=system_prompt,
            acting_mode=ACTING_MODE_INSTRUCTION,
            task_text=task_text,
            belief_json=belief_text,
            recent_tail=recent_tail_text,
            current_observation=current_obs_text,
            available_tools=tools_text,
        )

    def _render_belief(self, belief: Any) -> str:
        if hasattr(belief, "to_context_str"):
            return belief.to_context_str()
        if hasattr(belief, "to_json"):
            return belief.to_json()
        if hasattr(belief, "to_dict"):
            return json.dumps(belief.to_dict(), indent=2)
        return str(belief)

    def _render_recent_tail(self, tail: list[Any]) -> str:
        if not tail:
            return "(none)"
        max_chars = self._config.max_observation_chars
        parts = []
        for item in tail[-self._config.recent_tail_steps:]:
            if isinstance(item, Observation):
                text = item.text or json.dumps(item.structured)
                if len(text) > max_chars:
                    text = text[:max_chars] + " ... [truncated, see artifact_refs]"
                parts.append(f"[obs step={item.step_id} kind={item.kind}]\n{text}")
            elif isinstance(item, Action):
                text = item.text or json.dumps(item.structured)
                parts.append(f"[act step={item.step_id} kind={item.kind}]\n{text}")
        return "\n\n".join(parts)

    def _render_observation(self, obs: Observation | None) -> str:
        if obs is None:
            return "(none)"
        text = obs.text or json.dumps(obs.structured)
        max_chars = self._config.max_observation_chars
        if len(text) > max_chars:
            text = text[:max_chars] + " ... [truncated, see artifact_refs]"
        refs = ""
        if obs.artifact_refs:
            refs = "\n[artifact_refs: " + ", ".join(r.artifact_id for r in obs.artifact_refs) + "]"
        return f"[obs step={obs.step_id} kind={obs.kind}]\n{text}{refs}"

    def encode(self, text: str) -> list[int] | None:
        """Tokenize text if a tokenizer is available."""
        if self._tokenizer is None:
            return None
        try:
            return self._tokenizer.tokenize(text)
        except Exception:
            return None
