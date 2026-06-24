# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import time
from typing import Any

import torch

from .algorithm import CompactionAlgorithm, DeterministicAlgorithm
from .config import PomdpConfig
from .context_builder import ContextBuilder
from ..kv.compressors import KVCompressor
from ..kv.megatron_hook import MegatronInferenceHook, NullHook
from ..kv.types import KVMask
from .metrics import ShadowRunMetrics, ShadowStepMetrics
from .store import PomdpTraceStore
from .trigger import AlwaysTrigger, CompactionTrigger
from .types import (
    Action,
    ArtifactRef,
    Observation,
    PomdpTransition,
    RolloutTrace,
    new_id,
)

# BeliefUpdater / BeliefMemory are duck-typed as Any at runtime so the recorder
# stays importable without torch.  Type annotations below are strings only.
_BELIEF_MEMORY_ARTIFACT_KIND = "belief_memory"


class PomdpRolloutRecorder:
    """Wraps a rollout loop with POMDP recording.

    Three pluggable components control behavior:

    * ``trigger``   — decides *when* to compact (defaults to every step).
    * ``algorithm`` — controls *how* the belief is updated (defaults to
                      DeterministicAlgorithm).
    * ``context_builder`` — renders the belief into the actor context string.

    When ``config.enabled`` is False all methods are no-ops and return sensible
    defaults so that existing agent code is completely unaffected.
    """

    def __init__(
        self,
        config: PomdpConfig,
        store: PomdpTraceStore,
        context_builder: ContextBuilder,
        trigger: CompactionTrigger | None = None,
        algorithm: CompactionAlgorithm | None = None,
        kv_algorithm: KVCompressor | None = None,
        kv_hook: MegatronInferenceHook | None = None,
        # Belief-Still: BeliefUpdater instance (duck-typed to avoid hard torch dep).
        belief_updater: Any | None = None,
        # Legacy kwarg for backward compat — ignored when algorithm is supplied.
        compactor: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._context_builder = context_builder
        self._tokenizer = tokenizer
        self._trigger: CompactionTrigger = trigger if trigger is not None else AlwaysTrigger()
        # Prefer explicit algorithm; fall back to wrapping a legacy compactor, then default.
        if algorithm is not None:
            self._algorithm: CompactionAlgorithm = algorithm
        elif compactor is not None:
            self._algorithm = _LegacyCompactorAdapter(compactor)
        else:
            self._algorithm = DeterministicAlgorithm()
        self._kv_algorithm = kv_algorithm
        self._kv_hook: MegatronInferenceHook = kv_hook if kv_hook is not None else NullHook()
        self._belief_updater = belief_updater
        # Per-run belief memory: run_id → current BeliefMemory (or None before first step).
        self._belief_memories: dict[str, Any] = {}
        self.shadow_metrics: dict[str, ShadowRunMetrics] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_run(
        self,
        task_text: str,
        task_id: str | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        """Initialise a new rollout run and return (run_id, initial_belief)."""
        run_id = new_id()
        belief = self._algorithm.initialize(run_id, task_text, task_metadata=task_metadata)

        if self._belief_updater is not None:
            self._belief_memories[run_id] = None   # initialised on first step

        if self._config.enabled:
            trace = RolloutTrace(
                run_id=run_id,
                task_id=task_id,
                initial_task_text=task_text,
                created_at_ms=time.time_ns() // 1_000_000,
            )
            self._store.create_run(trace)
            if hasattr(belief, "belief_id"):
                self._store.put_belief_state(belief)

        return run_id, belief

    def build_context_for_step(
        self,
        run_id: str,
        system_prompt: str,
        task_text: str,
        belief: Any,
        recent_tail: list[Any],
        current_observation: Observation | None,
        available_tools: list[dict[str, Any]],
        full_context_text: str | None = None,
    ) -> tuple[str, ArtifactRef | None, ArtifactRef | None]:
        """Build compact actor context.

        Returns (context_text, context_ref, context_tokens_ref).
        When ``mode == "shadow"`` and ``full_context_text`` is provided the
        compression ratio is recorded in shadow_metrics.
        """
        compact_context = self._context_builder.build_actor_context(
            system_prompt=system_prompt,
            task_text=task_text,
            belief=belief,
            recent_tail=recent_tail,
            current_observation=current_observation,
            available_tools=available_tools,
        )

        context_ref: ArtifactRef | None = None
        context_tokens_ref: ArtifactRef | None = None
        belief_id = getattr(belief, "belief_id", None)

        if self._config.enabled:
            context_ref = self._store.put_artifact(
                kind="raw_actor_context",
                payload=compact_context,
                metadata={"run_id": run_id, "belief_id": belief_id},
            )
            if self._config.store_actor_input_tokens:
                tokens = self._context_builder.encode(compact_context)
                if tokens is not None:
                    context_tokens_ref = self._store.put_artifact(
                        kind="tokenized_actor_context",
                        payload={"tokens": tokens},
                        metadata={"run_id": run_id, "belief_id": belief_id},
                    )

        if self._config.mode == "shadow" and full_context_text is not None:
            self._record_shadow_step(run_id, belief, compact_context, full_context_text)

        return compact_context, context_ref, context_tokens_ref

    def record_step(
        self,
        run_id: str,
        step_id: int,
        belief: Any,
        current_observation: Observation | None,
        action: Action,
        next_observation: Observation | None,
        reward: float | None,
        done: bool,
        actor_context: str,
        actor_context_tokens: list[int] | None,
        raw_history_text: str | None,
        actor_context_ref: ArtifactRef | None = None,
        actor_context_tokens_ref: ArtifactRef | None = None,
        memory_snapshot_ref: ArtifactRef | None = None,
        environment_snapshot_ref: ArtifactRef | None = None,
        shadow_metrics_ref: ShadowRunMetrics | None = None,
    ) -> Any:
        """Record one POMDP step synchronously and return the updated belief."""
        next_belief = self._maybe_update(belief, step_id, action, next_observation)
        kv_mask = self._maybe_apply_kv(run_id, step_id, actor_context_tokens)
        belief_memory = self._maybe_update_belief(run_id, step_id)
        self._maybe_record_shadow(shadow_metrics_ref, step_id, run_id, next_belief, kv_mask, belief_memory)
        if not self._config.enabled:
            return next_belief
        return self._persist(
            run_id, step_id, belief, next_belief,
            current_observation, action, next_observation, reward, done,
            actor_context, actor_context_tokens, raw_history_text,
            actor_context_ref, actor_context_tokens_ref,
            memory_snapshot_ref, environment_snapshot_ref,
            kv_mask=kv_mask,
            belief_memory=belief_memory,
        )

    async def record_step_async(
        self,
        run_id: str,
        step_id: int,
        belief: Any,
        current_observation: Observation | None,
        action: Action,
        next_observation: Observation | None,
        reward: float | None,
        done: bool,
        actor_context: str,
        actor_context_tokens: list[int] | None,
        raw_history_text: str | None,
        actor_context_ref: ArtifactRef | None = None,
        actor_context_tokens_ref: ArtifactRef | None = None,
        memory_snapshot_ref: ArtifactRef | None = None,
        environment_snapshot_ref: ArtifactRef | None = None,
        shadow_metrics_ref: ShadowRunMetrics | None = None,
    ) -> Any:
        """Record one POMDP step asynchronously (enables LLM-based compaction)."""
        next_belief = await self._maybe_update_async(belief, step_id, action, next_observation)
        kv_mask = self._maybe_apply_kv(run_id, step_id, actor_context_tokens)
        belief_memory = self._maybe_update_belief(run_id, step_id)
        self._maybe_record_shadow(shadow_metrics_ref, step_id, run_id, next_belief, kv_mask, belief_memory)
        if not self._config.enabled:
            return next_belief
        return self._persist(
            run_id, step_id, belief, next_belief,
            current_observation, action, next_observation, reward, done,
            actor_context, actor_context_tokens, raw_history_text,
            actor_context_ref, actor_context_tokens_ref,
            memory_snapshot_ref, environment_snapshot_ref,
            kv_mask=kv_mask,
            belief_memory=belief_memory,
        )

    def finish_run(
        self,
        run_id: str,
        final_reward: float | None,
        final_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._belief_memories.pop(run_id, None)
        if self._config.enabled:
            self._store.finish_run(run_id, final_reward, final_status, metadata)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_update(
        self,
        belief: Any,
        step_id: int,
        action: Action,
        observation: Observation | None,
    ) -> Any:
        if observation is None:
            return belief
        if self._trigger.should_compact(step_id, belief, action, observation):
            return self._algorithm.update(belief, action, observation)
        return belief

    async def _maybe_update_async(
        self,
        belief: Any,
        step_id: int,
        action: Action,
        observation: Observation | None,
    ) -> Any:
        if observation is None:
            return belief
        if await self._should_compact_async(step_id, belief, action, observation):
            return await self._algorithm.async_update(belief, action, observation)
        return belief

    async def _should_compact_async(
        self,
        step_id: int,
        belief: Any,
        action: Action,
        observation: Observation,
    ) -> bool:
        """Consult the trigger on the async path.

        A trigger may optionally implement ``async_should_compact`` (same
        signature as ``should_compact``) to make a model-driven decision — e.g.
        asking an LLM whether the context should be compacted now. If absent, we
        fall back to the synchronous ``should_compact``. This keeps every
        existing sync trigger working unchanged while enabling model-gated ones.
        """
        async_fn = getattr(self._trigger, "async_should_compact", None)
        if async_fn is not None:
            return await async_fn(step_id, belief, action, observation)
        return self._trigger.should_compact(step_id, belief, action, observation)

    def _maybe_apply_kv(
        self,
        run_id: str,
        step_id: int,
        actor_context_tokens: list[int] | None,
    ) -> KVMask | None:
        if self._kv_algorithm is None:
            return None
        kv = self._kv_hook.get_kv_matrices()
        if kv is None:
            return None
        keys_per_layer, vals_per_layer = kv
        keys_agg = torch.stack([k[0].float() for k in keys_per_layer]).mean(0)
        vals_agg = torch.stack([v[0].float() for v in vals_per_layer]).mean(0)
        n = keys_agg.shape[0]
        budget = max(1, int(n * self._config.kv_budget_ratio))
        result = self._kv_algorithm.compress(keys_agg, vals_agg, budget,
                                             run_id=run_id, step_id=step_id)
        if self._config.mode == "live":
            self._kv_hook.apply_mask(result.to_kv_mask())
        return result.to_kv_mask()

    def _maybe_update_belief(self, run_id: str, step_id: int) -> Any | None:
        """Run the BeliefUpdater if configured, returning the new BeliefMemory.

        On the first step for a run (no prior memory), calls
        ``belief_updater.initial_compress`` to bootstrap M_0.
        On subsequent steps, calls ``belief_updater(M_{t-1}, R_t)`` to
        produce M_t, where R_t is the current step's raw KV matrices from
        ``kv_hook.get_kv_matrices()``.

        In "live" mode, injects M_t back into the inference engine via
        ``kv_hook.apply_belief_memory(M_t)`` so the next forward pass uses
        the compact synthetic cache instead of the full accumulated KV.

        Returns None when belief_updater is not configured or KV data is
        unavailable.
        """
        if self._belief_updater is None:
            return None
        kv = self._kv_hook.get_kv_matrices()
        if kv is None:
            return None
        keys_per_layer, values_per_layer = kv
        current_memory = self._belief_memories.get(run_id)
        if current_memory is None:
            # Step 0: bootstrap from the first chunk.
            new_memory = self._belief_updater.initial_compress(keys_per_layer, values_per_layer)
        else:
            # Step t > 0: recurrent update M_{t-1} → M_t.
            new_memory = self._belief_updater(current_memory, keys_per_layer, values_per_layer)
        self._belief_memories[run_id] = new_memory
        if self._config.mode == "live":
            self._kv_hook.apply_belief_memory(new_memory)
        return new_memory

    def _maybe_record_shadow(
        self,
        shadow_metrics_ref: ShadowRunMetrics | None,
        step_id: int,
        run_id: str,
        belief: Any,
        kv_mask: KVMask | None = None,
        belief_memory: Any | None = None,
    ) -> None:
        if shadow_metrics_ref is None:
            return
        uncertainty_level = None
        token_estimate = None
        if hasattr(belief, "uncertainty"):
            uncertainty_level = (belief.uncertainty or {}).get("level")
        if hasattr(belief, "token_estimate"):
            token_estimate = belief.token_estimate

        kv_retention = kv_mask.retention_ratio() if kv_mask is not None else None
        # When both KV mask and belief memory are present, prefer belief memory
        # retention ratio (more precise — reflects synthetic compaction budget).
        if belief_memory is not None and hasattr(belief_memory, "budget"):
            kv_retention = None   # belief memory doesn't have a selection ratio

        step_metric = ShadowStepMetrics(
            step_id=step_id,
            run_id=run_id,
            full_context_tokens=None,
            compact_context_tokens=None,
            compression_ratio=None,
            belief_json_valid=True,
            belief_token_estimate=token_estimate,
            compactor_error=False,
            fallback_used=False,
            uncertainty_level=uncertainty_level,
            kv_retention_ratio=kv_retention,
        )
        shadow_metrics_ref.steps.append(step_metric)

    def _record_shadow_step(
        self,
        run_id: str,
        belief: Any,
        compact_context: str,
        full_context_text: str,
    ) -> None:
        if run_id not in self.shadow_metrics:
            self.shadow_metrics[run_id] = ShadowRunMetrics(run_id=run_id)
        compact_tokens = self._context_builder.encode(compact_context)
        full_tokens = self._context_builder.encode(full_context_text)
        compact_count = len(compact_tokens) if compact_tokens is not None else None
        full_count = len(full_tokens) if full_tokens is not None else None
        ratio: float | None = None
        if compact_count and full_count:
            ratio = full_count / compact_count
        token_estimate = getattr(belief, "token_estimate", None)
        uncertainty_level = None
        if hasattr(belief, "uncertainty"):
            uncertainty_level = (belief.uncertainty or {}).get("level")
        step_metric = ShadowStepMetrics(
            step_id=getattr(belief, "step_id", 0),
            run_id=run_id,
            full_context_tokens=full_count,
            compact_context_tokens=compact_count,
            compression_ratio=ratio,
            belief_json_valid=True,
            belief_token_estimate=token_estimate,
            compactor_error=False,
            fallback_used=False,
            uncertainty_level=uncertainty_level,
        )
        self.shadow_metrics[run_id].steps.append(step_metric)

    def _persist(
        self,
        run_id: str,
        step_id: int,
        belief: Any,
        next_belief: Any,
        current_observation: Observation | None,
        action: Action,
        next_observation: Observation | None,
        reward: float | None,
        done: bool,
        actor_context: str,
        actor_context_tokens: list[int] | None,
        raw_history_text: str | None,
        actor_context_ref: ArtifactRef | None,
        actor_context_tokens_ref: ArtifactRef | None,
        memory_snapshot_ref: ArtifactRef | None,
        environment_snapshot_ref: ArtifactRef | None,
        kv_mask: KVMask | None = None,
        belief_memory: Any | None = None,
    ) -> Any:
        if current_observation is not None:
            self._store.put_observation(current_observation)
        self._store.put_action(action)
        if next_observation is not None:
            self._store.put_observation(next_observation)

        if actor_context_ref is None:
            actor_context_ref = self._store.put_artifact(
                kind="raw_actor_context",
                payload=actor_context,
                metadata={"run_id": run_id, "step_id": step_id},
            )
        if actor_context_tokens_ref is None and actor_context_tokens is not None:
            actor_context_tokens_ref = self._store.put_artifact(
                kind="tokenized_actor_context",
                payload={"tokens": actor_context_tokens},
                metadata={"run_id": run_id, "step_id": step_id},
            )

        raw_history_ref: ArtifactRef | None = None
        if self._config.store_raw_trace and raw_history_text:
            raw_history_ref = self._store.put_artifact(
                kind="raw_observation",
                payload=raw_history_text,
                metadata={"run_id": run_id, "step_id": step_id},
            )

        kv_mask_ref: ArtifactRef | None = None
        if kv_mask is not None:
            kv_mask_ref = self._store.put_artifact(
                kind="other",
                payload=kv_mask.to_dict(),
                metadata={"run_id": run_id, "step_id": step_id, "kind": "kv_mask"},
            )

        belief_memory_ref: ArtifactRef | None = None
        if belief_memory is not None:
            # Store compact metadata only; full tensors would be too large for JSON.
            # The tensors themselves must be checkpointed separately (e.g. torch.save).
            bm_payload: dict = {
                "step":     belief_memory.step,
                "n_layers": belief_memory.n_layers,
                "budget":   belief_memory.budget,
                "d_model":  belief_memory.d_model,
            }
            belief_memory_ref = self._store.put_artifact(
                kind=_BELIEF_MEMORY_ARTIFACT_KIND,
                payload=bm_payload,
                metadata={"run_id": run_id, "step_id": step_id},
            )

        if hasattr(next_belief, "belief_id"):
            self._store.put_belief_state(next_belief)

        belief_id = getattr(belief, "belief_id", str(id(belief)))
        next_belief_id = getattr(next_belief, "belief_id", str(id(next_belief)))

        transition = PomdpTransition(
            transition_id=new_id(),
            run_id=run_id,
            step_id=step_id,
            belief_state_id=belief_id,
            observation_id=current_observation.observation_id if current_observation else None,
            action_id=action.action_id,
            next_observation_id=next_observation.observation_id if next_observation else None,
            next_belief_state_id=next_belief_id,
            reward=reward,
            done=done,
            actor_context_ref=actor_context_ref,
            actor_context_tokens_ref=actor_context_tokens_ref,
            actor_context_token_count=len(actor_context_tokens) if actor_context_tokens else None,
            raw_history_ref=raw_history_ref,
            memory_snapshot_ref=memory_snapshot_ref,
            environment_snapshot_ref=environment_snapshot_ref,
            kv_mask_ref=kv_mask_ref,
            belief_memory_ref=belief_memory_ref,
            behavior_policy_version=action.behavior_policy_version,
            tokenizer_version=action.tokenizer_version,
            created_at_ms=time.time_ns() // 1_000_000,
        )
        self._store.put_transition(transition)
        self._store.append_transition_to_run(run_id, transition.transition_id)
        return next_belief


# ---------------------------------------------------------------------------
# Legacy adapter — wraps a BeliefCompactor as a CompactionAlgorithm
# ---------------------------------------------------------------------------

class _LegacyCompactorAdapter:
    """Adapts the old BeliefCompactor interface to CompactionAlgorithm."""

    def __init__(self, compactor: Any) -> None:
        self._c = compactor

    def initialize(self, run_id: str, task_text: str, **kwargs: Any) -> Any:
        meta = kwargs.get("task_metadata")
        return self._c.initialize_belief(run_id, task_text, meta)

    def update(self, belief: Any, action: Any, observation: Any) -> Any:
        return self._c.update_belief(belief, action, observation)

    async def async_update(self, belief: Any, action: Any, observation: Any) -> Any:
        if hasattr(self._c, "update_belief_async"):
            return await self._c.update_belief_async(belief, action, observation)
        return self.update(belief, action, observation)
