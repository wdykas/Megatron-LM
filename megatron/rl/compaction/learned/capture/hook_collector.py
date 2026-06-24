# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inline Trajectory collection via MegatronInferenceHook.

Instead of a standalone model call, this collector intercepts KV matrices
from the live Megatron inference engine during normal rollouts.  No separate
model load or HF dependency required.

Usage
-----
    from megatron.rl.compaction.learned.capture.hook_collector import HookTrajectoryCollector
    from megatron.rl.compaction.learned import PipelineConfig

    collector = HookTrajectoryCollector(hook, PipelineConfig(chunk_size=256))
    collector.begin_rollout()

    # During prefill:
    engine.prefill(prompt_tokens)
    collector.on_step(len(prompt_tokens), teacher_logits=engine.last_logits,
                      query_tokens=prompt_tokens)

    # During decode:
    for _ in range(max_new_tokens):
        engine.step(...)
        collector.on_step(1)

    trajectory = collector.end_rollout()

KV format
---------
Keys and values are stored *as returned by the hook* — typically with RoPE
already applied.  This is fine for BeliefUpdater training (the updater learns
to compress positional keys).  If you need position-free keys for a
standalone FrozenModelAdapter student_fn, use kv/rope.py to strip them
before saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from megatron.rl.compaction.learned.training.data import CompactKV, Trajectory, TrainingProbe
from megatron.rl.compaction.learned.training.data import PipelineConfig
from megatron.rl.compaction.kv.megatron_hook import MegatronInferenceHook


class HookTrajectoryCollector:
    """Accumulate a Trajectory by querying MegatronInferenceHook at each step.

    The hook returns *cumulative* KV tensors (the full KV cache up to the
    current token).  This collector slices each completed chunk out of the
    cumulative cache, storing only the new tokens' KV.

    Parameters
    ----------
    hook:   Live MegatronInferenceHook from the Megatron inference engine.
    config: Controls chunk_size, probe_stride, probe_query_len, max_probes.
            Defaults to PipelineConfig().
    """

    def __init__(
        self,
        hook: MegatronInferenceHook,
        config: PipelineConfig = PipelineConfig(),
    ) -> None:
        self._hook = hook
        self._cfg = config
        self._chunks: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        self._probes_by_chunk: dict[int, list[TrainingProbe]] = {}
        self._chunk_start_len: int = 0
        self._cumul_kv_len: int = 0
        self._probe_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_rollout(self) -> None:
        """Reset state — call once before each new rollout."""
        self._chunks = []
        self._probes_by_chunk = {}
        self._chunk_start_len = 0
        self._cumul_kv_len = 0
        self._probe_count = 0

    def on_step(
        self,
        n_new_tokens: int,
        teacher_logits: torch.Tensor | None = None,  # (B, n_new, vocab)
        query_tokens: torch.Tensor | None = None,    # (B, n_new)
        answer_tokens: torch.Tensor | None = None,   # (B, n_ans) — gold targets for task loss
    ) -> None:
        """Record one inference step.

        Call this after each prefill or decode step.  The hook is queried
        for the cumulative KV cache; any newly completed chunks are stored.

        Parameters
        ----------
        n_new_tokens:    Number of new tokens added to the KV cache this step.
        teacher_logits:  Model output logits (B, n_new, vocab) for this step.
                         When None, task loss requires answer_tokens instead.
        query_tokens:    Input token IDs for this step (B, n_new).
        answer_tokens:   Gold answer token IDs for task/retrieval loss.
                         When set (even without teacher_logits), enables task-loss
                         value-directed training without a live teacher model.
        """
        kv = self._hook.get_kv_matrices()
        if kv is None:
            self._cumul_kv_len += n_new_tokens
            return

        keys_per_layer, values_per_layer = kv
        # Use actual tensor length (authoritative even if n_new_tokens is wrong)
        self._cumul_kv_len = keys_per_layer[0].shape[1]

        cfg = self._cfg
        while self._cumul_kv_len - self._chunk_start_len >= cfg.chunk_size:
            chunk_end = self._chunk_start_len + cfg.chunk_size
            chunk_idx = len(self._chunks)

            chunk_keys = [
                k[:, self._chunk_start_len:chunk_end, :].detach()
                for k in keys_per_layer
            ]
            chunk_vals = [
                v[:, self._chunk_start_len:chunk_end, :].detach()
                for v in values_per_layer
            ]
            self._chunks.append((chunk_keys, chunk_vals))

            self._maybe_add_probe(chunk_idx, teacher_logits, query_tokens, answer_tokens)
            self._chunk_start_len = chunk_end

    def end_rollout(self) -> Trajectory:
        """Flush any remaining partial chunk and return the Trajectory.

        The final partial chunk (< chunk_size tokens) is included so that
        nothing is silently dropped.  Training code should handle variable-
        length chunks.
        """
        kv = self._hook.get_kv_matrices()
        remaining = self._cumul_kv_len - self._chunk_start_len

        if kv is not None and remaining > 0:
            keys_per_layer, values_per_layer = kv
            chunk_keys = [
                k[:, self._chunk_start_len:self._cumul_kv_len, :].detach()
                for k in keys_per_layer
            ]
            chunk_vals = [
                v[:, self._chunk_start_len:self._cumul_kv_len, :].detach()
                for v in values_per_layer
            ]
            self._chunks.append((chunk_keys, chunk_vals))

        if not self._chunks:
            import warnings
            warnings.warn(
                "HookTrajectoryCollector.end_rollout: no KV chunks captured. "
                "The KV hook may have returned None for every on_step call. "
                "This trajectory will be empty and skipped during training.",
                stacklevel=2,
            )
        return Trajectory(
            chunks=self._chunks,
            probes_by_chunk=self._probes_by_chunk,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_add_probe(
        self,
        chunk_idx: int,
        teacher_logits: torch.Tensor | None,
        query_tokens: torch.Tensor | None,
        answer_tokens: torch.Tensor | None = None,
    ) -> None:
        cfg = self._cfg
        if chunk_idx % cfg.probe_stride != 0:
            return
        if cfg.max_probes is not None and self._probe_count >= cfg.max_probes:
            return
        # Need at least query_tokens; teacher_logits optional (task loss works without them)
        if query_tokens is None:
            return

        S_q = min(cfg.probe_query_len, query_tokens.shape[1])
        probe = TrainingProbe(
            query_tokens=query_tokens[:, -S_q:],
            teacher_logits=teacher_logits[:, -S_q:, :].detach().clone() if teacher_logits is not None else None,
            answer_tokens=answer_tokens,
        )
        self._probes_by_chunk[chunk_idx] = [probe]
        self._probe_count += 1
