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
    collector.on_step(teacher_logits=engine.last_logits, query_tokens=prompt_tokens)

    # During decode:
    for _ in range(max_new_tokens):
        engine.step(...)
        collector.on_step()

    trajectory = collector.end_rollout()

KV format
---------
Keys and values are stored *as returned by the hook* — typically with RoPE
already applied.  This is fine for BeliefUpdater training (the updater learns
to compress positional keys).  If you need position-free keys for a
student forward pass, use kv/rope.py to strip them before saving.
"""

from __future__ import annotations


import torch

from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe
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
        self._probe_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_rollout(self) -> None:
        """Reset state — call once before each new rollout."""
        self._chunks = []
        self._probes_by_chunk = {}
        self._chunk_start_len = 0
        self._probe_count = 0

    def on_step(
        self,
        teacher_logits: torch.Tensor | None = None,  # (B, n_new, vocab)
        query_tokens: torch.Tensor | None = None,    # (B, n_new)
        answer_tokens: torch.Tensor | None = None,   # (B, n_ans) — gold targets for task loss
    ) -> None:
        """Record one inference step, cutting any newly-completed chunks.

        Call this after each prefill or decode step.  The hook is queried for
        the cumulative KV cache; the new tokens' length is read from the cache
        tensor itself, so no token count is passed in.

        Parameters
        ----------
        teacher_logits:  Model output logits (B, n_new, vocab) for this step.
                         When None, task loss requires answer_tokens instead.
        query_tokens:    Input token IDs for this step (B, n_new).
        answer_tokens:   Gold answer token IDs for task/retrieval loss.
                         When set (even without teacher_logits), enables task-loss
                         value-directed training without a live teacher model.
        """
        keys_per_layer, values_per_layer = self._require_kv()
        cumul = keys_per_layer[0].shape[1]

        cfg = self._cfg
        while cumul - self._chunk_start_len >= cfg.chunk_size:
            chunk_end = self._chunk_start_len + cfg.chunk_size
            chunk_idx = len(self._chunks)
            self._chunks.append(
                self._slice_chunk(keys_per_layer, values_per_layer, self._chunk_start_len, chunk_end)
            )
            self._maybe_add_probe(chunk_idx, teacher_logits, query_tokens, answer_tokens)
            self._chunk_start_len = chunk_end

    def end_rollout(self) -> Trajectory:
        """Flush any remaining partial chunk and return the Trajectory.

        The final partial chunk (< chunk_size tokens) is included so that
        nothing is silently dropped.  Training code should handle variable-
        length chunks.
        """
        keys_per_layer, values_per_layer = self._require_kv()
        cumul = keys_per_layer[0].shape[1]
        if cumul > self._chunk_start_len:
            self._chunks.append(
                self._slice_chunk(keys_per_layer, values_per_layer, self._chunk_start_len, cumul)
            )

        if not self._chunks:
            raise RuntimeError(
                "HookTrajectoryCollector captured no KV: the cache was empty for the "
                "whole rollout. The compactor cannot train on an empty trajectory."
            )
        return Trajectory(
            chunks=self._chunks,
            probes_by_chunk=self._probes_by_chunk,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_kv(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        kv = self._hook.get_kv_matrices()
        if kv is None:
            raise RuntimeError(
                "MegatronInferenceHook returned no KV matrices; the hook is not "
                "capturing. Check that KV capture is enabled on this rank."
            )
        return kv

    @staticmethod
    def _slice_chunk(
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
        start: int,
        end: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        keys = [k[:, start:end, :].detach() for k in keys_per_layer]
        vals = [v[:, start:end, :].detach() for v in values_per_layer]
        return keys, vals

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
        # A probe needs query_tokens; teacher_logits is optional (task loss works without).
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
