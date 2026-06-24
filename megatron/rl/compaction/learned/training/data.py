# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training data types, type aliases, and dataset utilities for Still."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Per-layer compact KV cache: n_layers × (K (B, C, d), V (B, C, d))
CompactKV = list[tuple[torch.Tensor, torch.Tensor]]

# Frozen model forward: (query_tokens, compact_kv) → logits (B, S_q, vocab)
StudentFn = Callable[[torch.Tensor, CompactKV], torch.Tensor]


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class TrainingProbe:
    """A single evaluation point for a training step.

    Attributes
    ----------
    query_tokens:     Token IDs for the probe query — (B, S_q).
    teacher_logits:   Logits from the frozen model with full KV context —
                      (B, S_q, vocab).  Pre-computed once and reused.
    answer_tokens:    Gold answer token IDs — (B, S_a) with -100 for ignored
                      positions.  Required when is_exact_retrieval=True or
                      when the task/retrieval loss weight is non-zero.
    is_exact_retrieval: If True, use CompactorLossWeights.retrieval for this
                      probe's answer loss instead of the task weight.
    advantage:        Per-probe GRPO advantage for value-directed training.
                      When set, replaces the uniform teacher_kl weight with an
                      advantage-proportional weight.  None means no weighting.
    """

    query_tokens:      torch.Tensor              # (B, S_q)
    teacher_logits:    torch.Tensor | None = None  # (B, S_q, vocab) — None disables teacher KL
    answer_tokens:     torch.Tensor | None = None
    is_exact_retrieval: bool = False
    advantage:         float | None = None


@dataclass
class Trajectory:
    """A full training trajectory: KV chunks + probes.

    chunks:          One (keys_per_layer, values_per_layer) tuple per chunk.
                     Each list has n_layers elements of shape (B, S_chunk, d).
    probes_by_chunk: Mapping from chunk index to the probes that should be
                     evaluated after that chunk has been incorporated into
                     the belief memory M_{chunk_idx}.
    """

    chunks: list[tuple[list[torch.Tensor], list[torch.Tensor]]]
    probes_by_chunk: dict[int, list[TrainingProbe]] = field(default_factory=dict)
    rollout_return:          float | None = None
    # Per-token log-probs from the teacher (full-KV rollout).  Stored as a flat
    # list over response tokens.  Used by STILL training to weight kv_recon by
    # teacher confidence rather than task reward.
    teacher_logprob_return:  float | None = None

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def probes_at(self, chunk_idx: int) -> list[TrainingProbe]:
        return self.probes_by_chunk.get(chunk_idx, [])

    @property
    def device(self) -> torch.device:
        k, _ = self.chunks[0]
        return k[0].device


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

class TrajectoryDataset(Dataset):
    """Dataset of Trajectory objects for BeliefCompactorTrainer.

    Each element is a complete offline Trajectory (KV chunks + probes)
    collected from a frozen model by TrajectoryBuilder.
    """

    def __init__(self, trajectories: list[Trajectory]) -> None:
        self.trajectories = trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]


def trajectory_collate_fn(batch: list[Trajectory]) -> list[Trajectory]:
    """Pass-through collation for DataLoader.

    Trajectories have variable chunk counts so stacking is not possible.
    Pass as ``collate_fn`` to DataLoader:
        DataLoader(dataset, collate_fn=trajectory_collate_fn)
    """
    return batch


# ---------------------------------------------------------------------------
# Trainer config (moved here from trainer.py so rl_utils.py can import it
# without pulling in the full trainer module)
# ---------------------------------------------------------------------------

@dataclass
class CompactorTrainerConfig:
    """Shared configuration for SinglePassCompactorTrainer and BeliefCompactorTrainer.

    Attributes
    ----------
    loss_weights:        Per-term loss weights.
    temperature:         KL distillation temperature.
    weighted_kl_rho:     Confidence-weighting strength (section 10 of design doc).
    truncated_bptt_steps: Detach belief memory and step optimizer every N
                         chunks during BeliefCompactorTrainer.train_trajectory.
                         Smaller values reduce memory but cut gradient horizon.
    clip_grad_norm:      Max gradient norm. None disables clipping.
    vd_cfg:              Optional ValueDirectedConfig for value-directed training.
    """

    loss_weights:              "CompactorLossWeights | None" = None   # lazy import; see __post_init__
    temperature:               float = 1.0
    weighted_kl_rho:           float = 1.0
    truncated_bptt_steps:      int = 8
    clip_grad_norm:            float | None = 1.0
    vd_cfg:                    "ValueDirectedConfig | None" = None
    # When True, weight kv_recon by teacher_logprob_return (teacher confidence)
    # instead of rollout_return (task reward).  Implements the STILL paper objective:
    # compress harder where the full-KV model was most confident.
    use_teacher_logprob_weight: bool = False
    # When True, use the differentiable student forward (attention hook) to compute
    # CE(model(response | compact_kv), response_tokens) as the training signal.
    # This is the true STILL paper objective.  Requires a live model via compactor_student_model.
    use_teacher_kl: bool = False
    # Discount factor applied across future chunks when computing a horizon-weighted
    # accuracy signal.  1.0 = no discounting (uniform weight across all future chunks).
    future_horizon_gamma:       float = 1.0
    # When True, weight the loss for each chunk by its discounted future accuracy
    # (how well the compact memory supports future probes) rather than uniform weighting.
    use_future_accuracy_weight: bool = False
    # Probability that two adjacent chunks are merged into a single larger chunk
    # during trajectory construction.  0.0 = no merging.
    merged_chunk_prob:          float = 0.0

    def __post_init__(self) -> None:
        if self.loss_weights is None:
            from megatron.rl.compaction.learned.training.losses import CompactorLossWeights
            self.loss_weights = CompactorLossWeights()


# ---------------------------------------------------------------------------
# FrozenModelAdapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FrozenModelAdapter(Protocol):
    """Interface to a frozen LLM for data collection and student evaluation.

    Implementers handle:
        • KV cache extraction from the forward pass
        • RoPE stripping (prefill returns position-FREE keys)
        • KV cache injection for the student forward pass

    All tensors use batch-first layout: (B, S, ...).
    """

    @property
    def n_layers(self) -> int: ...
    @property
    def d_head(self) -> int: ...
    @property
    def n_heads(self) -> int: ...
    @property
    def vocab_size(self) -> int: ...

    def prefill(
        self,
        token_ids: torch.Tensor,           # (B, S)
        kv_prefix: "CompactKV | None",     # position-free KV to prepend
        logical_start: int,                # logical position of token_ids[0]
    ) -> "tuple[CompactKV, torch.Tensor]":  # (position-free kv, logits (B, S, vocab))
        """Run the frozen LLM on token_ids, optionally with a KV prefix."""
        ...

    def student_logits(
        self,
        query_tokens: torch.Tensor,   # (B, S_q)
        compact_kv: "CompactKV",      # position-free compact memory
        logical_kv_start: int,        # logical position of first compact KV slot
        logical_query_start: int,     # logical position of first query token
    ) -> torch.Tensor:               # (B, S_q, vocab)
        """Evaluate the LLM on query_tokens conditioned on compact KV memory."""
        ...


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for TrajectoryBuilder."""

    chunk_size:     int = 256
    # Variable-size chunking: if chunk_size_max > chunk_size, each chunk's
    # length is sampled uniformly from [chunk_size, chunk_size_max].
    # This trains the updater to handle compaction at arbitrary boundaries.
    chunk_size_max: int = 0        # 0 = fixed (use chunk_size for all chunks)
    probe_stride:   int = 1
    probe_query_len: int = 32
    max_probes:     int | None = None


# ---------------------------------------------------------------------------
# TrajectoryBuilder
# ---------------------------------------------------------------------------

class TrajectoryBuilder:
    """Collect KV chunks and teacher logits from a frozen model.

    Usage:
        config  = PipelineConfig(chunk_size=512, probe_stride=2)
        builder = TrajectoryBuilder(config)
        traj    = builder.build(token_ids, adapter)
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.cfg = config

    def build(
        self,
        token_ids: torch.Tensor,          # (B, total_len)
        adapter: FrozenModelAdapter,
    ) -> Trajectory:
        """Segment token_ids into chunks and collect KV + teacher logits.

        Chunk sizes are fixed (cfg.chunk_size) unless cfg.chunk_size_max > 0,
        in which case each chunk length is sampled uniformly from
        [cfg.chunk_size, cfg.chunk_size_max].  Variable sizes train the updater
        to compact at arbitrary boundaries, matching on-demand inference.
        """
        import random
        B, total_len = token_ids.shape
        cfg = self.cfg
        variable = cfg.chunk_size_max > cfg.chunk_size

        # Build chunk boundary list
        chunk_starts: list[int] = []
        pos = 0
        while pos < total_len:
            chunk_starts.append(pos)
            if variable:
                pos += random.randint(cfg.chunk_size, cfg.chunk_size_max)
            else:
                pos += cfg.chunk_size

        chunks: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        probes_by_chunk: dict[int, list[TrainingProbe]] = {}
        n_probes = 0

        for chunk_idx, start in enumerate(chunk_starts):
            end = min(
                chunk_starts[chunk_idx + 1] if chunk_idx + 1 < len(chunk_starts) else total_len,
                total_len,
            )
            chunk_kv, _ = adapter.prefill(
                token_ids[:, start:end], kv_prefix=None, logical_start=start,
            )
            chunks.append(([k for k, _ in chunk_kv], [v for _, v in chunk_kv]))

            if chunk_idx % cfg.probe_stride == 0:
                if cfg.max_probes is None or n_probes < cfg.max_probes:
                    probe = self._make_probe(chunk_idx, chunk_starts, token_ids, adapter, cfg, end)
                    if probe is not None:
                        probes_by_chunk[chunk_idx] = [probe]
                        n_probes += 1

        return Trajectory(chunks=chunks, probes_by_chunk=probes_by_chunk)

    def _make_probe(self, chunk_idx, chunk_starts, token_ids, adapter, cfg, chunk_end=None):
        B, total_len = token_ids.shape
        if chunk_end is None:
            chunk_end = min(chunk_starts[chunk_idx] + cfg.chunk_size, total_len)
        q_start = chunk_end
        q_end   = min(q_start + cfg.probe_query_len, total_len)
        if q_start >= total_len:
            return None
        query_tokens = token_ids[:, q_start:q_end]
        _, context_logits = adapter.prefill(
            token_ids[:, :q_end], kv_prefix=None, logical_start=0,
        )
        S_q = q_end - q_start
        return TrainingProbe(
            query_tokens=query_tokens,
            teacher_logits=context_logits[:, -S_q:, :].detach(),
        )
