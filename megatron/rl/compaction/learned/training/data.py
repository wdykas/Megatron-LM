# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training data types, type aliases, and dataset utilities for Still."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Per-layer compact KV cache: n_layers × (K (B, C, d), V (B, C, d))
CompactKV = list[tuple[torch.Tensor, torch.Tensor]]


class StudentOutput(NamedTuple):
    """One frozen-model forward over probe tokens with compact KV injected.

    logits: (B, S_q, vocab) — differentiable w.r.t. the compact KV.
    hidden: (B, S_q, d_model) — final decoder output (post final norm), the
        target space of the NextLat future-latent loss (NextLat future-latent).
    """

    logits: torch.Tensor
    hidden: torch.Tensor


# Frozen model forward: (query_tokens, compact_kv) → StudentOutput
StudentFn = Callable[[torch.Tensor, CompactKV], "StudentOutput"]


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
    teacher_hidden:    torch.Tensor | None = None  # (B, S_q, d_model) full-KV final hidden — None disables future_latent
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
    loaded from disk (saved during rollout collection).
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


def save_trajectory(
    trajectory: Trajectory, trajectory_dir: str, iteration: int, prompt_idx: int
) -> str:
    """Pickle a Trajectory to ``trajectory_dir`` for the offline pipeline.

    Tensors are moved to CPU first so the file loads on any machine. Returns
    the written path (``iter<NNNNNNN>_prompt<NNNNN>.pt``).
    """
    import dataclasses
    import os
    import pickle

    cpu_chunks = [
        ([k.cpu() for k in chunk_keys], [v.cpu() for v in chunk_vals])
        for chunk_keys, chunk_vals in trajectory.chunks
    ]
    cpu_probes: dict[int, list[TrainingProbe]] = {}
    for chunk_idx, probes in trajectory.probes_by_chunk.items():
        cpu_probes[chunk_idx] = [
            dataclasses.replace(p, **{
                f.name: getattr(p, f.name).cpu()
                for f in dataclasses.fields(p)
                if isinstance(getattr(p, f.name), torch.Tensor)
            })
            for p in probes
        ]
    cpu_traj = Trajectory(
        chunks=cpu_chunks,
        probes_by_chunk=cpu_probes,
        rollout_return=trajectory.rollout_return,
        teacher_logprob_return=trajectory.teacher_logprob_return,
    )

    os.makedirs(trajectory_dir, exist_ok=True)
    path = os.path.join(trajectory_dir, f"iter{iteration:07d}_prompt{prompt_idx:05d}.pt")
    with open(path, "wb") as f:
        pickle.dump(cpu_traj, f)
    return path


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


