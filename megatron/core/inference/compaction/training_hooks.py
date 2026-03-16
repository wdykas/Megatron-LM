# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Post-training hooks for compaction-in-the-loop (Phase 3).

Provides:
  - Compaction consistency loss (KL divergence after compaction)
  - Teacher distillation loss (student vs AM-teacher compaction)
  - Compaction-in-the-loop rollout integration
  - RL cost terms for compaction frequency/budget
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .am_compaction import am_compact, AMCompactionResult
from .streaming_compactor import StreamingClusterCompactor


@dataclass
class CompactionTrainingConfig:
    """Configuration for compaction-in-the-loop training."""

    # Compaction consistency loss weight
    kl_weight: float = 1.0

    # Teacher distillation loss weight
    distill_weight: float = 0.1

    # RL compaction cost weights
    compaction_frequency_penalty: float = 0.01
    memory_budget_penalty: float = 0.001

    # How often to compute teacher AM compaction for distillation
    teacher_every_n_steps: int = 10

    # Compaction checkpoint interval during rollouts (in tokens)
    compact_checkpoint_interval: int = 512

    # Number of reference queries for teacher
    teacher_num_ref_queries: int = 32

    # Layers to apply consistency loss (None = all)
    consistency_loss_layers: Optional[List[int]] = None

    # Temperature for KL divergence
    kl_temperature: float = 1.0


class CompactionConsistencyLoss(torch.nn.Module):
    """KL divergence loss between full-cache and compacted-cache logits.

    Applied at compaction checkpoints during training rollouts.
    The model learns to produce consistent outputs even after compaction.
    """

    def __init__(self, config: CompactionTrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        logits_full: Tensor,
        logits_compact: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute KL(full || compact) loss.

        Args:
            logits_full: (B, S, V) logits from full cache (teacher/reference).
            logits_compact: (B, S, V) logits from compacted cache.
            mask: (B, S) binary mask for valid positions.

        Returns:
            Scalar KL loss.
        """
        T = self.config.kl_temperature

        log_p = F.log_softmax(logits_full.float() / T, dim=-1)
        log_q = F.log_softmax(logits_compact.float() / T, dim=-1)
        p = log_p.exp()

        # KL(p || q) = sum p * (log_p - log_q)
        kl = (p * (log_p - log_q)).sum(dim=-1)  # (B, S)

        if mask is not None:
            kl = kl * mask.float()
            loss = kl.sum() / mask.float().sum().clamp(min=1)
        else:
            loss = kl.mean()

        return loss * self.config.kl_weight * (T ** 2)


class TeacherDistillationLoss(torch.nn.Module):
    """Distill AM-teacher compaction into the streaming student compactor.

    Periodically runs the expensive AM solver and regresses the student's
    K_mem, V_mem toward the teacher's.
    """

    def __init__(self, config: CompactionTrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        K_student: Tensor,
        V_student: Tensor,
        K_teacher: Tensor,
        V_teacher: Tensor,
    ) -> Tensor:
        """Compute L2 distillation loss.

        Args:
            K_student: (M, H, D) student compacted keys.
            V_student: (M, H, D) student compacted values.
            K_teacher: (M, H, D) teacher (AM) compacted keys.
            V_teacher: (M, H, D) teacher (AM) compacted values.

        Returns:
            Scalar distillation loss.
        """
        k_loss = F.mse_loss(K_student.float(), K_teacher.float())
        v_loss = F.mse_loss(V_student.float(), V_teacher.float())
        return (k_loss + v_loss) * self.config.distill_weight


class AttentionOutputDistillationLoss(torch.nn.Module):
    """Distill attention output matching from teacher to student.

    Instead of raw K/V regression, matches attention outputs for a set
    of probe queries. This is more semantically meaningful.
    """

    def __init__(self, config: CompactionTrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        Q_probe: Tensor,
        K_student: Tensor,
        V_student: Tensor,
        biases_student: Optional[Tensor],
        K_teacher: Tensor,
        V_teacher: Tensor,
        biases_teacher: Optional[Tensor],
        scale: Optional[float] = None,
    ) -> Tensor:
        """Compute attention output matching loss.

        Args:
            Q_probe: (R, H, D) probe queries.
            K_student/V_student: Student compacted KV.
            biases_student: Student biases (optional).
            K_teacher/V_teacher: Teacher compacted KV.
            biases_teacher: Teacher biases (optional).
            scale: Attention scale.

        Returns:
            Scalar loss.
        """
        R, H, D = Q_probe.shape
        if scale is None:
            scale = 1.0 / math.sqrt(D)

        total_loss = torch.tensor(0.0, device=Q_probe.device)

        for h in range(H):
            Qh = Q_probe[:, h, :].float()

            # Teacher output
            Kt = K_teacher[:, h, :].float()
            Vt = V_teacher[:, h, :].float()
            scores_t = Qh @ Kt.T * scale
            if biases_teacher is not None:
                scores_t = scores_t + biases_teacher.float().unsqueeze(0)
            O_teacher = F.softmax(scores_t, dim=-1) @ Vt

            # Student output
            Ks = K_student[:, h, :].float()
            Vs = V_student[:, h, :].float()
            scores_s = Qh @ Ks.T * scale
            if biases_student is not None:
                scores_s = scores_s + biases_student.float().unsqueeze(0)
            O_student = F.softmax(scores_s, dim=-1) @ Vs

            total_loss = total_loss + F.mse_loss(O_student, O_teacher)

        return total_loss / H * self.config.distill_weight


class CompactionRLCost(torch.nn.Module):
    """RL cost term for compaction frequency and memory budget.

    Adds penalties to encourage efficient compaction:
      cost = lambda * frequency_cost + mu * drift_cost

    Used with GRPO/PPO to optimize:
      maximize reward - lambda * compaction_cost - mu * drift/KL
    """

    def __init__(self, config: CompactionTrainingConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        compaction_count: int,
        memory_budget: int,
        kl_drift: float,
        max_sequence_length: int,
    ) -> Tensor:
        """Compute compaction cost for RL.

        Args:
            compaction_count: Number of compaction rounds performed.
            memory_budget: M, number of memory tokens used.
            kl_drift: KL divergence after compaction.
            max_sequence_length: Maximum sequence length for normalization.

        Returns:
            Scalar cost.
        """
        # Frequency cost: penalize frequent compaction
        freq_cost = compaction_count * self.config.compaction_frequency_penalty

        # Budget cost: penalize large memory budgets
        budget_cost = (memory_budget / max_sequence_length) * self.config.memory_budget_penalty

        return torch.tensor(
            freq_cost + budget_cost,
            dtype=torch.float32,
        )


class CompactionInTheLoopTrainer:
    """Orchestrates compaction-in-the-loop during SFT/RL rollouts.

    During rollouts:
      1. Run trajectory normally
      2. At compaction checkpoints, apply student compactor
      3. Continue decoding with compacted cache
      4. Compute consistency + distillation losses
    """

    def __init__(
        self,
        config: CompactionTrainingConfig,
        student_compactor: StreamingClusterCompactor,
        consistency_loss: CompactionConsistencyLoss,
        distill_loss: Optional[TeacherDistillationLoss] = None,
        attn_distill_loss: Optional[AttentionOutputDistillationLoss] = None,
        rl_cost: Optional[CompactionRLCost] = None,
    ):
        self.config = config
        self.student = student_compactor
        self.consistency_loss = consistency_loss
        self.distill_loss = distill_loss
        self.attn_distill_loss = attn_distill_loss
        self.rl_cost = rl_cost
        self.step_count = 0

    def compaction_checkpoint(
        self,
        K_cold: Tensor,
        V_cold: Tensor,
        logits_full: Optional[Tensor] = None,
        logits_compact: Optional[Tensor] = None,
        Q_ref: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Process a compaction checkpoint during a rollout.

        Args:
            K_cold: (T, H, D) cold KV to compact.
            V_cold: (T, H, D) cold values.
            logits_full: Full-cache logits (for consistency loss).
            logits_compact: Compact-cache logits (for consistency loss).
            Q_ref: Reference queries (for teacher distillation).

        Returns:
            Dict of losses and the compacted K_mem, V_mem.
        """
        losses = {}

        # Student compaction
        K_student, V_student = self.student.compact(K_cold, V_cold)

        # Consistency loss
        if logits_full is not None and logits_compact is not None:
            losses["consistency"] = self.consistency_loss(
                logits_full.unsqueeze(0),
                logits_compact.unsqueeze(0),
            )

        # Teacher distillation (periodic)
        self.step_count += 1
        if (
            self.distill_loss is not None
            and Q_ref is not None
            and self.step_count % self.config.teacher_every_n_steps == 0
        ):
            T, H, D = K_cold.shape
            M = self.student.num_anchors
            with torch.no_grad():
                teacher_result = am_compact(
                    K_cold, V_cold, Q_ref, M,
                    method="top_attention",
                    nnls_iters=50,
                )

            losses["distill_kv"] = self.distill_loss(
                K_student, V_student,
                teacher_result.K_mem.to(K_student.dtype),
                teacher_result.V_mem.to(V_student.dtype),
            )

            if self.attn_distill_loss is not None and Q_ref is not None:
                losses["distill_attn"] = self.attn_distill_loss(
                    Q_ref, K_student, V_student, None,
                    teacher_result.K_mem, teacher_result.V_mem,
                    teacher_result.biases, None,
                )

        losses["K_mem"] = K_student
        losses["V_mem"] = V_student

        return losses

    def compute_total_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        """Sum all compaction-related losses."""
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for key, val in loss_dict.items():
            if key in ("K_mem", "V_mem"):
                continue
            if isinstance(val, Tensor) and val.dim() == 0:
                total = total + val
        return total
