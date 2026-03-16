#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Example: RL training with compaction-in-the-loop.

This script demonstrates how to train a model to be robust to KV cache
compaction using GRPO-style RL. The key idea:

  1. Generate rollouts normally (build full KV cache)
  2. At compaction checkpoints, compact the KV cache
  3. Continue generation with the compacted cache
  4. Compute rewards on the full trajectory
  5. Add compaction consistency loss (KL between full and compacted logits)
  6. Optionally distill AM-teacher compaction into the streaming student

This integrates with Megatron-LM's existing GRPO training loop.

Usage:
    # Standalone example (no distributed, for demonstration):
    python examples/compaction/train_compaction_rl.py

    # With Megatron-LM GRPO training (modify train_rl.py):
    # See integration instructions at the bottom of this file.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Import compaction modules
# ---------------------------------------------------------------------------
from megatron.core.inference.compaction import (
    CompactionConfig,
    CompactionManager,
    StreamingClusterCompactor,
    StreamingCompactorConfig,
    am_compact,
    gather_kv,
    write_kv,
    validate_attention_output,
)
from megatron.core.inference.compaction.training_hooks import (
    AttentionOutputDistillationLoss,
    CompactionConsistencyLoss,
    CompactionInTheLoopTrainer,
    CompactionRLCost,
    CompactionTrainingConfig,
    TeacherDistillationLoss,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompactionRLConfig:
    """Full configuration for compaction-aware RL training."""

    # --- Model dimensions (set from model config) ---
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    block_size: int = 64

    # --- Compaction settings ---
    memory_budget: int = 512       # M: tokens in compact memory
    hot_window: int = 256          # W: recent tokens kept uncompacted
    compact_every_n: int = 256     # Compact when hot window overflows by this

    # --- Student compactor ---
    num_anchors: int = 512         # Anchors for streaming compactor
    routing: str = "top1"          # "top1" or "top2"
    heads_per_group: int = 1       # Head grouping for compactor
    learnable_anchors: bool = True # Train anchor positions

    # --- Training losses ---
    kl_weight: float = 1.0         # Weight for KL(full || compact)
    distill_weight: float = 0.1    # Weight for teacher distillation
    rl_frequency_penalty: float = 0.01
    rl_budget_penalty: float = 0.001
    teacher_every_n_steps: int = 10  # Run AM teacher this often

    # --- GRPO settings (standard) ---
    grpo_kl_beta: float = 0.1      # KL penalty in GRPO
    grpo_clip_eps: float = 0.2     # Ratio clipping
    entropy_weight: float = 0.01

    # --- Training ---
    lr: float = 1e-5
    num_epochs: int = 3
    max_seq_length: int = 8192


# ---------------------------------------------------------------------------
# Compaction-aware rollout wrapper
# ---------------------------------------------------------------------------

class CompactionAwareRolloutEngine:
    """Wraps the inference engine to apply compaction during generation.

    During RL rollouts, this engine:
    1. Generates tokens normally until the hot window fills
    2. Applies the student compactor to compress cold KV
    3. Continues generation with the two-tier [mem + hot] cache
    4. Records logits at compaction boundaries for consistency loss
    """

    def __init__(
        self,
        config: CompactionRLConfig,
        student_compactor: StreamingClusterCompactor,
        memory_buffer: Tensor,
        block_allocator,
    ):
        self.config = config
        self.student = student_compactor

        self.compaction_mgr = CompactionManager(
            config=CompactionConfig(
                memory_budget=config.memory_budget,
                hot_window=config.hot_window,
                compact_every_n=config.compact_every_n,
                method="top_attention",
                use_mass_matching=True,
                store_biases=True,
            ),
            memory_buffer=memory_buffer,
            block_allocator=block_allocator,
            block_size=config.block_size,
            num_layers=config.num_layers,
        )

        # Buffers for collecting training signal
        self.logits_at_boundaries: Dict[int, List[Tuple[Tensor, Tensor]]] = {}

    def on_generation_step(
        self,
        seq_id: int,
        step: int,
        block_ids: Tensor,
        token_count: int,
        logits: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        """Called after each token generation step.

        Returns updated (block_table, kv_len) if compaction occurred, else (None, None).
        """
        state = self.compaction_mgr.seq_states.get(seq_id)
        if state is None:
            self.compaction_mgr.register_sequence(seq_id, initial_pos=token_count)
            state = self.compaction_mgr.seq_states[seq_id]
            state.hot_token_count = token_count

        self.compaction_mgr.advance_position(seq_id)

        # Check if compaction is needed
        if not self.compaction_mgr.should_compact(seq_id):
            return None, None

        # Record pre-compaction logits for consistency loss
        if logits is not None:
            if seq_id not in self.logits_at_boundaries:
                self.logits_at_boundaries[seq_id] = []
            self.logits_at_boundaries[seq_id].append(("pre", logits.detach().clone()))

        # Use student compactor (fast) instead of full AM
        new_block_table, new_kv_len = self._compact_with_student(
            seq_id, block_ids, token_count
        )

        return new_block_table, new_kv_len

    def _compact_with_student(
        self,
        seq_id: int,
        block_ids: Tensor,
        token_count: int,
    ) -> Tuple[Tensor, int]:
        """Run student compaction on the cold prefix."""
        M = self.config.memory_budget
        W = self.config.hot_window
        bs = self.config.block_size

        hot_start = max(0, token_count - W)
        cold_token_count = hot_start

        if cold_token_count <= M:
            return block_ids, token_count

        cold_num_blocks = math.ceil(cold_token_count / bs)
        cold_block_ids = block_ids[:cold_num_blocks]
        hot_block_ids = block_ids[cold_num_blocks:]
        hot_token_count = token_count - cold_token_count

        # Allocate memory blocks
        num_mem_blocks = math.ceil(M / bs)
        new_mem_blocks = self.compaction_mgr.block_allocator.allocate_memory_blocks(num_mem_blocks)
        if new_mem_blocks is None:
            return block_ids, token_count

        buf = self.compaction_mgr.memory_buffer

        for layer in range(self.config.num_layers):
            # Student compactor: stream over cold pages directly
            K_mem, V_mem = self.student.compact_from_pages(
                buf, layer, cold_block_ids, bs, cold_token_count,
            )
            write_kv(buf, layer, new_mem_blocks, bs, K_mem, V_mem)

        # Free cold blocks
        self.compaction_mgr.block_allocator.release_memory_blocks(cold_block_ids)

        # Update state
        state = self.compaction_mgr.seq_states[seq_id]
        if state.mem_block_ids is not None:
            self.compaction_mgr.block_allocator.release_memory_blocks(state.mem_block_ids)
        state.mem_block_ids = new_mem_blocks
        state.mem_token_count = M
        state.hot_block_ids = hot_block_ids
        state.hot_token_count = hot_token_count
        state.compaction_count += 1

        new_block_table = torch.cat([new_mem_blocks, hot_block_ids])
        new_kv_len = M + hot_token_count

        return new_block_table, new_kv_len


# ---------------------------------------------------------------------------
# Training step: compaction-augmented GRPO loss
# ---------------------------------------------------------------------------

def compute_compaction_augmented_loss(
    # Standard GRPO inputs
    current_logprobs: Tensor,       # (B, S) policy logprobs
    old_logprobs: Tensor,           # (B, S) rollout logprobs
    ref_logprobs: Tensor,           # (B, S) reference model logprobs
    advantages: Tensor,             # (B,) GRPO advantages
    loss_mask: Tensor,              # (B, S) valid token mask
    # Compaction inputs
    logits_full: Optional[Tensor],  # (B, S, V) full-cache logits at boundary
    logits_compact: Optional[Tensor],  # (B, S, V) compact-cache logits at boundary
    compaction_count: int,          # Number of compactions in this trajectory
    # Student compactor for distillation
    student: Optional[StreamingClusterCompactor] = None,
    K_cold: Optional[Tensor] = None,
    V_cold: Optional[Tensor] = None,
    Q_ref: Optional[Tensor] = None,
    # Config
    config: CompactionRLConfig = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute GRPO loss augmented with compaction consistency terms.

    Loss = GRPO_loss + kl_weight * KL(full || compact) + distill_weight * distill_loss

    Args:
        Standard GRPO tensors plus compaction-specific inputs.

    Returns:
        total_loss: Scalar loss for backward.
        metrics: Dict of logging metrics.
    """
    if config is None:
        config = CompactionRLConfig()

    metrics = {}

    # ------------------------------------------------------------------
    # 1. Standard GRPO loss
    # ------------------------------------------------------------------
    ratios = (current_logprobs - old_logprobs).exp()
    clamped_ratios = ratios.clamp(1 - config.grpo_clip_eps, 1 + config.grpo_clip_eps)

    adv = advantages.unsqueeze(1)  # (B, 1) broadcast to (B, S)
    surrogate = clamped_ratios * adv

    # KL penalty (ref model)
    ref_diff = ref_logprobs - current_logprobs
    kl_penalty = ref_diff.exp() - ref_diff - 1

    # Entropy bonus
    entropy = -current_logprobs.exp() * current_logprobs

    grpo_loss = -(surrogate - config.grpo_kl_beta * kl_penalty + config.entropy_weight * entropy)

    # Mask and reduce
    grpo_loss = (grpo_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    metrics["grpo_loss"] = grpo_loss.item()

    # Only include GRPO loss if it has gradients (i.e., real model forward)
    if grpo_loss.requires_grad:
        total_loss = grpo_loss
    else:
        total_loss = torch.tensor(0.0, device=grpo_loss.device, requires_grad=False)

    # ------------------------------------------------------------------
    # 2. Compaction consistency loss: KL(full_logits || compact_logits)
    # ------------------------------------------------------------------
    if logits_full is not None and logits_compact is not None:
        consistency_fn = CompactionConsistencyLoss(
            CompactionTrainingConfig(kl_weight=config.kl_weight)
        )
        consistency_loss = consistency_fn(logits_full, logits_compact)
        total_loss = total_loss + consistency_loss
        metrics["compaction_consistency_loss"] = consistency_loss.item()

    # ------------------------------------------------------------------
    # 3. Teacher distillation loss (periodic)
    # ------------------------------------------------------------------
    if (
        student is not None
        and K_cold is not None
        and V_cold is not None
        and Q_ref is not None
        and config.distill_weight > 0
    ):
        M = config.memory_budget
        K_student, V_student = student(K_cold, V_cold)  # Uses compact_soft in train mode

        with torch.no_grad():
            teacher_result = am_compact(
                K_cold, V_cold, Q_ref, M,
                method="top_attention", nnls_iters=0,
            )

        distill_fn = TeacherDistillationLoss(
            CompactionTrainingConfig(distill_weight=config.distill_weight)
        )
        distill_loss = distill_fn(
            K_student, V_student,
            teacher_result.K_mem.to(K_student.dtype),
            teacher_result.V_mem.to(V_student.dtype),
        )
        total_loss = total_loss + distill_loss
        metrics["distill_loss"] = distill_loss.item()

    # ------------------------------------------------------------------
    # 4. Compaction cost (for RL reward shaping)
    # ------------------------------------------------------------------
    if compaction_count > 0:
        cost_fn = CompactionRLCost(CompactionTrainingConfig(
            compaction_frequency_penalty=config.rl_frequency_penalty,
            memory_budget_penalty=config.rl_budget_penalty,
        ))
        compaction_cost = cost_fn(
            compaction_count, config.memory_budget,
            metrics.get("compaction_consistency_loss", 0.0),
            config.max_seq_length,
        )
        metrics["compaction_cost"] = compaction_cost.item()

    metrics["total_loss"] = total_loss.item()
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Standalone training example
# ---------------------------------------------------------------------------

def create_synthetic_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: torch.device,
) -> Dict[str, Tensor]:
    """Create a synthetic batch for demonstration."""
    return {
        "tokens": torch.randint(0, vocab_size, (batch_size, seq_len), device=device),
        "current_logprobs": torch.randn(batch_size, seq_len - 1, device=device) * 0.1 - 5.0,
        "old_logprobs": torch.randn(batch_size, seq_len - 1, device=device) * 0.1 - 5.0,
        "ref_logprobs": torch.randn(batch_size, seq_len - 1, device=device) * 0.1 - 5.0,
        "advantages": torch.randn(batch_size, device=device),
        "loss_mask": torch.ones(batch_size, seq_len - 1, device=device),
    }


def create_synthetic_kv(
    num_layers: int, T: int, H: int, D: int, device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Create synthetic KV cache tensors for demonstration."""
    K = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    return K, V


def run_standalone_example():
    """Run a standalone compaction RL training example.

    This demonstrates the training loop without requiring a full
    Megatron-LM distributed setup.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running compaction RL training example on {device}")

    # ---- Configuration ----
    config = CompactionRLConfig(
        num_layers=4,
        num_kv_heads=8,
        head_dim=64,
        block_size=16,
        memory_budget=64,
        hot_window=128,
        compact_every_n=128,
        num_anchors=64,
        routing="top1",
        heads_per_group=1,
        learnable_anchors=True,
        kl_weight=1.0,
        distill_weight=0.1,
        teacher_every_n_steps=5,
        lr=1e-4,
        num_epochs=3,
        max_seq_length=512,
    )

    vocab_size = 1000
    batch_size = 2
    seq_len = 256

    # ---- Create student compactor (trainable) ----
    student = StreamingClusterCompactor(
        head_dim=config.head_dim,
        num_heads=config.num_kv_heads,
        config=StreamingCompactorConfig(
            num_anchors=config.num_anchors,
            routing=config.routing,
            heads_per_group=config.heads_per_group,
            learnable_anchors=config.learnable_anchors,
        ),
    ).to(device)

    # Initialize anchors from synthetic data
    K_init = torch.randn(256, config.num_kv_heads, config.head_dim, device=device)
    student.initialize_anchors_from_data(K_init)

    # ---- Optimizer (trains the student compactor) ----
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)

    # ---- Training loop ----
    print(f"\nTraining for {config.num_epochs} epochs...")
    print(f"  Student compactor: {sum(p.numel() for p in student.parameters())} params")
    print(f"  Memory budget: {config.memory_budget} tokens")
    print(f"  Hot window: {config.hot_window} tokens")
    print()

    for epoch in range(config.num_epochs):
        student.train()  # Enable soft routing for differentiable training
        epoch_metrics = {
            "total_loss": 0.0,
            "grpo_loss": 0.0,
            "consistency_loss": 0.0,
            "distill_loss": 0.0,
        }
        num_steps = 10

        for step in range(num_steps):
            optimizer.zero_grad()

            # Simulate a GRPO batch
            batch = create_synthetic_batch(batch_size, seq_len, vocab_size, device)

            # Simulate KV cache state at a compaction boundary
            T = 256  # Tokens before compaction point
            K_cold, V_cold = create_synthetic_kv(
                config.num_layers, T, config.num_kv_heads, config.head_dim, device,
            )
            Q_ref = torch.randn(
                16, config.num_kv_heads, config.head_dim, device=device, dtype=torch.bfloat16,
            )

            # Simulate full vs compact logits at boundary
            logits_full = torch.randn(batch_size, 32, vocab_size, device=device)
            logits_compact = logits_full + torch.randn_like(logits_full) * 0.3

            # Compute augmented loss
            # In standalone mode, always distill so student gets gradients.
            # In full integration, GRPO loss backprops through the model and
            # distillation provides gradients for the student compactor.
            loss, metrics = compute_compaction_augmented_loss(
                current_logprobs=batch["current_logprobs"],
                old_logprobs=batch["old_logprobs"],
                ref_logprobs=batch["ref_logprobs"],
                advantages=batch["advantages"],
                loss_mask=batch["loss_mask"],
                logits_full=logits_full,
                logits_compact=logits_compact,
                compaction_count=1,
                student=student,
                K_cold=K_cold,
                V_cold=V_cold,
                Q_ref=Q_ref,
                config=config,
            )

            # Backward
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]

        # Report
        for k in epoch_metrics:
            epoch_metrics[k] /= num_steps

        print(
            f"Epoch {epoch+1}/{config.num_epochs}: "
            f"total_loss={epoch_metrics['total_loss']:.4f}, "
            f"grpo={epoch_metrics['grpo_loss']:.4f}, "
            f"consistency={epoch_metrics.get('consistency_loss', 0):.4f}, "
            f"distill={epoch_metrics.get('distill_loss', 0):.6f}"
        )

    # ---- Evaluate student quality ----
    print("\nEvaluating student compactor quality...")
    student.eval()  # Switch to hard routing for inference
    with torch.no_grad():
        K_eval, V_eval = create_synthetic_kv(
            1, 512, config.num_kv_heads, config.head_dim, device,
        )
        Q_eval = torch.randn(
            16, config.num_kv_heads, config.head_dim, device=device, dtype=torch.bfloat16,
        )

        # Student compaction
        K_student, V_student = student.compact(K_eval, V_eval)

        # AM teacher compaction
        teacher_result = am_compact(
            K_eval, V_eval, Q_eval, config.memory_budget,
            method="top_attention", nnls_iters=0,
        )

        # Evaluate both against full cache
        m_student = validate_attention_output(
            K_eval, V_eval, K_student, V_student, Q_eval,
        )
        m_teacher = validate_attention_output(
            K_eval, V_eval,
            teacher_result.K_mem, teacher_result.V_mem, Q_eval,
            teacher_result.biases,
        )

        print(f"  Student attention error:  {m_student.mean_relative_l2:.4f}")
        print(f"  Teacher attention error:  {m_teacher.mean_relative_l2:.4f}")
        print(f"  Student/teacher ratio:    {m_student.mean_relative_l2 / max(m_teacher.mean_relative_l2, 1e-8):.2f}x")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Integration instructions for train_rl.py
# ---------------------------------------------------------------------------

INTEGRATION_NOTES = """
==========================================================================
HOW TO INTEGRATE COMPACTION INTO MEGATRON-LM's GRPO TRAINING (train_rl.py)
==========================================================================

1. In get_environment_rollouts(), wrap the inference engine:

    from megatron.core.inference.compaction import (
        StreamingClusterCompactor, StreamingCompactorConfig,
    )
    from examples.compaction.train_compaction_rl import CompactionAwareRolloutEngine

    # Create student compactor (once, at init)
    student = StreamingClusterCompactor(head_dim, num_kv_heads, config).cuda()

    # Wrap the inference context
    rollout_engine = CompactionAwareRolloutEngine(
        config=compaction_config,
        student_compactor=student,
        memory_buffer=inference_context.memory_buffer,
        block_allocator=inference_context.block_allocator,
    )

    # In the generation loop, call:
    new_bt, new_kv_len = rollout_engine.on_generation_step(
        seq_id, step, block_ids, token_count, logits,
    )
    if new_bt is not None:
        # Update block table and kv_len in the inference context
        update_block_table(seq_id, new_bt, new_kv_len)

2. In forward_step(), add compaction loss:

    from examples.compaction.train_compaction_rl import (
        compute_compaction_augmented_loss,
    )

    # Replace the standard calculate_grpo_loss with:
    loss, metrics = compute_compaction_augmented_loss(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        logits_full=logits_at_boundary_full,
        logits_compact=logits_at_boundary_compact,
        compaction_count=num_compactions,
        student=student,
        K_cold=K_cold,
        V_cold=V_cold,
        Q_ref=Q_ref,
        config=compaction_config,
    )

3. Add student compactor parameters to the optimizer:

    # In the optimizer setup:
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(student.parameters()),
        lr=args.lr,
    )

4. Monitor compaction metrics in TensorBoard:

    for key, value in metrics.items():
        writer.add_scalar(f"compaction/{key}", value, global_step)
==========================================================================
"""


if __name__ == "__main__":
    run_standalone_example()
