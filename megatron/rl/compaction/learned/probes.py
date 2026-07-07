# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sufficiency-KL probes: how much does compaction change the policy?

The core measurement of the compaction research plan: per query position,

    D_KL( pi(. | full context)  ||  pi(. | compacted context) )

computed through the REAL Megatron model — the teacher distribution comes from
a full-context forward and the student from ``student_outputs`` (the same
compact-KV injection used for compactor training). Low KL at a position means
the compacted cache is a sufficient statistic for that prediction; a spike
localizes exactly where compression lost something the policy needed.

Uses: eval metric for any compressor at matched budget, reward proxy for
RL-trained eviction (eviction-policy RL), and trigger labels for hierarchical-memory
retrieval (the negative-cache trigger).
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from megatron.rl.compaction.learned.capture.student_forward import student_outputs
from megatron.rl.compaction.learned.training.losses import per_token_kl


@torch.no_grad()
def sufficiency_kl(
    model,
    query_tokens: torch.Tensor,
    compact_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    teacher_logits: torch.Tensor,
    gather_logits: bool = False,
) -> torch.Tensor:
    """Per-position sufficiency KL of a compacted cache.

    Runs the query tokens through the frozen model with each attention layer's
    context replaced by ``compact_kv`` (list of (K, V) per attention layer,
    each (B, C, d_kv)) and compares against ``teacher_logits`` — the logits the
    model produced for the same query positions with the FULL context.

    Returns (B, S_q) fp32 KL per position. Mean it for a scalar sufficiency
    score; argmax it to localize the worst-hit positions.
    """
    student = student_outputs(model, query_tokens, compact_kv,
                               gather_logits=gather_logits).logits
    return per_token_kl(teacher_logits, student)
