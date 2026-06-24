# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Still and Belief-Still training objectives.

The original Still paper trains with a single teacher-KL distillation loss:

    L_teacher_KL = KL(p_full || p_compact)   for the current step

where p_full = softmax(logits from frozen base model with full KV cache) and
p_compact = softmax(logits from frozen base model with compact KV cache).

Belief-Still adds three further terms that enforce the POMDP interpretation
of the compact memory as a sufficient information state:

    L_future_KL  = Σ_{j=1..n} KL(p_full[t+j] || p_compact[t+j])
                   — compact memory should support predictions n steps ahead

    L_consistency = ||M_{t+1} - M_t||^2
                   — smooth belief trajectory: big jumps are penalised

    L_task        = -E[R]  or  cross-entropy on target tokens
                   — end-to-end task signal (zero-weight until RL training)

The full objective is:

    L = w_teacher * L_teacher_KL
      + w_future  * L_future_KL
      + w_consist * L_consistency
      + w_task    * L_task

Reference: value-directed belief compression (Poupart & Boutilier, NeurIPS 2002).
The key principle is that we optimise for what matters for future behavior,
not for KV reconstruction fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from megatron.rl.compaction.learned.models.belief import BeliefMemory
from megatron.rl.compaction.learned.training.data import CompactKV, StudentFn


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------

@dataclass
class CompactorLossWeights:
    """Per-term weights for the combined training objective.

    Start with ``teacher_kl=1.0`` and gradually introduce ``future_kl``
    after warm-up to avoid destabilising early training.
    """

    teacher_kl:                float = 1.0
    future_kl:                 float = 0.5
    consistency:               float = 0.1
    task:                      float = 0.0    # enable after switching to RL training
    retrieval:                 float = 0.1   # exact-recall probes (names, numbers, dates)
    weighted_kl:               float = 0.0   # low-entropy token weighting; enable after warm-up
    path_consistency:          float = 0.0   # sequential vs combined path agreement
    predictive:                float = 0.1   # weight for POMDP predictive coding loss
    kv_reconstruction:         float = 0.0  # attention-output matching loss; use when student_fn is not available
    future_kv_reconstruction:  float = 0.0
    dynamics:                  float = 0.0
    future_horizon_kl:         float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "teacher_kl":               self.teacher_kl,
            "future_kl":                self.future_kl,
            "consistency":              self.consistency,
            "task":                     self.task,
            "retrieval":                self.retrieval,
            "weighted_kl":              self.weighted_kl,
            "path_consistency":         self.path_consistency,
            "predictive":               self.predictive,
            "kv_reconstruction":        self.kv_reconstruction,
            "future_kv_reconstruction": self.future_kv_reconstruction,
            "dynamics":                 self.dynamics,
            "future_horizon_kl":        self.future_horizon_kl,
        }


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def teacher_kl_loss(
    full_logits:    torch.Tensor,    # (B, seq, vocab)
    compact_logits: torch.Tensor,    # (B, seq, vocab)
    temperature:    float = 1.0,
    reduction:      str = "mean",
) -> torch.Tensor:
    """KL(p_full || p_compact) — the core Still distillation loss.

    The probability distribution p_full comes from the frozen base model
    running with the full KV cache (teacher). p_compact comes from the
    same frozen base model running with the compact KV cache (student).

    Temperature scaling is applied before computing the KL; the T² factor
    re-scales the gradient magnitude to be temperature-independent (standard
    practice in knowledge distillation).

    Parameters
    ----------
    full_logits:    Logits from the frozen model with full context.
    compact_logits: Logits from the frozen model with compact KV.
    temperature:    Softmax temperature for softening distributions (default 1.0).
    reduction:      "mean" (default) or "sum" or "none".
    """
    T = temperature
    # Upcast to fp32: the softmax/exp and vocab-axis sum lose precision in bf16
    # (8-bit mantissa) over a 100k+ vocab.
    full_logits = full_logits.float()
    compact_logits = compact_logits.float()
    # Work in log-space for numerical stability
    log_p = F.log_softmax(full_logits / T, dim=-1)    # teacher
    log_q = F.log_softmax(compact_logits / T, dim=-1)  # student
    p = log_p.exp()

    kl = (p * (log_p - log_q)).sum(-1)                 # (B, seq)

    if reduction == "mean":
        return kl.mean() * T * T
    elif reduction == "sum":
        return kl.sum() * T * T
    else:
        return kl * T * T


def advantage_weighted_kl_loss(
    full_logits:    torch.Tensor,
    compact_logits: torch.Tensor,
    advantage:      float,
    temperature:    float = 1.0,
    advantage_clip: float = 5.0,
    min_weight:     float = 0.1,
) -> torch.Tensor:
    """Teacher KL loss weighted by GRPO advantage.

    Weight mapping: advantage=+clip → w=2.0 (compress this context faithfully),
    advantage=0 → w=1.0 (baseline), advantage=-clip → w=min_weight (nearly ignore).

    Parameters
    ----------
    advantage:       Per-probe GRPO advantage (from calculate_grpo_advantages).
    advantage_clip:  Clip extreme advantage values.
    min_weight:      Minimum weight (prevents zeroing out low-advantage probes entirely).
    """
    adv = float(advantage)
    if adv != adv:  # NaN guard
        return teacher_kl_loss(full_logits, compact_logits, temperature)
    adv = max(-advantage_clip, min(advantage_clip, adv))
    # Linear map: [-clip, +clip] → [min_weight, 2.0]
    w = min_weight + (2.0 - min_weight) * (adv + advantage_clip) / (2.0 * advantage_clip)
    return w * teacher_kl_loss(full_logits, compact_logits, temperature)


def future_kl_loss(
    full_logits_list:    list[torch.Tensor],   # one (B, seq, vocab) per future step
    compact_logits_list: list[torch.Tensor],
    temperature:         float = 1.0,
    step_weights:        list[float] | None = None,
) -> torch.Tensor:
    """Mean KL loss over multiple future steps.

    Implements the key POMDP insight: the compact belief M_t should support
    accurate predictions not just at step t but also at steps t+1, t+2, etc.

    Parameters
    ----------
    full_logits_list:    One tensor per future step (teacher with full context).
    compact_logits_list: One tensor per future step (student with compact M_{t+j}).
    step_weights:        Optional per-step discount weights. Default: uniform.
    """
    assert len(full_logits_list) == len(compact_logits_list), \
        "full and compact future logit lists must be the same length"

    n = len(full_logits_list)
    weights = step_weights or [1.0 / n] * n

    total = torch.zeros(1, device=full_logits_list[0].device)
    for full_l, compact_l, w in zip(full_logits_list, compact_logits_list, weights):
        total = total + w * teacher_kl_loss(full_l, compact_l, temperature)
    return total.squeeze(0)


def retrieval_loss(
    logits:    torch.Tensor,   # (B, seq, vocab)
    answer_ids: torch.Tensor,  # (B, seq) long  — positions outside the answer should be -100
    ignore_index: int = -100,
) -> torch.Tensor:
    """NLL loss on exact-answer token IDs.

    Used to train the compact memory to preserve brittle facts (names, numbers,
    dates, citations) that KL distillation may underweight because the full
    model assigns high probability across many plausible continuations.

    Identical computation to ``task_loss``; named separately so callers can
    control weighting via ``CompactorLossWeights.retrieval``.
    """
    return task_loss(logits, answer_ids, ignore_index=ignore_index)


def weighted_kl_loss(
    full_logits:    torch.Tensor,   # (B, seq, vocab)
    compact_logits: torch.Tensor,   # (B, seq, vocab)
    temperature:    float = 1.0,
    rho:            float = 1.0,
    reduction:      str = "mean",
) -> torch.Tensor:
    """KL distillation weighted by teacher confidence.

    Positions where the teacher is near-deterministic (low entropy) get a
    higher weight:

        w_i = 1 + ρ · (1 − H(p_teacher_i) / H_max)

    so the compact memory is penalised more strongly for missing tokens the
    full model answers confidently — exact answers, rare facts, code symbols.

    Parameters
    ----------
    full_logits:    Teacher logits (frozen LLM with full KV).
    compact_logits: Student logits (frozen LLM with compact KV).
    temperature:    Distillation temperature (applied to both).
    rho:            Weighting strength. 0.0 → uniform (same as teacher_kl_loss).
    reduction:      "mean" (default) or "sum".
    """
    T = temperature
    full_logits = full_logits.float()      # fp32 for the vocab-axis softmax/sum
    compact_logits = compact_logits.float()
    log_p = F.log_softmax(full_logits / T, dim=-1)
    log_q = F.log_softmax(compact_logits / T, dim=-1)
    p = log_p.exp()

    kl_per_token = (p * (log_p - log_q)).sum(-1)           # (B, seq)

    if rho > 0.0:
        H = -(p * log_p).sum(-1)                            # (B, seq) entropy of teacher
        vocab = full_logits.shape[-1]
        H_max = float(torch.log(torch.tensor(float(vocab))))
        w = 1.0 + rho * (1.0 - H / H_max)                  # (B, seq)
        kl_per_token = w * kl_per_token

    if reduction == "mean":
        return kl_per_token.mean() * T * T
    elif reduction == "sum":
        return kl_per_token.sum() * T * T
    else:
        return kl_per_token * T * T


def consistency_loss(
    memory_t:  BeliefMemory,
    memory_t1: BeliefMemory,
    gates:     torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalise belief change in slots the gate decided to protect.

    Without gates: ||M_{t+1} - M_t||^2 — penalises ALL change uniformly.
    This conflicts with the distillation objective: slots that correctly open
    their gate to absorb new content also incur consistency penalty.

    With gates (shape n_layers, B, C, d): weight the diff by (1 - g).
    A slot with g≈0 (gate closed, should protect) is penalised heavily for
    changing.  A slot with g≈1 (gate open, should update) is penalised
    barely at all.  Gates are detached so this loss does not train the gate
    to game the weighting.

    Pass the gates tensor returned by GatedRecurrentUpdater.update() here.
    """
    diff_k = memory_t1.keys  - memory_t.keys
    diff_v = memory_t1.values - memory_t.values
    if gates is not None:
        # (1 - g): protected slots may not change; updating slots may change freely
        w = (1.0 - gates).detach()
        return (w * diff_k).pow(2).mean() + (w * diff_v).pow(2).mean()
    return diff_k.pow(2).mean() + diff_v.pow(2).mean()


def path_consistency_loss(
    updater,   # GatedRecurrentUpdater / BeliefUpdater
    chunk_a_keys:   list[torch.Tensor],   # n_layers × (B, T_a, d)
    chunk_a_values: list[torch.Tensor],
    chunk_b_keys:   list[torch.Tensor],   # n_layers × (B, T_b, d)
    chunk_b_values: list[torch.Tensor],
    query_tokens: torch.Tensor,           # (B, S_q)
    student_fn: StudentFn,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Penalise disagreement between sequential vs combined compression.

    The ideal compressor should satisfy the path-independence property:
    processing chunks A then B should give the same model behavior as
    processing the concatenation [A, B] in a single pass.

    Two belief update paths:
        Path S (sequential): M_A = compress(A)
                             M_AB = update(M_A, B)
        Path C (combined):   M_AB' = compress(cat(A, B))

    The loss is KL(p(·|M_AB') || p(·|M_AB)):
        - M_AB' is treated as the reference (detached)
        - M_AB  is trained to match it

    Parameters
    ----------
    updater:        BeliefUpdater or GatedRecurrentUpdater.
    chunk_a/b_*:    Position-free keys/values for the two chunks.
    query_tokens:   (B, S_q) token IDs for the student forward pass.
    student_fn:     CompactKV → logits (same interface as in SinglePassCompactorTrainer).
    temperature:    KL distillation temperature.
    """
    # --- Path S: compress A, then update with B ----------------------------
    mem_a  = updater.initial_compress(chunk_a_keys, chunk_a_values)
    mem_ab = updater(mem_a, chunk_b_keys, chunk_b_values)
    kv_seq: CompactKV = list(zip(mem_ab.keys_list(), mem_ab.values_list()))
    logits_seq = student_fn(query_tokens, kv_seq)

    # --- Path C: compress cat(A, B) in one shot ----------------------------
    combined_keys   = [torch.cat([a, b], dim=1) for a, b in zip(chunk_a_keys,   chunk_b_keys)]
    combined_values = [torch.cat([a, b], dim=1) for a, b in zip(chunk_a_values, chunk_b_values)]
    mem_combined = updater.initial_compress(combined_keys, combined_values)
    kv_comb: CompactKV = list(zip(mem_combined.keys_list(), mem_combined.values_list()))
    logits_comb = student_fn(query_tokens, kv_comb)

    # KL(combined || sequential): train sequential path to match combined reference
    return teacher_kl_loss(logits_comb.detach(), logits_seq, temperature=temperature)


def task_loss(
    logits:       torch.Tensor,   # (B, seq, vocab)
    target_ids:   torch.Tensor,   # (B, seq) long
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss on target token IDs.

    Can substitute for RL reward signal during initial supervised training.
    Replace with policy-gradient reward signal (negative expected return) in
    RL fine-tuning.
    """
    B, S, V = logits.shape
    return F.cross_entropy(
        logits.view(B * S, V),
        target_ids.view(B * S),
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
# Predictive coding loss (POMDP observation model)
# ---------------------------------------------------------------------------

def kv_reconstruction_loss(
    compact_keys:   list[torch.Tensor],   # n_layers × (B, C, d)
    compact_values: list[torch.Tensor],   # n_layers × (B, C, d)
    full_keys:      list[torch.Tensor],   # n_layers × (B, T, d)
    full_values:    list[torch.Tensor],   # n_layers × (B, T, d)
    n_queries:      int = 32,
) -> torch.Tensor:
    """Attention output matching: compact KV should reproduce full KV attention outputs.

    Samples ``n_queries`` random linear combinations of the original keys as query
    vectors, computes attention output from both compact and full KV, and returns
    MSE between them.  This is the same objective minimised analytically by OMP
    and TopK compressors, here applied as a gradient loss so the compactor can
    learn to minimise it end-to-end.

    Does not require a live language model — only the KV tensors from the
    trajectory are needed.  Use this as the primary training signal when
    ``student_fn`` is not available.
    """
    n_layers = len(compact_keys)
    total = torch.zeros([], device=compact_keys[0].device, dtype=compact_keys[0].dtype)
    for l in range(n_layers):
        ck = compact_keys[l]    # (B, C, d)
        cv = compact_values[l]  # (B, C, d)
        fk = full_keys[l]       # (B, T, d)
        fv = full_values[l]     # (B, T, d)
        B, T, d = fk.shape

        # Sample query vectors as random linear combinations of original keys.
        # Shape: (B, n_queries, d)
        n_q = min(n_queries, T)
        coeff = torch.randn(B, n_q, T, device=fk.device, dtype=fk.dtype)
        coeff = coeff.softmax(dim=-1)
        Q = coeff @ fk  # (B, n_q, d)

        scale = d ** -0.5
        # Attention output with compact KV
        a_compact = F.softmax(Q @ ck.transpose(-2, -1) * scale, dim=-1) @ cv  # (B, n_q, d)
        # Attention output with full KV (no grad — full KV is fixed data)
        with torch.no_grad():
            a_full = F.softmax(Q @ fk.transpose(-2, -1) * scale, dim=-1) @ fv  # (B, n_q, d)

        total = total + (a_compact - a_full).pow(2).mean()
    return total / n_layers


def future_kv_reconstruction_loss(
    memory_keys, memory_values, future_chunk_keys, future_chunk_values, n_queries=32,
):
    """NextLat belief-state test: does M_{t-1} answer queries from future chunk t?

    Uses chunk t keys as queries against old memory M_{t-1}, comparing to
    chunk t own attention. A belief state should support future queries
    without being updated first.
    """
    return kv_reconstruction_loss(
        memory_keys, memory_values,
        future_chunk_keys, future_chunk_values,
        n_queries=n_queries,
    )


def dynamics_prediction_loss(pred_keys, pred_values, target_keys, target_values):
    """Train dynamics head: predict M_{t+1} from M_t (NextLat-style).

    SmoothL1 against stop-gradient targets so only the dynamics head
    trains, not the memory representation.
    """
    n_layers = len(pred_keys)
    total = torch.zeros([], device=pred_keys[0].device, dtype=pred_keys[0].dtype)
    for pk, pv, tk, tv in zip(pred_keys, pred_values, target_keys, target_values):
        total = total + F.smooth_l1_loss(pk, tk.detach())
        total = total + F.smooth_l1_loss(pv, tv.detach())
    return total / (n_layers * 2)


def future_horizon_kl_loss(full_logits, compact_logits, temperature=1.0, gamma=0.9):
    """KL loss with exponentially increasing position weights.

    Later positions in the probe window (furthest from compressed prefix)
    get higher weight. gamma < 1.0 upweights later positions.
    gamma = 1.0 gives uniform weights (same as teacher_kl_loss).
    """
    T = temperature
    full_logits = full_logits.float()      # fp32 for the vocab-axis softmax/sum
    compact_logits = compact_logits.float()
    log_p = F.log_softmax(full_logits / T, dim=-1)
    log_q = F.log_softmax(compact_logits / T, dim=-1)
    p = log_p.exp()
    kl_per_token = (p * (log_p - log_q)).sum(-1)
    S_q = kl_per_token.shape[1]
    if gamma < 1.0 and S_q > 1:
        exponents = torch.arange(S_q, device=kl_per_token.device, dtype=kl_per_token.dtype)
        base = torch.tensor(1.0 / gamma, device=kl_per_token.device, dtype=kl_per_token.dtype)
        weights = torch.pow(base, exponents)
        weights = weights / weights.sum()
        kl_per_token = (kl_per_token * weights.unsqueeze(0)).sum(-1).unsqueeze(-1)
        return kl_per_token.mean() * T * T
    return kl_per_token.mean() * T * T


def predictive_coding_loss(
    predictions: torch.Tensor,    # (n_layers, B, C, d) — slot predictions of R_t
    actual_keys: list[torch.Tensor],  # n_layers × (B, T, d) — actual R_t keys
) -> torch.Tensor:
    """POMDP observation model loss: penalise inaccurate chunk predictions.

    Each memory slot predicted the incoming chunk's key content.
    Penalise the MSE between the prediction and the actual chunk mean.
    This forces the belief to carry forward-looking information.
    """
    n_layers = predictions.shape[0]
    total = torch.zeros([], device=predictions.device, dtype=predictions.dtype)
    for l in range(n_layers):
        pred_l = predictions[l]                            # (B, C, d)
        actual_mean = actual_keys[l].mean(dim=1, keepdim=True)  # (B, 1, d)
        total = total + (pred_l - actual_mean).pow(2).mean()
    return total / n_layers


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

@dataclass
class CompactorLossTerms:
    """All computed loss terms returned by CompactorLosses.compute()."""

    teacher_kl:               torch.Tensor | None
    future_kl:                torch.Tensor | None
    consistency:              torch.Tensor | None
    task:                     torch.Tensor | None
    retrieval:                torch.Tensor | None
    weighted_kl:              torch.Tensor | None
    path_consistency:         torch.Tensor | None
    predictive:               torch.Tensor | None
    kv_reconstruction:        torch.Tensor | None
    future_kv_reconstruction: torch.Tensor | None
    dynamics:                 torch.Tensor | None
    future_horizon_kl:        torch.Tensor | None
    total:                    torch.Tensor

    def as_dict(self, include_total: bool = False) -> dict[str, float]:
        """For logging. Returns {name: float} (detaches from graph).

        ``total`` is excluded by default because it is a weighted sum of the
        component terms — including it alongside the components causes
        double-counting in downstream loggers that sum all values.  Pass
        ``include_total=True`` if you need the weighted total explicitly.
        """
        d: dict[str, float] = {}
        for name, val in [
            ("teacher_kl",               self.teacher_kl),
            ("future_kl",                self.future_kl),
            ("consistency",              self.consistency),
            ("task",                     self.task),
            ("retrieval",                self.retrieval),
            ("weighted_kl",              self.weighted_kl),
            ("path_consistency",         self.path_consistency),
            ("predictive",               self.predictive),
            ("kv_reconstruction",        self.kv_reconstruction),
            ("future_kv_reconstruction", self.future_kv_reconstruction),
            ("dynamics",                 self.dynamics),
            ("future_horizon_kl",        self.future_horizon_kl),
        ]:
            if val is not None:
                d[name] = val.item()
        if include_total:
            d["total"] = self.total.item()
        return d


class CompactorLosses:
    """Compute the combined Belief-Still training objective.

    Usage — minimal (Still distillation only):
        terms = CompactorLosses.compute(
            weights=CompactorLossWeights(),
            full_logits=full_logits,
            compact_logits=compact_logits,
        )
        terms.total.backward()

    Usage — full Belief-Still with future KL and consistency:
        terms = CompactorLosses.compute(
            weights=CompactorLossWeights(future_kl=0.5, consistency=0.1),
            full_logits=full_logits_t,
            compact_logits=compact_logits_t,
            full_logits_future=[fl_t1, fl_t2, fl_t3],
            compact_logits_future=[cl_t1, cl_t2, cl_t3],
            memory_t=memory_t,
            memory_t1=memory_t1,
        )
    """

    @staticmethod
    def compute(
        weights:               CompactorLossWeights,
        full_logits:           torch.Tensor | None = None,
        compact_logits:        torch.Tensor | None = None,
        temperature:           float = 1.0,
        # Belief-Still extras (all optional):
        full_logits_future:    list[torch.Tensor] | None = None,
        compact_logits_future: list[torch.Tensor] | None = None,
        memory_t:              BeliefMemory | None = None,
        memory_t1:             BeliefMemory | None = None,
        target_ids:            torch.Tensor | None = None,
        retrieval_ids:         torch.Tensor | None = None,
        weighted_kl_rho:       float = 1.0,
        # POMDP predictive coding (optional):
        predictions:           torch.Tensor | None = None,       # (n_layers, B, C, d)
        actual_keys:           list[torch.Tensor] | None = None, # n_layers × (B, T, d)
        # KV reconstruction (optional — use when student_fn unavailable):
        compact_keys:          list[torch.Tensor] | None = None, # n_layers × (B, C, d)
        compact_values:        list[torch.Tensor] | None = None, # n_layers × (B, C, d)
        full_keys:             list[torch.Tensor] | None = None, # n_layers × (B, T, d)
        full_values:           list[torch.Tensor] | None = None, # n_layers × (B, T, d)
        # Future KV reconstruction (NextLat belief-state test):
        future_memory_keys:    list[torch.Tensor] | None = None, # n_layers × (B, C, d)
        future_memory_values:  list[torch.Tensor] | None = None, # n_layers × (B, C, d)
        # Dynamics prediction (NextLat-style):
        pred_keys:             list[torch.Tensor] | None = None,
        pred_values:           list[torch.Tensor] | None = None,
        dyn_target_keys:       list[torch.Tensor] | None = None,
        dyn_target_values:     list[torch.Tensor] | None = None,
        # Future horizon KL (position-weighted):
        future_horizon_gamma:  float = 1.0,
    ) -> "CompactorLossTerms":
        """Compute all requested loss terms and return their weighted sum.

        Parameters
        ----------
        weights:               Per-term loss weights.
        full_logits:           Logits from full-context forward pass (teacher). Optional
                               when only kv_reconstruction or self-supervised losses are used.
        compact_logits:        Logits from compact-KV forward pass (student).
        temperature:           Distillation temperature.
        full_logits_future:    Per-step teacher logits for n future steps.
        compact_logits_future: Per-step student logits for n future steps.
        memory_t:              Belief state at step t (for consistency loss).
        memory_t1:             Belief state at step t+1 (for consistency loss).
        target_ids:            Ground-truth token IDs (for supervised task CE loss).
        retrieval_ids:         Token IDs for exact-recall probes (names, numbers, etc).
        weighted_kl_rho:       Confidence-weighting strength for weighted_kl term.
        predictions:           Slot predictions of R_t from GatedRecurrentUpdater.update().
        actual_keys:           Actual chunk keys R_t per layer.
        compact_keys/values:   Compact memory KV for kv_reconstruction loss.
        full_keys/values:      Original full KV for kv_reconstruction loss.
        future_memory_keys/values: Old memory M_{t-1} KV for future_kv_reconstruction loss.
        pred_keys/values:      Dynamics head predictions of M_{t+1}.
        dyn_target_keys/values: Stop-gradient targets (actual M_{t+1}) for dynamics loss.
        future_horizon_gamma:  Exponential position weight decay for future_horizon_kl.
        """
        device = (
            full_logits.device if full_logits is not None
            else compact_keys[0].device if compact_keys is not None
            else actual_keys[0].device if actual_keys is not None
            else predictions.device if predictions is not None
            else compact_logits.device if compact_logits is not None
            else torch.device("cpu")
        )

        t_kl: torch.Tensor | None = None
        if weights.teacher_kl > 0.0 and full_logits is not None and compact_logits is not None:
            t_kl = teacher_kl_loss(full_logits, compact_logits, temperature)

        f_kl: torch.Tensor | None = None
        if (weights.future_kl > 0.0
                and full_logits_future is not None
                and compact_logits_future is not None):
            f_kl = future_kl_loss(full_logits_future, compact_logits_future, temperature)

        consist: torch.Tensor | None = None
        if weights.consistency > 0.0 and memory_t is not None and memory_t1 is not None:
            consist = consistency_loss(memory_t, memory_t1)

        t_loss: torch.Tensor | None = None
        if weights.task > 0.0 and target_ids is not None and compact_logits is not None:
            t_loss = task_loss(compact_logits, target_ids)

        retr: torch.Tensor | None = None
        if weights.retrieval > 0.0 and retrieval_ids is not None and compact_logits is not None:
            retr = retrieval_loss(compact_logits, retrieval_ids)

        wkl: torch.Tensor | None = None
        if weights.weighted_kl > 0.0 and full_logits is not None and compact_logits is not None:
            wkl = weighted_kl_loss(full_logits, compact_logits, temperature, rho=weighted_kl_rho)

        pred_loss: torch.Tensor | None = None
        if weights.predictive > 0.0 and predictions is not None and actual_keys is not None:
            pred_loss = predictive_coding_loss(predictions, actual_keys)

        kv_recon: torch.Tensor | None = None
        if (weights.kv_reconstruction > 0.0
                and compact_keys is not None and compact_values is not None
                and full_keys is not None and full_values is not None):
            kv_recon = kv_reconstruction_loss(compact_keys, compact_values, full_keys, full_values)

        fut_kv_recon: torch.Tensor | None = None
        if (weights.future_kv_reconstruction > 0.0
                and future_memory_keys is not None and future_memory_values is not None
                and full_keys is not None and full_values is not None):
            fut_kv_recon = future_kv_reconstruction_loss(
                future_memory_keys, future_memory_values, full_keys, full_values,
            )

        dyn_loss: torch.Tensor | None = None
        if (weights.dynamics > 0.0
                and pred_keys is not None and pred_values is not None
                and dyn_target_keys is not None and dyn_target_values is not None):
            dyn_loss = dynamics_prediction_loss(pred_keys, pred_values, dyn_target_keys, dyn_target_values)

        fh_kl: torch.Tensor | None = None
        if (weights.future_horizon_kl > 0.0
                and full_logits is not None and compact_logits is not None):
            fh_kl = future_horizon_kl_loss(full_logits, compact_logits, temperature, gamma=future_horizon_gamma)

        total = torch.zeros([], device=device)
        if t_kl is not None:
            total = total + weights.teacher_kl * t_kl
        if f_kl is not None:
            total = total + weights.future_kl * f_kl
        if consist is not None:
            total = total + weights.consistency * consist
        if t_loss is not None:
            total = total + weights.task * t_loss
        if retr is not None:
            total = total + weights.retrieval * retr
        if wkl is not None:
            total = total + weights.weighted_kl * wkl
        if pred_loss is not None:
            total = total + weights.predictive * pred_loss
        if kv_recon is not None:
            total = total + weights.kv_reconstruction * kv_recon
        if fut_kv_recon is not None:
            total = total + weights.future_kv_reconstruction * fut_kv_recon
        if dyn_loss is not None:
            total = total + weights.dynamics * dyn_loss
        if fh_kl is not None:
            total = total + weights.future_horizon_kl * fh_kl

        return CompactorLossTerms(
            teacher_kl=t_kl,
            future_kl=f_kl,
            consistency=consist,
            task=t_loss,
            retrieval=retr,
            weighted_kl=wkl,
            path_consistency=None,  # computed separately via path_consistency_loss()
            predictive=pred_loss,
            kv_reconstruction=kv_recon,
            future_kv_reconstruction=fut_kv_recon,
            dynamics=dyn_loss,
            future_horizon_kl=fh_kl,
            total=total,
        )
