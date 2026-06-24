# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Curriculum scheduler for staged Belief-Still training.

The design doc specifies four training stages that gradually add objectives:

    Stage 1 — Still warm-up
        teacher_kl only.  Trains the PerceiverCompactor to produce compact KV
        that preserves the base model's token probabilities.

    Stage 2 — Belief-Still recurrence
        + future_kl, + consistency.  Trains the recurrent updater to maintain
        a belief state that predicts not just the current step but future steps.

    Stage 3 — Retrieval focus
        + retrieval, + weighted_kl.  Upweights confident / exact-recall positions
        so brittle facts (names, numbers, dates) are not lost under compression.

    Stage 4 — Value-directed
        + path_consistency, + task.  Trains end-to-end on task signal and
        enforces path-independence (sequential == combined chunk processing).

Usage:
    scheduler = CurriculumScheduler.default_4stage(steps_per_stage=5000)

    for batch in dataloader:
        weights = scheduler.step()           # get weights for this step
        loss = trainer.train_step(..., weights)
        loss.total.backward()
        ...
        print(scheduler.stage_idx, scheduler.stage_name)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megatron.rl.compaction.learned.training.losses import CompactorLossWeights


# ---------------------------------------------------------------------------
# CurriculumStage
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """A single training stage with its loss weights and step budget.

    Attributes
    ----------
    name:     Human-readable stage identifier (used in logs).
    n_steps:  Gradient steps to spend in this stage.
    weights:  CompactorLossWeights active during this stage.
    """

    name: str
    n_steps: int
    weights: CompactorLossWeights


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Step-based training curriculum that advances through CurriculumStages.

    Call ``step()`` once per gradient update.  It returns the loss weights
    for the *current* step and then advances the internal counter.

    The scheduler stays in the last stage indefinitely once all stage budgets
    are exhausted — there is no hard stop.

    Example:
        scheduler = CurriculumScheduler.default_4stage(steps_per_stage=10_000)

        for batch in loader:
            weights = scheduler.step()
            ...  # use weights in loss computation
    """

    def __init__(self, stages: list[CurriculumStage]) -> None:
        if not stages:
            raise ValueError("stages must be non-empty")
        self.stages = stages
        self._total_steps: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step(self) -> CompactorLossWeights:
        """Return current loss weights and advance the step counter.

        After this call, ``total_steps`` has been incremented by one.
        """
        weights = self.current_weights
        self._total_steps += 1
        return weights

    @property
    def current_weights(self) -> CompactorLossWeights:
        """Loss weights for the current step (before calling step())."""
        return self.stages[self.stage_idx].weights

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.stage_idx]

    @property
    def stage_idx(self) -> int:
        """Zero-based index of the current stage."""
        cumulative = 0
        for i, stage in enumerate(self.stages):
            cumulative += stage.n_steps
            if self._total_steps < cumulative:
                return i
        return len(self.stages) - 1   # last stage never ends

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    @property
    def total_steps(self) -> int:
        """Total gradient steps taken so far."""
        return self._total_steps

    @property
    def steps_in_stage(self) -> int:
        """Steps taken within the current stage."""
        offset = sum(s.n_steps for s in self.stages[:self.stage_idx])
        return self._total_steps - offset

    @property
    def stage_progress(self) -> float:
        """Fraction of the current stage completed: steps_in_stage / n_steps."""
        stage = self.current_stage
        return min(1.0, self.steps_in_stage / max(stage.n_steps, 1))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default_4stage(cls, steps_per_stage: int = 10_000) -> "CurriculumScheduler":
        """Build the four-stage curriculum from the Belief-Still design doc.

        Parameters
        ----------
        steps_per_stage: Gradient steps to spend in each stage (default 10 000).
                         Multiply by batch accumulation if using gradient accumulation.
        """
        return cls([
            CurriculumStage(
                name="still_warmup",
                n_steps=steps_per_stage,
                # Stage 1: distillation only.  No consistency or predictive — the
                # compactor must first learn to compress before being regularised.
                weights=CompactorLossWeights(
                    teacher_kl=1.0,
                    future_kl=0.0,
                    consistency=0.0,
                    predictive=0.1,   # train the prediction head from the start
                    task=0.0,
                    retrieval=0.0,
                    weighted_kl=0.0,
                    path_consistency=0.0,
                ),
            ),
            CurriculumStage(
                name="belief_still",
                n_steps=steps_per_stage,
                # Stage 2: add gate-weighted consistency + future KL once the
                # compactor baseline is stable.
                weights=CompactorLossWeights(
                    teacher_kl=1.0,
                    future_kl=0.3,
                    consistency=0.01,  # gate already handles smoothness;
                    predictive=0.1,    # consistency just guards truly stable slots
                    task=0.0,
                    retrieval=0.0,
                    weighted_kl=0.0,
                    path_consistency=0.0,
                ),
            ),
            CurriculumStage(
                name="retrieval_focus",
                n_steps=steps_per_stage,
                weights=CompactorLossWeights(
                    teacher_kl=1.0,
                    future_kl=0.3,
                    consistency=0.01,
                    predictive=0.1,
                    task=0.0,
                    retrieval=0.5,
                    weighted_kl=0.2,
                    path_consistency=0.0,
                ),
            ),
            CurriculumStage(
                name="value_directed",
                n_steps=steps_per_stage,
                weights=CompactorLossWeights(
                    teacher_kl=1.0,
                    future_kl=0.3,
                    consistency=0.01,
                    predictive=0.1,
                    task=0.1,
                    retrieval=0.5,
                    weighted_kl=0.2,
                    path_consistency=0.2,
                ),
            ),
        ])

    # ------------------------------------------------------------------
    # Serialisation (for checkpointing scheduler state)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"total_steps": self._total_steps}

    def load_state_dict(self, state: dict) -> None:
        self._total_steps = state["total_steps"]
