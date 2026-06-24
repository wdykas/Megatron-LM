# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CompactionTrigger(Protocol):
    """Decides whether to run the compaction algorithm at a given step.

    Implementations receive the current step context and return True if the
    compaction algorithm should be called.

    Optional async hook
    -------------------
    A trigger may also define::

        async def async_should_compact(self, step_id, belief, action, observation) -> bool

    with the same arguments. When present, ``PomdpRolloutRecorder`` awaits it on
    the async record path instead of ``should_compact``. This is the extension
    point for *model-driven* triggers (e.g. asking an LLM "should I compact
    now?"), which need an await. Sync-only triggers omit it and are awaited via
    a transparent fallback to ``should_compact``.
    """

    def should_compact(
        self,
        step_id: int,
        belief: Any,
        action: Any,
        observation: Any,
    ) -> bool: ...


class AlwaysTrigger:
    """Compact on every step (default)."""

    def should_compact(self, step_id: int, belief: Any, action: Any, observation: Any) -> bool:
        return True


class NeverTrigger:
    """Never compact — belief passes through unchanged. Useful for record-only baselines."""

    def should_compact(self, step_id: int, belief: Any, action: Any, observation: Any) -> bool:
        return False


class EveryNStepsTrigger:
    """Compact every N steps (step_id 0, N, 2N, …)."""

    def __init__(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._n = n

    def should_compact(self, step_id: int, belief: Any, action: Any, observation: Any) -> bool:
        return (step_id % self._n) == 0


class TokenBudgetTrigger:
    """Compact when the belief context string exceeds a token/character budget.

    If a tokenizer is supplied its ``tokenize`` method is used for an exact
    token count. Otherwise a rough 4-chars-per-token estimate is applied.
    """

    def __init__(self, budget: int, tokenizer: Any = None) -> None:
        self._budget = budget
        self._tokenizer = tokenizer

    def should_compact(self, step_id: int, belief: Any, action: Any, observation: Any) -> bool:
        text = belief.to_context_str() if hasattr(belief, "to_context_str") else str(belief)
        if self._tokenizer is not None:
            try:
                tokens = self._tokenizer.tokenize(text)
                return len(tokens) >= self._budget
            except Exception:
                pass
        return len(text) >= self._budget * 4
