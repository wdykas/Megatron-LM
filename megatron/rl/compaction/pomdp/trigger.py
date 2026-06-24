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
    """Compact when the belief's context exceeds a token budget.

    The authoritative size is ``belief.token_estimate`` — the real, tokenizer-
    measured token count of the actor context, which the recorder computes once
    per step when it builds and tokenizes that context (see
    ``PomdpRolloutRecorder.build_context_for_step``). Reading it here is O(1) and
    reflects exactly what the model consumed, so this is the normal path.

    The fallbacks below only apply when no upstream count is available (e.g. the
    context has not been built yet, or the recorder has no tokenizer): if a
    tokenizer was supplied to this trigger, tokenize the belief's context string;
    otherwise apply a coarse 4-chars-per-token estimate. Both are last resorts —
    they re-derive a worse proxy and should not be relied on in steady state.
    """

    def __init__(self, budget: int, tokenizer: Any = None) -> None:
        self._budget = budget
        self._tokenizer = tokenizer

    def should_compact(self, step_id: int, belief: Any, action: Any, observation: Any) -> bool:
        # Primary path: the real token count maintained on the belief upstream.
        token_estimate = getattr(belief, "token_estimate", None)
        if token_estimate is not None:
            return token_estimate >= self._budget

        # Fallback: no upstream count available — re-derive from the context string.
        text = belief.to_context_str() if hasattr(belief, "to_context_str") else str(belief)
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.tokenize(text)) >= self._budget
            except Exception:
                pass
        return len(text) >= self._budget * 4
