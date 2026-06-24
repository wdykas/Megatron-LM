# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.rl.compaction.pomdp.trigger import (
    AlwaysTrigger,
    CompactionTrigger,
    EveryNStepsTrigger,
    NeverTrigger,
    TokenBudgetTrigger,
)
from megatron.rl.compaction.pomdp.types import Action, Observation, new_id


def _obs(step_id: int = 0) -> Observation:
    return Observation(observation_id=new_id(), step_id=step_id, kind="other")


def _act(step_id: int = 0) -> Action:
    return Action(action_id=new_id(), step_id=step_id, kind="other")


class TestAlwaysTrigger:
    def test_always_returns_true(self):
        t = AlwaysTrigger()
        for step in range(5):
            assert t.should_compact(step, None, _act(), _obs()) is True

    def test_satisfies_protocol(self):
        assert isinstance(AlwaysTrigger(), CompactionTrigger)


class TestNeverTrigger:
    def test_always_returns_false(self):
        t = NeverTrigger()
        for step in range(5):
            assert t.should_compact(step, None, _act(), _obs()) is False

    def test_satisfies_protocol(self):
        assert isinstance(NeverTrigger(), CompactionTrigger)


class TestEveryNStepsTrigger:
    def test_compact_on_zero(self):
        t = EveryNStepsTrigger(n=3)
        assert t.should_compact(0, None, _act(), _obs()) is True

    def test_skips_intermediate_steps(self):
        t = EveryNStepsTrigger(n=3)
        assert t.should_compact(1, None, _act(), _obs()) is False
        assert t.should_compact(2, None, _act(), _obs()) is False

    def test_compact_on_multiple_of_n(self):
        t = EveryNStepsTrigger(n=3)
        assert t.should_compact(3, None, _act(), _obs()) is True
        assert t.should_compact(6, None, _act(), _obs()) is True

    def test_n_equals_one_is_always(self):
        t = EveryNStepsTrigger(n=1)
        for step in range(5):
            assert t.should_compact(step, None, _act(), _obs()) is True

    def test_invalid_n_raises(self):
        import pytest
        with pytest.raises(ValueError):
            EveryNStepsTrigger(n=0)

    def test_satisfies_protocol(self):
        assert isinstance(EveryNStepsTrigger(n=2), CompactionTrigger)


class TestTokenBudgetTrigger:
    def test_triggers_when_text_is_long(self):
        t = TokenBudgetTrigger(budget=10)

        class _Belief:
            def to_context_str(self):
                return "x" * 500  # >> 10 * 4 = 40 chars

        assert t.should_compact(0, _Belief(), _act(), _obs()) is True

    def test_no_trigger_when_text_is_short(self):
        t = TokenBudgetTrigger(budget=1000)

        class _Belief:
            def to_context_str(self):
                return "short text"

        assert t.should_compact(0, _Belief(), _act(), _obs()) is False

    def test_uses_tokenizer_when_available(self):
        class _Tok:
            def tokenize(self, text):
                return list(text)  # 1 char = 1 token

        t = TokenBudgetTrigger(budget=5, tokenizer=_Tok())

        class _Short:
            def to_context_str(self):
                return "abc"  # 3 tokens < 5

        class _Long:
            def to_context_str(self):
                return "abcdefghij"  # 10 tokens >= 5

        assert t.should_compact(0, _Short(), _act(), _obs()) is False
        assert t.should_compact(0, _Long(), _act(), _obs()) is True

    def test_prefers_belief_token_estimate(self):
        """The authoritative token_estimate is used directly — no re-tokenization."""
        t = TokenBudgetTrigger(budget=100)

        class _Belief:
            token_estimate = 150  # >= budget
            def to_context_str(self):
                raise AssertionError("must not re-tokenize when token_estimate is set")

        assert t.should_compact(0, _Belief(), _act(), _obs()) is True

        class _Small:
            token_estimate = 50  # < budget
            def to_context_str(self):
                raise AssertionError("must not re-tokenize when token_estimate is set")

        assert t.should_compact(0, _Small(), _act(), _obs()) is False

    def test_falls_back_when_token_estimate_is_none(self):
        """When token_estimate is absent/None, fall back to the context string."""
        t = TokenBudgetTrigger(budget=10)

        class _Belief:
            token_estimate = None
            def to_context_str(self):
                return "x" * 500  # >> 10 * 4

        assert t.should_compact(0, _Belief(), _act(), _obs()) is True

    def test_no_to_context_str_falls_back_to_str(self):
        t = TokenBudgetTrigger(budget=1)
        assert t.should_compact(0, "a" * 100, _act(), _obs()) is True

    def test_satisfies_protocol(self):
        assert isinstance(TokenBudgetTrigger(budget=100), CompactionTrigger)
