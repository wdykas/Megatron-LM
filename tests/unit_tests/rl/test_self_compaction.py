# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SWE-1.7-style self-compaction training: segmented rollouts + alternating
length penalty.

Covers the full agent-side loop with a scripted inference interface (segment
→ summary → resume message plumbing, per-turn masks/flags, reward from the
final task segment), the trainer-side truncated-turn handling, and the
alternating length penalty phase/cost/penalty math.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from megatron.rl.__init__ import GenericGenerationArgs
from megatron.rl.agent.api import GroupedRolloutRequest, RolloutRequest, TokenRollout
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.inference import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from megatron.rl.rl_utils import apply_alternating_length_penalty, compute_group_stats

EOD = 7


class ScriptedInterface(ReturnsTokens, ReturnsRaw):
    """Returns scripted (finish_reason, text) responses and logs requests."""

    script: list[tuple[str, str]]
    calls: list[InferenceRequest] = []

    async def agenerate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls.append(request)
        finish_reason, text = self.script[len(self.calls) - 1]
        # Fabricated tokens: prompt scales with message count so each turn has
        # a distinct prompt_length; generation is 5 tokens ending in EOD when
        # the segment finished naturally.
        prompt_len = 10 * len(request.prompt)
        gen = [2, 2, 2, 2, EOD] if finish_reason == 'stop' else [2, 2, 2, 2, 2]
        return InferenceResponse(
            response=LLMChatMessage(role='assistant', content=text),
            raw_text='raw:' + text,
            token_ids=[1] * prompt_len + gen,
            prompt_length=prompt_len,
            logprobs=[-0.1] * len(gen),
            policy_staleness=[0] * len(gen),
            kv_cache_staleness=[0] * len(gen),
            completed_at_step=0,
            num_evictions=0,
            finish_reason=finish_reason,
        )


class CountingAgent(RewardOnlyAgent):
    env_id: str = 'test_env'

    async def get_prompt(self, validation: bool) -> tuple[str, Any]:
        return "Solve the problem.", {'answer': 'ANSWER'}

    async def get_reward(self, response: str, golden: Any) -> float:
        return 1.0 if golden['answer'] in response else 0.0


def run_rollout(agent, script):
    iface = ScriptedInterface(script=script, calls=[])
    request = RolloutRequest(
        num_rollouts=1, inference_interface=iface,
        generation_args=GenericGenerationArgs(temperature=0.9))
    rollout = asyncio.run(agent.rollout(request))
    return rollout, iface


class TestSelfCompactionRollout:
    def test_off_by_default_single_turn(self):
        agent = CountingAgent()
        rollout, iface = run_rollout(agent, [('stop', 'the ANSWER')])
        assert len(iface.calls) == 1
        # No cap injected when self-compaction is off.
        assert iface.calls[0].generation_args.max_tokens is None
        assert isinstance(rollout, TokenRollout)
        assert len(rollout.trajectory) == 1
        assert rollout.reward == 1.0
        assert rollout.truncated == [False]

    def test_segments_summary_resume_flow(self):
        agent = CountingAgent(
            self_compaction_segment_tokens=100,
            self_compaction_summary_tokens=20,
            self_compaction_max_segments=3,
        )
        rollout, iface = run_rollout(agent, [
            ('length', 'partial work...'),
            ('stop', 'SUMMARY: did half the steps'),
            ('stop', 'continuing... the ANSWER'),
        ])
        assert len(iface.calls) == 3

        seg1, summary, seg2 = iface.calls
        # Segment and summary calls carry their own generation caps.
        assert seg1.generation_args.max_tokens == 100
        assert summary.generation_args.max_tokens == 20
        assert seg2.generation_args.max_tokens == 100
        # Temperature merged through, not clobbered.
        assert seg1.generation_args.temperature == 0.9

        # Summary request = truncated context + partial output + instruction.
        # The assistant message must carry the segment's exact serving tokens
        # (the chat endpoint's prevent-retokenization patch requires them and
        # splices them for a prefix-cache hit).
        assert summary.prompt[0].content == "Solve the problem."
        assert summary.prompt[1].role == 'assistant'
        assert summary.prompt[1].content == 'partial work...'
        seg1_resp = 10 * len(seg1.prompt)
        assert summary.prompt[1].prompt_token_ids == [1] * seg1_resp
        assert summary.prompt[1].generation_token_ids == [2, 2, 2, 2, 2]
        assert summary.prompt[2].role == 'user'
        assert 'summar' in summary.prompt[2].content.lower()

        # Resume request = task + self-authored summary EMBEDDED AS TEXT in a
        # user message — no assistant message (its serving tokens contain the
        # pre-compaction context and would undo the compaction if spliced).
        assert seg2.prompt[0].content == "Solve the problem."
        assert len(seg2.prompt) == 2
        assert seg2.prompt[1].role == 'user'
        assert 'SUMMARY: did half the steps' in seg2.prompt[1].content
        assert all(m.role != 'assistant' for m in seg2.prompt)
        assert all('partial work' not in m.content for m in seg2.prompt)

        # Rollout: 3 turns, all trained with the final answer's reward.
        assert len(rollout.trajectory) == 3
        assert rollout.reward == 1.0
        assert rollout.truncated == [True, False, False]
        for turn, resp_call in zip(rollout.trajectory, iface.calls):
            assert len(turn) == 10 * len(resp_call.prompt) + 5
        # Masks mark exactly the generated tokens of each turn.
        for mask, turn in zip(rollout.generation_mask, rollout.trajectory):
            assert sum(mask) == 5 and mask[-5:] == [True] * 5

    def test_reward_ignores_trailing_summary(self):
        # If the run ends right after a summary (max segments hit), the reward
        # must come from the last TASK segment, not the summary text.
        agent = CountingAgent(
            self_compaction_segment_tokens=100, self_compaction_max_segments=2)
        rollout, iface = run_rollout(agent, [
            ('length', 'work with the ANSWER buried'),
            ('stop', 'summary without the token'),
            ('length', 'still unfinished'),
        ])
        # 2 task segments + 1 summary; final segment truncated, no answer.
        assert len(iface.calls) == 3
        assert rollout.reward == 0.0
        assert rollout.truncated == [True, False, True]

    def test_max_segments_caps_generation(self):
        agent = CountingAgent(
            self_compaction_segment_tokens=100, self_compaction_max_segments=3)
        rollout, iface = run_rollout(agent, [
            ('length', 'seg0'), ('stop', 'sum0'),
            ('length', 'seg1'), ('stop', 'sum1'),
            ('length', 'seg2'), ('stop', 'never-requested'),
        ])
        # 3 task segments with summaries between: seg, sum, seg, sum, seg —
        # no summary after the last segment (max segments reached).
        assert len(iface.calls) == 5
        assert rollout.truncated == [True, False, True, False, True]

    def test_evaluation_uses_segmented_generation(self):
        from megatron.rl.agent.api import EvaluationRequest
        agent = CountingAgent(self_compaction_segment_tokens=100)
        iface = ScriptedInterface(script=[
            ('length', 'partial'),
            ('stop', 'summary'),
            ('stop', 'the ANSWER'),
        ], calls=[])
        request = EvaluationRequest(
            inference_interface=iface, num_prompts=1,
            generation_args=GenericGenerationArgs())
        response = asyncio.run(agent._evaluation("prompt", {'answer': 'ANSWER'}, request))
        assert len(iface.calls) == 3
        assert response.results[0].reward == 1.0
        # The reported response is the final task segment, not the summary.
        assert response.results[0].response.content == 'the ANSWER'

    def test_group_rollout_preserves_rollout_index(self):
        agent = CountingAgent(self_compaction_segment_tokens=100)
        iface = ScriptedInterface(
            script=[('stop', 'ANSWER')] * 2, calls=[])
        request = GroupedRolloutRequest(
            num_groups=1, rollouts_per_group=2, inference_interface=iface,
            generation_args=GenericGenerationArgs())
        rollouts = asyncio.run(agent.group_rollout(request))
        assert len(rollouts) == 2
        assert sorted(c.rollout_index for c in iface.calls) == [0, 1]


def make_token_rollout(gen_tokens_per_turn, reward=1.0, truncated=None, seq_len=None):
    turns = []
    masks = []
    for gen in gen_tokens_per_turn:
        prompt = 3
        turn = [1] * prompt + [2] * (gen - 1) + [EOD]
        turns.append(turn)
        masks.append([False] * prompt + [True] * gen)
    return TokenRollout(
        trajectory=turns, reward=reward, generation_mask=masks,
        logprobs=[[-0.1] * len(t) for t in turns],
        policy_staleness=[[0]] * len(turns),
        kv_cache_staleness=[[0]] * len(turns),
        completed_at_step=[0] * len(turns),
        num_evictions=[0] * len(turns),
        truncated=truncated,
    )


def penalty_args(iteration, phase_len=10, budget=100, turn_cost=0.0, weight=1.0):
    return SimpleNamespace(
        curr_iteration=iteration,
        rl_length_penalty_phase_len=phase_len,
        rl_length_penalty_budget=budget,
        rl_length_penalty_turn_cost=turn_cost,
        rl_length_penalty_weight=weight,
    )


class TestAlternatingLengthPenalty:
    def test_disabled_returns_none(self):
        rollouts = [[make_token_rollout([50])]]
        assert apply_alternating_length_penalty(
            rollouts, penalty_args(iteration=0, phase_len=0)) is None
        assert rollouts[0][0].reward == 1.0

    def test_unconstrained_phase_no_penalty(self):
        # Iterations [0, phase_len) are unconstrained even over budget.
        rollouts = [[make_token_rollout([500])]]
        metrics = apply_alternating_length_penalty(
            rollouts, penalty_args(iteration=3, budget=100))
        assert metrics['length_penalty_phase'] == 0
        assert metrics['mean_length_penalty'] == 0.0
        assert metrics['frac_over_budget'] == 1.0
        assert rollouts[0][0].reward == 1.0

    def test_budget_phase_penalizes_overshoot_only(self):
        over = make_token_rollout([150])   # cost 150, budget 100
        under = make_token_rollout([80])
        metrics = apply_alternating_length_penalty(
            [[over, under]], penalty_args(iteration=13, budget=100, weight=2.0))
        assert metrics['length_penalty_phase'] == 1
        assert over.reward == pytest.approx(1.0 - 2.0 * 0.5)
        assert under.reward == 1.0
        assert metrics['mean_rollout_cost'] == pytest.approx(115.0)

    def test_turn_cost_charges_restarts(self):
        # 3 turns x 40 generated tokens; turn_cost 50 adds 100 for the two
        # extra turns: cost 220 vs budget 200 -> penalty 0.1.
        rollout = make_token_rollout([40, 40, 40])
        apply_alternating_length_penalty(
            [[rollout]], penalty_args(iteration=13, budget=200, turn_cost=50.0))
        assert rollout.reward == pytest.approx(1.0 - 0.1)

    def test_phase_alternation_schedule(self):
        phases = [apply_alternating_length_penalty(
            [[make_token_rollout([10])]],
            penalty_args(iteration=i, phase_len=2))['length_penalty_phase']
            for i in range(8)]
        assert phases == [0, 0, 1, 1, 0, 0, 1, 1]

    def test_missing_budget_raises(self):
        with pytest.raises(ValueError, match='rl-length-penalty-budget'):
            apply_alternating_length_penalty(
                [[make_token_rollout([10])]],
                penalty_args(iteration=0, budget=None))


class TestTruncatedTurnStats:
    def _tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.eod = EOD
        tokenizer.detokenize = lambda toks: '<detok>'
        return tokenizer

    def test_truncated_turn_without_eod_accepted(self):
        rollout = make_token_rollout([20, 20])
        rollout.trajectory[0][-1] = 2  # cut at the cap: no EOD
        rollout.truncated = [True, False]
        stats = compute_group_stats([[rollout]], self._tokenizer(), seq_len=512)
        assert stats.num_turns == [[2]]

    def test_untruncated_short_turn_without_eod_asserts(self):
        rollout = make_token_rollout([20])
        rollout.trajectory[0][-1] = 2
        with pytest.raises(AssertionError):
            compute_group_stats([[rollout]], self._tokenizer(), seq_len=512)
