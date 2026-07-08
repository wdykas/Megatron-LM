# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm

from ..__init__ import GenericGenerationArgs
from ..inference import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from .api import (
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    RewardEvaluationResult,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from .pass_at_evaluation_agent import PassAtEvaluationAgent


class RewardOnlyEvaluationResponse(EvaluationResponse[RewardEvaluationResult]):
    type_name: str = 'RewardOnlyEvaluationResponse'

    def metrics(self):
        return {'reward': [el.reward for el in self.results]}


DEFAULT_SUMMARY_PROMPT = (
    "You are about to run out of context. Write a working-state summary you "
    "can resume from with no other context: restate what the problem asks, "
    "record the key intermediate results and decisions you have reached so "
    "far, and state exactly what remains to be done. Be concise and complete."
)

DEFAULT_RESUME_PROMPT = (
    "Here is your own working-state summary from your earlier progress on "
    "this problem:\n\n{summary}\n\nResume from it — do not restart from "
    "scratch. Finish the remaining steps and give the final answer."
)


class RewardOnlyAgent(RolloutGenerator, GroupedRolloutGenerator, PassAtEvaluationAgent):
    """Agent that returns rollouts generated via default inference with a fixed reward function.

    Self-compaction (SWE-1.7 / Kevin style): when ``self_compaction_segment_tokens``
    is set, generation runs in segments. A segment that stops at the token cap
    (finish_reason 'length') triggers a summary turn — the model writes its own
    working-state summary from the truncated context — and the next segment
    resumes from [task, self-authored summary] alone. Every turn (segments AND
    summaries) is trained with the task reward, so the model simultaneously
    learns to write informative summaries and to work from them.
    """

    env_id: str | None = None

    # --- self-compaction (off unless segment_tokens is set via env config) ---
    self_compaction_segment_tokens: int | None = None
    self_compaction_summary_tokens: int = 512
    self_compaction_max_segments: int = 4
    self_compaction_summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    self_compaction_resume_prompt: str = DEFAULT_RESUME_PROMPT

    def get_dataset(self, validation: bool = False):
        """Return validation or train dataset."""
        raise NotImplementedError("Derived class must implement get_dataset.")

    async def get_reward(self, response: str, golden: Any) -> float:
        """Given the LLM response and the golden data, provide a reward."""
        raise NotImplementedError("Derived class must implement get_reward")

    async def get_prompt(self, validation: bool) -> tuple[str, Any]:
        """Return a tuple with the prompt string and the golden data."""
        raise NotImplementedError("Derived class must implement get_prompt")

    async def evaluation_prompts(
        self, num_prompts: int, validation: bool = False
    ) -> list[tuple[str, Any]]:
        """Get evaluation prompts for the agent. This method should be overridden by subclasses."""
        raise NotImplementedError

    def _get_rank_subset(
        self, prompts: list[tuple[str, Any]], num_prompts: int, rank: int, world_size: int
    ) -> list[tuple[str, Any]]:
        """Helper method to get the subset of prompts for a given rank.

        Args:
            prompts: List of all prompts
            num_prompts: Total number of prompts to use
            rank: Current process rank
            world_size: Total number of processes

        Returns:
            Subset of prompts for the current rank
        """
        # Take first num_prompts from all prompts
        prompts = prompts[:num_prompts]

        # Split prompts into chunks for each rank
        chunk_size = (len(prompts) + world_size - 1) // world_size
        start_idx = rank * chunk_size
        end_idx = min(start_idx + chunk_size, len(prompts))

        return prompts[start_idx:end_idx]

    async def _generate_segments(
        self, request: RolloutRequest | GroupedRolloutRequest | EvaluationRequest,
        inference_request: InferenceRequest,
    ) -> tuple[list[InferenceResponse], list[bool]]:
        """Run generation, self-compacting across the context limit.

        Returns (responses, is_summary): one response per turn, with summary
        turns flagged. With self-compaction off this is a single agenerate.
        """
        iface = request.inference_interface
        if self.self_compaction_segment_tokens is None:
            return [await iface.agenerate(inference_request)], [False]

        assert isinstance(iface, ReturnsTokens), (
            "Self-compaction training needs token-level turns; the inference "
            "interface must support token return.")

        segment_args = request.generation_args.add(
            GenericGenerationArgs(max_tokens=self.self_compaction_segment_tokens))
        summary_args = request.generation_args.add(
            GenericGenerationArgs(max_tokens=self.self_compaction_summary_tokens))

        task_messages = inference_request.prompt
        messages = task_messages
        responses: list[InferenceResponse] = []
        is_summary: list[bool] = []
        for segment in range(self.self_compaction_max_segments):
            response = await iface.agenerate(inference_request.model_copy(
                update={"prompt": messages, "generation_args": segment_args}))
            responses.append(response)
            is_summary.append(False)
            if (response.finish_reason != 'length'
                    or segment == self.self_compaction_max_segments - 1):
                break

            # Summary turn: the model writes its working state from the
            # truncated context. The assistant message carries the segment's
            # exact serving tokens — the chat endpoint splices them into the
            # prompt (prefix-cache hit on the segment's KV) and its
            # prevent-retokenization patch requires them.
            summary_request = inference_request.model_copy(update={
                "prompt": messages + [
                    LLMChatMessage(
                        role='assistant',
                        content=response.response.content,
                        prompt_token_ids=response.token_ids[:response.prompt_length],
                        generation_token_ids=response.token_ids[response.prompt_length:],
                    ),
                    LLMChatMessage(role='user',
                                   content=self.self_compaction_summary_prompt),
                ],
                "generation_args": summary_args,
            })
            try:
                summary = await iface.agenerate(summary_request)
            except Exception:
                # Most likely the summary request overflowed the context
                # window (segment cap left no headroom). Keep the truncated
                # rollout rather than killing the whole collection; a dead
                # server still surfaces on the next rollout's segment call.
                logging.getLogger(__name__).exception(
                    "self-compaction summary generation failed; ending "
                    "rollout at %d segment(s)", segment + 1)
                break
            responses.append(summary)
            is_summary.append(True)

            # The next segment resumes from the task plus the self-authored
            # summary alone — the raw working context is gone. The summary is
            # embedded as TEXT in a user message (no assistant message: its
            # serving tokens contain the pre-compaction context, and splicing
            # them back would silently undo the compaction).
            messages = task_messages + [
                LLMChatMessage(
                    role='user',
                    content=self.self_compaction_resume_prompt.format(
                        summary=summary.response.content)),
            ]
        return responses, is_summary

    async def rollout_from_responses(
        self, request: RolloutRequest, responses: list[InferenceResponse],
        is_summary: list[bool], golden: Any
    ) -> Rollout:
        assert isinstance(
            request.inference_interface, ReturnsRaw
        ), "InferenceInterface must support raw_text return to provide rollouts."

        # The answer lives in the last task segment; summary turns share its
        # reward (that is the point: summaries are trained by task outcome).
        final_text = next(
            r.response.content
            for r, summary in zip(reversed(responses), reversed(is_summary))
            if not summary)
        reward = await self.get_reward(final_text, golden)

        if isinstance(request.inference_interface, ReturnsTokens):
            rollout = TokenRollout(
                trajectory=[r.token_ids for r in responses],
                reward=reward,
                logprobs=[r.logprobs for r in responses],
                generation_mask=[
                    [x >= r.prompt_length for x in range(len(r.token_ids))]
                    for r in responses
                ],
                env_id=self.env_id,
                problem_id=golden['problem_id'] if 'problem_id' in golden else None,
                policy_staleness=[r.policy_staleness for r in responses],
                kv_cache_staleness=[r.kv_cache_staleness for r in responses],
                completed_at_step=[r.completed_at_step for r in responses],
                num_evictions=[r.num_evictions for r in responses],
                kv_compacted=[r.kv_compacted for r in responses],
                truncated=[r.finish_reason == 'length' for r in responses],
            )
        else:
            assert len(responses) == 1, (
                "Self-compaction requires a token-returning interface; raw "
                "rollouts are single-turn.")
            rollout = Rollout(
                trajectory=[responses[0].raw_text],
                reward=reward,
                env_id=self.env_id,
                problem_id=golden['problem_id'] if 'problem_id' in golden else None,
                policy_staleness=[responses[0].policy_staleness],
                kv_cache_staleness=[responses[0].kv_cache_staleness],
                completed_at_step=[responses[0].completed_at_step],
                num_evictions=[responses[0].num_evictions],
                kv_compacted=[responses[0].kv_compacted],
            )

        return rollout

    async def rollout(self, request: RolloutRequest) -> Rollout:

        prompt, golden = await self.get_prompt(validation=request.validation)

        inference_request = request.inference_interface.prepare_request(
            prompt, request.generation_args
        )

        responses, is_summary = await self._generate_segments(request, inference_request)

        return await self.rollout_from_responses(request, responses, is_summary, golden)

    async def group_rollout(self, request: GroupedRolloutRequest) -> list[Rollout]:

        prompt, golden = await self.get_prompt(validation=request.validation)

        inference_request = request.inference_interface.prepare_request(
            prompt, request.generation_args
        )

        # Tag each group member with its index so the inference side can make
        # deterministic within-group treatment splits (e.g. split-group kv_compact arms).
        segment_lists = await asyncio.gather(*[
            self._generate_segments(
                request, inference_request.model_copy(update={"rollout_index": i})
            )
            for i in range(request.rollouts_per_group)
        ])
        return [
            await self.rollout_from_responses(request, responses, is_summary, golden)
            for responses, is_summary in segment_lists
        ]

    async def _evaluation(
        self, prompt: str, golden: Any, request: EvaluationRequest
    ) -> RewardOnlyEvaluationResponse:

        inference_request = request.inference_interface.prepare_request(
            prompt, request.generation_args
        )

        # Evaluation measures the same regime training runs in: with
        # self-compaction on, eval rollouts also segment and resume.
        responses, is_summary = await self._generate_segments(request, inference_request)
        final = next(
            r for r, summary in zip(reversed(responses), reversed(is_summary))
            if not summary)

        result = RewardEvaluationResult(
            env_id=self.env_id,
            prompt=[prompt] if isinstance(prompt, LLMChatMessage) else prompt,
            response=final.response,
            reward=await self.get_reward(final.response.content, golden),
            problem_id=golden['problem_id'] if 'problem_id' in golden else None,
        )

        return RewardOnlyEvaluationResponse(results=[result], env_id=self.env_id)

    async def run_evaluation(self, request: EvaluationRequest):

        # Get all prompts first
        all_prompts = list(
            await self.evaluation_prompts(
                num_prompts=request.num_prompts, validation=request.validation
            )
        )

        # Then get this rank's subset if needed
        if request.rank_info is not None:
            prompts_to_evaluate = self._get_rank_subset(
                all_prompts, request.num_prompts, request.rank_info[0], request.rank_info[1]
            )
        else:
            prompts_to_evaluate = all_prompts

        results = await tqdm.gather(
            *[self.evaluation(p, g, request) for p, g in prompts_to_evaluate],
            desc="Evaluating prompts..",
        )
        return type(results[0])(
            results=sum([result.results for result in results], []), env_id=self.env_id
        )
