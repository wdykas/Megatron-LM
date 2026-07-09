# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging

import httpx
import torch.distributed as dist
from openai import AsyncOpenAI, DefaultAioHttpClient
from pydantic import PrivateAttr

try:
    import h2  # noqa: F401
    use_http2 = True
except ImportError:
    use_http2 = False

from megatron.core.inference.config import KVCacheManagementMode
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tokenizer

from ..inference.inference_interface import (
    InferenceRequest,
    InferenceResponse,
    LLMChatMessage,
    ReturnsRaw,
    ReturnsTokens,
)
from ..server.api import InferenceServer

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class MegatronLocal(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Interface to use MCoreEngine directly as an inference engine."""

    host: str
    port: int

    _client: InferenceClient = PrivateAttr(None)
    _inference_engine: DynamicInferenceEngine = PrivateAttr(None)
    _rl_kv_cache_management_mode: KVCacheManagementMode = PrivateAttr(None)
    _openai_client: AsyncOpenAI = PrivateAttr(None)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        tokenizer = get_tokenizer()
        args = get_args()

        # Use the shared, optimized client instead of spinning up a new one
        client = self._openai_client

        # split-group counterfactual: draw this rollout's compaction arm.
        # GRPO groups are issued as rollouts_per_group concurrent identical
        # requests, so a per-request Bernoulli split partitions every group into
        # compact and full-cache arms; the arm is recorded on the response.
        kv_compacted = None
        if args.rl_compaction_split_fraction is not None:
            if request.rollout_index is None:
                # A per-turn coin flip here would let one rollout mix compact
                # and full-cache turns, contaminating both arms of the
                # counterfactual. The arm must be a per-rollout property.
                raise ValueError(
                    "--rl-compaction-split-fraction requires the agent to tag "
                    "requests with rollout_index (see "
                    "RewardOnlyAgent.group_rollout) so the arm is constant "
                    "across a rollout's turns.")
            # Deterministic within-group split: exact arm proportions in
            # every group, stable across the rollout's turns.
            from megatron.rl.inference.api import deterministic_split_arm
            kv_compacted = deterministic_split_arm(
                request.rollout_index, args.rl_compaction_split_fraction)

        # Things that may be problematic when doing this switch
        # - Add BOS token
        # - Skip prompt logprobs
        response = await client.chat.completions.create(
            model="",
            # exclude_none: plain messages must NOT carry null token-id keys —
            # the endpoint checks key membership before splicing them.
            messages=[message.model_dump(exclude_none=True) for message in request.prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            **({} if request.generation_args.max_tokens is None
               else {"max_tokens": request.generation_args.max_tokens}),
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
                **({} if kv_compacted is None else {"kv_compact": kv_compacted}),
            },
        )

        choice = response.choices[0]

        return InferenceResponse(
            # TODO: Handle tool calls and reasoning in LLMChatMessage
            response=LLMChatMessage(**choice.message.model_dump(include={'role', 'content'})),
            raw_text=choice.raw_text,
            token_ids=choice.prompt_token_ids + choice.generation_token_ids,
            logprobs=choice.generation_log_probs,
            prompt_length=len(choice.prompt_token_ids),
            policy_staleness=choice.policy_staleness,
            kv_cache_staleness=choice.kv_cache_staleness,
            completed_at_step=args.curr_iteration,
            num_evictions=getattr(choice, 'num_evictions', 0),
            kv_compacted=kv_compacted,
            finish_reason=choice.finish_reason,
        )

    @classmethod
    async def launch(cls, model: GPTModel, **kwargs):
        # Import here to avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        inference_engine: DynamicInferenceEngine = get_dynamic_inference_engine(model=model)
        dp_addr = await inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=41521, launch_inference_coordinator=True,
        )

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import start_text_gen_server

            client = InferenceClient(inference_coordinator_address=dp_addr)
            client.start()

            start_text_gen_server(
                coordinator_addr=dp_addr,
                tokenizer=inference_engine.controller.tokenizer,
                rank=dist.get_rank(),
                server_port=kwargs.get('port', 8294),
                parsers=[],
                verbose=kwargs.get('verbose', False),
            )
        else:
            client = None

        launched_server = cls(**kwargs)
        launched_server._client = client
        launched_server._inference_engine = inference_engine
        launched_server._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )

        # The split-group counterfactual only means anything when live compaction actually runs;
        # validate UNCONDITIONALLY so a split fraction without --rl-compaction-
        # enabled (or with a non-live mode) hard-fails instead of silently
        # tagging arms that were never compacted — a pure-noise counterfactual.
        if args.rl_compaction_split_fraction is not None and (
                not args.rl_compaction_enabled
                or args.rl_compaction_mode != "live"
                or not 0.0 < args.rl_compaction_split_fraction <= 1.0):
            raise ValueError(
                "--rl-compaction-split-fraction needs --rl-compaction-enabled, "
                "--rl-compaction-mode live and a value in (0, 1]; got "
                f"enabled={args.rl_compaction_enabled} "
                f"mode={args.rl_compaction_mode!r} "
                f"fraction={args.rl_compaction_split_fraction}.")

        if args.rl_compaction_enabled:
            # Live mode: every rollout decodes over a compacted cache — the
            # engine prunes (or belief_still-synthesizes) each request's prompt
            # KV right after its prefill, identically to the serving path.
            if args.rl_compaction_mode == "live":
                from megatron.rl.compaction.kv.serving.live import LiveKVCompactor
                if (args.rl_compaction_strategy == "snapkv"
                        and not args.decode_only_cuda_graphs):
                    raise ValueError(
                        "--rl-compaction-strategy snapkv in live mode needs eager "
                        "prefill for observation-window Q capture: add "
                        "--decode-only-cuda-graphs.")
                inference_engine.kv_compactor = LiveKVCompactor(
                    inference_engine,
                    strategy=args.rl_compaction_strategy,
                    budget_ratio=args.rl_compaction_kv_budget_ratio,
                    n_compress=args.rl_compaction_n_compress,
                    compactor_checkpoint=args.rl_compaction_compactor_checkpoint,
                    oracle_checkpoint=args.rl_compaction_oracle_checkpoint,
                    budget_final=args.rl_compaction_budget_final,
                    budget_anneal_iters=args.rl_compaction_budget_anneal_iters,
                    score_weighting=args.rl_compaction_score_weighting,
                )
                log_single_rank(logger, logging.INFO,
                                f"[kv-compaction] live rollout compaction: "
                                f"{args.rl_compaction_strategy}")
                # NOTE: if archive mode ever lands in training, the engine
                # must capture its decode graphs AFTER the compactor wraps
                # the attention (see run_dynamic_text_generation_server.py
                # defer_capture) — a post-hoc re-capture doubles graph pools.

        concurrency_limit = args.grpo_prompts_per_step * args.grpo_group_size * args.rl_parallel_generation_tasks
        custom_limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        http_client = DefaultAioHttpClient(
            timeout=None,
            limits=custom_limits,
            http2=use_http2
        )

        launched_server._openai_client = AsyncOpenAI(
            base_url=f"http://{launched_server.host}:{launched_server.port}",
            api_key="NONE",
            http_client=http_client
        )

        return launched_server

    async def kill(self):
        # Gracefully close the shared OpenAI client connections
        if self._openai_client is not None:
            await self._openai_client.close()

        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.stop_engines()
        await self._inference_engine.wait_until(EngineState.STOPPED)

        if dist.get_rank() == 0:
            self._client.shutdown_coordinator()
            self._client.stop()

        if dist.get_rank() == 0:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import stop_text_gen_server
            stop_text_gen_server()

    def increment_staleness(self):
        if dist.get_rank() == 0:
            self._client.increment_staleness()

    async def suspend(self):
        if dist.get_rank() == 0:
            self._client.pause_engines()
        await self._inference_engine.wait_until(EngineState.PAUSED)

        if dist.get_rank() == 0:
            self._client.suspend_engines()
        await self._inference_engine.wait_until(EngineState.SUSPENDED)

    async def resume(self):
        if self._inference_engine._state_events[EngineState.RUNNING].is_set():
            return

        if dist.get_rank() == 0:
            self._client.resume_engines()
        await self._inference_engine.wait_until(EngineState.RESUMED)

        if dist.get_rank() == 0:
            self._client.unpause_engines()
        await self._inference_engine.wait_until(EngineState.RUNNING)
