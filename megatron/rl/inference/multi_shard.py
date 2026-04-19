# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-shard inference serving for heterogeneous RL rollout.

Each registered ``InferenceShard`` runs its own ``DynamicInferenceEngine``,
``DataParallelInferenceCoordinator``, and text-generation HTTP server. A single
``MegatronLocalMulti`` instance, returned on every rank from ``launch``, fans
lifecycle calls (resume/suspend/kill/set_generation_epoch) out to every shard
and routes ``base_generate`` requests round-robin across their HTTP front-ends.

Reachability:
- Every rank learns every shard's coordinator ZMQ address and HTTP URL via
  an ``all_gather_object`` at launch, so any process can address any other
  shard's coordinator or HTTP server directly.
- Cross-shard torch process groups can be built on top via
  :func:`megatron.rl.parallel_utils.build_cross_shard_group`.
"""
import asyncio
import logging
from typing import List, Optional

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
from megatron.core.inference.shards import InferenceShard
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


class MegatronLocalMulti(InferenceServer, ReturnsTokens, ReturnsRaw):
    """Fanout inference server across heterogeneous inference shards.

    Parameters:
        host: Hostname / bind address for the HTTP servers.
        base_port: First shard's HTTP port; shard ``i`` serves on
            ``base_port + i``.
    """

    host: str
    base_port: int

    # Filled in at launch on every rank so the whole world sees the full shard
    # table — enabling any rank to reach any shard's coordinator / HTTP server.
    _shards: List[InferenceShard] = PrivateAttr(default_factory=list)

    # Rank-local state (only populated on ranks that belong to a shard).
    _my_shard_index: Optional[int] = PrivateAttr(default=None)
    _my_engine: Optional[DynamicInferenceEngine] = PrivateAttr(default=None)
    _my_client: Optional[InferenceClient] = PrivateAttr(default=None)
    _rl_kv_cache_management_mode: Optional[KVCacheManagementMode] = PrivateAttr(default=None)

    # Routing state for base_generate (lives on every rank but is only
    # exercised on the rollout-driver rank; per-rank lock is fine).
    _openai_clients: List[AsyncOpenAI] = PrivateAttr(default_factory=list)
    _next_shard: int = PrivateAttr(default=0)
    _route_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    # ---- Public API -----------------------------------------------------

    @classmethod
    async def launch(
        cls,
        model: Optional[GPTModel],
        shards: List[InferenceShard],
        host: str = "0.0.0.0",
        base_port: int = 8294,
        verbose: bool = False,
    ) -> "MegatronLocalMulti":
        """Bring up one engine + coordinator + text-gen server per shard.

        Args:
            model: The rank-local inference model (the one belonging to the
                shard this rank is in). ``None`` on idle ranks that aren't in
                any shard.
            shards: Full shard layout, shared across all ranks. Each shard's
                ``coordinator_addr`` / ``http_url`` will be populated in place.
            host: Hostname for HTTP bind and ZMQ addresses that must be
                reachable from other ranks.
            base_port: Port of the first shard's text-gen server. Shard ``i``
                will serve on ``base_port + i``.

        Returns:
            A ``MegatronLocalMulti`` instance usable on every rank. Idle ranks
            receive an instance too, so all ranks can still route / observe
            the shard table — but ``_my_engine`` is ``None`` on idle ranks.
        """
        # Avoid circular imports
        from megatron.inference.utils import get_dynamic_inference_engine

        args = get_args()
        tokenizer = get_tokenizer()

        if tokenizer.bos is None:
            log_single_rank(
                logger,
                logging.WARNING,
                "WARNING: Tokenizer has no BOS token so prompt will not have BOS token",
            )

        rank = dist.get_rank()
        my_shard: Optional[InferenceShard] = None
        for s in shards:
            if s.owns_rank(rank):
                my_shard = s
                break

        # Each rank that owns a shard builds an engine + wires it into its
        # shard's coordinator. `start_listening_to_data_parallel_coordinator`
        # internally uses the shard's dp/mp groups (via the engine's pg_collection)
        # so coordinator spawning happens once per shard automatically.
        my_engine: Optional[DynamicInferenceEngine] = None
        my_dp_addr: str = ""
        if my_shard is not None:
            assert model is not None, (
                f"rank {rank} owns shard {my_shard.index} but was given model=None"
            )
            my_engine = get_dynamic_inference_engine(model=model)
            # Pass None port so each shard auto-picks an unused port; collisions
            # between shards (which would happen with a fixed port) become the
            # rule rather than the exception in multi-shard mode.
            my_dp_addr = await my_engine.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=None,
                launch_inference_coordinator=True,
            )

        # --- Exchange coordinator addresses across all ranks --------------
        #
        # We all_gather_object by default, then pull the value reported by
        # each shard's rank_offset rank (which is the shard's dp_coordinator
        # and therefore the sole source of truth for its dp_addr).
        world_size = dist.get_world_size()
        all_dp_addrs: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(all_dp_addrs, my_dp_addr or "")
        for s in shards:
            addr = all_dp_addrs[s.rank_offset] or None
            s.coordinator_addr = addr if addr else None

        # --- Start per-shard text-gen server on the shard's rank_offset ---
        my_http_port: int = -1
        my_client: Optional[InferenceClient] = None
        if my_shard is not None and rank == my_shard.rank_offset:
            my_http_port = base_port + my_shard.index
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
                start_text_gen_server,
            )

            my_client = InferenceClient(inference_coordinator_address=my_dp_addr)
            my_client.start()
            start_text_gen_server(
                coordinator_addr=my_dp_addr,
                tokenizer=my_engine.controller.tokenizer,
                rank=rank,
                server_port=my_http_port,
                parsers=[],
                verbose=verbose,
                hostname=host,
            )

        # --- Exchange HTTP ports so every rank knows every shard's URL ----
        all_http_ports: List[Optional[int]] = [None] * world_size
        dist.all_gather_object(all_http_ports, my_http_port)
        for s in shards:
            port = all_http_ports[s.rank_offset]
            if port is not None and port >= 0:
                s.http_url = f"http://{host}:{port}"
            else:
                s.http_url = None

        # --- Build OpenAI clients pointing at every shard's HTTP server ---
        # We build these on all ranks for symmetry; only the rollout-driver
        # rank actually exercises them, but having them everywhere keeps the
        # future cross-shard direct-call path trivial.
        concurrency_limit = (
            args.grpo_prompts_per_step
            * args.grpo_group_size
            * args.rl_parallel_generation_tasks
        )
        custom_limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        openai_clients: List[AsyncOpenAI] = []
        for s in shards:
            if s.http_url is None:
                openai_clients.append(None)  # type: ignore[arg-type]
                continue
            http_client = DefaultAioHttpClient(
                timeout=None,
                limits=custom_limits,
                http2=use_http2,
            )
            openai_clients.append(
                AsyncOpenAI(
                    base_url=s.http_url,
                    api_key="NONE",
                    http_client=http_client,
                )
            )

        instance = cls(host=host, base_port=base_port)
        instance._shards = shards
        instance._my_shard_index = my_shard.index if my_shard is not None else None
        instance._my_engine = my_engine
        instance._my_client = my_client
        instance._openai_clients = openai_clients
        instance._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )
        instance._route_lock = asyncio.Lock()
        return instance

    # ---- Generation routing --------------------------------------------

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Round-robin request across shards that have an HTTP server up."""
        tokenizer = get_tokenizer()
        args = get_args()

        reachable = [i for i, c in enumerate(self._openai_clients) if c is not None]
        if not reachable:
            raise RuntimeError("No inference shards are reachable for generation.")

        async with self._route_lock:
            idx = reachable[self._next_shard % len(reachable)]
            self._next_shard += 1

        client = self._openai_clients[idx]
        response = await client.chat.completions.create(
            model="",
            messages=[message.model_dump() for message in request.prompt],
            temperature=request.generation_args.temperature or 1.0,
            top_p=request.generation_args.top_p or 0.0,
            n=1,
            logprobs=True,
            extra_body={
                "skip_prompt_log_probs": True,
                "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
            },
        )

        choice = response.choices[0]
        return InferenceResponse(
            response=LLMChatMessage(
                **choice.message.model_dump(include={"role", "content"})
            ),
            raw_text=choice.raw_text,
            token_ids=choice.prompt_token_ids + choice.generation_token_ids,
            logprobs=choice.generation_log_probs,
            prompt_length=len(choice.prompt_token_ids),
            policy_epoch=choice.policy_epoch,
            kv_cache_epoch=choice.kv_cache_epoch,
            num_evictions=getattr(choice, "num_evictions", 0),
        )

    # ---- Lifecycle (fanned out across shards) --------------------------

    def set_generation_epoch(self, generation_epoch: int) -> None:
        """Each shard's local client sets its own epoch."""
        if self._my_client is not None:
            self._my_client.set_generation_epoch(generation_epoch)

    async def resume(self) -> None:
        if self._my_engine is None:
            return
        if self._my_engine._state_events[EngineState.RUNNING].is_set():
            return
        if self._my_client is not None:
            self._my_client.resume_engines()
        await self._my_engine.wait_until(EngineState.RESUMED)
        if self._my_client is not None:
            self._my_client.unpause_engines()
        await self._my_engine.wait_until(EngineState.RUNNING)

    async def suspend(self) -> None:
        if self._my_engine is None:
            return
        if self._my_client is not None:
            self._my_client.pause_engines()
        await self._my_engine.wait_until(EngineState.PAUSED)
        if self._my_client is not None:
            self._my_client.suspend_engines()
        await self._my_engine.wait_until(EngineState.SUSPENDED)

    async def kill(self) -> None:
        # Close all HTTP connections on every rank so idle clients don't leak.
        for c in self._openai_clients:
            if c is not None:
                try:
                    await c.close()
                except Exception:
                    pass

        if self._my_engine is None:
            return

        if self._my_client is not None:
            self._my_client.pause_engines()
        await self._my_engine.wait_until(EngineState.PAUSED)

        if self._my_client is not None:
            self._my_client.stop_engines()
        await self._my_engine.wait_until(EngineState.STOPPED)

        if self._my_client is not None:
            self._my_client.shutdown_coordinator()
            self._my_client.stop()

        # The text-gen server lives on the shard's first rank; only that rank
        # needs to stop it.
        my_shard = self._shards[self._my_shard_index] if self._my_shard_index is not None else None
        if my_shard is not None and dist.get_rank() == my_shard.rank_offset:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
                stop_text_gen_server,
            )

            stop_text_gen_server()

    # ---- Reachability introspection ------------------------------------

    def shard_urls(self) -> List[Optional[str]]:
        """Full table of shard HTTP URLs as seen by this rank."""
        return [s.http_url for s in self._shards]

    def shard_coordinator_addrs(self) -> List[Optional[str]]:
        """Full table of shard coordinator ZMQ addresses as seen by this rank."""
        return [s.coordinator_addr for s in self._shards]
