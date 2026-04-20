# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-shard inference serving for heterogeneous RL rollout.

Each registered ``InferenceShard`` runs its own ``DynamicInferenceEngine``,
``DataParallelInferenceCoordinator``, and text-generation HTTP server. A single
``MegatronLocalMulti`` instance, returned on every rank from ``launch``, drives
lifecycle (resume/suspend/kill/set_generation_epoch) from rank 0 and routes
``base_generate`` requests across shards with 1/latency weighting.

Reachability:
- Every rank learns every shard's coordinator ZMQ address and HTTP URL via
  an ``all_gather_object`` at launch, so any process can address any other
  shard's coordinator or HTTP server directly.
- Cross-shard torch process groups can be built on top via
  :func:`megatron.rl.parallel_utils.build_cross_shard_group`.
"""
import asyncio
import logging
import time
from collections import deque
from typing import Deque, List, Optional

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
    # Lifecycle clients — one per shard, populated **only** on global rank 0
    # (the rollout driver). Shard-local rank_offsets only launch the HTTP
    # server; they do not drive pause/resume because the lifecycle calls run
    # on every rank concurrently, and if each shard's rank_offset paused its
    # own shard, non-driver shards would pause before rank 0 finished issuing
    # requests to them.
    _lifecycle_clients: List[Optional[InferenceClient]] = PrivateAttr(default_factory=list)
    _rl_kv_cache_management_mode: Optional[KVCacheManagementMode] = PrivateAttr(default=None)

    # Routing state for base_generate (lives on every rank but is only
    # exercised on the rollout-driver rank; per-rank lock is fine).
    _openai_clients: List[Optional[AsyncOpenAI]] = PrivateAttr(default_factory=list)
    _next_shard: int = PrivateAttr(default=0)
    _route_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    # Sliding windows of recent per-shard response durations (seconds); used
    # for 1/latency-weighted routing once each shard has some history.
    # _in_flight counts pending requests per shard and biases the weighted
    # pick against shards that are currently saturated, so a fast shard
    # doesn't get piled on before a response lands in _recent_latencies.
    _recent_latencies: List[Deque[float]] = PrivateAttr(default_factory=list)
    _in_flight: List[int] = PrivateAttr(default_factory=list)
    _latency_window: int = PrivateAttr(default=32)

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
        # Only each shard's rank_offset (its dp_coordinator) knows the real
        # dp_addr, so we all_gather and pluck by rank_offset.
        world_size = dist.get_world_size()
        all_dp_addrs: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(all_dp_addrs, my_dp_addr or "")
        for s in shards:
            s.coordinator_addr = all_dp_addrs[s.rank_offset] or None

        # --- Start per-shard text-gen server on the shard's rank_offset ---
        # The HTTP server's subprocesses create their own InferenceClient against
        # `coordinator_addr`, so no outer client is needed here for HTTP. The
        # lifecycle client (pause/resume/...) is created separately on rank 0
        # below, independent of shard membership.
        my_http_port: int = -1
        if my_shard is not None and rank == my_shard.rank_offset:
            my_http_port = base_port + my_shard.index
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
                start_text_gen_server,
            )

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
        all_http_ports: List[int] = [-1] * world_size
        dist.all_gather_object(all_http_ports, my_http_port)
        for s in shards:
            port = all_http_ports[s.rank_offset]
            s.http_url = f"http://{host}:{port}" if port >= 0 else None

        # --- Build lifecycle InferenceClients (global rank 0 only) ---
        # Only global rank 0 drives pause / resume / stop / shutdown so that
        # these collective state transitions happen once per shard, not once
        # per shard-rank_offset. rank 0 connects to every shard's coordinator.
        lifecycle_clients: List[Optional[InferenceClient]] = [None] * len(shards)
        if rank == 0:
            for s in shards:
                if s.coordinator_addr:
                    c = InferenceClient(inference_coordinator_address=s.coordinator_addr)
                    c.start()
                    lifecycle_clients[s.index] = c

        # --- Build OpenAI clients pointing at every shard's HTTP server ---
        # Built on every rank so any rank can drive rollouts.
        concurrency_limit = (
            args.grpo_prompts_per_step
            * args.grpo_group_size
            * args.rl_parallel_generation_tasks
        )
        custom_limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        openai_clients: List[Optional[AsyncOpenAI]] = []
        for s in shards:
            if s.http_url is None:
                openai_clients.append(None)
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
        instance._lifecycle_clients = lifecycle_clients
        instance._openai_clients = openai_clients
        instance._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )
        instance._route_lock = asyncio.Lock()
        instance._recent_latencies = [deque(maxlen=instance._latency_window) for _ in shards]
        instance._in_flight = [0 for _ in shards]
        return instance

    # ---- Generation routing --------------------------------------------

    def _pick_shard(self, reachable: List[int]) -> int:
        """Pick a reachable shard to route the next request to.

        Falls back to round-robin until every reachable shard has at least one
        recorded latency sample; after that, routes by 1/(latency * (1 +
        in_flight)) — i.e. fast shards get more requests, and a shard that's
        currently saturated (large in-flight queue) gets down-weighted so a
        warm-up burst doesn't pile onto a single fast shard.
        """
        if not all(self._recent_latencies[i] for i in reachable):
            idx = reachable[self._next_shard % len(reachable)]
            self._next_shard += 1
            return idx

        def score(i: int) -> float:
            samples = sorted(self._recent_latencies[i])
            latency = max(samples[len(samples) // 2], 1e-6)
            return 1.0 / (latency * (1 + self._in_flight[i]))

        return max(reachable, key=score)

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        """Route request across reachable shards using 1/latency weights."""
        tokenizer = get_tokenizer()
        args = get_args()

        reachable = [i for i, c in enumerate(self._openai_clients) if c is not None]
        if not reachable:
            raise RuntimeError("No inference shards are reachable for generation.")

        async with self._route_lock:
            idx = self._pick_shard(reachable)
            self._in_flight[idx] += 1

        client = self._openai_clients[idx]
        start = time.monotonic()
        try:
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
        finally:
            elapsed = time.monotonic() - start
            async with self._route_lock:
                self._in_flight[idx] -= 1
                self._recent_latencies[idx].append(elapsed)

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
    #
    # Every rank calls these methods, but only rank 0 actually sends
    # pause/resume/stop/shutdown commands — one per shard — through its
    # `_lifecycle_clients`. Non-driver ranks just wait on their local engine
    # state (``_my_engine.wait_until``) so the engine-side broadcast from the
    # shard's coordinator brings them along. This matches the single-shard
    # MegatronLocal pattern (rank 0 drives, others wait) but generalises it
    # across N heterogeneous shards.

    def _drive_all(self, fn_name: str, *args) -> None:
        """Call `fn_name(*args)` on every shard's lifecycle client (rank 0 only)."""
        if dist.get_rank() != 0:
            return
        for c in self._lifecycle_clients:
            if c is not None:
                getattr(c, fn_name)(*args)

    def set_generation_epoch(self, generation_epoch: int) -> None:
        self._drive_all("set_generation_epoch", generation_epoch)

    async def resume(self) -> None:
        if self._my_engine is None:
            return
        if self._my_engine._state_events[EngineState.RUNNING].is_set():
            return
        self._drive_all("resume_engines")
        await self._my_engine.wait_until(EngineState.RESUMED)
        self._drive_all("unpause_engines")
        await self._my_engine.wait_until(EngineState.RUNNING)

    async def suspend(self) -> None:
        if self._my_engine is None:
            return
        self._drive_all("pause_engines")
        await self._my_engine.wait_until(EngineState.PAUSED)
        self._drive_all("suspend_engines")
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

        self._drive_all("pause_engines")
        await self._my_engine.wait_until(EngineState.PAUSED)

        self._drive_all("stop_engines")
        await self._my_engine.wait_until(EngineState.STOPPED)

        if dist.get_rank() == 0:
            for c in self._lifecycle_clients:
                if c is not None:
                    c.shutdown_coordinator()
                    c.stop()

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

    def shard_routing_stats(self) -> List[dict]:
        """Per-shard routing telemetry: median latency + in-flight count.

        Useful for debugging load imbalance across heterogeneous shards. Values
        are collected from the rank-local view, so on non-driver ranks they
        are always zero/empty.
        """
        stats = []
        for i, samples in enumerate(self._recent_latencies):
            median = sorted(samples)[len(samples) // 2] if samples else None
            stats.append(
                {
                    "shard": i,
                    "samples": len(samples),
                    "median_latency_s": median,
                    "in_flight": self._in_flight[i],
                }
            )
        return stats
