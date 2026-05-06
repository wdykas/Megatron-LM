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
import torch
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


async def _spawn_unified_coord(
    shards: "List[InferenceShard]",
    *,
    host: str,
    engine: "Optional[DynamicInferenceEngine]",
    data_parallel_size: int,
) -> str:
    """Spawn a single DataParallelInferenceCoordinator subprocess for all shards.

    rank 0 owns the subprocess; every other rank receives the coord's
    ZMQ address via a world-wide broadcast. ``engine`` must be non-None
    on rank 0 because inference-context settings that travel with the
    coord (tokenizer, block size, prefix-caching policy) are pulled
    from its InferenceContext.

    Returns as soon as the coord has its address bound. The coord's
    initial-registration barrier is satisfied later by each engine's
    ``start_listening_to_data_parallel_coordinator`` call — waiting
    for it here would deadlock, since this rank's engine hasn't
    registered yet.
    """
    import multiprocessing
    from megatron.core.inference.data_parallel_inference_coordinator import (
        DataParallelInferenceCoordinator,
    )

    rank = dist.get_rank()
    dp_addr: Optional[str] = None
    if rank == 0:
        assert engine is not None, (
            "rank 0 must own an engine to seed the unified coord with "
            "inference-context settings (tokenizer, block size, prefix "
            "caching policy)"
        )
        ctx = engine.context
        spawn_context = multiprocessing.get_context('spawn')
        coord_pipe, coord_process_pipe = spawn_context.Pipe()
        ready_event = spawn_context.Event()  # kept for entrypoint API; not awaited here
        coord_process = spawn_context.Process(
            target=DataParallelInferenceCoordinator.entrypoint,
            kwargs={
                "pipe_connection": coord_process_pipe,
                "ready_event": ready_event,
                "data_parallel_size": data_parallel_size,
                "tokenizer": engine.controller.tokenizer,
                "max_requests": ctx.max_requests,
                "inference_coordinator_port": None,
                "deterministic_mode": torch.are_deterministic_algorithms_enabled(),
                "block_size_tokens": ctx.block_size_tokens,
                "enable_prefix_caching": ctx.enable_prefix_caching,
                "prefix_caching_coordinator_policy": ctx.prefix_caching_coordinator_policy,
                "prefix_caching_routing_alpha": ctx.prefix_caching_routing_alpha,
                "schedule_output_path": None,
                "hostname": host,
            },
        )
        coord_process.start()
        dp_addr = coord_pipe.recv()
        coord_pipe.close()
        logger.info("Unified inference coordinator bound at %s", dp_addr)

    bcast = [dp_addr]
    dist.broadcast_object_list(bcast, src=0)
    [dp_addr] = bcast
    assert dp_addr is not None
    return dp_addr


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
    # Unified lifecycle client held only by global rank 0 (the rollout
    # driver). Kept as a one-element list so existing call sites that
    # iterate the list keep working; non-rank-0 processes leave it empty.
    # Pause/resume broadcast to every engine through the single coord,
    # which is why one client suffices regardless of shard count.
    _lifecycle_clients: List[Optional[InferenceClient]] = PrivateAttr(default_factory=list)
    _rl_kv_cache_management_mode: Optional[KVCacheManagementMode] = PrivateAttr(default=None)

    # Routing state for base_generate (lives on every rank but is only
    # exercised on the rollout-driver rank; per-rank lock is fine).
    _openai_clients: List[Optional[AsyncOpenAI]] = PrivateAttr(default_factory=list)
    _next_shard: int = PrivateAttr(default=0)
    _route_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    # Auto-disagg scheduler state — coord-mediated design.
    #
    # The scheduler runs as a normal asyncio task on the **decider
    # rank** of each watched src shard (the shard's ``rank_offset``).
    # No separate process group, no separate thread, no cross-rank
    # broadcast. Each tick the decider inspects its local engine's
    # request dict, builds migration plans, and submits them to the
    # coordinator via ``InferenceClient.migrate_request_batch(...)``.
    # The coord forwards a ``Headers.MIGRATE_BATCH`` signal to the
    # engines in src + dst shards, which run the registered migration
    # handler in their own run loop — sync NCCL on
    # ``cross_shard_group``. No collective decision-making, no
    # cross-PG NCCL ordering hazard.
    #
    # Earlier designs used a periodic broadcast on a dedicated PG to
    # coordinate the migration trigger across ranks; that produced
    # cross-PG NCCL deadlocks (PG 18 forward pass + scheduler PG
    # broadcast competing for serialization order on the GPU).
    # Routing through the coord eliminates that entirely — the coord
    # is just ZMQ, and the only NCCL involved is the migration
    # transport itself, which only the participating engines do.
    _scheduler_configs: List[tuple] = PrivateAttr(default_factory=list)
    _auto_disagg_tasks: List[asyncio.Task] = PrivateAttr(default_factory=list)
    _scheduler_stop: bool = PrivateAttr(default=False)
    # Per-shard ``InferenceClient`` keyed by shard_index. Each shard's
    # ``rank_offset`` rank holds a client so it can post migration
    # triggers (and ``UPDATE_REQUEST_RANK`` follow-ups) to the coord
    # without going through rank 0. Empty on ranks that aren't a
    # ``rank_offset`` for any shard.
    _shard_clients: dict = PrivateAttr(default_factory=dict)
    # Migration metadata snapshot, computed at launch on every rank
    # whose engine participates in any cross-shard migration. Maps
    # ``(src_shard_index, dst_shard_index)`` → dict with the
    # cross-shard process group, layouts and head offsets needed by
    # ``migrate_requests_cross_shard_batch``. Pre-computing avoids
    # paying the layout-build cost (and the ``build_cross_shard_group``
    # collective) on every migration.
    _migration_meta: dict = PrivateAttr(default_factory=dict)
    # When set, base_generate stamps every HTTP request with
    # ``disagg_pair=[src, dst]`` so the coord routes to the prefill shard
    # and the auto-disagg scheduler migrates after first token. Populated
    # by the AUTO_DISAGG_SRC_SHARD/AUTO_DISAGG_DST_SHARD env gate in launch.
    _disagg_rollout_pair: Optional[tuple] = PrivateAttr(default=None)
    # When set, base_generate stamps requests with ``tail_cut=[dst, n]``
    # so the auto-disagg scheduler does a *second* migration once the
    # request has produced ``n`` tokens — typically pulling long-tail
    # rollouts off the throughput decode shard onto a smaller, latency-
    # optimized decode shard. ``(dst_shard_index, min_tokens)`` tuple.
    _tail_cut_rollout_config: Optional[tuple] = PrivateAttr(default=None)
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

        # --- Unified coordinator (single coord across all shards) ---------
        # One DataParallelInferenceCoordinator subprocess owns routing and
        # request-ownership bookkeeping for every shard. Heterogeneous TP
        # still works because each engine tags its registration with a
        # shard_index; the coord routes SUBMIT with an optional
        # target_shard_index and rewrites the owning engine on migration
        # via Headers.UPDATE_REQUEST_RANK. This keeps the HTTP path
        # transparent across mid-flight migration and disaggregation:
        # the client's ENGINE_REPLY future resolves regardless of which
        # shard's engine actually produced the final tokens.
        #
        # NVSHMEM init for cross-shard migration. Must happen *before*
        # engine creation because the engine's KV ``memory_buffer`` and
        # Mamba state buffers are allocated on the symmetric heap when
        # NVSHMEM is initialized — direct one-sided put for migration
        # (no staging, no barrier_all) requires symmetric memory.
        # Collective: every rank participates.
        if len(shards) >= 2:
            from megatron.core.inference import nvshmem_migration as _nvshmem_mig

            _nvshmem_mig.maybe_init_nvshmem()

        # Build the engine first so rank 0 can seed the coord from its
        # InferenceContext (tokenizer, block size, prefix-caching config).
        my_engine: Optional[DynamicInferenceEngine] = None
        if my_shard is not None:
            assert model is not None, (
                f"rank {rank} owns shard {my_shard.index} but was given model=None"
            )
            my_engine = get_dynamic_inference_engine(model=model)

        total_dp_size = sum(int(s.spec.get("dp", 1)) for s in shards)
        unified_coord_addr = await _spawn_unified_coord(
            shards,
            host=host,
            engine=my_engine,
            data_parallel_size=total_dp_size,
        )

        if my_shard is not None:
            await my_engine.start_listening_to_data_parallel_coordinator(
                inference_coordinator_port=None,
                launch_inference_coordinator=False,
                coordinator_addr=unified_coord_addr,
                shard_index=my_shard.index,
            )

        # Every shard exposes the same unified coord address; downstream
        # code (HTTP server, lifecycle client, OpenAI routing) stays
        # shard-indexed for legibility but ultimately talks to one coord.
        for s in shards:
            s.coordinator_addr = unified_coord_addr

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
                coordinator_addr=unified_coord_addr,
                tokenizer=my_engine.controller.tokenizer,
                rank=rank,
                server_port=my_http_port,
                parsers=[],
                verbose=verbose,
                hostname=host,
            )

        # --- Exchange HTTP ports so every rank knows every shard's URL ----
        world_size = dist.get_world_size()
        all_http_ports: List[int] = [-1] * world_size
        dist.all_gather_object(all_http_ports, my_http_port)
        for s in shards:
            port = all_http_ports[s.rank_offset]
            s.http_url = f"http://{host}:{port}" if port >= 0 else None

        # --- Build per-shard InferenceClients on each shard's rank_offset ---
        # Each shard's first rank gets a client. Rank 0's client drives
        # the lifecycle (pause/resume/stop/shutdown via _drive_all);
        # other shards' clients are used by their auto-disagg scheduler
        # to post migration triggers (``MIGRATE_BATCH``) without going
        # through rank 0. Multiple clients on one coord is fine — the
        # coord identifies them by ZMQ identity at CONNECT time.
        shard_clients: dict = {}
        for s in shards:
            if rank == s.rank_offset:
                c = InferenceClient(inference_coordinator_address=unified_coord_addr)
                c.start()
                shard_clients[s.index] = c
        # Backward-compat ``lifecycle_clients`` list — single-element on
        # rank 0 (which is shard 0's rank_offset by construction), empty
        # elsewhere. ``_unified_client`` continues to read from this.
        lifecycle_clients: List[Optional[InferenceClient]] = (
            [shard_clients[0]] if rank == 0 and 0 in shard_clients else []
        )

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
        instance._shard_clients = shard_clients
        instance._openai_clients = openai_clients
        instance._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )
        instance._route_lock = asyncio.Lock()
        instance._recent_latencies = [deque(maxlen=instance._latency_window) for _ in shards]
        instance._in_flight = [0 for _ in shards]

        # Post-boot smoke tests, both opt-in via env. Run once per
        # launch, before the caller starts driving real rollouts.
        import os

        if os.environ.get("MIGRATION_SMOKE_TEST", "") == "1" and len(shards) >= 2:
            log_single_rank(
                logger,
                logging.INFO,
                "[migration-smoke] MIGRATION_SMOKE_TEST=1 set; running "
                "one-shot migration test across shards 0 → 1",
            )
            await instance.run_migration_smoke_test()

        if os.environ.get("DISAGG_SMOKE_TEST", "") == "1" and len(shards) >= 2:
            log_single_rank(
                logger,
                logging.INFO,
                "[disagg-smoke] DISAGG_SMOKE_TEST=1 set; running one-shot "
                "disaggregated (prefill on shard 0, decode on shard 1) test",
            )
            await instance.run_disaggregated_smoke_test()

        # Opt-in end-to-end disagg for every rollout. Setting both env
        # vars starts the auto-disagg scheduler watching src_shard and
        # stamps every subsequent base_generate with
        # ``disagg_pair=[src, dst]`` — so each HTTP request lands on src,
        # reaches first-token, then gets migrated to dst.
        src_env = os.environ.get("AUTO_DISAGG_SRC_SHARD")
        dst_env = os.environ.get("AUTO_DISAGG_DST_SHARD")
        if src_env is not None and dst_env is not None and len(shards) >= 2:
            src_idx = int(src_env)
            dst_idx = int(dst_env)
            assert src_idx != dst_idx, (
                "AUTO_DISAGG_SRC_SHARD and AUTO_DISAGG_DST_SHARD must differ"
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[auto-disagg] enabling scheduler: src=%d dst=%d — every "
                "base_generate call will stamp disagg_pair=[%d, %d]",
                src_idx,
                dst_idx,
                src_idx,
                dst_idx,
            )
            instance._disagg_rollout_pair = (src_idx, dst_idx)
            instance._register_migration_pair(src_idx, dst_idx)
            instance.enable_auto_disagg(src_shard_index=src_idx)

        # Two-stage tail-cut. When TAIL_CUT_DST_SHARD is set, every
        # rollout is also stamped with ``tail_cut=[dst, n]`` and a
        # second scheduler is spun up watching the disagg dst shard
        # (the throughput decode shard) — once a request there has
        # produced ``n`` tokens, the second scheduler migrates it to
        # the latency-optimized shard.
        tail_dst_env = os.environ.get("TAIL_CUT_DST_SHARD")
        tail_min_env = os.environ.get("TAIL_CUT_MIN_TOKENS")
        if (
            tail_dst_env is not None
            and tail_min_env is not None
            and dst_env is not None
            and len(shards) >= 3
        ):
            tail_dst = int(tail_dst_env)
            tail_min = int(tail_min_env)
            throughput_decode = int(dst_env)
            assert tail_dst != throughput_decode, (
                "TAIL_CUT_DST_SHARD must differ from AUTO_DISAGG_DST_SHARD"
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[auto-disagg] enabling tail-cut: src=%d dst=%d min_tokens=%d",
                throughput_decode,
                tail_dst,
                tail_min,
            )
            instance._tail_cut_rollout_config = (tail_dst, tail_min)
            instance._register_migration_pair(throughput_decode, tail_dst)
            instance.enable_auto_disagg(src_shard_index=throughput_decode)

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
        extra_body = {
            "skip_prompt_log_probs": True,
            "add_BOS": (not args.rl_skip_bos_token and tokenizer.bos is not None),
        }
        # When disagg is configured, every request gets tagged with the
        # (src, dst) shard pair. The HTTP endpoint routes the submit to
        # src via the coord's target_shard_index and stamps
        # disagg_dst_shard_index on the engine-side request so the
        # auto-disagg scheduler migrates it at first token. The HTTP
        # server's choice of shard (idx above) becomes irrelevant —
        # the coord overrides with target_shard_index=src.
        if self._disagg_rollout_pair is not None:
            extra_body["disagg_pair"] = list(self._disagg_rollout_pair)
        # Two-stage tail-cut: when set, the scheduler migrates rollouts
        # off the throughput decode shard onto a latency-optimized
        # decode shard once they exceed the threshold.
        if self._tail_cut_rollout_config is not None:
            extra_body["tail_cut"] = list(self._tail_cut_rollout_config)
        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model="",
                messages=[message.model_dump() for message in request.prompt],
                temperature=request.generation_args.temperature or 1.0,
                top_p=request.generation_args.top_p or 0.0,
                n=1,
                logprobs=True,
                extra_body=extra_body,
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
    # pause/resume/stop/shutdown commands through its unified
    # :class:`InferenceClient`. Non-driver ranks just wait on their
    # local engine state so the engine-side broadcast from the
    # coordinator brings them along.

    @property
    def _unified_client(self) -> Optional[InferenceClient]:
        """rank 0's single unified lifecycle client, or ``None`` elsewhere."""
        return self._lifecycle_clients[0] if self._lifecycle_clients else None

    def _drive_all(self, fn_name: str, *args, **kwargs) -> None:
        """Call ``fn_name(*args, **kwargs)`` on the unified lifecycle
        client (rank 0 only). The unified coord then broadcasts to every
        engine, so a single call reaches all shards.
        """
        client = self._unified_client
        if client is not None:
            getattr(client, fn_name)(*args, **kwargs)

    def set_generation_epoch(self, generation_epoch: int) -> None:
        self._drive_all("set_generation_epoch", generation_epoch)

    def _is_in_scope(self, shard_indices: Optional[List[int]]) -> bool:
        """True if this rank's engine is affected by a scoped lifecycle op.

        Ranks outside the scope must NOT call ``wait_until`` on their
        local engine — the coord won't send the signal to that engine,
        so the state never transitions and the wait would hang.
        """
        if shard_indices is None:
            return True
        return self._my_shard_index in shard_indices

    async def resume(self, *, shard_indices: Optional[List[int]] = None) -> None:
        # Whole-world resume: drive engines to RUNNING (engine-owning
        # ranks), then respawn the auto-disagg scheduler so it ticks
        # only inside this synchronized inference window. Idle ranks
        # still need ``_resume_scheduler`` so their broadcast partner
        # is alive — hence the finally block.
        try:
            if self._my_engine is None:
                return
            if not self._is_in_scope(shard_indices):
                return
            if self._my_engine._state_events[EngineState.RUNNING].is_set():
                return
            self._drive_all("resume_engines", shard_indices=shard_indices)
            await self._my_engine.wait_until(EngineState.RESUMED)
            self._drive_all("unpause_engines", shard_indices=shard_indices)
            await self._my_engine.wait_until(EngineState.RUNNING)
        finally:
            if shard_indices is None:
                self._resume_scheduler()

    async def suspend(self, *, shard_indices: Optional[List[int]] = None) -> None:
        # Cancel scheduler tasks BEFORE driving engines to PAUSED so an
        # in-flight scheduler tick doesn't try to start a migration
        # (which would itself call scoped suspend/resume) while the
        # outer whole-world suspend is racing it. Scoped suspends from
        # the migration primitive itself leave the scheduler alone —
        # they have ``shard_indices`` set.
        if shard_indices is None:
            await self._pause_scheduler()
        if self._my_engine is None:
            return
        if not self._is_in_scope(shard_indices):
            return
        self._drive_all("pause_engines", shard_indices=shard_indices)
        await self._my_engine.wait_until(EngineState.PAUSED)
        self._drive_all("suspend_engines", shard_indices=shard_indices)
        await self._my_engine.wait_until(EngineState.SUSPENDED)

    async def kill(self) -> None:
        # Stop the auto-disagg scheduler before engines so in-flight
        # polling doesn't race with teardown.
        await self.disable_auto_disagg()

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

        # Each rank that owns a shard's rank_offset stops its
        # per-shard client. Rank 0's lifecycle client (which is also
        # shard 0's client) is closed via the shutdown path below.
        for sidx, c in list(self._shard_clients.items()):
            if dist.get_rank() == 0 and sidx == 0:
                continue
            c.stop()
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

    # ---- Request migration ---------------------------------------------

    async def migrate_request(
        self,
        request_id: int,
        src_shard_index: int,
        dst_shard_index: int,
        *,
        request_id_dst: Optional[int] = None,
    ) -> Optional[int]:
        """Move an in-flight request from one shard's engine to another.

        Collective: every rank calls this; rank 0 decides. The call
        broadcasts the parameters to every rank, builds (or reuses) the
        cross-shard process group spanning the two shards, quiesces both
        engines, runs the KV-migration primitive, and resumes. The
        returned dst-side request id (if this rank owns the dst shard)
        lets the caller track the request's new identity.

        Args:
            request_id: Source-side request id on the src shard's engine.
                Ignored on ranks that are not in the src shard.
            src_shard_index: Index into ``_shards`` of the shard that
                currently holds the request.
            dst_shard_index: Index into ``_shards`` of the shard to move
                to. Must differ from ``src_shard_index`` (same-shard
                migration is a no-op at this API).
            request_id_dst: Optional explicit id to register on the
                destination. Defaults to ``request_id``. Useful when
                the caller's id namespace overlaps between shards.

        Returns:
            The dst-side request id on ranks in the dst shard; ``None``
            on other ranks.
        """
        # Delegate to the batch primitive so the layout/head-offset/
        # suspend-bracket/UPDATE_REQUEST_RANK plumbing lives in exactly
        # one place.
        migrated_ids = await self.migrate_requests_batch(
            request_ids=[request_id],
            src_shard_index=src_shard_index,
            dst_shard_index=dst_shard_index,
            request_ids_dst=[request_id_dst] if request_id_dst is not None else None,
        )
        if migrated_ids:
            return migrated_ids[0]
        return None

    def _layout_and_head_offset(
        self,
        shard: InferenceShard,
        engine: DynamicInferenceEngine,
        rank: int,
        *,
        participates: bool,
    ) -> "Tuple[object, int]":
        """Build a :class:`KVLayout` for ``shard`` plus this rank's head offset.

        The layout is whole-model invariant (same on every rank, src or
        dst); the head offset is this rank's position inside ``shard``'s
        TP window, used by the transport to translate global head
        ranges to local buffer indices. Ranks not in ``shard`` get
        offset 0 (they never read/write the buffer for that side).
        """
        from megatron.core.inference.engines.request_migration import KVLayout

        model_config = engine.controller.inference_wrapped_model.model.config
        ctx = engine.context
        num_kv_heads_total = (
            model_config.num_query_groups or model_config.num_attention_heads
        )
        tp = int(shard.spec["tp"])
        pp = int(shard.spec.get("pp", 1))
        layout = KVLayout(
            tp_size=tp,
            pp_size=pp,
            num_layers_total=model_config.num_layers,
            num_kv_heads_total=num_kv_heads_total,
            head_dim=ctx.hidden_size_per_attention_head,
            block_size_tokens=ctx.block_size_tokens,
            is_mla=getattr(ctx, "is_mla", False),
            kv_reduced_dim=getattr(ctx, "kv_reduced_dim", None),
        )
        # Rank layout within a shard is TP-major: (tp_rank, pp_rank)
        # sits at ``pp_rank * tp_size + tp_rank`` from rank_offset.
        # Head offset is driven by the TP component only; the PP
        # layer offset is computed inside
        # ``migrate_requests_cross_shard_batch`` from each rank's
        # position.
        head_offset = 0
        if participates:
            tp_rank = (rank - shard.rank_offset) % tp
            head_offset = tp_rank * (num_kv_heads_total // tp)
        return layout, head_offset

    async def migrate_requests_batch(
        self,
        request_ids: List[int],
        src_shard_index: int,
        dst_shard_index: int,
        *,
        request_ids_dst: Optional[List[int]] = None,
    ) -> Optional[List[int]]:
        """Migrate ``N`` in-flight requests in a single cross-shard collective.

        Same contract as :meth:`migrate_request` but the engine-level
        transport fuses all ``N`` KV plans into one NCCL call — the
        big win over looping :meth:`migrate_request` N times.
        Suspend/resume still brackets the whole batch once, not N times.

        Returns the dst-side request ids on ranks in the dst shard;
        ``None`` elsewhere. Empty batches are a no-op that returns
        ``[]`` on dst / ``None`` otherwise without any collective.
        """
        from megatron.core.inference.engines.request_migration import (
            migrate_requests_cross_shard_batch,
        )
        from megatron.core.inference.shards import build_cross_shard_group

        if not request_ids:
            return [] if (
                self._my_shard_index == dst_shard_index
            ) else None

        assert src_shard_index != dst_shard_index, (
            f"src and dst shard indices must differ (got {src_shard_index})"
        )
        assert 0 <= src_shard_index < len(self._shards)
        assert 0 <= dst_shard_index < len(self._shards)
        if request_ids_dst is not None:
            assert len(request_ids_dst) == len(request_ids), (
                "request_ids_dst length must match request_ids"
            )

        src_shard = self._shards[src_shard_index]
        dst_shard = self._shards[dst_shard_index]
        rank = dist.get_rank()
        in_src = src_shard.owns_rank(rank)
        in_dst = dst_shard.owns_rank(rank)

        cross_shard_group = build_cross_shard_group(
            self._shards, [src_shard_index, dst_shard_index]
        )

        if not (in_src or in_dst):
            return None

        my_engine = self._my_engine
        assert my_engine is not None
        src_layout, my_src_head_offset = self._layout_and_head_offset(
            src_shard, my_engine, rank, participates=in_src
        )
        dst_layout, my_dst_head_offset = self._layout_and_head_offset(
            dst_shard, my_engine, rank, participates=in_dst
        )

        scope = [src_shard_index, dst_shard_index]
        await self.suspend(shard_indices=scope)
        try:
            migrated_ids = migrate_requests_cross_shard_batch(
                role="src" if in_src else "dst",
                engine=my_engine,
                request_ids_src=request_ids if in_src else None,
                src_layout=src_layout,
                dst_layout=dst_layout,
                src_ranks=src_shard.ranks(),
                dst_ranks=dst_shard.ranks(),
                cross_shard_group=cross_shard_group,
                my_src_head_offset=my_src_head_offset,
                my_dst_head_offset=my_dst_head_offset,
                request_ids_dst=request_ids_dst,
            )
        finally:
            await self.resume(shard_indices=scope)

        # One UPDATE_REQUEST_RANK per request so the coord's dispatch
        # table follows the batch — the coord has no batched variant of
        # this message yet (each request has its own client future).
        client = self._unified_client
        if client is not None:
            for req_id in (request_ids_dst or request_ids):
                client.update_request_rank(
                    request_id=req_id,
                    new_shard_index=dst_shard_index,
                    new_dp_rank_within_shard=0,
                )

        return migrated_ids

    # ---- Auto-disagg scheduler -----------------------------------------
    #
    # Per-request opt-in: a request gets migrated iff it was submitted
    # with ``disagg_dst_shard_index`` set (via HTTP's ``disagg_pair`` or
    # :meth:`InferenceClient.add_request(disagg_dst_shard_index=...)`).
    # Untagged traffic on the same shard is left alone.
    #
    # **Coord-mediated design.** The scheduler runs as a single asyncio
    # task on the *decider rank* (the watched src shard's
    # ``rank_offset``). Each tick the decider inspects its local
    # engine's request dict, builds migration plans, and submits each
    # plan to the coordinator via
    # ``InferenceClient.migrate_request_batch(...)``. The coord
    # forwards a ``Headers.MIGRATE_BATCH`` signal to the engines in
    # src + dst shards; each engine pops the signal off its
    # ``_pending_signals`` queue and invokes the registered migration
    # handler (:meth:`_on_migrate_batch_signal`), which runs sync
    # NCCL on the cross-shard process group to move KV blocks.
    #
    # No cross-rank broadcast is required to coordinate the migration
    # decision — only the decider decides, and the trigger flows over
    # the coord's ZMQ control channel (same path as PAUSE / SUSPEND).
    # That avoids the cross-PG NCCL ordering deadlock the prior
    # broadcast-based designs hit.

    def enable_auto_disagg(
        self,
        src_shard_index: int,
        *,
        poll_interval_s: float = 0.05,
        max_batch_size: int = 16,
    ) -> None:
        """Register an auto-disagg scheduler watching one src shard.

        Collective: every rank must call (so each rank's instance has
        the same ``_scheduler_configs`` for symmetric pause/resume,
        even though only the decider rank actually runs the loop).

        Per-request opt-in: a request on ``src_shard_index`` is
        migrated when one of its tags fires — either
        ``disagg_dst_shard_index`` at first decode token, or
        (``late_dst_shard_index``, ``late_dst_min_tokens``) once the
        token threshold is met. Untagged requests stay put.

        Args:
            src_shard_index: Shard whose engine the scheduler polls.
            max_batch_size: Upper bound on migrations fired per tick.
        """
        assert 0 <= src_shard_index < len(self._shards)
        assert max_batch_size >= 1, "max_batch_size must be >= 1"
        for cfg in self._scheduler_configs:
            if cfg[0] == src_shard_index:
                raise RuntimeError(
                    f"auto-disagg scheduler already running for src shard "
                    f"{src_shard_index}"
                )
        # Tasks are spawned lazily by the next whole-world
        # :meth:`resume`. We don't spawn at enable time because
        # ``_resume_scheduler`` needs an actively-running asyncio
        # loop, and ``enable_auto_disagg`` is called from
        # :meth:`launch` while the launch coroutine is still running
        # (loop is alive there, but tasks should only run during
        # actual inference windows so we don't accumulate stale work
        # between launch and the first resume).
        self._scheduler_configs.append(
            (src_shard_index, poll_interval_s, max_batch_size)
        )

    def _register_migration_pair(
        self, src_shard_index: int, dst_shard_index: int
    ) -> None:
        """Pre-compute migration metadata for ``(src, dst)`` and wire
        the engine's MIGRATE_BATCH handler.

        Collective: every rank must call. ``build_cross_shard_group``
        creates a torch process group covering ranks in src + dst,
        which has to happen with all ranks in lockstep. The metadata
        (layouts, head offsets, group handle) is then cached so the
        per-migration cost is just the NCCL transport itself.

        Bystander ranks (not in src or dst) still call this — their
        local metadata for this pair is never consulted (engine
        handler returns early for bystanders), but the
        ``build_cross_shard_group`` collective requires their
        participation.
        """
        from megatron.core.inference.shards import build_cross_shard_group

        assert 0 <= src_shard_index < len(self._shards)
        assert 0 <= dst_shard_index < len(self._shards)
        assert src_shard_index != dst_shard_index
        key = (src_shard_index, dst_shard_index)
        if key in self._migration_meta:
            return  # already registered

        src_shard = self._shards[src_shard_index]
        dst_shard = self._shards[dst_shard_index]
        rank = dist.get_rank()
        in_src = src_shard.owns_rank(rank)
        in_dst = dst_shard.owns_rank(rank)

        # Collective on every rank, even bystanders — the PG creation
        # itself is world-synchronized.
        cross_shard_group = build_cross_shard_group(
            self._shards, [src_shard_index, dst_shard_index]
        )

        meta = {
            "src_shard": src_shard,
            "dst_shard": dst_shard,
            "cross_shard_group": cross_shard_group,
            "in_src": in_src,
            "in_dst": in_dst,
        }
        if in_src or in_dst:
            assert self._my_engine is not None
            src_layout, my_src_head_offset = self._layout_and_head_offset(
                src_shard, self._my_engine, rank, participates=in_src
            )
            dst_layout, my_dst_head_offset = self._layout_and_head_offset(
                dst_shard, self._my_engine, rank, participates=in_dst
            )
            meta.update(
                {
                    "src_layout": src_layout,
                    "dst_layout": dst_layout,
                    "my_src_head_offset": my_src_head_offset,
                    "my_dst_head_offset": my_dst_head_offset,
                }
            )
        self._migration_meta[key] = meta

        # Wire the engine handler once. Subsequent pairs share it.
        if self._my_engine is not None and getattr(
            self._my_engine, '_migration_handler', None
        ) is None:
            self._my_engine.set_migration_handler(self._on_migrate_batch_signal)

    def _on_migrate_batch_signal(
        self,
        request_ids: List[int],
        src_shard_index: int,
        dst_shard_index: int,
        bundles: List[dict],
        dst_dp_rank: int = 0,
    ) -> None:
        """Engine-side handler for ``Headers.MIGRATE_BATCH`` —
        **non-blocking** on both src and dst.

        - **Bundle delivery via coord (no cross-shard broadcast):**
          ``bundles`` is the full per-request migration envelope shipped
          inline by the auto-disagg scheduler.
        - **Src side:** gather KV slices into staging slots and issue
          ``put_signal`` on the migration stream. Record a CUDA event
          covering the gather; push a pending entry whose ``poll``
          callback detaches the request (frees src blocks) once the
          event fires.
        - **Dst side:** ``inject_request`` to allocate dst KV blocks +
          register the request, then schedule ``signal_wait`` + scatter
          on the migration stream and ``wait_stream`` the engine's
          compute stream onto it. Future forward passes touching this
          request's KV implicitly wait for scatter to complete.

        No ``barrier_all``, no cross-shard broadcast, no engine pause.
        """
        meta = self._migration_meta.get((src_shard_index, dst_shard_index))
        if meta is None:
            logger.error(
                "[auto-disagg] no migration metadata for (%d, %d) — "
                "registered pairs: %s",
                src_shard_index,
                dst_shard_index,
                list(self._migration_meta.keys()),
            )
            return
        if not (meta["in_src"] or meta["in_dst"]):
            return  # bystander; coord shouldn't forward to us anyway

        try:
            self._do_async_migrate_batch(
                request_ids,
                src_shard_index,
                dst_shard_index,
                bundles,
                meta,
                dst_dp_rank,
            )
        except Exception as e:
            logger.error(
                "[auto-disagg] async migration of %d requests (%d → %d) "
                "raised %s — engine continues",
                len(request_ids),
                src_shard_index,
                dst_shard_index,
                e,
            )

    def _do_async_migrate_batch(
        self,
        request_ids: List[int],
        src_shard_index: int,
        dst_shard_index: int,
        bundles: List[dict],
        meta: dict,
        dst_dp_rank: int = 0,
    ) -> None:
        """Body of :meth:`_on_migrate_batch_signal` — split out so the
        outer wrapper can catch exceptions without losing the run
        loop. See the parent docstring for the design."""
        from megatron.core.inference.engines.request_migration import (
            _gather_kv_slice,
            _scatter_kv_slice,
            build_kv_migration_plan,
            deserialize_bundle,
        )
        from megatron.core.inference import nvshmem_migration as _nv

        src_shard = meta["src_shard"]
        dst_shard = meta["dst_shard"]
        src_layout = meta["src_layout"]
        dst_layout = meta["dst_layout"]
        src_root = src_shard.ranks()[0]
        # ``dst_root`` is the global rank of TP-rank-0 of the *target
        # dp replica* — not always rank_offset when dst has DP > 1.
        # Within a shard ranks are laid out TP-major: rank
        # ``rank_offset + dp * tp_size + tp``. The coord forwards
        # MIGRATE_BATCH only to that one dp rank, and the migration
        # plan must address the same rank for ``put_signal`` and
        # scatter; mismatched ``dst_root`` would have the dst engine
        # call ``inject_request`` (allocating blocks) but skip
        # scatter, leaving its KV uninitialized.
        dst_root = dst_shard.rank_offset + dst_dp_rank * dst_layout.tp_size

        def _src_rank_of(tp: int, pp: int) -> int:
            return src_root + pp * src_layout.tp_size + tp

        def _dst_rank_of(tp: int, pp: int) -> int:
            return dst_root + pp * dst_layout.tp_size + tp

        # Deserialize bundles inline (small dicts).
        parsed_bundles = [deserialize_bundle(b) for b in bundles]
        engine = self._my_engine
        memory_buffer = engine.context.memory_buffer
        layout = src_layout if meta["in_src"] else dst_layout

        # Per-rank PP layer offsets for the migration plan's local
        # head/layer slicing.
        rank = dist.get_rank()
        my_src_pp_stage = (
            (rank - src_root) // src_layout.tp_size if meta["in_src"] else 0
        )
        my_dst_pp_stage = (
            (rank - dst_root) // dst_layout.tp_size if meta["in_dst"] else 0
        )
        src_layers_per_pp = src_layout.num_layers_total // src_layout.pp_size
        dst_layers_per_pp = dst_layout.num_layers_total // dst_layout.pp_size
        my_src_pp_layer_offset = my_src_pp_stage * src_layers_per_pp
        my_dst_pp_layer_offset = my_dst_pp_stage * dst_layers_per_pp
        my_src_head_offset = meta["my_src_head_offset"]
        my_dst_head_offset = meta["my_dst_head_offset"]

        stream = _nv.migration_stream()

        # Src detaches each migrated request *now* (before the put_signal is
        # scheduled) so the engine's main loop can't run another async_step
        # and emit a stale ENGINE_REPLY for an already-migrated request.
        # ``keep_blocks=True`` keeps the KV blocks alive for the gather
        # kernels we'll enqueue below; the poll callback releases them once
        # the migration's CUDA event has fired.
        src_block_ids_to_free: List[torch.Tensor] = []
        if meta["in_src"]:
            for req_id in request_ids:
                if req_id in engine.requests:
                    src_block_ids_to_free.append(
                        engine.detach_request(req_id, keep_blocks=True)
                    )

        # Dst injects each bundle (allocates dst blocks, registers
        # request) and accumulates the (bundle, dst_blocks) pairs.
        injected: List[Tuple[object, List[int]]] = []
        if meta["in_dst"]:
            for bundle in parsed_bundles:
                dst_blocks = engine.inject_request(bundle).tolist()
                injected.append((bundle, dst_blocks))

        # Build per-bundle KV migration plans. Both src and dst walk
        # the bundles in identical order so slot/flag assignment
        # stays consistent.
        all_ops_by_bundle: List[List] = []
        for i, bundle in enumerate(parsed_bundles):
            if meta["in_dst"]:
                _, dst_blocks = injected[i]
            else:
                # On src we don't have dst_blocks; the bundle's
                # ``num_kv_blocks`` is the count, the dst-side block
                # ids are unknowable here. But the migration plan
                # only needs (src_block_ids, dst_block_ids) at the
                # src for the gather slice — the put_signal lands at
                # the SAME staging-slot offset on dst regardless of
                # which dst_block_ids dst will use. So the dst block
                # list doesn't actually need to match between sides.
                #
                # Use src_block_ids as a placeholder; the migration
                # plan's dst_block_ids field is consulted only on the
                # dst side (during scatter), so a placeholder here is
                # fine.
                dst_blocks = list(bundle.src_block_ids)
            ops = build_kv_migration_plan(
                bundle,
                src_global_rank_of=_src_rank_of,
                dst_global_rank_of=_dst_rank_of,
                dst_block_ids=dst_blocks,
            )
            all_ops_by_bundle.append(ops)

        # Walk every (bundle, op) deterministically to assign slot +
        # flag indices. Same order on src and dst → same slot/flag
        # for each transfer.
        arena = _nv.StagingArena()
        op_meta: List[dict] = []
        elem_size = memory_buffer.element_size()
        dtype = memory_buffer.dtype
        for ops in all_ops_by_bundle:
            for op in ops:
                num_blocks = len(op.src_block_ids)
                layer_span = op.layer_range[1] - op.layer_range[0]
                head_span = op.head_range[1] - op.head_range[0]
                if layout.is_mla:
                    shape = (
                        layer_span,
                        num_blocks,
                        layout.block_size_tokens,
                        head_span,
                    )
                else:
                    shape = (
                        2,
                        layer_span,
                        num_blocks,
                        layout.block_size_tokens,
                        head_span,
                        layout.head_dim,
                    )
                nelems = 1
                for d in shape:
                    nelems *= d
                nbytes = nelems * elem_size
                op_meta.append(
                    {
                        "op": op,
                        "shape": shape,
                        "nbytes": nbytes,
                        "slot": arena.take(nbytes),
                        "flag": _nv.acquire_flag_slot(),
                    }
                )

        # Schedule everything on the migration stream and return.
        with torch.cuda.stream(stream):
            if meta["in_src"]:
                # Gather + put_signal per op this rank participates in.
                for entry in op_meta:
                    op = entry["op"]
                    if rank != op.src_rank:
                        continue
                    src_blocks = torch.as_tensor(
                        op.src_block_ids, device=memory_buffer.device, dtype=torch.long
                    )
                    local_head_range = (
                        op.head_range[0] - my_src_head_offset,
                        op.head_range[1] - my_src_head_offset,
                    )
                    local_layer_range = (
                        op.layer_range[0] - my_src_pp_layer_offset,
                        op.layer_range[1] - my_src_pp_layer_offset,
                    )
                    send_tensor = _gather_kv_slice(
                        memory_buffer,
                        local_layer_range,
                        src_blocks,
                        local_head_range,
                        layout.is_mla,
                    )
                    slot = _nv.staging_slot(entry["slot"])
                    slot[: entry["nbytes"]].view(dtype).reshape(
                        entry["shape"]
                    ).copy_(send_tensor, non_blocking=True)
                    _nv.put_slot_with_signal(
                        entry["slot"],
                        entry["flag"],
                        op.dst_rank,
                        nbytes=entry["nbytes"],
                        stream=stream,
                    )

            if meta["in_dst"]:
                # signal_wait + scatter per op this rank participates in.
                for entry in op_meta:
                    op = entry["op"]
                    if rank != op.dst_rank:
                        continue
                    _nv.wait_slot_signal(
                        entry["flag"], expected_value=1, stream=stream
                    )
                    dst_blocks = torch.as_tensor(
                        op.dst_block_ids, device=memory_buffer.device, dtype=torch.long
                    )
                    local_head_range = (
                        op.head_range[0] - my_dst_head_offset,
                        op.head_range[1] - my_dst_head_offset,
                    )
                    local_layer_range = (
                        op.layer_range[0] - my_dst_pp_layer_offset,
                        op.layer_range[1] - my_dst_pp_layer_offset,
                    )
                    slot = _nv.staging_slot(entry["slot"])
                    recv_view = (
                        slot[: entry["nbytes"]]
                        .view(dtype)
                        .reshape(entry["shape"])
                    )
                    _scatter_kv_slice(
                        memory_buffer,
                        local_layer_range,
                        dst_blocks,
                        local_head_range,
                        layout.is_mla,
                        recv_view,
                    )

            # Mark the end of this migration on the stream so the
            # tick poll can query a CUDA event.
            done_event = torch.cuda.Event()
            done_event.record(stream)

        # Engine compute stream waits for the migration stream — any
        # forward pass touching the migrated KV blocks (on either
        # side) implicitly waits without us having to pause the
        # engine here.
        torch.cuda.default_stream().wait_stream(stream)

        # Push pending entry whose poll resets the flags + (on src)
        # detaches the freed requests once the event has fired.
        flags_used = [e["flag"] for e in op_meta]

        def _poll() -> bool:
            if not done_event.query():
                return False
            for slot in flags_used:
                _nv.reset_flag(slot)
            # Src detached each request synchronously above with
            # ``keep_blocks=True`` so the gather kernel could read them
            # safely. Now that the gather has completed, return the blocks
            # to the KV allocator.
            if meta["in_src"] and src_block_ids_to_free:
                allocator = engine.context.kv_block_allocator
                for blocks in src_block_ids_to_free:
                    if blocks.numel() > 0:
                        allocator.release_memory_blocks(blocks)
            return True

        engine.push_pending_migration({"poll": _poll})

    async def _pause_scheduler(self) -> None:
        """Stop all scheduler asyncio tasks. Configs stay in
        ``_scheduler_configs`` so :meth:`_resume_scheduler` can respawn
        them at the next whole-world :meth:`resume`."""
        if not self._auto_disagg_tasks:
            return
        self._scheduler_stop = True
        for t in self._auto_disagg_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[auto-disagg] task raised on pause: %s", e)
        self._auto_disagg_tasks.clear()
        self._scheduler_stop = False

    def _resume_scheduler(self) -> None:
        """Spawn one asyncio task per stored config. No-op on ranks
        that aren't a decider for any config (the task loop returns
        immediately on non-decider ranks anyway, but skipping the
        spawn avoids cluttering the loop)."""
        if self._auto_disagg_tasks:
            return
        if not self._scheduler_configs:
            return
        self._scheduler_stop = False
        loop = asyncio.get_running_loop()
        rank = dist.get_rank()
        for src_shard_index, poll, batch in self._scheduler_configs:
            decider_rank = self._shards[src_shard_index].rank_offset
            if rank != decider_rank:
                continue  # non-decider ranks don't participate
            self._auto_disagg_tasks.append(
                loop.create_task(
                    self._auto_disagg_loop(src_shard_index, poll, batch),
                    name=f"megatron_local_multi.auto_disagg@s{src_shard_index}",
                )
            )

    async def disable_auto_disagg(self) -> None:
        """Stop every auto-disagg scheduler instance and forget configs.
        Idempotent. Distinct from :meth:`_pause_scheduler` in that
        ``_scheduler_configs`` is cleared, so a subsequent
        :meth:`_resume_scheduler` won't respawn anything."""
        self._scheduler_configs.clear()
        await self._pause_scheduler()

    @staticmethod
    def _migration_target(entry, current_shard_index: int) -> Optional[int]:
        """Pick where the request in ``entry`` should migrate to from
        ``current_shard_index``.

        Takes a :class:`RequestEntry` (the value type stored in
        ``engine.requests``) and reads the live request from
        ``entry.record[-1]``. Returns the dst shard index, or ``None``
        if no trigger fires. First-token disagg takes priority over
        the tail-cut trigger so a freshly-prefilled request lands on
        the throughput decode shard before any late-stage move.
        """
        # ``engine.requests`` maps id → RequestEntry; the live request
        # is the last item in ``entry.record`` (records grow on
        # cross-shard migrations to track ancestor states).
        record = getattr(entry, "record", None)
        if not record:
            return None
        req = record[-1]
        gen_tokens = getattr(req, "generated_tokens", None)
        if gen_tokens is None:
            return None
        n = (
            gen_tokens.numel()
            if isinstance(gen_tokens, torch.Tensor)
            else len(gen_tokens)
        )
        # First-token disagg.
        disagg = getattr(req, "disagg_dst_shard_index", None)
        if disagg is not None and disagg != current_shard_index and n >= 1:
            return disagg
        # Tail-cut: only fires once ``late_dst_min_tokens`` tokens have
        # accumulated AND we're not already on the late dst shard.
        late_dst = getattr(req, "late_dst_shard_index", None)
        late_min = getattr(req, "late_dst_min_tokens", None)
        if (
            late_dst is not None
            and late_min is not None
            and late_dst != current_shard_index
            and n >= late_min
        ):
            return late_dst
        return None

    async def _auto_disagg_loop(
        self,
        src_shard_index: int,
        poll_interval_s: float,
        max_batch_size: int,
    ) -> None:
        """Per-tick decision loop. **Runs only on the decider rank**
        (``self._shards[src_shard_index].rank_offset``).

        Each tick: inspect the local engine's request dict, build per-
        dst migration plans, and submit each plan to the coordinator
        via ``InferenceClient.migrate_request_batch``. The coord
        forwards a ``MIGRATE_BATCH`` signal to engines in src + dst,
        which run :meth:`_on_migrate_batch_signal` in their run loop
        to perform the NCCL transport. Nothing on this loop blocks on
        NCCL — the migration is asynchronous from the scheduler's
        perspective.
        """
        src_shard = self._shards[src_shard_index]
        decider_rank = src_shard.rank_offset
        rank = dist.get_rank()
        assert rank == decider_rank, (
            f"_auto_disagg_loop should only run on decider rank "
            f"{decider_rank} (got rank {rank})"
        )
        if self._my_engine is None:
            return
        client = self._shard_clients.get(src_shard_index)
        if client is None:
            logger.error(
                "[auto-disagg] no InferenceClient for shard %d on rank %d "
                "— scheduler cannot post migrations",
                src_shard_index,
                rank,
            )
            return

        # Already-fired ids; pruned each tick to the set still resident
        # on this engine. A persistent migration failure leaves an id in
        # this set so we don't retry-storm. Successful migrations move
        # the id off this engine, so the next active_ids intersection
        # drops it naturally.
        migrated_ids: set = set()

        while not self._scheduler_stop:
            await asyncio.sleep(poll_interval_s)
            if self._scheduler_stop:
                return
            if not self._my_engine._state_events[EngineState.RUNNING].is_set():
                continue

            # Snapshot active requests; tolerate races with the engine
            # task by retrying on RuntimeError (dict changed size
            # during iteration).
            groups: dict = {}
            try:
                active_ids = set(self._my_engine.requests.keys())
                migrated_ids &= active_ids
                for req_id, entry in list(self._my_engine.requests.items()):
                    if req_id in migrated_ids:
                        continue
                    dst = self._migration_target(entry, src_shard_index)
                    if dst is None:
                        continue
                    bucket = groups.setdefault(dst, [])
                    if len(bucket) < max_batch_size:
                        bucket.append(int(req_id))
            except RuntimeError:
                continue

            if not groups:
                continue

            # Submit one batched migration per dst. We snapshot each
            # request on the local (decider) engine to build the
            # bundle and ship it inline through the coord — so the
            # dst engine doesn't need a cross-shard broadcast to
            # learn the request's metadata. This is what makes the
            # whole migration handler non-blocking on both sides.
            from megatron.core.inference.engines.request_migration import (
                serialize_bundle,
            )

            for dst_shard_index, request_ids in groups.items():
                try:
                    bundles = []
                    for req_id in request_ids:
                        bundle, _ = self._my_engine.snapshot_request(req_id)
                        # Stamp the layouts so the dst can build
                        # the migration plan without re-deriving them.
                        meta = self._migration_meta.get(
                            (src_shard_index, dst_shard_index)
                        )
                        if meta is not None:
                            bundle.src_layout = meta["src_layout"]
                            bundle.dst_layout = meta["dst_layout"]
                        bundles.append(serialize_bundle(bundle))
                    client.migrate_request_batch(
                        request_ids,
                        src_shard_index,
                        dst_shard_index,
                        bundles=bundles,
                    )
                    for req_id in request_ids:
                        client.update_request_rank(
                            request_id=req_id,
                            new_shard_index=dst_shard_index,
                            new_dp_rank_within_shard=0,
                        )
                    migrated_ids.update(request_ids)
                except Exception as e:
                    logger.error(
                        "[auto-disagg] failed to submit migration of %d "
                        "requests (%d → %d): %s",
                        len(request_ids),
                        src_shard_index,
                        dst_shard_index,
                        e,
                    )

    # ---- Disaggregated inference ---------------------------------------

    async def _submit_via_coord(
        self,
        prompt: str,
        sampling_params: "SamplingParams",
        shard_index: int,
    ):
        """Submit a request to the unified coordinator and broadcast the
        server-side request id to every rank.

        Only rank 0 (which holds the unified lifecycle client) actually
        sends; every rank receives the server id so subsequent collective
        operations (e.g. migration) target the same engine-visible id.

        Returns:
            A tuple ``(server_request_id, completion_future)``. The
            future is non-``None`` only on rank 0; its eventual result
            is the full ENGINE_REPLY payload from whichever engine
            finishes the request (possibly after a cross-shard migration).
        """
        tokenizer = get_tokenizer()
        server_request_id = -1
        completion_future: Optional[asyncio.Future] = None
        client = self._unified_client
        if client is not None:
            prompt_tokens = tokenizer.tokenize(prompt)
            client_request_id = client.next_request_id
            completion_future = client.add_request(
                prompt_tokens, sampling_params, target_shard_index=shard_index
            )
            # The coord assigns the server-side id; engines and the
            # UPDATE_REQUEST_RANK control path both speak server ids, so
            # we have to wait for the ack before the caller drives any
            # server-side operation on this request.
            server_request_id = await client.wait_for_server_id(client_request_id)

        id_tensor = torch.tensor(
            [server_request_id if rank == 0 else 0],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.broadcast(id_tensor, src=0)
        return int(id_tensor.item()), completion_future

    async def _await_request_progress(
        self,
        request_id: int,
        shard_index: int,
        *,
        min_generated_tokens: int = 1,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.1,
    ) -> int:
        """Block until ``request_id`` has ≥ ``min_generated_tokens`` on
        the given shard's engine, or ``timeout_s`` elapses.

        Called on every rank; only the shard's rank_offset actually
        inspects the engine and broadcasts the observed token count so
        the rest of the world agrees. Returns the observed count.
        """
        shard = self._shards[shard_index]
        rank = dist.get_rank()
        deadline = asyncio.get_event_loop().time() + timeout_s
        observed = 0

        while asyncio.get_event_loop().time() < deadline:
            if rank == shard.rank_offset:
                engine = self._my_engine
                if request_id in engine.requests:
                    observed = len(engine.get_request(request_id).generated_tokens)

            count_tensor = torch.tensor(
                [observed if rank == shard.rank_offset else 0],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            dist.broadcast(count_tensor, src=shard.rank_offset)
            observed = int(count_tensor.item())
            if observed >= min_generated_tokens:
                break
            await asyncio.sleep(poll_interval_s)

        return observed

    async def submit_disaggregated_request(
        self,
        prompt: str,
        sampling_params: "SamplingParams",
        *,
        prefill_shard_index: int = 0,
        decode_shard_index: int = 1,
        prefill_timeout_s: float = 30.0,
        poll_interval_s: float = 0.1,
    ):
        """Run prefill on one shard and decode on another.

        Collective: every rank must call. The flow:

            1. rank 0 submits the prompt to the unified coordinator with
               ``target_shard_index=prefill_shard_index`` (same ZMQ path
               HTTP would use).
            2. All ranks poll the prefill shard's engine until the
               request has produced at least one generated token — i.e.
               prefill has completed and written KV for every prompt
               position, and the first decode step has sampled the
               initial output token.
            3. The request is migrated to the decode shard using the
               same KV-transport collective that powers mid-flight
               migration; rank 0 also sends UPDATE_REQUEST_RANK so the
               coord's owner-of-request mapping follows.
            4. The decode shard continues generation and eventually
               sends ENGINE_REPLY through the unified coord back to the
               original client. That makes the client's completion
               future resolve exactly as if no migration had happened.

        Returns:
            A tuple ``(server_request_id, completion_future)``. The
            future is non-``None`` only on rank 0 (which holds the
            submitting client); its result is the final ENGINE_REPLY
            payload from the decode shard's engine.

        Preconditions:
          - Engines must be RUNNING (call after :meth:`resume`); the
            prefill shard must be able to pick up the submit.
        """
        assert prefill_shard_index != decode_shard_index, (
            "prefill and decode shards must differ for disaggregation"
        )

        request_id, completion_future = await self._submit_via_coord(
            prompt, sampling_params, prefill_shard_index
        )
        observed_tokens = await self._await_request_progress(
            request_id,
            prefill_shard_index,
            min_generated_tokens=1,
            timeout_s=prefill_timeout_s,
            poll_interval_s=poll_interval_s,
        )
        if observed_tokens < 1:
            raise RuntimeError(
                f"disaggregated submit: prefill did not complete within "
                f"{prefill_timeout_s}s on shard {prefill_shard_index} "
                f"(request {request_id} not seen on engine)"
            )
        await self.migrate_request(
            request_id=request_id,
            src_shard_index=prefill_shard_index,
            dst_shard_index=decode_shard_index,
        )
        return request_id, completion_future

    # ---- Smoke tests ---------------------------------------------------

    async def run_disaggregated_smoke_test(
        self,
        *,
        prefill_shard_index: int = 0,
        decode_shard_index: int = 1,
        prompt: str = "The quick brown fox jumps over the lazy dog",
        post_decode_sleep_s: float = 5.0,
    ) -> None:
        """One-shot self-test for :meth:`submit_disaggregated_request`.

        Gated on ``DISAGG_SMOKE_TEST=1`` from :meth:`launch`. Logs the
        prefill → migrate → decode hop and the decode-side token count
        after a short grace period, so a human can eyeball that
        disaggregation is actually running end-to-end.
        """
        from megatron.core.inference.sampling_params import SamplingParams

        rank = dist.get_rank()
        decode_shard = self._shards[decode_shard_index]

        # Drive engines to RUNNING before the coord can see our submit.
        await self.resume()

        sampling = SamplingParams(
            num_tokens_to_generate=2048,
            termination_id=-1,
            return_log_probs=False,
            skip_prompt_log_probs=True,
        )

        if rank == 0:
            logger.info(
                "[disagg-smoke] prefill→decode shards = %d → %d (prompt=%r)",
                prefill_shard_index,
                decode_shard_index,
                prompt,
            )

        request_id, completion_future = await self.submit_disaggregated_request(
            prompt,
            sampling,
            prefill_shard_index=prefill_shard_index,
            decode_shard_index=decode_shard_index,
        )

        if rank == decode_shard.rank_offset:
            engine = self._my_engine
            if request_id in engine.requests:
                n_at_handoff = len(engine.get_request(request_id).generated_tokens)
                logger.info(
                    "[disagg-smoke] at handoff: request %d landed on decode "
                    "shard (%d) with %d generated tokens (should be ≥1 "
                    "— the first-decode token produced during prefill)",
                    request_id,
                    decode_shard_index,
                    n_at_handoff,
                )
            else:
                logger.error(
                    "[disagg-smoke] handoff FAILED: request %d not on "
                    "decode shard %d",
                    request_id,
                    decode_shard_index,
                )

        # The whole point of the unified coord is that the client's
        # completion future resolves transparently after migration. Cap
        # the wait so a broken topology still fails fast.
        if rank == 0 and completion_future is not None:
            try:
                reply = await asyncio.wait_for(
                    completion_future, timeout=post_decode_sleep_s + 30.0
                )
                n_final = len(reply.get("generated_tokens") or [])
                logger.info(
                    "[disagg-smoke] HTTP-path future resolved for request "
                    "%d: %d generated tokens",
                    request_id,
                    n_final,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[disagg-smoke] HTTP-path future DID NOT resolve for "
                    "request %d within the grace window — coord-side rank "
                    "update likely not reaching the decode shard",
                    request_id,
                )
        else:
            # Non-rank-0 ranks still participate in the polling so shard
            # collective barriers line up, but they don't own the client
            # future.
            await asyncio.sleep(post_decode_sleep_s)

        if rank == 0:
            logger.info("[disagg-smoke] smoke test completed")

    async def run_migration_smoke_test(
        self,
        *,
        src_shard_index: int = 0,
        dst_shard_index: int = 1,
        prompt: str = "The quick brown fox jumps over the lazy dog",
        pre_migrate_sleep_s: float = 10.0,
        post_migrate_sleep_s: float = 5.0,
    ) -> None:
        """Submit a dummy request to one shard, migrate it, verify.

        A one-shot self-test that proves the migration primitive works
        against the live engines launched by :meth:`launch`. Useful as a
        post-boot assertion in RL runs to catch migration regressions
        before they surface mid-rollout. Runs collectively: every rank
        must call it.

        The flow on a healthy system:

            1. rank 0 of the src shard submits a short prompt via the
               shard's :class:`InferenceClient` (same path HTTP takes).
            2. After ``pre_migrate_sleep_s`` the engine has prefilled
               and produced at least one decode token.
            3. All ranks enter :meth:`migrate_request`; the primitive
               pauses both engines, runs the collective transport, and
               resumes.
            4. After ``post_migrate_sleep_s`` the dst shard has
               continued generation on the migrated KV.
            5. rank 0 logs the before/after token counts.

        The originating HTTP/coord future is deliberately abandoned —
        coordinator-mediated migration plumbing (so the src coord
        forwards the reply to the dst, or the caller learns where the
        request went) is a separate follow-up. This test exercises the
        engine-level + transport-level path only.
        """
        from megatron.core.inference.sampling_params import SamplingParams

        rank = dist.get_rank()
        src_shard = self._shards[src_shard_index]
        dst_shard = self._shards[dst_shard_index]

        # ``launch`` leaves engines in RESUMED but not RUNNING (the
        # caller normally does that via ``resume`` right after). Drive
        # both engines to RUNNING here so the coord can forward our
        # dummy submit to the engine-side MP group before the sleep.
        await self.resume()

        # Generous token budget so the request is still mid-decode when
        # the migration collective fires (a short request would finish
        # and be cleaned out of engine.requests before we snapshot it).
        smoke_sampling = SamplingParams(
            num_tokens_to_generate=2048,
            termination_id=-1,
            return_log_probs=False,
            skip_prompt_log_probs=True,
        )
        # The migration smoke test is about the engine-level transport —
        # it doesn't need to observe the coord's completion future.
        submitted_request_id, _ = await self._submit_via_coord(
            prompt, smoke_sampling, src_shard_index
        )
        if rank == 0:
            logger.info(
                "[migration-smoke] rank 0 submitted request %d via shard %d's "
                "coordinator",
                submitted_request_id,
                src_shard_index,
            )

        # Let the coordinator push the request to the shard's MP group,
        # then prefill + some decode steps.
        await asyncio.sleep(pre_migrate_sleep_s)

        if rank == src_shard.rank_offset:
            engine = self._my_engine
            n_before = len(engine.get_request(submitted_request_id).generated_tokens)
            logger.info(
                "[migration-smoke] pre-migration: request %d on src shard has "
                "%d generated tokens — migrating to shard %d",
                submitted_request_id,
                n_before,
                dst_shard_index,
            )

        # Collective migration. Every rank calls; primitive handles the
        # role dispatch internally (src vs dst vs bystander).
        await self.migrate_request(
            request_id=submitted_request_id,
            src_shard_index=src_shard_index,
            dst_shard_index=dst_shard_index,
        )

        if rank == dst_shard.rank_offset:
            engine = self._my_engine
            if submitted_request_id in engine.requests:
                n_after = len(engine.get_request(submitted_request_id).generated_tokens)
                logger.info(
                    "[migration-smoke] post-migration: dst shard has request "
                    "%d with %d generated tokens (migration succeeded)",
                    submitted_request_id,
                    n_after,
                )
            else:
                logger.error(
                    "[migration-smoke] migration appears to have FAILED: "
                    "dst shard engine does not have request %d",
                    submitted_request_id,
                )

        # Give dst shard a moment to continue generating on the migrated KV.
        await asyncio.sleep(post_migrate_sleep_s)

        if rank == dst_shard.rank_offset:
            engine = self._my_engine
            if submitted_request_id in engine.requests:
                n_final = len(engine.get_request(submitted_request_id).generated_tokens)
                logger.info(
                    "[migration-smoke] after %.1fs of continued decode on dst "
                    "shard: request %d has %d generated tokens",
                    post_migrate_sleep_s,
                    submitted_request_id,
                    n_final,
                )

        if rank == 0:
            logger.info("[migration-smoke] smoke test completed")

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
