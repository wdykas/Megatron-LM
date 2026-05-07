# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multi-shard inference serving for heterogeneous RL rollout.

Heterogeneous shards share a single ``DataParallelInferenceCoordinator``;
each shard runs its own ``DynamicInferenceEngine`` and text-generation
HTTP server. ``MegatronLocalMulti`` (returned from ``launch`` on every
rank) drives lifecycle from rank 0 and routes ``base_generate`` across
shards with 1/latency weighting.

Cross-shard request migration runs over NVSHMEM one-sided
``put_signal`` / ``signal_wait`` on a dedicated stream, so neither the
src nor dst engine pauses its run loop. Each registered
:class:`MigrationPolicy` runs on its src shard's decider rank, picks
candidates, and posts ``MIGRATE_BATCH`` to the coord, which forwards
the signal to the participating ranks in src + the chosen dst dp_rank.
"""
import asyncio
import logging
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

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
        # ``entrypoint`` calls ``ready_event.set()``; we don't read it
        # here (we wait on ``coord_pipe.recv()`` for the bound address).
        ready_event = spawn_context.Event()
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
    # driver). Pause/resume / shutdown broadcast through the single
    # coord, so one client suffices regardless of shard count;
    # everywhere else this is None.
    _lifecycle_client: Optional[InferenceClient] = PrivateAttr(default=None)

    # Routing state for base_generate (lives on every rank but is only
    # exercised on the rollout-driver rank; per-rank lock is fine).
    _openai_clients: List[Optional[AsyncOpenAI]] = PrivateAttr(default_factory=list)
    _next_shard: int = PrivateAttr(default=0)
    _route_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    # Migration scheduler state — coord-mediated, policy-driven.
    #
    # Each registered :class:`MigrationPolicy` runs as an asyncio task
    # on the decider rank of its ``src_shard_index`` (the shard's
    # ``rank_offset``). Each tick, the policy's ``is_eligible`` is
    # called per request on that shard's engine; eligible requests
    # are bundled and submitted as ``MIGRATE_BATCH`` via the shard's
    # ``InferenceClient``. The coord forwards the signal to participating
    # ranks in src + the chosen dst dp_rank, which run the migration
    # handler in their run loop — NVSHMEM one-sided put_signal /
    # signal_wait on a dedicated stream, no torch process group, no
    # host-side pause.
    _migration_policies: list = PrivateAttr(default_factory=list)
    _migration_tasks: List[asyncio.Task] = PrivateAttr(default_factory=list)
    _scheduler_stop: bool = PrivateAttr(default=False)
    # Per-shard ``InferenceClient`` keyed by shard_index. Each shard's
    # ``rank_offset`` rank holds a client; the decider rank of any
    # registered :class:`MigrationPolicy` uses its client to post
    # ``MIGRATE_BATCH`` and ``UPDATE_REQUEST_RANKS_BATCH`` to the coord
    # without going through rank 0. Empty on ranks that aren't a
    # ``rank_offset`` for any shard.
    _shard_clients: dict = PrivateAttr(default_factory=dict)
    # Migration metadata snapshot. Maps
    # ``(src_shard_index, dst_shard_index)`` → dict with the src/dst
    # layouts and head offsets the engine handler needs. Pre-computing
    # avoids paying the layout-build cost on every migration.
    _migration_meta: dict = PrivateAttr(default_factory=dict)
    # Round-robin counter used by the migration scheduler to spread
    # batches across the destination shard's DP replicas (keyed by
    # ``(src_shard, dst_shard)``). Without this every batch lands on
    # dp_rank 0, which cuts effective dst capacity in half on
    # ``tp=K + tp=1,dp=N`` configs.
    _disagg_dst_dp_counter: dict = PrivateAttr(default_factory=dict)
    # When set, base_generate stamps every HTTP request with
    # ``disagg_pair=[src, dst]`` so the coord routes to the prefill shard
    # and the registered :class:`FirstTokenDisaggPolicy` migrates after
    # first token. Populated by the
    # ``--rl-auto-disagg-src-shard`` / ``--rl-auto-disagg-dst-shard``
    # CLI args in :meth:`launch`.
    _disagg_rollout_pair: Optional[tuple] = PrivateAttr(default=None)
    # When set, base_generate stamps requests with ``tail_cut=[dst, n]``
    # so the registered :class:`TailCutPolicy` migrates the request a
    # second time once it has produced ``n`` tokens — typically pulling
    # long-tail rollouts off the throughput decode shard onto a smaller,
    # latency-optimized decode shard. ``(dst_shard_index, min_tokens)``
    # tuple.
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
                ``http_url`` will be populated in place.
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

        # --- Start per-shard text-gen server on the shard's rank_offset ---
        # The HTTP server's subprocesses create their own InferenceClient
        # against ``unified_coord_addr``, so no outer client is needed here
        # for HTTP. The lifecycle client (pause/resume/...) is created
        # separately on rank 0 below, independent of shard membership.
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
                disagg_length_threshold=getattr(
                    args, "rl_disagg_length_threshold", None
                ),
            )

        # --- Exchange HTTP ports so every rank knows every shard's URL ----
        world_size = dist.get_world_size()
        all_http_ports: List[int] = [-1] * world_size
        dist.all_gather_object(all_http_ports, my_http_port)
        for s in shards:
            port = all_http_ports[s.rank_offset]
            s.http_url = f"http://{host}:{port}" if port >= 0 else None

        # --- Build per-shard InferenceClients on each shard's rank_offset ---
        # Each shard's first rank gets a client. Rank 0's drives the
        # lifecycle (pause/resume/stop/shutdown via _drive_all). The
        # decider rank of any shard with a registered :class:`MigrationPolicy`
        # uses its client to post ``MIGRATE_BATCH`` directly to
        # the coord without going through rank 0. Clients on rank_offsets
        # that never become deciders are vestigial but harmless — the
        # coord identifies clients by ZMQ identity at CONNECT time and
        # idle ones consume nothing.
        shard_clients: dict = {}
        for s in shards:
            if rank == s.rank_offset:
                c = InferenceClient(inference_coordinator_address=unified_coord_addr)
                c.start()
                shard_clients[s.index] = c
        # rank 0 (which is shard 0's rank_offset by construction) holds
        # the lifecycle client for pause / resume / shutdown.
        lifecycle_client = shard_clients[0] if rank == 0 and 0 in shard_clients else None

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
        instance._lifecycle_client = lifecycle_client
        instance._shard_clients = shard_clients
        instance._openai_clients = openai_clients
        instance._rl_kv_cache_management_mode = KVCacheManagementMode(
            args.rl_kv_cache_management_mode
        )
        instance._route_lock = asyncio.Lock()
        instance._recent_latencies = [deque(maxlen=instance._latency_window) for _ in shards]
        instance._in_flight = [0 for _ in shards]

        # Opt-in end-to-end disagg for every rollout. When
        # --rl-auto-disagg-{src,dst}-shard are set, the scheduler
        # watches src_shard and stamps every base_generate with
        # ``disagg_pair=[src, dst]`` — each HTTP request lands on src,
        # reaches first-token, then is migrated to dst for decode.
        src_idx_arg = getattr(args, "rl_auto_disagg_src_shard", None)
        dst_idx_arg = getattr(args, "rl_auto_disagg_dst_shard", None)
        if src_idx_arg is not None and dst_idx_arg is not None and len(shards) >= 2:
            assert src_idx_arg != dst_idx_arg, (
                "--rl-auto-disagg-src-shard and --rl-auto-disagg-dst-shard "
                "must differ"
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[auto-disagg] enabling scheduler: src=%d dst=%d — every "
                "base_generate call will stamp disagg_pair=[%d, %d]",
                src_idx_arg,
                dst_idx_arg,
                src_idx_arg,
                dst_idx_arg,
            )
            from .migration_policy import FirstTokenDisaggPolicy

            instance._disagg_rollout_pair = (src_idx_arg, dst_idx_arg)
            instance._register_migration_pair(src_idx_arg, dst_idx_arg)
            instance.register_migration_policy(
                FirstTokenDisaggPolicy(
                    src_shard_index=src_idx_arg, dst_shard_index=dst_idx_arg
                )
            )

        # Two-stage tail-cut. When --rl-tail-cut-{dst-shard,min-tokens}
        # are set, every rollout is also stamped with
        # ``tail_cut=[dst, n]`` and a second scheduler watches the
        # throughput-decode shard — once a request there has produced
        # ``n`` tokens, the second scheduler migrates it to the
        # latency-optimized shard.
        tail_dst_arg = getattr(args, "rl_tail_cut_dst_shard", None)
        tail_min_arg = getattr(args, "rl_tail_cut_min_tokens", None)
        if (
            tail_dst_arg is not None
            and tail_min_arg is not None
            and dst_idx_arg is not None
            and len(shards) >= 3
        ):
            assert tail_dst_arg != dst_idx_arg, (
                "--rl-tail-cut-dst-shard must differ from "
                "--rl-auto-disagg-dst-shard"
            )
            log_single_rank(
                logger,
                logging.INFO,
                "[tail-cut] enabling: src=%d dst=%d min_tokens=%d",
                dst_idx_arg,
                tail_dst_arg,
                tail_min_arg,
            )
            from .migration_policy import TailCutPolicy

            instance._tail_cut_rollout_config = (tail_dst_arg, tail_min_arg)
            instance._register_migration_pair(dst_idx_arg, tail_dst_arg)
            instance.register_migration_policy(
                TailCutPolicy(
                    src_shard_index=dst_idx_arg,
                    dst_shard_index=tail_dst_arg,
                    min_tokens=int(tail_min_arg),
                )
            )

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
        # registered FirstTokenDisaggPolicy migrates it at first token. The HTTP
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

    def _drive_all(self, fn_name: str, *args, **kwargs) -> None:
        """Call ``fn_name(*args, **kwargs)`` on the unified lifecycle
        client (rank 0 only — ``_lifecycle_client`` is ``None``
        elsewhere). The unified coord then broadcasts to every engine,
        so a single call reaches all shards.
        """
        if self._lifecycle_client is not None:
            getattr(self._lifecycle_client, fn_name)(*args, **kwargs)

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
        # ranks), then respawn the migration-policy scheduler tasks so
        # they tick only inside this synchronized inference window.
        # Idle ranks still need ``_resume_scheduler`` so their broadcast
        # partner is alive — hence the finally block.
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
        # Cancel migration-policy scheduler tasks before driving engines
        # to PAUSED so an in-flight policy tick doesn't try to schedule
        # a migration during the outer whole-world suspend. Scoped
        # suspends (``shard_indices`` set) are reserved for downstream
        # callers that lifecycle individual shards independently; they
        # leave the scheduler tasks alone since other shards are still
        # serving.
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
        # Stop migration-policy tasks before engines so an in-flight
        # policy tick doesn't race with teardown.
        await self.disable_migration_policies()

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
        if dist.get_rank() == 0 and self._lifecycle_client is not None:
            self._lifecycle_client.shutdown_coordinator()
            self._lifecycle_client.stop()

        # The text-gen server lives on the shard's first rank; only that rank
        # needs to stop it.
        my_shard = self._shards[self._my_shard_index] if self._my_shard_index is not None else None
        if my_shard is not None and dist.get_rank() == my_shard.rank_offset:
            from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
                stop_text_gen_server,
            )

            stop_text_gen_server()

    # ---- Request migration ---------------------------------------------

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
        # layer offset is computed inside the migration handler from
        # each rank's position.
        head_offset = 0
        if participates:
            tp_rank = (rank - shard.rank_offset) % tp
            head_offset = tp_rank * (num_kv_heads_total // tp)
        return layout, head_offset

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
    # the src shard + the chosen dst dp_rank; each engine pops the
    # signal off its ``_pending_signals`` queue and invokes the
    # registered migration handler (:meth:`_on_migrate_batch_signal`),
    # which runs the NVSHMEM ``put_signal`` / ``signal_wait`` transport
    # on the migration stream — no NCCL collective, no engine pause.
    #
    # No cross-rank broadcast is required to coordinate the migration
    # decision — only the decider decides, and the trigger flows over
    # the coord's ZMQ control channel (same path as PAUSE / SUSPEND).

    def register_migration_policy(self, policy) -> None:
        """Register a :class:`MigrationPolicy` instance.

        Collective: every rank must call (so each rank's instance has
        the same ``_migration_policies`` list for symmetric pause /
        resume, even though only the decider rank for each policy
        actually runs the loop).

        The policy's asyncio task is spawned lazily by the next
        whole-world :meth:`resume`. We don't spawn at register time
        because ``_resume_scheduler`` needs an actively-running
        asyncio loop, and registration is typically called from
        :meth:`launch` while the launch coroutine is still running
        (loop is alive there, but tasks should only run during
        actual inference windows so we don't accumulate stale work
        between launch and the first resume).

        Multiple policies can target the same src shard; they each
        get their own asyncio task and their own ``migrated_ids``
        memo. Policies are independent — first to fire on a given
        request wins, and the others see the request gone from
        ``engine.requests`` on the next tick.
        """
        assert 0 <= policy.src_shard_index < len(self._shards)
        assert 0 <= policy.dst_shard_index < len(self._shards)
        assert policy.src_shard_index != policy.dst_shard_index
        assert policy.max_batch_size >= 1
        self._migration_policies.append(policy)

    def _register_migration_pair(
        self, src_shard_index: int, dst_shard_index: int
    ) -> None:
        """Pre-compute migration metadata for ``(src, dst)`` and wire
        the engine's MIGRATE_BATCH handler. Cheap and rank-local — the
        NVSHMEM migration path uses one-sided put_signal / signal_wait,
        no torch process group.
        """
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

        meta = {
            "src_shard": src_shard,
            "dst_shard": dst_shard,
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

        - **Src:** gather KV slices into staging slots and issue
          ``put_signal`` on the migration stream; push a poll that
          frees src blocks once the migration event fires.
        - **Dst:** ``inject_request`` to allocate dst KV blocks +
          register the request, then ``signal_wait`` + scatter on the
          migration stream and ``wait_stream`` the engine's compute
          stream onto it.

        No ``barrier_all``, no cross-shard broadcast, no engine pause.
        Exceptions propagate to the engine's run-loop catch.
        """
        meta = self._migration_meta.get((src_shard_index, dst_shard_index))
        if meta is None:
            logger.error(
                "[migration] no metadata for (%d, %d) — "
                "registered pairs: %s",
                src_shard_index,
                dst_shard_index,
                list(self._migration_meta.keys()),
            )
            return
        if not (meta["in_src"] or meta["in_dst"]):
            return  # bystander; coord shouldn't forward to us anyway

        from megatron.core.inference import nvshmem_migration as _nv
        from megatron.core.inference.engines.request_migration import (
            _gather_kv_slice,
            _scatter_kv_slice,
            build_kv_migration_plan,
            deserialize_bundle,
        )

        src_shard = meta["src_shard"]
        dst_shard = meta["dst_shard"]
        src_layout = meta["src_layout"]
        dst_layout = meta["dst_layout"]
        src_root = src_shard.ranks()[0]
        # Within a shard, ranks are laid out TP-major:
        # ``rank_offset + dp * tp_size + tp``. ``dst_root`` is TP-0 of
        # the target dp replica; coord forwards MIGRATE_BATCH only to
        # that one dp rank and the migration plan must address the
        # same rank for ``put_signal`` and scatter.
        dst_root = dst_shard.rank_offset + dst_dp_rank * dst_layout.tp_size

        def _src_rank_of(tp: int, pp: int) -> int:
            return src_root + pp * src_layout.tp_size + tp

        def _dst_rank_of(tp: int, pp: int) -> int:
            return dst_root + pp * dst_layout.tp_size + tp

        # Deserialize bundles inline (small dicts). Layouts aren't on the
        # wire — restamp from the local meta so ``build_kv_migration_plan``
        # can read them off the bundle.
        parsed_bundles = [deserialize_bundle(b) for b in bundles]
        for b in parsed_bundles:
            b.src_layout = src_layout
            b.dst_layout = dst_layout
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

        # Detach src synchronously so async_step can't emit a stale
        # ENGINE_REPLY for an already-migrated request. ``keep_blocks=True``
        # keeps the KV blocks alive for the gather kernels we'll enqueue
        # below; the poll callback releases them once the event fires.
        try:
            src_block_ids_to_free: List[torch.Tensor] = []
            injected: List[Tuple[object, List[int]]] = []
            if meta["in_src"]:
                for req_id in request_ids:
                    if req_id in engine.requests:
                        src_block_ids_to_free.append(
                            engine.detach_request(req_id, keep_blocks=True)
                        )

            # Dst injects each bundle (allocates dst blocks, registers
            # request) and accumulates the (bundle, dst_blocks) pairs.
            if meta["in_dst"]:
                for bundle in parsed_bundles:
                    dst_blocks = engine.inject_request(bundle).tolist()
                    injected.append((bundle, dst_blocks))

            # Build per-bundle KV migration plans. Both src and dst walk
            # the bundles in identical order so slot/flag assignment
            # stays consistent. On src ``dst_block_ids`` is unknown and
            # left to the plan helper to fill.
            all_ops_by_bundle: List[List] = []
            for i, bundle in enumerate(parsed_bundles):
                dst_blocks = injected[i][1] if meta["in_dst"] else None
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
            # Flag slots are derived deterministically from
            # ``(request_id, op_index_within_bundle)``: every src and
            # dst rank looking at the same op picks the same flag,
            # so we don't need a per-PE counter (which round-robin
            # routing would desync, since not every dst replica
            # participates in every batch).
            MAX_OPS_PER_REQ = 32
            for bundle_idx, ops in enumerate(all_ops_by_bundle):
                req_id = parsed_bundles[bundle_idx].request_id
                for op_idx_in_bundle, op in enumerate(ops):
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
                    flag_key = req_id * MAX_OPS_PER_REQ + op_idx_in_bundle
                    op_meta.append(
                        {
                            "op": op,
                            "shape": shape,
                            "nbytes": nbytes,
                            "slot": arena.take(nbytes),
                            "flag": _nv.flag_slot_for(flag_key),
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

            # Once the migration event fires: reset the flags and (on src)
            # return the kept blocks to the allocator.
            flags_used = [e["flag"] for e in op_meta]

            def _poll() -> bool:
                if not done_event.query():
                    return False
                for slot in flags_used:
                    _nv.reset_flag(slot)
                if meta["in_src"] and src_block_ids_to_free:
                    allocator = engine.context.kv_block_allocator
                    for blocks in src_block_ids_to_free:
                        if blocks.numel() > 0:
                            allocator.release_memory_blocks(blocks)
                return True

            engine.push_pending_migration(_poll)
        except BaseException:
            # Catch ``BaseException`` (not ``Exception``) so
            # ``KeyboardInterrupt`` mid-migration still triggers the
            # block-release path before propagating — otherwise an
            # interrupt during the gather/put leaks GPU blocks.
            # Roll back the partially-applied migration so the engine doesn't
            # leak GPU blocks. ``release_memory_blocks`` returns the kept
            # blocks to the allocator; ``detach_request(keep_blocks=False)``
            # does the same for any dst requests that were injected before
            # the failure point.
            if src_block_ids_to_free:
                allocator = engine.context.kv_block_allocator
                for blocks in src_block_ids_to_free:
                    if blocks.numel() > 0:
                        allocator.release_memory_blocks(blocks)
            for bundle, _ in injected:
                if bundle.request_id in engine.requests:
                    engine.detach_request(
                        bundle.request_id, keep_blocks=False
                    )
            raise

    async def _pause_scheduler(self) -> None:
        """Stop all migration-policy asyncio tasks. Policies stay in
        ``_migration_policies`` so :meth:`_resume_scheduler` can respawn
        them at the next whole-world :meth:`resume`."""
        if not self._migration_tasks:
            return
        self._scheduler_stop = True
        for t in self._migration_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[migration] task raised on pause: %s", e)
        self._migration_tasks.clear()
        self._scheduler_stop = False

    def _resume_scheduler(self) -> None:
        """Spawn one asyncio task per registered policy. No-op on ranks
        that aren't a decider for any policy."""
        if self._migration_tasks:
            return
        if not self._migration_policies:
            return
        self._scheduler_stop = False
        loop = asyncio.get_running_loop()
        rank = dist.get_rank()
        for policy in self._migration_policies:
            decider_rank = self._shards[policy.src_shard_index].rank_offset
            if rank != decider_rank:
                continue  # non-decider ranks don't participate
            self._migration_tasks.append(
                loop.create_task(
                    self._run_policy_loop(policy),
                    name=(
                        f"megatron_local_multi.{type(policy).__name__}"
                        f"@s{policy.src_shard_index}->s{policy.dst_shard_index}"
                    ),
                )
            )

    async def disable_migration_policies(self) -> None:
        """Stop every migration-policy task and forget registered
        policies. Idempotent. Distinct from :meth:`_pause_scheduler`
        in that ``_migration_policies`` is cleared, so a subsequent
        :meth:`_resume_scheduler` won't respawn anything."""
        self._migration_policies.clear()
        await self._pause_scheduler()

    @staticmethod
    def _live_request(entry):
        """Return the live :class:`DynamicInferenceRequest` from a
        :class:`RequestEntry`, or ``None`` if the entry has no record.

        ``engine.requests`` maps id → RequestEntry; the live request
        is the last item in ``entry.record`` (records grow on
        cross-shard migrations to track ancestor states).
        """
        record = getattr(entry, "record", None)
        if not record:
            return None
        return record[-1]

    async def _run_policy_loop(self, policy) -> None:
        """Generic per-tick decision loop for one :class:`MigrationPolicy`.
        **Runs only on the decider rank**
        (``self._shards[policy.src_shard_index].rank_offset``).

        Each tick: ask the policy which requests on the watched
        engine are eligible, batch them up to ``policy.max_batch_size``,
        and submit one ``MIGRATE_BATCH`` to the coord. Nothing on this
        loop blocks on the actual transport — the migration is
        asynchronous from the scheduler's perspective.
        """
        src_shard_index = policy.src_shard_index
        dst_shard_index = policy.dst_shard_index
        decider_rank = self._shards[src_shard_index].rank_offset
        rank = dist.get_rank()
        assert rank == decider_rank, (
            f"_run_policy_loop should only run on decider rank "
            f"{decider_rank} (got rank {rank})"
        )
        if self._my_engine is None:
            return
        client = self._shard_clients.get(src_shard_index)
        if client is None:
            logger.error(
                "[migration:%s] no InferenceClient for shard %d on rank %d "
                "— scheduler cannot post migrations",
                type(policy).__name__,
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
            await asyncio.sleep(policy.poll_interval_s)
            if self._scheduler_stop:
                return
            if not self._my_engine._state_events[EngineState.RUNNING].is_set():
                continue

            # Snapshot active requests; tolerate races with the engine
            # task by retrying on RuntimeError (dict changed size
            # during iteration).
            eligible: List[int] = []
            try:
                active_ids = set(self._my_engine.requests.keys())
                migrated_ids &= active_ids
                for req_id, entry in list(self._my_engine.requests.items()):
                    if req_id in migrated_ids:
                        continue
                    req = self._live_request(entry)
                    if req is None:
                        continue
                    if not policy.is_eligible(req):
                        continue
                    eligible.append(int(req_id))
                    if len(eligible) >= policy.max_batch_size:
                        break
            except RuntimeError:
                continue

            if not eligible:
                continue

            # Snapshot each request on the local (decider) engine to
            # build the bundle and ship it inline through the coord —
            # so the dst engine doesn't need a cross-shard broadcast
            # to learn the request's metadata. This is what makes the
            # whole migration handler non-blocking on both sides.
            from megatron.core.inference.engines.request_migration import serialize_bundle

            try:
                # Layouts are not stamped here: they're invariant per
                # (src, dst) pair and the receiving handler restamps
                # from its own ``_migration_meta`` after deserialize.
                bundles = []
                for req_id in eligible:
                    bundle, _ = self._my_engine.snapshot_request(req_id)
                    bundles.append(serialize_bundle(bundle))
                # Round-robin one batch at a time across the dst
                # shard's DP replicas so we don't stack every migration
                # on dp_rank 0.
                dst_shard = self._shards[dst_shard_index]
                dst_dp_size = max(int(dst_shard.spec.get("dp", 1)), 1)
                counter_key = (src_shard_index, dst_shard_index)
                counter = self._disagg_dst_dp_counter.get(counter_key, 0)
                dst_dp_rank = counter % dst_dp_size
                self._disagg_dst_dp_counter[counter_key] = counter + 1
                client.migrate_request_batch(
                    eligible,
                    src_shard_index,
                    dst_shard_index,
                    bundles=bundles,
                    dst_dp_rank_within_shard=dst_dp_rank,
                )
                client.update_request_ranks_batch(
                    request_ids=eligible,
                    new_shard_index=dst_shard_index,
                    new_dp_rank_within_shard=dst_dp_rank,
                )
            except Exception as e:
                logger.error(
                    "[migration:%s] failed to submit migration of %d "
                    "requests (%d → %d): %s",
                    type(policy).__name__,
                    len(eligible),
                    src_shard_index,
                    dst_shard_index,
                    e,
                )
            finally:
                # Always mark the ids migrated — including on failure —
                # so a persistent error doesn't retry-storm the same
                # batch every poll tick. The set is pruned each tick to
                # ids still resident on this engine, so successful
                # migrations drop out naturally on the next pass.
                migrated_ids.update(eligible)

