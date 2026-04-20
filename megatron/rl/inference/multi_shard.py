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
        from megatron.core.inference.engines.request_migration import (
            KVLayout,
            migrate_request_cross_shard,
        )
        from megatron.core.inference.shards import build_cross_shard_group

        assert src_shard_index != dst_shard_index, (
            f"src and dst shard indices must differ (got {src_shard_index})"
        )
        assert 0 <= src_shard_index < len(self._shards)
        assert 0 <= dst_shard_index < len(self._shards)

        src_shard = self._shards[src_shard_index]
        dst_shard = self._shards[dst_shard_index]
        rank = dist.get_rank()
        in_src = src_shard.owns_rank(rank)
        in_dst = dst_shard.owns_rank(rank)

        # Cross-shard group spanning src + dst. The cache in
        # ``build_cross_shard_group`` means subsequent migrations
        # between the same pair reuse one group.
        cross_shard_group = build_cross_shard_group(
            self._shards, [src_shard_index, dst_shard_index]
        )

        if not (in_src or in_dst):
            # Bystander rank — nothing to orchestrate, but the
            # cross_shard_group construction above was world-collective.
            return None

        # Build KV layouts from the shard specs. MLA / head-dim config
        # comes from the engine's model/context.
        my_engine = self._my_engine
        assert my_engine is not None, (
            f"rank {rank} is in shard but has no engine — multi_shard.launch "
            "wasn't run"
        )
        model_config = my_engine.controller.inference_wrapped_model.model.config
        ctx = my_engine.context
        num_kv_heads_total = (
            model_config.num_query_groups or model_config.num_attention_heads
        )
        head_dim = ctx.hidden_size_per_attention_head
        layout_kwargs = dict(
            pp_size=1,
            num_layers_total=model_config.num_layers,
            num_kv_heads_total=num_kv_heads_total,
            head_dim=head_dim,
            block_size_tokens=ctx.block_size_tokens,
            is_mla=getattr(ctx, "is_mla", False),
            kv_reduced_dim=getattr(ctx, "kv_reduced_dim", None),
        )
        src_layout = KVLayout(tp_size=src_shard.spec["tp"], **layout_kwargs)
        dst_layout = KVLayout(tp_size=dst_shard.spec["tp"], **layout_kwargs)

        # Head offsets for this rank's shard-local buffer slice.
        src_heads_per_tp = num_kv_heads_total // src_shard.spec["tp"]
        dst_heads_per_tp = num_kv_heads_total // dst_shard.spec["tp"]
        my_src_head_offset = (
            (rank - src_shard.rank_offset) * src_heads_per_tp if in_src else 0
        )
        my_dst_head_offset = (
            (rank - dst_shard.rank_offset) * dst_heads_per_tp if in_dst else 0
        )

        # Quiesce both engines so the migration collective can run at a
        # safe step boundary. suspend() is idempotent.
        await self.suspend()
        try:
            migrated_id = migrate_request_cross_shard(
                role="src" if in_src else "dst",
                engine=my_engine,
                request_id_src=request_id if in_src else None,
                src_layout=src_layout,
                dst_layout=dst_layout,
                src_ranks=src_shard.ranks(),
                dst_ranks=dst_shard.ranks(),
                cross_shard_group=cross_shard_group,
                my_src_head_offset=my_src_head_offset,
                my_dst_head_offset=my_dst_head_offset,
                request_id_dst=request_id_dst,
            )
        finally:
            await self.resume()

        return migrated_id

    # ---- Disaggregated inference ---------------------------------------

    async def _submit_via_coord(
        self,
        prompt: str,
        sampling_params: "SamplingParams",
        shard_index: int,
    ) -> int:
        """Submit a request to one shard's coordinator and broadcast the
        engine-visible request id to every rank.

        Only rank 0 (which holds the lifecycle clients) actually sends;
        every rank receives the id so subsequent collective operations
        (e.g. migration) have a consistent target.
        """
        tokenizer = get_tokenizer()
        rank = dist.get_rank()
        submitted_request_id = -1
        if rank == 0:
            client = self._lifecycle_clients[shard_index]
            assert client is not None, (
                "submission requires rank 0 to hold a lifecycle client "
                f"for shard {shard_index}"
            )
            prompt_tokens = tokenizer.tokenize(prompt)
            submitted_request_id = client.next_request_id
            client.add_request(prompt_tokens, sampling_params)

        id_tensor = torch.tensor(
            [submitted_request_id if rank == 0 else 0],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.broadcast(id_tensor, src=0)
        return int(id_tensor.item())

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
    ) -> int:
        """Run prefill on one shard and decode on another.

        Collective: every rank must call. The flow:

            1. rank 0 submits the prompt to the prefill shard's
               coordinator (same ZMQ path HTTP would use).
            2. All ranks poll the prefill shard's engine until the
               request has produced at least one generated token — i.e.
               prefill has completed and written KV for every prompt
               position, and the first decode step has sampled the
               initial output token.
            3. The request is migrated to the decode shard using the
               same KV-transport collective that powers mid-flight
               migration. The decode shard continues generation from
               the migrated KV.

        Returns the request id (identical on both shards — no id
        renaming by default).

        Caveats (same as migrate_request):
          - The prefill shard's coord loses visibility of the request
            after migration; any originating HTTP future won't resolve
            on it. Full HTTP-transparent disaggregation needs the
            coord-mediated reply-forwarding work tracked alongside the
            migration primitive.
          - Call this *after* engines are RUNNING (post-``resume``);
            the prefill shard must be able to pick up the submit.
        """
        assert prefill_shard_index != decode_shard_index, (
            "prefill and decode shards must differ for disaggregation"
        )

        request_id = await self._submit_via_coord(
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
        return request_id

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

        request_id = await self.submit_disaggregated_request(
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

        await asyncio.sleep(post_decode_sleep_s)

        if rank == decode_shard.rank_offset:
            engine = self._my_engine
            if request_id in engine.requests:
                n_final = len(engine.get_request(request_id).generated_tokens)
                logger.info(
                    "[disagg-smoke] after %.1fs of decode on shard %d: "
                    "request %d has %d generated tokens",
                    post_decode_sleep_s,
                    decode_shard_index,
                    request_id,
                    n_final,
                )

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
        submitted_request_id = await self._submit_via_coord(
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
