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
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

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


@dataclass
class MigrationMeta:
    """Per-(src_shard, dst_shard) snapshot of everything the migration
    handler needs to schedule transfers. Built once when a migration
    pair is registered (or its first policy lands); read on every
    ``Headers.MIGRATE_BATCH`` for that pair.

    Layout / head-offset fields are populated only on ranks that
    participate (``in_src or in_dst``). Mamba-layout fields are
    populated only on hybrid models.
    """

    src_shard: InferenceShard
    dst_shard: InferenceShard
    in_src: bool
    in_dst: bool
    src_layout: Optional[object] = None
    dst_layout: Optional[object] = None
    my_src_head_offset: int = 0
    my_dst_head_offset: int = 0
    src_mamba_layout: Optional[object] = None
    dst_mamba_layout: Optional[object] = None


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


# ---- Migration handler helpers ---------------------------------------------
#
# These factor metadata construction and the poll/rollback closures out of
# :meth:`MegatronLocalMulti._on_migrate_batch_signal` so the handler reads as
# orchestration rather than ~480 lines of nested branches.


def _build_kv_op_meta(
    arena,
    flag_arena,
    all_ops_by_bundle,
    layout,
    memory_buffer,
) -> List[dict]:
    """Walk every (bundle, op) deterministically and assign a staging
    slot + flag index to each KV transfer.

    Both src and dst call this with identical inputs, so the
    ``(slot, flag)`` choice for each op matches across PEs without any
    cross-rank coordination. Flag identity is per-(src_pe, dst_pe)
    lane (via :class:`FlagArena`) so cross-pair acks can't be misread,
    and the ack handshake inside ``put_slot_with_signal`` serializes
    cross-migration reuse of the same flag.
    """

    op_meta: List[dict] = []
    elem_size = memory_buffer.element_size()
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
                    "flag": flag_arena.take(op.src_rank, op.dst_rank),
                }
            )
    return op_meta


def _build_mamba_op_meta(
    arena,
    flag_arena,
    parsed_bundles,
    engine_ctx,
    src_mamba_layout,
    dst_mamba_layout,
    src_global_rank_of,
    dst_global_rank_of,
) -> List[dict]:
    """Per-request mamba transfer metadata.

    Plan-driven — calls :func:`build_mamba_migration_plan` to enumerate
    one transfer per ``(src_pp × dst_pp × src_tp × dst_tp)`` overlap.
    Each op carries every block kind (``conv_x`` / ``conv_B`` /
    ``conv_C`` / ``ssm``) with non-empty overlap, packed into a single
    NVSHMEM staging slot at distinct byte offsets and shipped under one
    flag. This keeps slot-pool pressure at ``num_overlap_pairs × batch``
    rather than ``4 × num_overlap_pairs × batch``.

    Shares the migration's :class:`FlagArena` with KV ops, so flag
    indices keep counting up in the same per-(src_pe, dst_pe) lane —
    no KV-up / Mamba-down hand-partitioning needed (lane partitioning
    + ack handshake handles it).

    Returns a list of dicts, one per (bundle × mamba op); each
    carries a ``blocks`` sub-list with per-kind ``shape`` / ``dtype``
    / ``byte_offset`` / ``nbytes`` so gather / scatter can address
    its slice of the packed slot.
    """
    from megatron.core.inference.engines.request_migration import (
        build_mamba_migration_plan,
    )

    conv_states = engine_ctx.mamba_conv_states  # [layers, slots, conv_dim, d_conv]
    ssm_states = engine_ctx.mamba_ssm_states  # [layers, slots, nheads, headdim, d_state]
    headdim = src_mamba_layout.headdim
    d_state = src_mamba_layout.d_state
    d_conv = src_mamba_layout.d_conv

    # Per-kind metadata: (buffer-dtype, element-bytes, per-row-shape).
    # The block range's span × ``len(mamba_layer_indices)`` gives the
    # row count transferred; per-row shape is fixed by kind (conv
    # blocks → (d_conv,); ssm → (headdim, d_state)).
    conv_dtype = conv_states.dtype
    ssm_dtype = ssm_states.dtype
    conv_elem = torch.empty((), dtype=conv_dtype).element_size()
    ssm_elem = torch.empty((), dtype=ssm_dtype).element_size()
    kind_spec = {
        "conv_x": (conv_dtype, conv_elem, (d_conv,)),
        "conv_B": (conv_dtype, conv_elem, (d_conv,)),
        "conv_C": (conv_dtype, conv_elem, (d_conv,)),
        "ssm": (ssm_dtype, ssm_elem, (headdim, d_state)),
    }

    mamba_meta: List[dict] = []
    for bundle_idx, bundle in enumerate(parsed_bundles):
        ops = build_mamba_migration_plan(
            bundle,
            src_mamba_layout,
            dst_mamba_layout,
            src_global_rank_of,
            dst_global_rank_of,
        )
        for op in ops:
            num_op_layers = len(op.mamba_layer_indices)
            block_meta: List[dict] = []
            byte_offset = 0
            for block in op.blocks:
                dtype, elem_size, per_row_shape = kind_spec[block.kind]
                span = block.src_local_range[1] - block.src_local_range[0]
                # Per-block payload shape: (layers, span, *per_row).
                shape = (num_op_layers, span, *per_row_shape)
                nelems = 1
                for d in shape:
                    nelems *= d
                block_nbytes = nelems * elem_size
                block_meta.append(
                    {
                        "kind": block.kind,
                        "src_local_range": block.src_local_range,
                        "dst_local_range": block.dst_local_range,
                        "shape": shape,
                        "dtype": dtype,
                        "byte_offset": byte_offset,
                        "nbytes": block_nbytes,
                    }
                )
                byte_offset += block_nbytes
            total_nbytes = byte_offset
            mamba_meta.append(
                {
                    "bundle_idx": bundle_idx,
                    "op": op,
                    "src_rank": op.src_rank,
                    "dst_rank": op.dst_rank,
                    "slot": arena.take(total_nbytes),
                    "flag": flag_arena.take(op.src_rank, op.dst_rank),
                    "nbytes": total_nbytes,
                    "blocks": block_meta,
                }
            )
    return mamba_meta


def _execute_kv_src_side(
    rank: int,
    op_meta: List[dict],
    memory_buffer: torch.Tensor,
    dtype: torch.dtype,
    layout,
    my_head_offset: int,
    my_pp_layer_offset: int,
    stream: torch.cuda.Stream,
) -> None:
    """Src side of KV migration: for each KV op where ``rank`` is the src
    rank, gather the local KV slice into a staging slot and issue a
    ``put_signal`` to the dst rank. Stream-ordered."""
    from megatron.core.inference import migration_transport as _nv
    from megatron.core.inference.engines.request_migration import _gather_kv_slice

    for entry in op_meta:
        op = entry["op"]
        if rank != op.src_rank:
            continue
        src_blocks = torch.as_tensor(
            op.src_block_ids, device=memory_buffer.device, dtype=torch.long
        )
        local_head_range = (
            op.head_range[0] - my_head_offset,
            op.head_range[1] - my_head_offset,
        )
        local_layer_range = (
            op.layer_range[0] - my_pp_layer_offset,
            op.layer_range[1] - my_pp_layer_offset,
        )
        send_tensor = _gather_kv_slice(
            memory_buffer,
            local_layer_range,
            src_blocks,
            local_head_range,
            layout.is_mla,
        )
        slot = _nv.staging_slot(entry["slot"])
        slot[: entry["nbytes"]].view(dtype).reshape(entry["shape"]).copy_(
            send_tensor, non_blocking=True
        )
        _nv.put_slot_with_signal(
            entry["slot"],
            entry["flag"],
            op.dst_rank,
            nbytes=entry["nbytes"],
            stream=stream,
        )


def _execute_kv_dst_side(
    rank: int,
    op_meta: List[dict],
    memory_buffer: torch.Tensor,
    dtype: torch.dtype,
    layout,
    my_head_offset: int,
    my_pp_layer_offset: int,
    stream: torch.cuda.Stream,
) -> None:
    """Dst side of KV migration: for each KV op where ``rank`` is the dst
    rank, ``signal_wait`` and scatter the received slot into the local KV
    buffer. Stream-ordered."""
    from megatron.core.inference import migration_transport as _nv
    from megatron.core.inference.engines.request_migration import _scatter_kv_slice

    for entry in op_meta:
        op = entry["op"]
        if rank != op.dst_rank:
            continue
        _nv.wait_slot_signal(entry["flag"], expected_value=1, stream=stream)
        dst_blocks = torch.as_tensor(
            op.dst_block_ids, device=memory_buffer.device, dtype=torch.long
        )
        local_head_range = (
            op.head_range[0] - my_head_offset,
            op.head_range[1] - my_head_offset,
        )
        local_layer_range = (
            op.layer_range[0] - my_pp_layer_offset,
            op.layer_range[1] - my_pp_layer_offset,
        )
        slot = _nv.staging_slot(entry["slot"])
        recv_view = slot[: entry["nbytes"]].view(dtype).reshape(entry["shape"])
        _scatter_kv_slice(
            memory_buffer,
            local_layer_range,
            dst_blocks,
            local_head_range,
            layout.is_mla,
            recv_view,
        )
        # Tell src this flag slot is safe to reuse for the next round.
        # Stream-ordered after the scatter so src can't overwrite the
        # slot until scatter has finished reading it.
        _nv.send_ack(entry["flag"], op.src_rank, stream=stream)


def _execute_mamba_src_side(
    rank: int,
    mamba_meta: List[dict],
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    src_mamba_state_indices: List[Optional[int]],
    stream: torch.cuda.Stream,
) -> None:
    """Src side of mamba migration: for each packed-op where ``rank`` is
    the src, gather the conv/SSM block slices into one staging slot at
    distinct byte offsets and ``put_signal`` under one flag.

    Each ``entry`` packs multiple block kinds (``conv_x`` / ``conv_B`` /
    ``conv_C`` / ``ssm``) for one (src_rank, dst_rank) overlap.
    """
    from megatron.core.inference import migration_transport as _nv

    def _buffer_for(kind):
        return conv_states if kind.startswith("conv_") else ssm_states

    for entry in mamba_meta:
        op = entry["op"]
        if rank != op.src_rank:
            continue
        src_state_idx = src_mamba_state_indices[entry["bundle_idx"]]
        if src_state_idx is None:
            continue
        slot = _nv.staging_slot(entry["slot"])
        layer_idx = torch.tensor(
            op.mamba_layer_indices,
            device=conv_states.device,
            dtype=torch.long,
        )
        for b in entry["blocks"]:
            buf = _buffer_for(b["kind"])
            lo, hi = b["src_local_range"]
            send = buf[layer_idx, src_state_idx, lo:hi].contiguous()
            off = b["byte_offset"]
            slot[off : off + b["nbytes"]].view(b["dtype"]).reshape(
                b["shape"]
            ).copy_(send, non_blocking=True)
        _nv.put_slot_with_signal(
            entry["slot"],
            entry["flag"],
            entry["dst_rank"],
            nbytes=entry["nbytes"],
            stream=stream,
        )


def _execute_mamba_dst_side(
    rank: int,
    mamba_meta: List[dict],
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    dst_mamba_state_indices: List[Optional[int]],
    stream: torch.cuda.Stream,
) -> None:
    """Dst side of mamba migration: for each packed-op where ``rank`` is
    the dst, ``signal_wait`` and scatter each conv/SSM block slice from
    its byte offset in the staging slot into the local state buffers."""
    from megatron.core.inference import migration_transport as _nv

    def _buffer_for(kind):
        return conv_states if kind.startswith("conv_") else ssm_states

    for entry in mamba_meta:
        op = entry["op"]
        if rank != op.dst_rank:
            continue
        dst_state_idx = dst_mamba_state_indices[entry["bundle_idx"]]
        if dst_state_idx is None:
            continue
        _nv.wait_slot_signal(entry["flag"], expected_value=1, stream=stream)
        slot = _nv.staging_slot(entry["slot"])
        layer_idx = torch.tensor(
            op.mamba_layer_indices,
            device=conv_states.device,
            dtype=torch.long,
        )
        for b in entry["blocks"]:
            buf = _buffer_for(b["kind"])
            lo, hi = b["dst_local_range"]
            off = b["byte_offset"]
            recv = (
                slot[off : off + b["nbytes"]]
                .view(b["dtype"])
                .reshape(b["shape"])
            )
            buf[layer_idx, dst_state_idx, lo:hi] = recv
        # Tell src this flag slot is safe to reuse for the next round.
        # Stream-ordered after the scatter so src can't overwrite the
        # slot until scatter has finished reading it.
        _nv.send_ack(entry["flag"], op.src_rank, stream=stream)


def _release_src_state(
    meta: "MigrationMeta",
    is_hybrid: bool,
    src_block_ids_to_free: List[torch.Tensor],
    src_mamba_state_indices: List[Optional[int]],
    engine,
) -> None:
    """Return src-side KV blocks + mamba slots to their allocators.

    Used by both the success-path poll callback and the rollback path
    in the migration handler. Rank-local — only acts when this rank
    actually held src state.
    """
    if meta.in_src and src_block_ids_to_free:
        allocator = engine.context.kv_block_allocator
        for blocks in src_block_ids_to_free:
            if blocks.numel() > 0:
                allocator.release_memory_blocks(blocks)
    if meta.in_src and is_hybrid:
        for src_state_idx in src_mamba_state_indices:
            if src_state_idx is not None:
                engine.release_mamba_state_slot(src_state_idx)


def _make_migration_poll_callback(
    done_event: torch.cuda.Event,
    meta: "MigrationMeta",
    is_hybrid: bool,
    src_block_ids_to_free: List[torch.Tensor],
    src_mamba_state_indices: List[Optional[int]],
    injected: List[Tuple["RequestMigrationBundle", torch.Tensor]],
    dst_mamba_state_indices: List[Optional[int]],
    engine,
) -> Callable[[], bool]:
    """Build the poll callback the engine ticks each loop iteration.

    Returns ``True`` once the migration's ``done_event`` has fired,
    at which point:

    - **Src**: kept KV blocks and Mamba state slots go back to the
      allocators.
    - **Dst**: ``default_stream.wait_event(done_event)`` syncs the
      engine compute stream to the migration stream's writes, then
      :meth:`complete_migration` installs each migrated request into
      the active batch. The next call to ``step_modern`` includes it.

    Until the event fires, the migrating request is *in transit* — off
    src's active list (detached at handler entry) and not yet on dst's
    (reservation made but slot not installed). Non-migrating requests
    on both sides continue forward-passing unaffected.
    """

    def _poll() -> bool:
        if not done_event.query():
            return False
        # Src: release blocks + mamba state slots.
        _release_src_state(
            meta, is_hybrid, src_block_ids_to_free, src_mamba_state_indices, engine
        )
        # Dst: sync the default stream to the migration stream's
        # writes, then activate each reserved request. The wait_event
        # here is the cross-stream memory-ordering barrier that makes
        # the scattered KV visible to subsequent forward passes on the
        # default stream.
        if meta.in_dst and injected:
            torch.cuda.default_stream().wait_event(done_event)
            for (bundle, block_ids), mamba_state_idx in zip(
                injected, dst_mamba_state_indices
            ):
                engine.complete_migration(
                    bundle, block_ids, mamba_state_idx=mamba_state_idx
                )
        return True

    return _poll


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
    _migration_policies: List[object] = PrivateAttr(default_factory=list)
    _migration_tasks: List[asyncio.Task] = PrivateAttr(default_factory=list)
    _scheduler_stop: bool = PrivateAttr(default=False)
    # Per-shard ``InferenceClient`` keyed by shard_index. Each shard's
    # ``rank_offset`` rank holds a client; the decider rank of any
    # registered :class:`MigrationPolicy` uses its client to post
    # ``MIGRATE_BATCH`` and ``UPDATE_REQUEST_RANKS_BATCH`` to the coord
    # without going through rank 0. Empty on ranks that aren't a
    # ``rank_offset`` for any shard.
    _shard_clients: Dict[int, InferenceClient] = PrivateAttr(default_factory=dict)
    # Migration metadata snapshot. Maps
    # ``(src_shard_index, dst_shard_index)`` → dict with the src/dst
    # layouts and head offsets the engine handler needs. Pre-computing
    # avoids paying the layout-build cost on every migration.
    _migration_meta: Dict[Tuple[int, int], MigrationMeta] = PrivateAttr(default_factory=dict)
    # Round-robin counter used by the migration scheduler to spread
    # batches across the destination shard's DP replicas (keyed by
    # ``(src_shard, dst_shard)``). Lives on every rank but only the
    # decider rank for each ``(src, dst)`` pair mutates and reads its
    # own keys. Without this every batch lands on dp_rank 0, which cuts
    # effective dst capacity in half on ``tp=K + tp=1,dp=N`` configs.
    _disagg_dst_dp_counter: Dict[Tuple[int, int], int] = PrivateAttr(default_factory=dict)
    # When set, base_generate stamps every HTTP request with
    # ``disagg_pair=[src, dst]`` so the coord routes to the prefill shard
    # and the registered :class:`FirstTokenDisaggPolicy` migrates after
    # first token. Populated by the
    # ``--rl-auto-disagg-src-shard`` / ``--rl-auto-disagg-dst-shard``
    # CLI args in :meth:`launch`.
    _disagg_rollout_pair: Optional[Tuple[int, int]] = PrivateAttr(default=None)
    # When set, base_generate stamps requests with ``tail_cut=[dst, n]``
    # so the registered :class:`TailCutPolicy` migrates the request a
    # second time once it has produced ``n`` tokens — typically pulling
    # long-tail rollouts off the throughput decode shard onto a smaller,
    # latency-optimized decode shard. ``(dst_shard_index, min_tokens)``
    # tuple.
    _tail_cut_rollout_config: Optional[Tuple[int, int]] = PrivateAttr(default=None)
    # Sliding windows of recent per-shard response durations (seconds); used
    # for 1/latency-weighted routing once each shard has some history.
    # _in_flight counts pending requests per shard and biases the weighted
    # pick against shards that are currently saturated, so a fast shard
    # doesn't get piled on before a response lands in _recent_latencies.
    _recent_latencies: List[Deque[float]] = PrivateAttr(default_factory=list)
    _in_flight: List[int] = PrivateAttr(default_factory=list)
    _latency_window: int = PrivateAttr(default=32)
    # Optional :class:`RouteSelector` consulted by ``base_generate`` to
    # pick a per-request layer-kind disagg route. When ``None`` (default)
    # every request walks the stored layout route. Lives on every rank
    # but only the rollout-driver rank exercises it.
    _route_selector: Optional[object] = PrivateAttr(default=None)

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
            from megatron.core.inference import migration_transport as _mig

            _mig.maybe_init_migration_transport()

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
        shard_clients: Dict[int, InferenceClient] = {}
        for s in shards:
            if rank == s.rank_offset:
                c = InferenceClient(inference_coordinator_address=unified_coord_addr)
                c.start()
                shard_clients[s.index] = c
        # rank 0 (which is shard 0's rank_offset by construction) holds
        # the lifecycle client for pause / resume / shutdown. Defend
        # against a future shard layout where shard 0's rank_offset is
        # not rank 0 — the lifecycle path assumes rank 0 owns the
        # client.
        lifecycle_client: Optional[InferenceClient] = None
        if rank == 0:
            assert 0 in shard_clients, (
                "rank 0 must hold shard 0's InferenceClient (lifecycle "
                "client); shard layout puts shard 0's rank_offset "
                f"elsewhere (offset={shards[0].rank_offset})"
            )
            lifecycle_client = shard_clients[0]

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

        instance._register_rollout_policies(args)
        await instance._maybe_publish_disagg_route()
        return instance

    async def _maybe_publish_disagg_route(self) -> None:
        """If the layout has layer-kind disagg shards, compute the
        layout-wide route once and upload it to the coord via
        ``SET_DISAGG_ROUTE``. From then on, every SUBMIT_REQUEST the
        coord receives auto-fans ROUTE_REQUEST to participating shards
        before forwarding the SUBMIT — so each engine has a dispatcher
        registered before its forward fires.

        Layout-level rather than per-request: in v1 every request walks
        the same route. Per-request routes can be wired later by adding
        a route field to ``add_request`` and having the coord prefer
        it over the stored layout route. Only rank 0 publishes (it owns
        the lifecycle client); on non-disagg layouts this is a no-op.
        """
        shards = self._shards
        if not any(s.kinds is not None for s in shards):
            return
        if self._lifecycle_client is None:
            return  # not the publishing rank
        if self._my_engine is None:
            return  # rank 0 must own an engine to read layer_type_list

        # Hetero-TP requires that every pair of consecutive shards in
        # the route have TP sizes that divide one or the other (the
        # dispatcher's fanout / stride needs an integer ratio). Check
        # at launch with a CLI-friendly pointer; the dispatcher
        # re-checks per hop defensively.
        disagg_shards = [s for s in shards if s.kinds is not None]
        tps = sorted({int(s.spec["tp"]) for s in disagg_shards})
        for a in tps:
            for b in tps:
                if a == b:
                    continue
                lo, hi = sorted([a, b])
                assert hi % lo == 0, (
                    f"layer-kind disagg with hetero-TP requires every "
                    f"pair of disagg-shard TP sizes to have a divisibility "
                    f"relationship; layout has tp={tps} with non-dividing "
                    f"pair ({a}, {b}). Reconfigure --rl-inference-shards "
                    f"so every kinds= shard's tp= divides or is divided "
                    f"by every other disagg shard's tp=."
                )

        from megatron.rl.inference.route_planner import plan_route

        layer_type_list = tuple(self._my_engine.context.layer_type_list)
        if not layer_type_list:
            # Pure transformer model: every block is attention.
            layer_type_list = ("*",) * int(
                self._my_engine.context.num_attention_layers
                * int(disagg_shards[0].spec.get("pp", 1))
            )
        route = plan_route(shards, layer_type_list=layer_type_list)
        await self._lifecycle_client.set_layout_route(route)
        log_single_rank(
            logger,
            logging.INFO,
            "[disagg] published layout route: %d hops, entry=s%d exit=s%d",
            len(route.hops),
            route.entry_shard,
            route.exit_shard,
        )

    def _register_rollout_policies(self, args) -> None:
        """Wire optional CLI-driven migration policies.

        - ``--rl-auto-disagg-{src,dst}-shard``: tag every base_generate
          with ``disagg_pair=[src, dst]`` and register a
          :class:`FirstTokenDisaggPolicy` that migrates each request
          from src to dst after its first token.
        - ``--rl-tail-cut-{dst-shard,min-tokens}``: tag the same
          requests with ``tail_cut=[dst, n]`` and register a
          :class:`TailCutPolicy` that migrates them off the decode
          shard onto a latency-optimized one after ``n`` tokens.
        """
        shards = self._shards
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

            self._disagg_rollout_pair = (src_idx_arg, dst_idx_arg)
            self.register_migration_policy(
                FirstTokenDisaggPolicy(
                    src_shard_index=src_idx_arg, dst_shard_index=dst_idx_arg
                )
            )

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

            self._tail_cut_rollout_config = (tail_dst_arg, tail_min_arg)
            self.register_migration_policy(
                TailCutPolicy(
                    src_shard_index=dst_idx_arg,
                    dst_shard_index=tail_dst_arg,
                    min_tokens=int(tail_min_arg),
                )
            )

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
        # Adaptive per-request route: when a RouteSelector is registered
        # (see ``set_route_selector``), consult it for a per-request
        # disagg route override and stamp it onto extra_body. The
        # text-gen server forwards this to
        # ``InferenceClient.add_request(disagg_route=...)`` and the coord
        # prefers the per-request route over the stored layout route.
        if self._route_selector is not None:
            from megatron.rl.inference.route_planner import serialize_route
            from megatron.rl.inference.route_selector import RequestInfo

            # Tokenization happens on the text-gen server, not here, so
            # ``prompt_tokens`` is left unset — selectors that need it
            # should tokenize ``prompt_text`` themselves. The combined
            # raw text is concatenated message contents (selectors
            # typically only care about its length, not structure).
            prompt_text = " ".join(
                str(m.content) for m in request.prompt if m.content
            )
            info = RequestInfo(
                prompt_text=prompt_text,
                sampling_params={
                    "temperature": request.generation_args.temperature,
                    "top_p": request.generation_args.top_p,
                },
            )
            route = self._route_selector.select(info)
            if route is not None:
                extra_body["disagg_route"] = serialize_route(route)
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
        # ``num_layers_total`` for the KV plan is the count of *attention*
        # layers in the whole model — the KV cache only exists for
        # attention layers. For hybrid (Mamba) models this differs from
        # ``model_config.num_layers`` (which counts all layer types).
        # For pure transformers ``ctx.num_attention_layers`` is per-PP-rank,
        # so multiply back to whole-model.
        if ctx.is_hybrid_model:
            num_attention_layers_total = ctx.num_attention_layers
        else:
            num_attention_layers_total = ctx.num_attention_layers * pp
        layout = KVLayout(
            tp_size=tp,
            pp_size=pp,
            num_layers_total=num_attention_layers_total,
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

    def _mamba_layout_for(self, shard: InferenceShard) -> "object":
        """Build a :class:`MambaLayout` for the given shard from this
        rank's local engine context.

        ``mamba_ssm_states_shape = (nheads_local_tp, headdim, d_state)``
        gives ``nheads`` and the per-head dims directly.
        ``mamba_conv_states_shape = (conv_dim_local_tp, d_conv)``
        packs three blocks ``[x, B, C]`` of sizes
        ``[d_inner_local_tp, ngroups_local_tp*d_state,
        ngroups_local_tp*d_state]``; we recover ``ngroups`` from
        ``conv_dim_local_tp - d_inner_local_tp``. ``layer_type_list``
        is propagated so the plan-builder can compute per-PP-rank
        mamba ownership when src and dst use different PP sizes.
        """
        from megatron.core.inference.engines.request_migration import MambaLayout

        ctx = self._my_engine.context
        tp = int(shard.spec["tp"])
        pp = int(shard.spec.get("pp", 1))
        nheads_local_tp, headdim, d_state = ctx.mamba_ssm_states_shape
        conv_dim_local_tp, d_conv = ctx.mamba_conv_states_shape
        d_inner_local_tp = nheads_local_tp * headdim
        bc_bytes_local = conv_dim_local_tp - d_inner_local_tp
        assert bc_bytes_local % (2 * d_state) == 0, (
            f"conv_dim_local_tp={conv_dim_local_tp} - d_inner_local_tp="
            f"{d_inner_local_tp} = {bc_bytes_local} must be divisible by "
            f"2*d_state={2 * d_state}; mamba conv state layout doesn't "
            "match the expected (x, B, C) packing."
        )
        ngroups_local_tp = bc_bytes_local // (2 * d_state)
        return MambaLayout(
            tp_size=tp,
            pp_size=pp,
            num_layers_total=ctx.num_mamba_layers,
            d_inner_total=d_inner_local_tp * tp,
            nheads_total=nheads_local_tp * tp,
            ngroups_total=ngroups_local_tp * tp,
            d_conv=d_conv,
            headdim=headdim,
            d_state=d_state,
            layer_type_list=tuple(getattr(ctx, "layer_type_list", ()) or ()),
        )

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

    def set_route_selector(self, selector: Optional[object]) -> None:
        """Register a :class:`RouteSelector` consulted by ``base_generate``
        to pick a per-request layer-kind disagg route.

        ``None`` clears the selector and falls back to the stored layout
        route for every request. Lives on every rank but only the
        rollout-driver rank exercises it; idle ranks store the selector
        harmlessly so a future driver-rank handoff doesn't need a separate
        registration step.
        """
        self._route_selector = selector

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

        Also registers the ``(src, dst)`` migration pair so the
        engine handler is wired and ``_migration_meta`` is populated
        on every rank — callers shouldn't have to remember to call
        ``_register_migration_pair`` separately.
        """
        assert 0 <= policy.src_shard_index < len(self._shards)
        assert 0 <= policy.dst_shard_index < len(self._shards)
        assert policy.src_shard_index != policy.dst_shard_index
        assert policy.max_batch_size >= 1
        self._register_migration_pair(policy.src_shard_index, policy.dst_shard_index)
        self._migration_policies.append(policy)

    def _register_migration_pair(
        self, src_shard_index: int, dst_shard_index: int
    ) -> None:
        """Pre-compute migration metadata for ``(src, dst)`` and wire
        the engine's MIGRATE_BATCH handler. Cheap and rank-local — the
        NVSHMEM migration path uses one-sided put_signal / signal_wait,
        no torch process group.

        Also reserves NVSHMEM flag-pool lanes for every ``(src_pe,
        dst_pe)`` in the shard cross-product. This is what bounds the
        flag pool by *active* migration topology (typically tens to
        hundreds of pairs) instead of ``n_pes²`` (which would dominate
        startup at 32+ GPUs since each flag is its own collective
        symmetric allocation).
        """
        assert 0 <= src_shard_index < len(self._shards)
        assert 0 <= dst_shard_index < len(self._shards)
        assert src_shard_index != dst_shard_index
        key = (src_shard_index, dst_shard_index)
        if key in self._migration_meta:
            return  # already registered

        src_shard = self._shards[src_shard_index]
        dst_shard = self._shards[dst_shard_index]
        # Reserve flag-pool lanes for this shard pair before pools are
        # realized. Same registration set on every rank because the
        # caller (``register_migration_policy``) is collective.
        from megatron.core.inference import migration_transport as _mig
        if _mig._initialized:
            _mig.register_migration_shard_pair(src_shard.ranks(), dst_shard.ranks())
        rank = dist.get_rank()
        in_src = src_shard.owns_rank(rank)
        in_dst = dst_shard.owns_rank(rank)

        meta = MigrationMeta(
            src_shard=src_shard,
            dst_shard=dst_shard,
            in_src=in_src,
            in_dst=in_dst,
        )
        if in_src or in_dst:
            assert self._my_engine is not None
            meta.src_layout, meta.my_src_head_offset = self._layout_and_head_offset(
                src_shard, self._my_engine, rank, participates=in_src
            )
            meta.dst_layout, meta.my_dst_head_offset = self._layout_and_head_offset(
                dst_shard, self._my_engine, rank, participates=in_dst
            )
            # Mamba state migration supports heterogeneous TP and PP
            # via :func:`build_mamba_migration_plan`. The plan walks
            # ``layer_type_list`` to compute per-PP-rank mamba ownership
            # and intersects between src and dst PP ranks, so hetero PP
            # works as long as the model's transformer layers partition
            # uniformly across PP. All ranks in a shard share the same
            # layout (model is replicated within a shard's TP/PP block).
            if self._my_engine.context.is_hybrid_model:
                meta.src_mamba_layout = self._mamba_layout_for(src_shard)
                meta.dst_mamba_layout = self._mamba_layout_for(dst_shard)
        self._migration_meta[key] = meta

        # Wire the engine handler once. Subsequent pairs share it.
        # Engine guarantees both attributes exist (set to None in __init__);
        # _route_handler binding mirrors the migration-handler pattern.
        if self._my_engine is not None:
            if self._my_engine._migration_handler is None:
                self._my_engine.set_migration_handler(self._on_migrate_batch_signal)
            if self._my_engine._route_handler is None:
                self._my_engine.set_route_handler(self._on_route_request_signal)
            if getattr(self._my_engine, "_disagg_submit_handler", None) is None:
                self._my_engine.set_disagg_submit_handler(
                    self._on_disagg_submit_signal
                )

    def _on_route_request_signal(self, request_id: int, route) -> None:
        """Engine-side ROUTE_REQUEST handler.

        Builds a :class:`RouteDispatcher` for the (request_id, route)
        pair and registers it on the engine so the model's forward
        pass can consult it at each layer. Also attaches the engine
        to the model's :class:`HybridStack` so it has a reference
        to look up dispatchers.

        Called by :meth:`DynamicInferenceEngine.async_step` when a
        ``Headers.ROUTE_REQUEST`` arrives from the coordinator.
        """
        from megatron.core.inference.route_dispatcher import RouteDispatcher

        if self._my_engine is None or self._my_shard_index is None:
            # Not a participating engine; nothing to do. (The coord
            # shouldn't fan to non-participating ranks but be defensive.)
            return
        if not route.visits(self._my_shard_index):
            # This shard isn't on this request's route — skip.
            return

        # Hetero-TP between disagg shards is allowed when TP sizes
        # have a divisibility relationship (the dispatcher's per-pair
        # routing handles fanout / stride). Non-divisible ratios still
        # fail at dispatcher build time via tp_pair_routing's assertion.

        # Hidden-state shape + dtype come from the model's config. For
        # disagg, every shard agrees on the hidden dim (per the
        # ownership invariant); batch is the engine's per-step max.
        # ``max_requests`` is set by the engine config; nbytes is sized
        # for this upper bound so a put never overruns the slot.
        cfg = self._my_engine.controller.inference_wrapped_model.model.config
        hidden_dim = int(cfg.hidden_size)
        hidden_dtype = (
            getattr(cfg, "params_dtype", None)
            or getattr(cfg, "pipeline_dtype", None)
            or torch.bfloat16
        )

        # NVSHMEM PE id equals global rank under standard init. The
        # dispatcher computes per-hop send/receive peer PEs from the
        # per-shard TP topology — matched-TP collapses to 1-to-1, and
        # divisible hetero-TP fans out / strides.
        my_shard = self._shards[self._my_shard_index]
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        tp_offset = global_rank - my_shard.rank_offset
        shard_tp = [int(s.spec["tp"]) for s in self._shards]
        shard_rank_offset = [s.rank_offset for s in self._shards]

        dispatcher = RouteDispatcher(
            route=route,
            my_shard_idx=self._my_shard_index,
            my_pe=global_rank,
            my_tp_offset=tp_offset,
            shard_tp=shard_tp,
            shard_rank_offset=shard_rank_offset,
            hidden_shape=(self._my_engine.max_requests, hidden_dim),
            hidden_dtype=hidden_dtype,
        )
        self._my_engine.register_route_dispatcher(request_id, dispatcher)

    def _on_disagg_submit_signal(
        self,
        request_id: int,
        prompt: object,
        sampling_params: object,
        role: str,
    ) -> None:
        """Engine-side ``DISAGG_SUBMIT`` handler. Injects the
        participant request into the engine's active set so the
        controller drives forward in lockstep with the entry shard.

        Delegates to :meth:`DynamicInferenceEngine.inject_disagg_participant`,
        which builds the request, marks it
        ``_is_disagg_participant = True`` + role, and routes through
        ``_add_request`` so the controller drives forward (allocating
        per-shard state for owned layers). On ``role == "exit"`` the
        engine samples and emits ``ENGINE_REPLY``; non-exit
        participants run forward but have their reply filtered out.
        """
        if self._my_engine is None:
            return
        if request_id not in self._my_engine._route_dispatchers:
            logger.warning(
                "[disagg] DISAGG_SUBMIT for request %d arrived without a "
                "dispatcher — ROUTE_REQUEST race or out-of-order delivery",
                request_id,
            )
            return
        try:
            self._my_engine.inject_disagg_participant(
                request_id=request_id,
                prompt=prompt,
                sampling_params_serialized=sampling_params,
                role=role,
            )
        except Exception as e:
            logger.error(
                "[disagg] inject_disagg_participant failed for request "
                "%d: %s", request_id, e,
            )

    def _on_migrate_batch_signal(
        self,
        request_ids: List[int],
        src_shard_index: int,
        dst_shard_index: int,
        bundles: List[dict],
        dst_dp_rank: int = 0,
    ) -> None:
        """Engine-side handler for ``Headers.MIGRATE_BATCH`` —
        **non-blocking** on both src and dst, including for requests
        that aren't part of this migration.

        - **Src:** detach the migrating requests so the engine stops
          forwarding on them, gather KV slices into staging slots, and
          issue ``put_signal`` on the migration stream. A poll callback
          releases the src KV blocks once the migration event fires.
        - **Dst:** reserve dst KV blocks (and a Mamba state slot on
          hybrid models) without installing the request in the active
          batch, then ``signal_wait`` + scatter on the migration
          stream. A poll callback waits on the migration done event,
          syncs the engine compute stream to it via ``wait_event``,
          and calls ``complete_migration`` — at which point the request
          joins the active batch.

        Crucially, no ``wait_stream`` on the engine compute stream:
        the migrating request is *in transit* (off src's active list,
        not yet on dst's), and the staging-pool memory is on the
        migration stream — neither overlaps with the engine compute
        stream's accesses to non-migrating requests' KV. So those
        requests keep generating at full speed throughout the migration.

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
        if not (meta.in_src or meta.in_dst):
            return  # bystander; coord shouldn't forward to us anyway

        from megatron.core.inference import migration_transport as _nv
        from megatron.core.inference.engines.request_migration import (
            RequestMigrationBundle,
            build_kv_migration_plan,
            deserialize_bundle,
        )

        src_shard = meta.src_shard
        dst_shard = meta.dst_shard
        src_layout = meta.src_layout
        dst_layout = meta.dst_layout
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
        layout = src_layout if meta.in_src else dst_layout

        # Per-rank PP layer offsets for the migration plan's local
        # head/layer slicing.
        rank = dist.get_rank()
        my_src_pp_stage = (
            (rank - src_root) // src_layout.tp_size if meta.in_src else 0
        )
        my_dst_pp_stage = (
            (rank - dst_root) // dst_layout.tp_size if meta.in_dst else 0
        )
        src_layers_per_pp = src_layout.num_layers_total // src_layout.pp_size
        dst_layers_per_pp = dst_layout.num_layers_total // dst_layout.pp_size
        my_src_pp_layer_offset = my_src_pp_stage * src_layers_per_pp
        my_dst_pp_layer_offset = my_dst_pp_stage * dst_layers_per_pp
        my_src_head_offset = meta.my_src_head_offset
        my_dst_head_offset = meta.my_dst_head_offset

        stream = _nv.migration_stream()
        is_hybrid = engine.context.is_hybrid_model

        # Rollback targets — initialized before the try so the except
        # handler can always iterate them, even if a failure fires
        # before the first append.
        src_block_ids_to_free: List[torch.Tensor] = []
        src_mamba_state_indices: List[Optional[int]] = []
        injected: List[Tuple[RequestMigrationBundle, torch.Tensor]] = []
        dst_mamba_state_indices: List[Optional[int]] = []

        # Detach src synchronously so async_step can't emit a stale
        # ENGINE_REPLY for an already-migrated request. ``keep_blocks=True``
        # keeps the KV blocks alive for the gather kernels we'll enqueue
        # below; the poll callback releases them once the event fires.
        # ``keep_mamba_state=True`` does the same for the per-request
        # Mamba slot on hybrid models — the transport reads the slot's
        # conv/SSM state on the migration stream, and the poll callback
        # releases the slot once the event fires.
        try:
            if meta.in_src:
                for req_id in request_ids:
                    if req_id in engine.requests:
                        # Capture mamba state idx BEFORE detach — even
                        # with ``keep_mamba_state=True`` the slot index
                        # is no longer reachable through the request
                        # after detach (it's been removed from
                        # ``engine.requests``).
                        src_mamba_state_indices.append(
                            engine.get_mamba_state_idx_for(req_id) if is_hybrid else None
                        )
                        src_block_ids_to_free.append(
                            engine.detach_request(
                                req_id,
                                keep_blocks=True,
                                keep_mamba_state=is_hybrid,
                            )
                        )
                    else:
                        src_mamba_state_indices.append(None)

            # Dst reserves dst KV blocks + a fresh mamba state slot
            # (on hybrid models) for each bundle, but does NOT install
            # the request in the active batch yet. The two-phase
            # activation keeps the engine forwarding on other requests
            # while migration is in flight: the request only becomes
            # part of the active batch after the migration's done event
            # fires (in the poll callback below).
            if meta.in_dst:
                for bundle in parsed_bundles:
                    rsv = engine.prepare_migration_reservation(bundle)
                    injected.append((bundle, rsv["block_ids"]))
                    dst_mamba_state_indices.append(rsv["mamba_state_idx"])

            # Build per-bundle KV migration plans. Both src and dst walk
            # the bundles in identical order so slot/flag assignment
            # stays consistent. On src ``dst_block_ids`` is unknown and
            # left to the plan helper to fill.
            all_ops_by_bundle: List[List] = []
            for i, bundle in enumerate(parsed_bundles):
                dst_blocks = injected[i][1].tolist() if meta.in_dst else None
                ops = build_kv_migration_plan(
                    bundle,
                    src_global_rank_of=_src_rank_of,
                    dst_global_rank_of=_dst_rank_of,
                    dst_block_ids=dst_blocks,
                )
                all_ops_by_bundle.append(ops)

            # Build per-op staging-slot + flag-slot tables. Both sides
            # walk the bundles in identical order so the choices match
            # without coordination. Slots come from a linear arena;
            # flags come from a lane-partitioned arena (per-(src_pe,
            # dst_pe) sub-counter) so cross-pair acks can't be misread.
            arena = _nv.StagingArena()
            flag_arena = _nv.FlagArena()
            dtype = memory_buffer.dtype
            op_meta = _build_kv_op_meta(
                arena, flag_arena, all_ops_by_bundle, layout, memory_buffer
            )
            mamba_meta: List[dict] = []
            if is_hybrid:
                mamba_meta = _build_mamba_op_meta(
                    arena,
                    flag_arena,
                    parsed_bundles,
                    engine.context,
                    meta.src_mamba_layout,
                    meta.dst_mamba_layout,
                    _src_rank_of,
                    _dst_rank_of,
                )

            # Schedule everything on the migration stream and return.
            # A rank is in at most one of src/dst (shards are
            # non-overlapping rank windows), so the two branches are
            # mutually exclusive. Mamba transfers piggyback on the same
            # stream after KV so the done_event covers both.
            with torch.cuda.stream(stream):
                if meta.in_src:
                    _execute_kv_src_side(
                        rank, op_meta, memory_buffer, dtype, layout,
                        my_src_head_offset, my_src_pp_layer_offset, stream,
                    )
                if meta.in_dst:
                    _execute_kv_dst_side(
                        rank, op_meta, memory_buffer, dtype, layout,
                        my_dst_head_offset, my_dst_pp_layer_offset, stream,
                    )
                if is_hybrid and (meta.in_src or meta.in_dst):
                    conv_states = engine.context.mamba_conv_states
                    ssm_states = engine.context.mamba_ssm_states
                    if meta.in_src:
                        _execute_mamba_src_side(
                            rank, mamba_meta, conv_states, ssm_states,
                            src_mamba_state_indices, stream,
                        )
                    if meta.in_dst:
                        _execute_mamba_dst_side(
                            rank, mamba_meta, conv_states, ssm_states,
                            dst_mamba_state_indices, stream,
                        )

                # Mark the end of this migration on the stream so the
                # tick poll can query a CUDA event.
                done_event = torch.cuda.Event()
                done_event.record(stream)

            # No cross-stream barrier: src's other requests aren't
            # touching the migrating KV blocks (the request was detached
            # at the top of the handler), and dst's other requests don't
            # touch them either (the migrated request isn't in dst's
            # active batch yet — it gets installed in the poll callback
            # below, after wait_event syncs the default stream to the
            # migration stream's writes). Non-migrating requests on both
            # sides keep generating at full speed.
            engine.push_pending_migration(
                _make_migration_poll_callback(
                    done_event,
                    meta,
                    is_hybrid,
                    src_block_ids_to_free,
                    src_mamba_state_indices,
                    injected,
                    dst_mamba_state_indices,
                    engine,
                )
            )
        except BaseException:
            # Catch ``BaseException`` (not ``Exception``) so
            # ``KeyboardInterrupt`` mid-migration still triggers the
            # block-release path before propagating — otherwise an
            # interrupt during the gather/put leaks GPU blocks. Roll
            # back any partial state: src kept-blocks/slots back to
            # allocators, dst reservations released (blocks were
            # reserved but the requests were never installed in the
            # active batch, so a plain reservation release is enough —
            # no detach needed).
            _release_src_state(
                meta,
                is_hybrid,
                src_block_ids_to_free,
                src_mamba_state_indices,
                engine,
            )
            for (_, block_ids), mamba_state_idx in zip(
                injected, dst_mamba_state_indices
            ):
                engine.release_migration_reservation(
                    block_ids, mamba_state_idx
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
        that aren't a decider for any policy.

        Also realizes the migration_transport's flag/ack pools the
        first time it runs — collective across every rank that called
        :meth:`resume`. Idempotent on subsequent calls.
        """
        # Realize flag pools now that the active-pair set is known.
        # Collective: every rank's ``resume()`` reaches this, idle ranks
        # included (via ``resume()``'s finally block). Cheap no-op on
        # second call.
        from megatron.core.inference import migration_transport as _mig
        if _mig._initialized:
            _mig.realize_migration_pools()
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

        # Exponential backoff on capacity rejection. Each consecutive
        # rejection doubles the sleep up to ``MAX_BACKOFF_MULT``; one
        # accept resets to 1×. Without this, a saturated dst shard
        # produces a rejection on every poll tick (tens of ZMQ
        # roundtrips/s, all useless until a decode completes); with
        # the multiplier the decider naturally throttles to the rate
        # at which dst frees slots.
        backoff_mult = 1.0
        MAX_BACKOFF_MULT = 32.0

        while not self._scheduler_stop:
            await asyncio.sleep(policy.poll_interval_s * backoff_mult)
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
                # Two-phase commit: post the batch to the coord and
                # await its accept/reject decision before posting the
                # follow-up rank rewrite. Coord rejects when dst would
                # overflow its slot table, leaving src's request live
                # (no detach has happened yet — that fires only when
                # the engine sees MIGRATE_BATCH from the coord).
                ack_future = client.migrate_request_batch(
                    eligible,
                    src_shard_index,
                    dst_shard_index,
                    bundles=bundles,
                    dst_dp_rank_within_shard=dst_dp_rank,
                )
                accepted = await ack_future
                if accepted:
                    client.update_request_ranks_batch(
                        request_ids=eligible,
                        new_shard_index=dst_shard_index,
                        new_dp_rank_within_shard=dst_dp_rank,
                    )
                    # Mark migrated only on accept; the engine is now
                    # walking these ids off this rank, so the next
                    # ``migrated_ids &= active_ids`` will drop them.
                    migrated_ids.update(eligible)
                    # Reset backoff on a successful migration — dst has
                    # accepted, so the next tick should resume polling
                    # at the policy's base cadence.
                    backoff_mult = 1.0
                else:
                    # Rejected — dst was at capacity. Don't mark as
                    # migrated; next tick will retry these ids. Double
                    # the backoff so the decider naturally throttles
                    # to the rate dst is freeing slots instead of
                    # retry-storming on every base-cadence tick.
                    backoff_mult = min(backoff_mult * 2.0, MAX_BACKOFF_MULT)
                    logger.info(
                        "[migration:%s] dst shard %d rejected batch of %d "
                        "requests (capacity); will retry after %.2fs backoff",
                        type(policy).__name__,
                        dst_shard_index,
                        len(eligible),
                        policy.poll_interval_s * backoff_mult,
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
                # Hard error path (e.g., snapshot/serialize failure):
                # mark ids migrated to avoid retry-storming the same
                # broken batch every tick.
                migrated_ids.update(eligible)

