# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure a prefill/decode shard engine for the shared DP inference
coordinator (role, KV layouts, identity); called by MegatronAsyncLLM when
given inference_shards. Also holds the shared shard helpers (KV-shard layout
from process groups, role-layout validation, global KV dims)."""

from __future__ import annotations

import functools
from dataclasses import asdict
from typing import Any, List, Tuple

import torch.distributed as dist

from megatron.core.inference.config import PrefixCachingEvictionPolicy
from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout
from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    parse_inference_shards_spec,
    spec_declares_disaggregation,
)
from megatron.core.utils import get_pg_rank, get_pg_size

PREFILL = "prefill"
DECODE = "decode"


def layout_from_pg_collection(pg, num_layers: int, num_heads: int) -> KVShardLayout:
    """Build a KVShardLayout from a shard's ProcessGroupCollection.

    Reads attention TP/PP (which shard the KV) and EP/ETP (KV-replica
    dimensions, used only for source dedup) from the collection's groups.
    """
    return KVShardLayout(
        num_layers=num_layers,
        num_heads=num_heads,
        tp_size=get_pg_size(pg.tp),
        tp_rank=get_pg_rank(pg.tp),
        pp_size=get_pg_size(pg.pp),
        pp_rank=get_pg_rank(pg.pp),
        global_rank=dist.get_rank(),
        ep_size=get_pg_size(getattr(pg, "ep", None)),
        ep_rank=get_pg_rank(getattr(pg, "ep", None)),
        etp_size=get_pg_size(getattr(pg, "expt_tp", None)),
        etp_rank=get_pg_rank(getattr(pg, "expt_tp", None)),
    )


def _validate_disagg_specs(specs: List[InferenceShardSpec]) -> None:
    """Check the role layout.

    Any number of prefill and decode instances is allowed; each instance (a
    shard's dp replica) is an independent routing target.
    """
    prefill = [s for s in specs if s.role == PREFILL]
    decode = [s for s in specs if s.role == DECODE]
    untagged = [s for s in specs if s.role not in (PREFILL, DECODE)]
    assert not untagged, (
        f"every shard must declare role=prefill or role=decode for "
        f"disaggregation; {len(untagged)} shard(s) had none: {untagged}"
    )
    assert prefill and decode, (
        "disaggregation needs at least one prefill shard and one decode shard."
    )


@functools.lru_cache(maxsize=None)
def disagg_refit_pools(inference_shards, world_size: int, rank: int = None) -> Tuple[int, int]:
    """Map an --inference-shards spec to (num_dst_pools, dst_pool_index) for
    swap_model_weights.

    Disaggregated serving refits the training model into each shard's
    inference model (disjoint rank windows, possibly at different
    parallelism), so the refit runs one collective pass per shard. This
    returns the pool count and the window containing `rank`. Returns (1, 0)
    when the spec is absent or not disaggregated, so callers can pass the
    result unconditionally. Memoized: the result is a pure function of the
    process-constant spec, world size, and rank."""
    if rank is None:
        rank = dist.get_rank()
    if not (inference_shards and spec_declares_disaggregation(inference_shards)):
        return 1, 0
    specs = parse_inference_shards_spec(inference_shards, world_size)
    offset = 0
    for index, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            return len(specs), index
        offset += s.world_size
    raise RuntimeError(f"rank {rank} not in any disagg shard window")


def _global_kv_dims(engine, pg) -> Tuple[int, int]:
    """Global (num_layers, KV-head count) for the attention KV cache.

    num_layers counts attention layers only, not cfg.num_layers: a hybrid
    model's cfg.num_layers includes Mamba layers, which have no KV cache, and
    a reshard plan spanning them would build mismatched transfers. The local
    attention-layer count is read off the context's memory_buffer and summed
    over PP stages. KV heads = num_query_groups (GQA).
    """
    cfg = engine.controller.model_config
    num_heads = cfg.num_query_groups
    # A configured disagg engine always has an allocated memory_buffer.
    mb = engine.context.memory_buffer
    assert mb is not None, (
        "disaggregation requires a dynamic KV context with an allocated "
        "memory_buffer; got engine.context=%r" % engine.context
    )
    # memory_buffer's layer dim is this PP stage's local attention-layer count.
    # Hybrid models split attention layers unevenly across PP stages, so gather
    # the per-stage counts and sum rather than assuming a uniform split.
    local_layers = int(mb.shape[1])
    pp = get_pg_size(pg.pp)
    if pp <= 1:
        return local_layers, num_heads
    counts = [0] * pp
    dist.all_gather_object(counts, local_layers, group=pg.pp)
    return sum(counts), num_heads


def _mamba_layout_dict(engine, pg):
    """This rank's Mamba shard layout dict, or None for non-hybrid models.

    Structural dims (the dims sub-dict, a serialized MambaStateDims) come from
    the model config: ngroups is config.mamba_num_groups, the same source
    MambaMixer reads, rather than reverse-derived from the conv channel width.
    nheads/headdim/d_state/d_conv are read off the allocated conv/ssm shapes.
    The global Mamba-layer offset is the prefix sum of per-PP-stage local
    counts, gathered over the PP group.
    """
    ctx = engine.context
    if not ctx.is_hybrid_model:
        return None
    conv_shape = ctx.mamba_conv_states_shape
    ssm_shape = ctx.mamba_ssm_states_shape

    nheads_local, headdim, d_state = (int(x) for x in ssm_shape)
    d_conv = int(conv_shape[1])
    tp = get_pg_size(pg.tp)
    tp_rank = get_pg_rank(pg.tp)

    num_local = int(ctx.num_mamba_layers)
    pp = get_pg_size(pg.pp)
    pp_rank = get_pg_rank(pg.pp)
    counts = [0] * pp
    dist.all_gather_object(counts, num_local, group=pg.pp)
    layer_start = sum(counts[:pp_rank])

    return dict(
        global_rank=dist.get_rank(), tp_size=tp, tp_rank=tp_rank,
        layer_start=layer_start, num_layers=num_local,
        dims=dict(
            nheads=nheads_local * tp, headdim=headdim, d_state=d_state,
            ngroups=engine.controller.model_config.mamba_num_groups, d_conv=d_conv,
        ),
    )


def configure_prebuilt_disagg_engine(
    engine: Any, pg: Any, specs: List[InferenceShardSpec], disagg_router: str = "round_robin",
    kv_transport_backend: str = "nccl",
) -> None:
    """Configure an already-built engine for the shared coordinator.

    The caller built the model + engine against this rank's shard `pg`; this
    only derives the disagg config and sets it on the engine. The per-rank KV
    layout is read from the live `pg` and the full per-instance layout is
    gathered over the instance's MP group (tp x pp), so it is correct for any
    tp/dp/pp rank ordering.
    """
    _validate_disagg_specs(specs)
    # The decode side admits handed-off KV via a prefix-cache hit (the import
    # registers the block hashes), so prefix caching is required.
    ctx = engine.context
    assert ctx.enable_prefix_caching, (
        "disaggregation requires prefix caching (enable_prefix_caching=True); "
        "the decode side admits handed-off KV via a prefix-cache hit."
    )
    # ref_zero eviction deregisters blocks the moment their ref count hits 0,
    # which would discard the imported KV before admission sees it.
    assert ctx.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU, (
        "disaggregation requires the LRU prefix-cache eviction policy "
        "(--inference-dynamic-batching-prefix-caching-eviction-policy lru); "
        f"got {ctx.prefix_caching_eviction_policy!r}."
    )
    assert not ctx.cache_mla_latent, (
        "disaggregation does not support the MLA latent KV cache "
        "(cache_mla_latent=True)."
    )
    rank = dist.get_rank()

    # Locate this rank's shard. Shard windows are contiguous (tp*pp*dp ranks
    # each) regardless of the intra-shard rank ordering.
    offset = 0
    my_index = None
    my_spec = None
    for i, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            my_index, my_spec = i, s
            break
        offset += s.world_size
    assert my_spec is not None, f"rank {rank} not in any disagg shard window"
    role = my_spec.role

    num_layers, num_heads = _global_kv_dims(engine, pg)
    dp_rank = get_pg_rank(pg.dp)
    my_layout = asdict(layout_from_pg_collection(pg, num_layers, num_heads))
    # Hybrid models: attach this rank's Mamba shard layout so snapshots can be
    # paired alongside the attention KV.
    mamba = _mamba_layout_dict(engine, pg)
    if mamba is not None:
        my_layout["mamba"] = mamba
    # Gather every rank of this instance (the MP group spans exactly tp x pp).
    layouts = [None] * get_pg_size(pg.mp)
    dist.all_gather_object(layouts, my_layout, group=pg.mp)

    # Unique per instance (shard index + dp replica), so each prefill/decode
    # replica gets a distinct ZMQ identity and layout key; this is what lets
    # the coordinator address multiple replicas of a role.
    replica_id = f"{role}_s{my_index}_dp{dp_rank}"
    engine.set_disaggregation_config(
        role=role,
        instance_layouts=layouts,
        identity=replica_id,
        spawn_coordinator=(rank == 0),
        disagg_router=disagg_router,
        kv_transport_backend=kv_transport_backend,
    )
