# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure a prefill/decode shard engine for the shared DP inference
coordinator (role + KV layouts + identity); called by ``MegatronAsyncLLM`` when
given ``inference_shards``. Also holds the shared shard helpers (KV-shard layout
from process groups, role-layout validation, global KV dims)."""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import Any, List, Tuple

import torch.distributed as dist

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
    """Build a :class:`KVShardLayout` from a shard's ``ProcessGroupCollection``.

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


def _validate_disagg_specs(specs: List[InferenceShardSpec]) -> int:
    """Check the role layout; return the number of decode instances.

    Any number of prefill and decode instances is allowed (each instance --
    a shard's dp replica -- is an independent routing target). The coordinator
    round-robins submits across prefill instances and remembers, per request,
    which prefill held its KV, so the decode side pulls from the right source.
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
    return sum(s.dp for s in decode)


@functools.lru_cache(maxsize=None)
def disagg_refit_pools(inference_shards, world_size: int, rank: int = None) -> Tuple[int, int]:
    """Map an ``--inference-shards`` spec to ``(num_dst_pools, dst_pool_index)``
    for :func:`~megatron.core.resharding.refit.swap_model_weights`.

    Memoized: the result is a pure function of the (process-constant) spec,
    world size, and this rank, so callers can invoke it once per rollout without
    re-parsing the spec each time.

    Disaggregated serving refits the source (training) model into each shard's
    inference model -- disjoint rank windows, possibly at different parallelism
    -- so the refit runs one collective pass per shard. This returns the pool
    count and the window containing ``rank``. Returns ``(1, 0)`` (the
    single-destination default) when the spec is absent or not disaggregated, so
    callers can pass the result unconditionally."""
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
    """Global (num_layers, KV-head count) for the *attention* KV cache.

    num_layers is the number of attention layers in the KV cache -- NOT
    ``cfg.num_layers``: a hybrid Mamba-attention model's ``cfg.num_layers``
    counts Mamba layers too, but only attention layers have a KV cache, so
    using the total would make the reshard plan span layers the cache doesn't
    have (mismatched transfers -> NCCL abort on the hand-off). Read the local
    attention-layer count off the context's ``memory_buffer`` (shape
    ``(2, local_layers, ...)``) and scale to global by the PP size. KV heads =
    num_query_groups for GQA, else num_attention_heads.
    """
    cfg = engine.controller.model_config
    num_heads = getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads
    mb = getattr(getattr(engine, "context", None), "memory_buffer", None)
    # A configured disagg engine always wraps an initialized dynamic context, so
    # memory_buffer is allocated. Assert rather than fall back to cfg.num_layers:
    # that fallback over-counts on hybrid models (it includes Mamba layers) and
    # would silently produce a reshard plan spanning layers the cache lacks.
    assert mb is not None, (
        "disaggregation requires a dynamic KV context with an allocated "
        "memory_buffer; got engine.context=%r" % getattr(engine, "context", None)
    )
    # memory_buffer layer dim is this PP stage's local attention-layer count.
    # Hybrid models do not split attention layers evenly across PP stages, so
    # gather the per-stage counts and sum (same approach as the Mamba layer
    # offset) rather than assuming uniform distribution.
    local_layers = int(mb.shape[1])
    pp = get_pg_size(pg.pp)
    if pp <= 1:
        return local_layers, num_heads
    counts = [0] * pp
    dist.all_gather_object(counts, local_layers, group=pg.pp)
    return sum(counts), num_heads


def _mamba_layout_dict(engine, pg):
    """This rank's Mamba shard layout dict (or ``None`` for non-hybrid models).

    Structural dims (the ``dims`` sub-dict, a serialized :class:`MambaStateDims`)
    come straight from the model config -- notably ``ngroups`` is
    ``config.mamba_num_groups``, the same source MambaMixer reads, rather than
    reverse-derived from the conv channel width (which would silently break if
    the conv packing ever changed). nheads/headdim/d_state/d_conv are read off
    the allocated conv/ssm shapes, which are unambiguous. The global Mamba-layer
    offset is the prefix sum of per-PP-stage local counts (contiguous in global
    layer order), via an all-gather over the PP group.
    """
    ctx = getattr(engine, "context", None)
    if ctx is None or not getattr(ctx, "is_hybrid_model", False):
        return None
    conv_shape = getattr(ctx, "mamba_conv_states_shape", None)
    ssm_shape = getattr(ctx, "mamba_ssm_states_shape", None)
    if conv_shape is None or ssm_shape is None:
        return None

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


@dataclass
class DisaggCoordinatorSetup:
    """This rank's place in a coordinator-native disagg job."""

    role: str           # "prefill" / "decode"
    replica_id: str     # "prefill" / "decode_s{shard}_dp{dp}"
    engine: Any
    is_primary: bool    # global rank 0 -> owns the InferenceClient
    total_instances: int


def configure_prebuilt_disagg_engine(
    engine: Any, pg: Any, specs: List[InferenceShardSpec], disagg_router: str = "round_robin",
    kv_transport_backend: str = "nccl",
) -> DisaggCoordinatorSetup:
    """Configure an already-built engine for the shared coordinator.

    The caller built the model + engine against this rank's shard ``pg``
    outside (mirroring ``MegatronLLM(model=...)``); this only derives the disagg
    config and sets it on the engine. The per-rank KV layout is read from the
    live ``pg`` and the full per-instance layout is gathered over the instance's
    MP group (tp x pp), so it is correct for any tp/dp/pp rank ordering (no
    contiguity assumption).
    """
    total_instances = sum(s.dp for s in specs)
    _validate_disagg_specs(specs)  # role-layout checks
    # Disaggregation requires prefix caching: the decode side admits the
    # handed-off KV via a prefix-cache hit (import registers the block hashes).
    # Without it the imported blocks aren't matched and decode silently
    # re-prefills, wasting the hand-off.
    ctx = getattr(engine, "context", None)
    if ctx is not None:
        assert getattr(ctx, "enable_prefix_caching", False), (
            "disaggregation requires prefix caching (enable_prefix_caching=True); "
            "the decode side admits handed-off KV via a prefix-cache hit."
        )
        # MLA's latent KV cache isn't derivable header-free, so the decode side
        # can't reconstruct the schema. Reject it up front with a clear message
        # rather than crash the engine loop at send/recv time.
        assert not getattr(ctx, "cache_mla_latent", False), (
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
    # Hybrid models: attach this rank's Mamba shard layout so conv/ssm state can
    # be resharded alongside the attention KV.
    mamba = _mamba_layout_dict(engine, pg)
    if mamba is not None:
        my_layout["mamba"] = mamba
    # Gather every rank of this instance (the MP group spans exactly tp x pp).
    layouts = [None] * get_pg_size(pg.mp)
    dist.all_gather_object(layouts, my_layout, group=pg.mp)

    # Unique per instance (shard index + dp replica) so each prefill/decode
    # replica gets a distinct ZMQ identity + layout key -- this is what lets the
    # coordinator address multiple prefill replicas, not just one.
    replica_id = f"{role}_s{my_index}_dp{dp_rank}"
    engine.set_disaggregation_config(
        role=role,
        instance_layouts=layouts,
        identity=replica_id,
        world_group=None,  # default world group for the cross-shard addr broadcast
        spawn_coordinator=(rank == 0),
        disagg_router=disagg_router,
        kv_transport_backend=kv_transport_backend,
    )
    return DisaggCoordinatorSetup(
        role=role, replica_id=replica_id, engine=engine,
        is_primary=(rank == 0), total_instances=total_instances,
    )
