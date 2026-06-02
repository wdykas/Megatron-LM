# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""High-level prefill->decode disaggregation driver over inference shards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from megatron.core.inference.disaggregation.disagg_coordinator import DisaggCoordinator
from megatron.core.inference.disaggregation.kv_router import RequestInfo
from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout
from megatron.core.inference.disaggregation.kv_transport_backend import (
    KVTransportBackend,
    NcclTransportBackend,
)
from megatron.core.inference.shards import InferenceShard, build_inference_pg_collections_for_shards

PREFILL = "prefill"
DECODE = "decode"


@dataclass
class DisaggRequest:
    """Minimal request the run loops need. ``request_id`` is the stable id
    used on both sides (decode recovers prompts by it)."""

    request_id: int
    prompt_text: Any
    prompt_tokens: Sequence[int]
    sampling_params: Any


@dataclass
class DisaggSetup:
    """What :func:`setup_disagg` returns for this rank."""

    role: str            # "prefill" or "decode"
    replica_id: str      # "prefill" or "decode_s{shard}_dp{dp_rank}"
    engine: Any
    coordinator: DisaggCoordinator
    num_decode_instances: int


def layout_from_pg_collection(pg, num_layers: int, num_heads: int) -> KVShardLayout:
    """Build a :class:`KVShardLayout` from a shard's ``ProcessGroupCollection``.

    Reads attention TP/PP (which shard the KV) and EP/ETP (KV-replica
    dimensions, used only for source dedup) from the collection's groups.
    """
    import torch.distributed as dist

    from megatron.core.utils import get_pg_rank, get_pg_size

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


def _decode_replica_id(shard_index: int, dp_rank: int) -> str:
    """Unique id per decode *instance*. A dp>1 decode shard has one
    instance per dp rank; each is an independent routing target."""
    return f"decode_s{shard_index}_dp{dp_rank}"


def _validate_disagg_specs(specs: List[dict]) -> int:
    """Check the role layout; return the number of decode instances."""
    prefill = [s for s in specs if s.get("role") == PREFILL]
    decode = [s for s in specs if s.get("role") == DECODE]
    untagged = [s for s in specs if s.get("role") not in (PREFILL, DECODE)]
    assert not untagged, (
        f"every shard must declare role=prefill or role=decode for "
        f"disaggregation; {len(untagged)} shard(s) had none: {untagged}"
    )
    assert prefill and decode, (
        "disaggregation needs at least one prefill shard and one decode shard."
    )
    prefill_instances = sum(s.get("dp", 1) for s in prefill)
    assert len(prefill) == 1 and prefill_instances == 1, (
        "exactly one prefill instance is supported today (one prefill shard "
        f"with dp=1); got {len(prefill)} prefill shard(s) totalling "
        f"{prefill_instances} instance(s). Data-parallel prefill needs "
        "per-request prefill-instance scoping (a follow-up)."
    )
    return sum(s.get("dp", 1) for s in decode)


def _global_kv_dims(engine) -> Tuple[int, int]:
    """Global (num_layers, KV-head count) read off the built engine's model
    config. KV heads = num_query_groups for GQA, else num_attention_heads."""
    cfg = engine.controller.model_config
    num_heads = getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads
    return cfg.num_layers, num_heads


def setup_disagg(
    specs: List[dict],
    *,
    engine_builder: Callable[[Any], Any],
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    backend: Optional[KVTransportBackend] = None,
    router_name: str = "sticky",
    group: Optional[object] = None,
) -> DisaggSetup:
    """Build this rank's shard (engine + coordinator) and complete handshake.

    Args:
        specs: Parsed shard specs (see
            :func:`megatron.core.inference.shards_spec.parse_inference_shards_spec`),
            each tagged ``role=prefill``/``role=decode``.
        engine_builder: ``pg_collection -> engine``. Called once for this
            rank's shard; the caller owns model construction / checkpoint
            loading (the only framework-specific step).
        num_layers, num_heads: Global attention layer / KV-head counts. Default
            ``None`` -> derived from the built engine's model config; pass them
            only for engines that don't expose ``controller.model_config``.
        backend: KV transport backend (default: NCCL/P2P).
        router_name: Decode router policy. Must be deterministic for
            per-worker routing (default 'sticky').
    """
    import torch.distributed as dist

    from megatron.core.utils import get_pg_rank

    num_decode_instances = _validate_disagg_specs(specs)

    world = dist.get_world_size()
    # Every rank builds every shard's process groups (new_group is
    # world-collective); only this rank's shard gets a usable collection.
    shards: List[InferenceShard] = build_inference_pg_collections_for_shards(world, specs)
    my_shards = [s for s in shards if s.pg_collection is not None]
    assert len(my_shards) == 1, (
        f"rank {dist.get_rank()} belongs to {len(my_shards)} shards; expected 1 "
        "(shard specs must partition the world exactly)."
    )
    my = my_shards[0]
    role = my.spec["role"]

    pg = my.pg_collection
    if role == PREFILL:
        replica_id = PREFILL
    else:
        replica_id = _decode_replica_id(my.index, get_pg_rank(pg.dp))

    engine = engine_builder(pg)
    if num_layers is None or num_heads is None:
        num_layers, num_heads = _global_kv_dims(engine)
    layout = layout_from_pg_collection(pg, num_layers, num_heads)

    backend = backend or NcclTransportBackend()
    backend.init()
    coordinator = DisaggCoordinator(
        role=role, replica_id=replica_id, my_layout=layout,
        backend=backend, router_name=router_name, group=group,
    )
    coordinator.handshake()
    return DisaggSetup(
        role=role, replica_id=replica_id, engine=engine,
        coordinator=coordinator, num_decode_instances=num_decode_instances,
    )


def run_prefill_replica(
    coordinator: DisaggCoordinator, engine, requests: Sequence[DisaggRequest]
) -> None:
    """Prefill every request and hand its KV to the routed decode instance.

    Each step advances prefill; once a request has run, route + ship its KV
    (the coordinator's router picks the decode instance and reshards to it).
    """
    for req in requests:
        engine.add_request(req.request_id, req.prompt_text, req.sampling_params)
    handed = set()
    while len(handed) < len(requests):
        engine.step_modern()
        for req in requests:
            if req.request_id not in handed:
                _target, handoff = coordinator.prefill_handoff(
                    engine, req.request_id, req.prompt_tokens
                )
                if handoff is not None:
                    handoff.wait()
                handed.add(req.request_id)


def run_decode_replica(
    coordinator: DisaggCoordinator, engine, requests: Sequence[DisaggRequest]
) -> Dict[int, Any]:
    """Intake the KV for this instance's requests, generate, return outputs.

    Returns ``{request_id: finished_record}`` for the requests routed to
    THIS decode instance (decode replays the deterministic router to learn
    which, so no extra coordination is needed).
    """
    by_id = {req.request_id: req for req in requests}
    infos = [RequestInfo(req.request_id, num_prompt_tokens=len(req.prompt_tokens)) for req in requests]
    my_ids = coordinator.assigned_request_ids(infos)

    for _ in my_ids:
        request_id, _import_result = coordinator.decode_intake(engine)
        req = by_id[request_id]
        # KV import already registered the prefix-cache blocks; add_request
        # matches the cache, skips prefill, and continues generation.
        engine.add_request(req.request_id, req.prompt_text, req.sampling_params)

    finished: Dict[int, Any] = {}
    while len(finished) < len(my_ids):
        result = engine.step_modern()
        for record in result.get("finished_request_records", []):
            merged = record.merge()
            finished[merged.request_id] = merged
    return finished
