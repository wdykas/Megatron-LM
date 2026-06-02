# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""High-level driver for non-collocated prefill->decode disaggregation.

This is the reusable entry point: given a :class:`DisaggTopology` and a
callable that builds an inference engine for a process-group collection,
it stands up this rank's replica (mesh + engine + coordinator) and runs
the prefill or decode loop. The only Megatron-specific dependency is the
engine the caller passes in -- everything here is engine-agnostic over
the ``DynamicInferenceEngine`` interface (``add_request`` / ``step_modern``
/ finished-record), so a serving layer, an RL rollout, or a Dynamo
backend can use it without copying orchestration into their own tree.

Typical use::

    topo = DisaggTopology.from_specs(["tp=2"], ["tp=1", "tp=2,ep=2"])
    setup = setup_disagg(topo, engine_builder=build_my_engine,
                         num_layers=L, num_heads=H)
    if setup.role == "prefill":
        run_prefill_replica(setup.coordinator, setup.engine, requests)
    else:
        outputs = run_decode_replica(setup.coordinator, setup.engine, requests)

``requests`` is any sequence of objects exposing ``request_id``,
``prompt_text``, ``prompt_tokens`` and ``sampling_params`` (see
:class:`DisaggRequest`); it must be identical on every rank (decode
replays the deterministic router to learn its share).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from megatron.core.inference.disaggregation.disagg_coordinator import DisaggCoordinator
from megatron.core.inference.disaggregation.kv_router import RequestInfo
from megatron.core.inference.disaggregation.kv_transport_backend import (
    KVTransportBackend,
    NcclTransportBackend,
)
from megatron.core.inference.disaggregation.pg_setup import (
    DisaggTopology,
    build_role_pg_collection,
    layout_from_pg_collection,
)


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
    replica_id: str      # "prefill" or "decode{i}"
    engine: Any
    coordinator: DisaggCoordinator


def setup_disagg(
    topo: DisaggTopology,
    *,
    engine_builder: Callable[[Any], Any],
    num_layers: int,
    num_heads: int,
    backend: Optional[KVTransportBackend] = None,
    router_name: str = "sticky",
    group: Optional[object] = None,
) -> DisaggSetup:
    """Build this rank's replica and complete the layout handshake.

    Args:
        topo: Prefill replica + decode pool topology.
        engine_builder: ``pg_collection -> engine``. Called once for this
            rank's mesh; the caller owns model construction / checkpoint
            loading (the only framework-specific step).
        num_layers, num_heads: Global attention layer / KV-head counts
            (for the KV shard layout).
        backend: KV transport backend (default: NCCL/P2P).
        router_name: Decode router policy. Must be deterministic for
            per-worker routing (default 'sticky').
    """
    role, replica_id, pg = build_role_pg_collection(topo)
    engine = engine_builder(pg)
    layout = layout_from_pg_collection(pg, num_layers, num_heads)

    backend = backend or NcclTransportBackend()
    backend.init()
    coordinator = DisaggCoordinator(
        role=role, replica_id=replica_id, my_layout=layout,
        backend=backend, router_name=router_name, group=group,
    )
    coordinator.handshake()
    return DisaggSetup(role=role, replica_id=replica_id, engine=engine, coordinator=coordinator)


def run_prefill_replica(
    coordinator: DisaggCoordinator, engine, requests: Sequence[DisaggRequest]
) -> None:
    """Prefill every request and hand its KV to the routed decode replica.

    Each step advances prefill; once a request has run, route + ship its KV
    (the coordinator's router picks the decode replica and reshards to it).
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
    """Intake the KV for this replica's requests, generate, return outputs.

    Returns ``{request_id: finished_record}`` for the requests routed to
    THIS decode replica (decode replays the deterministic router to learn
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
