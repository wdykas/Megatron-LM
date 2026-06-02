# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end orchestration for native prefill->decode disaggregation.

Ties together the pieces built separately -- layout/reshard
(:mod:`kv_shard_layout`), routing (:mod:`kv_router`), transport
(:mod:`kv_transport_backend`), and the KV staging hooks
(``DynamicInferenceContext.export_request_kv`` / ``import_request_kv``
via :mod:`native_kv_handoff`) -- into one coordinator so a request's KV
flows from a prefill worker to a decode worker automatically.

Design (static pool, optimized for cleanliness + speed):

* **Layout handshake.** One all-gather at startup over the global group:
  every worker contributes ``(role, replica_id, layout)`` and ends up
  with the full prefill ``src_layouts`` and the decode pool grouped into
  :class:`~kv_router.DecodeTarget` replicas. No hand-passing of layouts.
* **Per-worker routing.** Each prefill worker runs the same deterministic
  router over the shared registry -- no central-coordinator hop.
* **Push notification.** The prefill replica leader sends a tiny
  ``(request_id, prompt_token_ids)`` control message to the chosen
  decode replica; the header-free data plane derives shapes from the
  prompt, so nothing else crosses the control plane.
* **Transport-agnostic.** Bulk KV moves over the injected
  :class:`KVTransportBackend` (NVSHMEM credit-ring for the fast static
  path; NCCL for portability/CI).

Control plane = ``torch.distributed`` object messaging; data plane = the
backend. The two are kept separate so the backend stays a pure tensor
mover.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.inference.kv_router import (
    DecodeTarget,
    RequestInfo,
    make_router,
    route_and_plan,
)
from megatron.core.inference.kv_shard_layout import KVShardLayout
from megatron.core.inference.kv_transport_backend import (
    KVTransportBackend,
    get_kv_transport_backend,
)
from megatron.core.inference import native_kv_handoff as _handoff

logger = logging.getLogger(__name__)

PREFILL = "prefill"
DECODE = "decode"


def _gather_layouts(entry: dict, group) -> List[dict]:
    """All-gather one small dict per rank over ``group``."""
    import torch.distributed as dist

    world = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
    out: List[Optional[dict]] = [None] * world
    dist.all_gather_object(out, entry, group=group)
    return [e for e in out if e is not None]


def _layout_from(e: dict) -> KVShardLayout:
    return KVShardLayout(
        num_layers=e["num_layers"], num_heads=e["num_heads"],
        tp_size=e["tp_size"], tp_rank=e["tp_rank"],
        pp_size=e["pp_size"], pp_rank=e["pp_rank"],
        global_rank=e["global_rank"], ep_size=e.get("ep_size", 1),
        ep_rank=e.get("ep_rank", 0),
    )


class DisaggCoordinator:
    """Per-worker disaggregation coordinator. One instance per worker."""

    def __init__(
        self,
        *,
        role: str,
        replica_id: str,
        my_layout: KVShardLayout,
        # Per-worker routing requires a DETERMINISTIC policy: every prefill
        # rank of a given request must independently pick the same decode
        # replica (they each ship a shard). Sticky-hash on request_id does
        # that. Stateful policies (round_robin, least_loaded) need a single
        # decider that broadcasts the choice -- a future extension.
        router_name: str = "sticky",
        backend: Optional[KVTransportBackend] = None,
        group: Optional[object] = None,
        max_prompt_tokens: Optional[int] = None,
    ) -> None:
        assert role in (PREFILL, DECODE)
        self.role = role
        self.replica_id = replica_id
        self.my_layout = my_layout
        self.router_name = router_name
        self.backend = backend or get_kv_transport_backend()
        self.group = group
        self.max_prompt_tokens = max_prompt_tokens
        self.src_layouts: List[KVShardLayout] = []
        self.decode_targets: List[DecodeTarget] = []
        self._router = None
        self._my_replica_layouts: List[KVShardLayout] = []
        self._prefill_leader_rank: Optional[int] = None

    # --- one-time setup ----------------------------------------------------

    def handshake(self) -> None:
        """All-gather every worker's layout and build the registry."""
        entry = {
            "role": self.role,
            "replica_id": self.replica_id,
            "global_rank": self.my_layout.global_rank,
            "num_layers": self.my_layout.num_layers,
            "num_heads": self.my_layout.num_heads,
            "tp_size": self.my_layout.tp_size, "tp_rank": self.my_layout.tp_rank,
            "pp_size": self.my_layout.pp_size, "pp_rank": self.my_layout.pp_rank,
            "ep_size": self.my_layout.ep_size, "ep_rank": self.my_layout.ep_rank,
            "max_prompt_tokens": self.max_prompt_tokens,
        }
        entries = _gather_layouts(entry, self.group)

        prefill = [e for e in entries if e["role"] == PREFILL]
        self.src_layouts = sorted(
            (_layout_from(e) for e in prefill), key=lambda l: l.global_rank
        )
        # prefill replica leader = tp0/pp0 (the rank that sends notifications)
        leaders = [e for e in prefill if e["tp_rank"] == 0 and e["pp_rank"] == 0]
        self._prefill_leader_rank = leaders[0]["global_rank"] if leaders else None

        # group decode ranks into replicas by replica_id
        by_replica: dict = {}
        for e in entries:
            if e["role"] != DECODE:
                continue
            by_replica.setdefault(e["replica_id"], []).append(e)
        self.decode_targets = []
        for rid, es in sorted(by_replica.items()):
            layouts = sorted((_layout_from(e) for e in es), key=lambda l: l.global_rank)
            cap = es[0].get("max_prompt_tokens")
            self.decode_targets.append(DecodeTarget(rid, layouts, max_prompt_tokens=cap))

        if self.role == PREFILL:
            self._router = make_router(self.router_name, self.decode_targets)
        else:
            self._my_replica_layouts = next(
                t.layouts for t in self.decode_targets if t.target_id == self.replica_id
            )

    @property
    def router(self):
        return self._router

    # --- prefill side ------------------------------------------------------

    def prefill_handoff(
        self, engine, request_id: int, prompt_token_ids
    ) -> Tuple[DecodeTarget, Optional["_handoff.PrefillHandoff"]]:
        """Route ``request_id`` to a decode replica and ship its KV.

        Returns ``(target, handoff)``; the caller waits ``handoff`` (or
        defers it) and may call ``router.on_admit`` once accepted.
        """
        assert self.role == PREFILL and self._router is not None
        info = RequestInfo(request_id, num_prompt_tokens=len(prompt_token_ids))
        target, _ = route_and_plan(self._router, info, self.src_layouts)

        # Notification (control plane): the prefill leader tells each decode
        # rank of the target a request is inbound + its prompt (for schema).
        if self.my_layout.global_rank == self._prefill_leader_rank:
            import torch.distributed as dist

            for d in target.layouts:
                dist.send_object_list(
                    [(request_id, list(prompt_token_ids), target.target_id)],
                    dst=d.global_rank, group=self.group,
                )

        # Data plane: this prefill rank sends its resharded sub-blocks.
        handoff = _handoff.send_request_kv_resharded(
            engine, request_id, self.my_layout, self.src_layouts, target.layouts,
            backend=self.backend, group=self.group,
        )
        self._router.on_admit(info, target)
        return target, handoff

    # --- decode side -------------------------------------------------------

    def decode_intake(self, engine) -> Tuple[int, Optional[dict]]:
        """Block for one inbound handoff, pull its KV, and import it.

        Returns ``(request_id, import_result)``. ``import_result`` is the
        dict from ``import_request_kv`` (block_ids etc.); the caller then
        admits ``request_id`` into the decode engine's active set.
        """
        assert self.role == DECODE
        import torch.distributed as dist

        holder = [None]
        dist.recv_object_list(holder, src=self._prefill_leader_rank, group=self.group)
        request_id, prompt, _rid = holder[0]

        result = _handoff.recv_request_kv_resharded(
            engine, self.my_layout, self.src_layouts, self._my_replica_layouts,
            prompt, backend=self.backend, group=self.group,
        )
        return request_id, result
