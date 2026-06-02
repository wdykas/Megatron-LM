# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group setup for non-collocated prefill/decode disaggregation.

Prefill and decode run as two *independent* model replicas on disjoint
rank sets, each at its own (possibly different) TP/PP/EP/ETP. We reuse
the RL inference machinery -- :func:`build_inference_pg_collection` +
``HyperCommGrid`` ``rank_offset`` -- which already constructs a second,
independent set of process groups within one job. (This is the same
infrastructure RL post-training uses to give the rollout/inference mesh
a different parallelism than the training mesh.)

Layout::

    ranks [0, Wp)        -> prefill replica, its own pg_collection
    ranks [Wp, Wp+Wd)    -> decode  replica, its own pg_collection
    ranks [Wp+Wd, world) -> idle

where ``Wp = prefill_tp * prefill_pp`` and ``Wd = decode_tp * decode_pp``
(one replica each, so dp = cp = 1; EP/ETP re-factor the same ranks).

``build_inference_pg_collection`` issues collective ``new_group`` calls,
so **every** rank must build **both** meshes; a rank keeps the collection
for the mesh it belongs to and discards the other. The KV cache itself is
sharded only by attention TP/PP, so EP/ETP differences never move bytes
(see :mod:`.kv_shard_layout`); they are carried here only so the handoff
dedupes EP/ETP-replicated KV to one source per attention shard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout

PREFILL = "prefill"
DECODE = "decode"
IDLE = "idle"


@dataclass(frozen=True)
class DisaggParallelConfig:
    """Per-replica parallelism for a disaggregated job."""

    prefill_tp: int
    prefill_pp: int = 1
    decode_tp: int = 1
    decode_pp: int = 1
    prefill_ep: int = 1
    prefill_etp: int = 1
    decode_ep: int = 1
    decode_etp: int = 1

    @property
    def prefill_world(self) -> int:
        return self.prefill_tp * self.prefill_pp

    @property
    def decode_world(self) -> int:
        return self.decode_tp * self.decode_pp

    @property
    def required_world(self) -> int:
        return self.prefill_world + self.decode_world

    def role_of(self, rank: int) -> str:
        if rank < self.prefill_world:
            return PREFILL
        if rank < self.required_world:
            return DECODE
        return IDLE


def build_role_pg_collection(cfg: DisaggParallelConfig):
    """Build both meshes (collective on all ranks); return this rank's role
    and its ``ProcessGroupCollection`` (``None`` for idle ranks).

    Returns ``(role, pg_collection_or_None)``.
    """
    import torch.distributed as dist

    from megatron.rl.parallel_utils import build_inference_pg_collection

    world = dist.get_world_size()
    if world < cfg.required_world:
        raise RuntimeError(
            f"non-collocated disaggregation needs {cfg.required_world} ranks "
            f"(prefill {cfg.prefill_world} + decode {cfg.decode_world}); got {world}."
        )

    # All ranks participate in BOTH builds -- new_group is collective.
    prefill_pg = build_inference_pg_collection(
        world_size=cfg.prefill_world,
        tp_size=cfg.prefill_tp, pp_size=cfg.prefill_pp,
        ep_size=cfg.prefill_ep, expt_tp_size=cfg.prefill_etp,
        rank_offset=0,
    )
    decode_pg = build_inference_pg_collection(
        world_size=cfg.decode_world,
        tp_size=cfg.decode_tp, pp_size=cfg.decode_pp,
        ep_size=cfg.decode_ep, expt_tp_size=cfg.decode_etp,
        rank_offset=cfg.prefill_world,
    )

    role = cfg.role_of(dist.get_rank())
    if role == PREFILL:
        return role, prefill_pg
    if role == DECODE:
        return role, decode_pg
    return role, None


def layout_from_pg_collection(pg, num_layers: int, num_heads: int) -> KVShardLayout:
    """Build a :class:`KVShardLayout` from a ``ProcessGroupCollection``.

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
