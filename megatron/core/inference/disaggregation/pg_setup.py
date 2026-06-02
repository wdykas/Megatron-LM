# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-group setup for non-collocated prefill/decode disaggregation.

Prefill and decode run as independent model replicas on disjoint rank
sets, each at its own (possibly different) TP/PP/EP/ETP. The decode side
is a *pool*: several replicas that may each use a different parallelism
(requests are routed across them by :mod:`.kv_router`). We reuse the RL
inference machinery -- :func:`build_inference_pg_collection` +
``HyperCommGrid`` ``rank_offset`` -- which already constructs independent
process-group sets within one job.

Rank layout (replicas stacked in order)::

    prefill          ranks [0, Wp)
    decode replica 0 ranks [Wp, Wp + Wd0)
    decode replica 1 ranks [Wp + Wd0, Wp + Wd0 + Wd1)
    ...

where each replica's world is ``tp * pp`` (one replica each, so dp = cp =
1; EP/ETP re-factor the same ranks). ``build_inference_pg_collection``
issues collective ``new_group`` calls, so **every** rank builds **every**
mesh; a rank keeps the collection for the mesh it belongs to. The KV
cache is sharded only by attention TP/PP, so EP/ETP differences between
replicas never move bytes (see :mod:`.kv_shard_layout`); they are carried
only so the handoff dedupes EP/ETP-replicated KV to one source per shard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout

PREFILL = "prefill"
DECODE = "decode"
IDLE = "idle"


@dataclass(frozen=True)
class ReplicaSpec:
    """Parallelism of a single replica. ``world == tp * pp``."""

    tp: int
    pp: int = 1
    ep: int = 1
    etp: int = 1

    @property
    def world(self) -> int:
        return self.tp * self.pp


def parse_replica_spec(spec: str) -> ReplicaSpec:
    """Parse ``"tp=2,pp=1,ep=2,etp=1"`` into a :class:`ReplicaSpec`.

    ``tp`` is required; ``pp``/``ep``/``etp`` default to 1. Keys are
    case-insensitive; whitespace is ignored.
    """
    kv = {}
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        key, sep, val = part.partition("=")
        if not sep:
            raise ValueError(f"bad parallelism spec '{spec}': expected key=value, got '{part}'")
        key = key.lower()
        if key not in ("tp", "pp", "ep", "etp"):
            raise ValueError(f"bad parallelism spec '{spec}': unknown key '{key}'")
        kv[key] = int(val)
    if "tp" not in kv:
        raise ValueError(f"parallelism spec '{spec}' must set tp (e.g. 'tp=2,pp=1')")
    return ReplicaSpec(tp=kv["tp"], pp=kv.get("pp", 1), ep=kv.get("ep", 1), etp=kv.get("etp", 1))


@dataclass(frozen=True)
class DisaggTopology:
    """One prefill replica feeding a pool of decode replicas."""

    prefill: ReplicaSpec
    decode: Tuple[ReplicaSpec, ...]

    @classmethod
    def from_specs(cls, prefill_specs: List[str], decode_specs: List[str]) -> "DisaggTopology":
        prefill = [parse_replica_spec(s) for s in prefill_specs]
        if len(prefill) != 1:
            raise ValueError(
                f"expected exactly one --disagg-prefill-parallelism spec; got {len(prefill)}. "
                "(A pool of prefill replicas needs prefill-side routing, a future extension.)"
            )
        decode = tuple(parse_replica_spec(s) for s in decode_specs)
        if not decode:
            raise ValueError("need at least one --disagg-decode-parallelism spec")
        return cls(prefill=prefill[0], decode=decode)

    @property
    def num_decode_replicas(self) -> int:
        return len(self.decode)

    @property
    def prefill_world(self) -> int:
        return self.prefill.world

    @property
    def decode_world(self) -> int:
        return sum(d.world for d in self.decode)

    @property
    def required_world(self) -> int:
        return self.prefill_world + self.decode_world

    def meshes(self):
        """Yield ``(replica_id, spec, rank_offset)`` for every replica, in
        rank order (prefill first, then each decode replica)."""
        yield (PREFILL, self.prefill, 0)
        off = self.prefill_world
        for i, d in enumerate(self.decode):
            yield (f"decode{i}", d, off)
            off += d.world

    def replica_at(self, rank: int) -> Tuple[str, Optional[str], Optional[ReplicaSpec], int]:
        """``(role, replica_id, spec, rank_offset)`` for ``rank``; role is
        IDLE with None fields if the rank is past the topology."""
        for replica_id, spec, off in self.meshes():
            if off <= rank < off + spec.world:
                role = PREFILL if replica_id == PREFILL else DECODE
                return role, replica_id, spec, off
        return IDLE, None, None, self.required_world


def build_role_pg_collection(topo: DisaggTopology):
    """Build every mesh (collective on all ranks); return this rank's role,
    replica_id, and its ``ProcessGroupCollection``.

    Returns ``(role, replica_id, pg_collection)``.
    """
    import torch.distributed as dist

    from megatron.rl.parallel_utils import build_inference_pg_collection

    world = dist.get_world_size()
    if world != topo.required_world:
        raise RuntimeError(
            f"disaggregation needs exactly {topo.required_world} ranks "
            f"(prefill {topo.prefill_world} + decode {topo.decode_world}); got {world}. "
            "No idle ranks are allowed -- the handoff handshake spans the whole world."
        )
    my_rank = dist.get_rank()
    mine = None
    # All ranks participate in every build -- new_group is collective.
    for replica_id, spec, off in topo.meshes():
        pg = build_inference_pg_collection(
            world_size=spec.world,
            tp_size=spec.tp, pp_size=spec.pp,
            ep_size=spec.ep, expt_tp_size=spec.etp,
            rank_offset=off,
        )
        if off <= my_rank < off + spec.world:
            role = PREFILL if replica_id == PREFILL else DECODE
            mine = (role, replica_id, pg)
    assert mine is not None, f"rank {my_rank} not placed in topology {topo}"
    return mine


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
