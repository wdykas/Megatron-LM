# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous TP/PP/EP resharding for the native KV handoff.

When the prefill and decode workers use *different* parallel layouts,
the KV cache is physically sharded differently on each side and the
handoff must reshard it:

* **Tensor parallel (TP)** shards the KV by attention/KV heads: TP rank
  ``t`` of ``Tp`` owns global heads ``[t*H/Tp, (t+1)*H/Tp)``.
* **Pipeline parallel (PP)** partitions by layers: PP rank ``p`` of
  ``Pp`` owns global layers ``[p*L/Pp, (p+1)*L/Pp)``.
* **Expert parallel (EP)** shards MoE experts, **not** the attention KV
  cache -- the KV is EP-replicated. So hetero-EP needs no data
  re-partition: it reduces to *replica selection* (one EP replica is the
  representative source; every destination EP replica pulls the same
  bytes). We therefore ignore ``ep_rank`` for KV data and only use it to
  pick a single source replica.

The resharding primitive is **range intersection** in *global*
coordinates. A destination rank owns a ``(layer_range x head_range)``
rectangle; each source rank owns another. The two exchange exactly the
intersection sub-block (if non-empty). This is fully general -- it
handles divisible *and* non-divisible TP/PP ratios and any PP change --
and it is derivable on both sides from the two layouts alone (exchanged
once at setup), so per-request transfers stay header-free.

Scope (first MR): attention KV resharding. Mamba/hybrid state has a
different sharding semantics and is left on the matched-layout path for
a future MR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class KVShardLayout:
    """A worker's KV-cache ownership within the global model.

    ``num_layers`` / ``num_heads`` are the *global* attention layer count
    and KV-head count (for GQA, the number of KV heads). ``global_rank``
    is the worker's torch rank (used as the transport peer id).
    """

    num_layers: int
    num_heads: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    global_rank: int
    ep_size: int = 1
    ep_rank: int = 0

    def __post_init__(self) -> None:
        # Standard Megatron requirement: TP divides heads. PP usually
        # divides layers. We support the divisible case (even splits);
        # range intersection itself is general, but even splits keep the
        # global<->local index map unambiguous.
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} not divisible by tp_size={self.tp_size}"
            )
        if self.num_layers % self.pp_size != 0:
            raise ValueError(
                f"num_layers={self.num_layers} not divisible by pp_size={self.pp_size}"
            )

    @property
    def is_kv_source_replica(self) -> bool:
        """Whether this rank is the representative EP replica for its KV.
        Only ep_rank 0 sources KV; other EP replicas hold identical bytes."""
        return self.ep_rank == 0

    def layer_range(self) -> Tuple[int, int]:
        per = self.num_layers // self.pp_size
        return (self.pp_rank * per, (self.pp_rank + 1) * per)

    def head_range(self) -> Tuple[int, int]:
        per = self.num_heads // self.tp_size
        return (self.tp_rank * per, (self.tp_rank + 1) * per)

    def local_num_layers(self) -> int:
        lo, hi = self.layer_range()
        return hi - lo

    def local_num_heads(self) -> int:
        lo, hi = self.head_range()
        return hi - lo


@dataclass(frozen=True)
class ReshardTransfer:
    """One sub-block exchange between a (src, dst) rank pair.

    Global coords identify the intersection; the local-slice helpers
    convert to each side's buffer offsets. There is at most one transfer
    per (src, dst) pair (each owns a contiguous rectangle, so the
    intersection is a single rectangle).
    """

    src_rank: int
    dst_rank: int
    g_layer0: int
    g_layer1: int
    g_head0: int
    g_head1: int

    def src_layer_slice(self, src: KVShardLayout) -> slice:
        off = src.layer_range()[0]
        return slice(self.g_layer0 - off, self.g_layer1 - off)

    def src_head_slice(self, src: KVShardLayout) -> slice:
        off = src.head_range()[0]
        return slice(self.g_head0 - off, self.g_head1 - off)

    def dst_layer_slice(self, dst: KVShardLayout) -> slice:
        off = dst.layer_range()[0]
        return slice(self.g_layer0 - off, self.g_layer1 - off)

    def dst_head_slice(self, dst: KVShardLayout) -> slice:
        off = dst.head_range()[0]
        return slice(self.g_head0 - off, self.g_head1 - off)

    def tag(self, num_layers: int, num_heads: int, base: int = 0) -> int:
        """Deterministic, collision-free tag for this sub-block.

        There is one transfer per (src, dst) pair, so P2P matching by
        (src, dst, tag) is already unique; the coordinate encoding makes
        it robust if a pair ever carried more than one block."""
        return base + (self.g_layer0 * num_heads + self.g_head0)


def _intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None


def plan_kv_reshard(
    srcs: List[KVShardLayout], dsts: List[KVShardLayout]
) -> List[ReshardTransfer]:
    """Full reshard plan: every sub-block that must move src -> dst.

    Both sides compute the same plan from the same layouts and filter to
    their own rank (``transfers_for_src`` / ``transfers_for_dst``).
    Only the representative EP replica (``ep_rank == 0``) sources KV.
    """
    if srcs and dsts:
        if srcs[0].num_layers != dsts[0].num_layers or srcs[0].num_heads != dsts[0].num_heads:
            raise ValueError("src and dst describe different global models")

    transfers: List[ReshardTransfer] = []
    for d in dsts:
        dl, dh = d.layer_range(), d.head_range()
        for s in srcs:
            if not s.is_kv_source_replica:
                continue
            li = _intersect(s.layer_range(), dl)
            if li is None:
                continue
            hi = _intersect(s.head_range(), dh)
            if hi is None:
                continue
            transfers.append(
                ReshardTransfer(
                    src_rank=s.global_rank,
                    dst_rank=d.global_rank,
                    g_layer0=li[0],
                    g_layer1=li[1],
                    g_head0=hi[0],
                    g_head1=hi[1],
                )
            )
    return transfers


def transfers_for_dst(plan: List[ReshardTransfer], dst_rank: int) -> List[ReshardTransfer]:
    return [t for t in plan if t.dst_rank == dst_rank]


def transfers_for_src(plan: List[ReshardTransfer], src_rank: int) -> List[ReshardTransfer]:
    return [t for t in plan if t.src_rank == src_rank]


def is_matched(src: KVShardLayout, dst: KVShardLayout) -> bool:
    """Whether the two layouts are identical in the dims that affect KV
    sharding (TP heads + PP layers). EP is irrelevant to KV."""
    return (
        src.num_layers == dst.num_layers
        and src.num_heads == dst.num_heads
        and src.tp_size == dst.tp_size
        and src.pp_size == dst.pp_size
    )
