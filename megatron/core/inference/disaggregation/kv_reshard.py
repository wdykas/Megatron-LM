# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TP/PP/EP/ETP KV-shard layouts and the range-intersection reshard planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from megatron.core.inference.disaggregation.utils import intersect


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
    # Expert dimensions. KV-replica dimensions only: they shard the MoE
    # expert weights, never the attention KV cache, so they don't affect
    # head_range/layer_range -- only representative (source) selection.
    ep_size: int = 1
    ep_rank: int = 0
    etp_size: int = 1
    etp_rank: int = 0
    # Optional explicit PP layer window for this stage. When None, an even split
    # of num_layers across pp_size is assumed -- correct for pure-attention
    # models. Models that do NOT split attention layers evenly across PP stages
    # (e.g. hybrid Mamba+attention) must pass an explicit (layer_start,
    # num_local_layers); the even-split default would otherwise map the wrong
    # global layer indices.
    layer_start: Optional[int] = None
    num_local_layers: Optional[int] = None

    def __post_init__(self) -> None:
        # TP must divide heads (the head split is always even).
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} not divisible by tp_size={self.tp_size}"
            )
        # Only the even-split path requires PP to divide layers; an explicit
        # window may be uneven across stages.
        if self.layer_start is None and self.num_layers % self.pp_size != 0:
            raise ValueError(
                f"num_layers={self.num_layers} not divisible by pp_size={self.pp_size}; "
                "pass an explicit (layer_start, num_local_layers) for uneven PP splits"
            )

    def kv_shard_key(self) -> Tuple[int, int]:
        """The attention shard this rank holds: ``(tp_rank, pp_rank)``.
        Ranks sharing a key hold identical KV (EP/ETP replicas of it)."""
        return (self.tp_rank, self.pp_rank)

    def layer_range(self) -> Tuple[int, int]:
        if self.layer_start is not None:
            n = self.num_local_layers
            if n is None:
                n = self.num_layers // self.pp_size
            return (self.layer_start, self.layer_start + n)
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


def plan_kv_reshard(
    srcs: List[KVShardLayout], dsts: List[KVShardLayout]
) -> List[ReshardTransfer]:
    """Full reshard plan: every sub-block that must move src -> dst.

    Both sides compute the same plan from the same layouts and filter to
    their own rank (``transfers_for_src`` / ``transfers_for_dst``).

    KV is replicated across the EP and ETP dimensions, so each attention
    shard ``(tp_rank, pp_rank)`` may be held by several source ranks. We
    source each shard from exactly one of them -- the smallest
    ``global_rank`` -- which avoids duplicate sends and is independent of
    how EP/ETP map onto ranks.
    """
    if srcs and dsts:
        if srcs[0].num_layers != dsts[0].num_layers or srcs[0].num_heads != dsts[0].num_heads:
            raise ValueError("src and dst describe different global models")

    # One representative source rank per attention shard (dedupe EP/ETP
    # replicas that hold identical KV).
    rep_rank: dict = {}
    for s in srcs:
        key = s.kv_shard_key()
        if key not in rep_rank or s.global_rank < rep_rank[key]:
            rep_rank[key] = s.global_rank
    source_ranks = set(rep_rank.values())

    transfers: List[ReshardTransfer] = []
    for d in dsts:
        dl, dh = d.layer_range(), d.head_range()
        for s in srcs:
            if s.global_rank not in source_ranks:
                continue
            li = intersect(s.layer_range(), dl)
            if li is None:
                continue
            hi = intersect(s.head_range(), dh)
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
