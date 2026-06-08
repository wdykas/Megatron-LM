# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous TP/PP reshard of Mamba conv/ssm state between prefill and
decode shard layouts (the Mamba analog of the attention KV reshard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from megatron.core.inference.disaggregation.utils import intersect

# Channel bands of a Mamba layer's state, in the order the conv state
# concatenates them on its channel axis (x, B, C); ssm is the head axis.
# (name, lives_in_conv). conv bands share one tensor; ssm is its own tensor.
_CONV_BANDS = ("x", "B", "C")


@dataclass(frozen=True)
class MambaShardLayout:
    """One rank's Mamba-state ownership: which global layers + TP rank.

    Global structural dims (``nheads``, ``headdim``, ``d_state``, ``ngroups``,
    ``d_conv``) are unsharded; per-rank locals follow by dividing by ``tp_size``.
    """

    global_rank: int
    tp_size: int
    tp_rank: int
    layer_start: int   # global Mamba-layer index of this rank's first layer
    num_layers: int    # Mamba layers held locally (this PP stage)
    nheads: int        # global
    headdim: int
    d_state: int
    ngroups: int       # global
    d_conv: int

    def __post_init__(self) -> None:
        # TP shards heads and groups; both must divide evenly or the local
        # conv/ssm band sizes truncate to the wrong (or zero) width silently.
        if self.nheads % self.tp_size != 0:
            raise ValueError(f"nheads={self.nheads} not divisible by tp_size={self.tp_size}")
        if self.ngroups % self.tp_size != 0:
            raise ValueError(f"ngroups={self.ngroups} not divisible by tp_size={self.tp_size}")

    def mamba_shard_key(self) -> Tuple[int, int]:
        """The Mamba shard this rank holds: ``(tp_rank, layer_start)``. Ranks
        sharing a key hold identical state (e.g. EP/DP replicas of it)."""
        return (self.tp_rank, self.layer_start)

    @property
    def d_inner(self) -> int:
        return self.nheads * self.headdim

    @property
    def nheads_local(self) -> int:
        return self.nheads // self.tp_size

    @property
    def d_inner_local(self) -> int:
        return self.d_inner // self.tp_size

    @property
    def ngroups_local(self) -> int:
        return self.ngroups // self.tp_size

    @property
    def conv_dim_local(self) -> int:
        return self.d_inner_local + 2 * self.ngroups_local * self.d_state

    def layer_range(self) -> Tuple[int, int]:
        return (self.layer_start, self.layer_start + self.num_layers)

    def _band(self, name: str) -> Tuple[int, int, int]:
        """``(global_total, local_size, conv_local_offset)`` for a band.

        ``conv_local_offset`` is the band's start on the local conv channel
        axis; for the ``ssm`` (head) band it is the start on the local head
        axis (always 0, heads are the whole tensor)."""
        if name == "x":
            g = self.d_inner
            return g, self.d_inner_local, 0
        if name == "B":
            g = self.ngroups * self.d_state
            return g, self.ngroups_local * self.d_state, self.d_inner_local
        if name == "C":
            g = self.ngroups * self.d_state
            return g, self.ngroups_local * self.d_state, self.d_inner_local + self.ngroups_local * self.d_state
        if name == "ssm":
            return self.nheads, self.nheads_local, 0
        raise KeyError(name)


@dataclass(frozen=True)
class MambaReshardTransfer:
    """One sub-block move for the reshard.

    ``band`` is ``"x"``/``"B"``/``"C"`` (conv channel axis) or ``"ssm"`` (head
    axis). ``src_layer``/``dst_layer`` are local layer indices on each side;
    ``*_lo``/``*_hi`` are the local channel/head slice bounds.
    """

    src_rank: int
    dst_rank: int
    band: str
    global_layer: int
    src_layer: int
    dst_layer: int
    src_lo: int
    src_hi: int
    dst_lo: int
    dst_hi: int

    @property
    def is_conv(self) -> bool:
        return self.band in _CONV_BANDS


def plan_mamba_reshard(
    src_layouts: List[MambaShardLayout], dst_layouts: List[MambaShardLayout]
) -> List[MambaReshardTransfer]:
    """Plan the conv/ssm sub-block moves from the prefill (src) layouts to the
    decode (dst) layouts. One transfer per (src rank, dst rank, global layer,
    band) where both the layer ranges and the channel ranges overlap."""
    # Dedupe replica sources: ranks sharing (tp_rank, layer_start) hold identical
    # Mamba state (e.g. EP/DP replicas), so source each shard from exactly one of
    # them -- the smallest global_rank -- to avoid duplicate sends.
    rep_rank: dict = {}
    for s in src_layouts:
        key = s.mamba_shard_key()
        if key not in rep_rank or s.global_rank < rep_rank[key]:
            rep_rank[key] = s.global_rank
    source_ranks = set(rep_rank.values())

    out: List[MambaReshardTransfer] = []
    for s in src_layouts:
        if s.global_rank not in source_ranks:
            continue
        s_lr = s.layer_range()
        for d in dst_layouts:
            layer_ov = intersect(s_lr, d.layer_range())
            if layer_ov is None:
                continue
            for band in (*_CONV_BANDS, "ssm"):
                _, s_size, s_off = s._band(band)
                _, d_size, d_off = d._band(band)
                s_glo = (s.tp_rank * s_size, s.tp_rank * s_size + s_size)
                d_glo = (d.tp_rank * d_size, d.tp_rank * d_size + d_size)
                chan_ov = intersect(s_glo, d_glo)
                if chan_ov is None:
                    continue
                lo, hi = chan_ov
                for g in range(layer_ov[0], layer_ov[1]):
                    out.append(
                        MambaReshardTransfer(
                            src_rank=s.global_rank, dst_rank=d.global_rank, band=band,
                            global_layer=g, src_layer=g - s.layer_start, dst_layer=g - d.layer_start,
                            src_lo=s_off + (lo - s_glo[0]), src_hi=s_off + (hi - s_glo[0]),
                            dst_lo=d_off + (lo - d_glo[0]), dst_hi=d_off + (hi - d_glo[0]),
                        )
                    )
    return out


def transfers_for_src(plan: List[MambaReshardTransfer], src_rank: int) -> List[MambaReshardTransfer]:
    return [t for t in plan if t.src_rank == src_rank]


def transfers_for_dst(plan: List[MambaReshardTransfer], dst_rank: int) -> List[MambaReshardTransfer]:
    return [t for t in plan if t.dst_rank == dst_rank]
