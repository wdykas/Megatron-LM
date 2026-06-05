# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous TP/PP reshard of Mamba conv/ssm state between prefill and
decode shard layouts (the Mamba analog of the attention KV reshard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from megatron.core.inference.disaggregation.reshard_ranges import intersect

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
class MambaTransfer:
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
) -> List[MambaTransfer]:
    """Plan the conv/ssm sub-block moves from the prefill (src) layouts to the
    decode (dst) layouts. One transfer per (src rank, dst rank, global layer,
    band) where both the layer ranges and the channel ranges overlap."""
    out: List[MambaTransfer] = []
    for s in src_layouts:
        s_lr = s.layer_range()
        for d in dst_layouts:
            layer_ov = intersect(s_lr, d.layer_range())
            if layer_ov is None:
                continue
            for band in (*_CONV_BANDS, "ssm"):
                g_total, s_size, s_off = s._band(band)
                _, d_size, d_off = d._band(band)
                s_glo = (s.tp_rank * s_size, s.tp_rank * s_size + s_size)
                d_glo = (d.tp_rank * d_size, d.tp_rank * d_size + d_size)
                chan_ov = intersect(s_glo, d_glo)
                if chan_ov is None:
                    continue
                lo, hi = chan_ov
                for g in range(layer_ov[0], layer_ov[1]):
                    out.append(
                        MambaTransfer(
                            src_rank=s.global_rank, dst_rank=d.global_rank, band=band,
                            global_layer=g, src_layer=g - s.layer_start, dst_layer=g - d.layer_start,
                            src_lo=s_off + (lo - s_glo[0]), src_hi=s_off + (hi - s_glo[0]),
                            dst_lo=d_off + (lo - d_glo[0]), dst_hi=d_off + (hi - d_glo[0]),
                        )
                    )
    return out


def transfers_for_src(plan: List[MambaTransfer], src_rank: int) -> List[MambaTransfer]:
    return [t for t in plan if t.src_rank == src_rank]


def transfers_for_dst(plan: List[MambaTransfer], dst_rank: int) -> List[MambaTransfer]:
    return [t for t in plan if t.dst_rank == dst_rank]


def apply_conv_transfer(t: MambaTransfer, src_conv, dst_conv) -> None:
    """Copy a conv sub-block (test/in-memory helper). conv tensors are
    ``(num_layers, conv_dim_local, d_conv)``; the band slices the channel axis."""
    dst_conv[t.dst_layer, t.dst_lo:t.dst_hi, :] = src_conv[t.src_layer, t.src_lo:t.src_hi, :]


def apply_ssm_transfer(t: MambaTransfer, src_ssm, dst_ssm) -> None:
    """Copy an ssm sub-block (test/in-memory helper). ssm tensors are
    ``(num_layers, nheads_local, headdim, d_state)``; the band slices heads."""
    dst_ssm[t.dst_layer, t.dst_lo:t.dst_hi, :, :] = src_ssm[t.src_layer, t.src_lo:t.src_hi, :, :]
