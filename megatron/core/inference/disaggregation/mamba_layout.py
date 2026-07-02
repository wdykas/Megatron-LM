# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-rank Mamba shard identity, used to pair identical shards for the
snapshot hand-off (see matching_mamba_peer in kv_transfer_push.py)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MambaStateDims:
    """The model's global (unsharded) Mamba structural dims.

    Carried as one unit so there is a single source and the dims cannot drift
    apart. The producer should read them from the model config (e.g.
    ngroups = config.mamba_num_groups) rather than deriving them from tensor
    shapes. TP shards nheads/ngroups; the rest are unsharded.
    """

    nheads: int
    headdim: int
    d_state: int
    ngroups: int
    d_conv: int


@dataclass(frozen=True)
class MambaShardLayout:
    """One rank's Mamba-state ownership: which global layers and TP rank,
    plus the model's structural dims."""

    global_rank: int
    tp_size: int
    tp_rank: int
    layer_start: int  # global Mamba-layer index of this rank's first layer
    num_layers: int  # Mamba layers held locally (this PP stage)
    dims: MambaStateDims

    def __post_init__(self) -> None:
        # Wire reconstruction (MambaShardLayout(**dict)) hands dims as a plain
        # dict; coerce it back to MambaStateDims.
        if isinstance(self.dims, dict):
            object.__setattr__(self, "dims", MambaStateDims(**self.dims))
        # TP shards heads and groups; both must divide evenly or the local
        # conv/ssm shard widths are wrong.
        if self.dims.nheads % self.tp_size != 0:
            raise ValueError(f"nheads={self.dims.nheads} not divisible by tp_size={self.tp_size}")
        if self.dims.ngroups % self.tp_size != 0:
            raise ValueError(f"ngroups={self.dims.ngroups} not divisible by tp_size={self.tp_size}")
