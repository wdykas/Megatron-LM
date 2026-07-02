# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MambaShardLayout wire round-trip, validation, and the disagg snapshot
pairing gate (identical shards only)."""

import dataclasses

import pytest

from megatron.core.inference.disaggregation.mamba_layout import MambaShardLayout, MambaStateDims

NHEADS, HEADDIM, DSTATE, NGROUPS, DCONV = 8, 4, 2, 2, 3
M = 4  # global Mamba layers


def _layouts(tp, pp, base=0):
    """One MambaShardLayout per rank for a (tp, pp) instance; rank = base+p*tp+r.
    PP splits the M layers evenly (contiguous per stage). `base` offsets the
    global ranks so a prefill instance and a decode instance occupy disjoint
    rank windows."""
    per = M // pp
    out = {}
    for p in range(pp):
        for r in range(tp):
            rank = base + p * tp + r
            out[rank] = MambaShardLayout(
                global_rank=rank,
                tp_size=tp,
                tp_rank=r,
                layer_start=p * per,
                num_layers=per,
                dims=MambaStateDims(
                    nheads=NHEADS, headdim=HEADDIM, d_state=DSTATE, ngroups=NGROUPS, d_conv=DCONV
                ),
            )
    return out


def test_mamba_rejects_indivisible_groups():
    """ngroups < tp_size would shard the B/C groups to zero width; reject it
    up front."""
    with pytest.raises(ValueError):
        MambaShardLayout(
            global_rank=0,
            tp_size=4,
            tp_rank=0,
            layer_start=0,
            num_layers=1,
            dims=MambaStateDims(
                nheads=8, headdim=HEADDIM, d_state=DSTATE, ngroups=2, d_conv=DCONV
            ),
        )


def test_layout_wire_roundtrip():
    """Layouts cross the coordinator as plain dicts (asdict) and are rebuilt
    via MambaShardLayout(**dict); the nested dims dict must coerce back to
    MambaStateDims."""
    lay = MambaShardLayout(
        global_rank=1,
        tp_size=2,
        tp_rank=1,
        layer_start=0,
        num_layers=M,
        dims=MambaStateDims(
            nheads=NHEADS, headdim=HEADDIM, d_state=DSTATE, ngroups=NGROUPS, d_conv=DCONV
        ),
    )
    rebuilt = MambaShardLayout(**dataclasses.asdict(lay))
    assert rebuilt == lay
    assert rebuilt.dims == lay.dims


def test_matching_mamba_peer_identical_shard_only():
    """Snapshot sends pair identical Mamba shards (same tp_rank/layer range/
    dims) and skip hetero remaps; the whole-slot snapshot tensors are only
    byte-compatible between identical shards."""
    from megatron.core.inference.disaggregation.kv_transfer_push import matching_mamba_peer

    src = list(_layouts(2, 1).values())  # prefill TP2 -> ranks {0,1}
    dst_same = list(_layouts(2, 1, base=2).values())  # decode TP2 -> ranks {2,3}
    dst_tp1 = list(_layouts(1, 1, base=2).values())  # decode TP1 -> rank {2}

    # Identical shards pair one-to-one (rank i <-> base+i) and symmetrically.
    for me, expect in zip(src, dst_same):
        peer = matching_mamba_peer(me, dst_same)
        assert peer is expect
        assert matching_mamba_peer(peer, src) is me
    # A hetero TP remap has no identical peer: snapshots are skipped.
    for me in src:
        assert matching_mamba_peer(me, dst_tp1) is None
    # Empty / absent peer list (non-hybrid instance).
    assert matching_mamba_peer(src[0], []) is None
    assert matching_mamba_peer(src[0], None) is None
