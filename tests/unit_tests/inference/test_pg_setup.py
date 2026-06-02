# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Non-collocated disagg parallel-config + pg_collection->layout factory.

The pure rank-partition math (DisaggParallelConfig) needs no runtime. The
layout factory reads sizes/ranks off a ProcessGroupCollection via
get_pg_size/get_pg_rank; we fake a collection and monkeypatch those
helpers (no torch.distributed init) to assert the layout maps each group
to the right KVShardLayout field and derives correct head/layer ranges.
"""

import pytest

from megatron.core.inference.disaggregation.pg_setup import (
    DECODE,
    IDLE,
    PREFILL,
    DisaggParallelConfig,
    layout_from_pg_collection,
)


def test_homogeneous_worlds_and_roles():
    cfg = DisaggParallelConfig(prefill_tp=1, decode_tp=1)
    assert (cfg.prefill_world, cfg.decode_world, cfg.required_world) == (1, 1, 2)
    assert [cfg.role_of(r) for r in range(3)] == [PREFILL, DECODE, IDLE]


def test_hetero_tp2_to_tp1_partition():
    cfg = DisaggParallelConfig(prefill_tp=2, prefill_pp=1, decode_tp=1, decode_pp=1)
    assert (cfg.prefill_world, cfg.decode_world, cfg.required_world) == (2, 1, 3)
    # ranks 0,1 = prefill; rank 2 = decode
    assert [cfg.role_of(r) for r in range(3)] == [PREFILL, PREFILL, DECODE]


def test_pp_and_expert_dims_count_only_attention_for_world():
    # world per replica is attention tp*pp; EP/ETP re-factor the same ranks.
    cfg = DisaggParallelConfig(
        prefill_tp=2, prefill_pp=2, decode_tp=2, decode_pp=1,
        prefill_ep=2, prefill_etp=1, decode_ep=1, decode_etp=2,
    )
    assert cfg.prefill_world == 4 and cfg.decode_world == 2 and cfg.required_world == 6


class _FakeGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank


class _FakePG:
    """Minimal ProcessGroupCollection stand-in."""

    def __init__(self, tp, pp, ep, expt_tp):
        self.tp, self.pp, self.ep, self.expt_tp = tp, pp, ep, expt_tp


def test_layout_from_pg_collection(monkeypatch):
    import megatron.core.utils as mcu

    # group -> (size, rank); fake the dist-backed helpers
    sizes = {}
    ranks = {}

    def fake_size(g=None):
        return sizes.get(id(g), 1)

    def fake_rank(g=None):
        return ranks.get(id(g), 0)

    monkeypatch.setattr(mcu, "get_pg_size", fake_size)
    monkeypatch.setattr(mcu, "get_pg_rank", fake_rank)

    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 5)

    tp_g, pp_g, ep_g, etp_g = _FakeGroup(4, 1), _FakeGroup(2, 0), _FakeGroup(2, 1), _FakeGroup(1, 0)
    for g, (s, r) in [(tp_g, (4, 1)), (pp_g, (2, 0)), (ep_g, (2, 1)), (etp_g, (1, 0))]:
        sizes[id(g)] = s
        ranks[id(g)] = r
    pg = _FakePG(tp_g, pp_g, ep_g, etp_g)

    lay = layout_from_pg_collection(pg, num_layers=8, num_heads=16)
    assert (lay.tp_size, lay.tp_rank) == (4, 1)
    assert (lay.pp_size, lay.pp_rank) == (2, 0)
    assert (lay.ep_size, lay.ep_rank) == (2, 1)
    assert (lay.etp_size, lay.etp_rank) == (1, 0)
    assert lay.global_rank == 5
    # attention TP=4 -> heads [4,8); PP=2 rank0 -> layers [0,4)
    assert lay.head_range() == (4, 8)
    assert lay.layer_range() == (0, 4)
    # EP/ETP are replica dims: the shard key is purely (tp_rank, pp_rank)
    assert lay.kv_shard_key() == (1, 0)
