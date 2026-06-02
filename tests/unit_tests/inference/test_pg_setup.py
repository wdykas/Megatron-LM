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
    DisaggTopology,
    ReplicaSpec,
    layout_from_pg_collection,
    parse_replica_spec,
)


def test_parse_replica_spec_defaults_and_full():
    assert parse_replica_spec("tp=2") == ReplicaSpec(tp=2, pp=1, ep=1, etp=1)
    assert parse_replica_spec("tp=4,pp=2,ep=2,etp=1") == ReplicaSpec(tp=4, pp=2, ep=2, etp=1)
    assert parse_replica_spec(" tp=2 , pp=3 ") == ReplicaSpec(tp=2, pp=3)
    with pytest.raises(ValueError):
        parse_replica_spec("pp=2")            # tp required
    with pytest.raises(ValueError):
        parse_replica_spec("tp=2,dp=2")       # unknown key
    with pytest.raises(ValueError):
        parse_replica_spec("tp2")             # not key=value


def test_topology_homogeneous_worlds_and_roles():
    topo = DisaggTopology.from_specs(["tp=1"], ["tp=1"])
    assert (topo.prefill_world, topo.decode_world, topo.required_world) == (1, 1, 2)
    assert topo.num_decode_replicas == 1
    assert [topo.replica_at(r)[0] for r in range(3)] == [PREFILL, DECODE, IDLE]


def test_topology_hetero_tp2_to_tp1_partition():
    topo = DisaggTopology.from_specs(["tp=2,pp=1"], ["tp=1,pp=1"])
    assert (topo.prefill_world, topo.decode_world, topo.required_world) == (2, 1, 3)
    # ranks 0,1 = prefill; rank 2 = decode0
    roles = [topo.replica_at(r) for r in range(3)]
    assert [x[0] for x in roles] == [PREFILL, PREFILL, DECODE]
    assert roles[2][1] == "decode0"


def test_topology_heterogeneous_decode_pool():
    # prefill TP2 + a decode pool of TWO different-parallelism replicas
    topo = DisaggTopology.from_specs(["tp=2"], ["tp=1", "tp=2,ep=2"])
    assert topo.num_decode_replicas == 2
    # world = prefill 2 + decode0 (TP1=1) + decode1 (TP2=2) = 5
    assert (topo.prefill_world, topo.decode_world, topo.required_world) == (2, 3, 5)
    placements = [topo.replica_at(r) for r in range(5)]
    assert [p[1] for p in placements] == ["prefill", "prefill", "decode0", "decode1", "decode1"]
    # rank offsets per replica
    assert [(rid, off) for rid, _, off in topo.meshes()] == [
        ("prefill", 0), ("decode0", 2), ("decode1", 3)
    ]


def test_topology_requires_single_prefill_and_some_decode():
    with pytest.raises(ValueError):
        DisaggTopology.from_specs(["tp=1", "tp=2"], ["tp=1"])   # >1 prefill
    with pytest.raises(ValueError):
        DisaggTopology.from_specs(["tp=1"], [])                 # no decode


def test_expert_dims_count_only_attention_for_world():
    # world per replica is attention tp*pp; EP/ETP re-factor the same ranks.
    topo = DisaggTopology.from_specs(["tp=2,pp=2,ep=2"], ["tp=2,etp=2"])
    assert topo.prefill_world == 4 and topo.decode_world == 2 and topo.required_world == 6


def test_decode_assignment_is_deterministic_and_partitions_the_pool():
    """Each decode replica replays the (deterministic) router to learn its
    share; the shares must be disjoint and cover every request."""
    from megatron.core.inference.disaggregation.disagg_coordinator import DisaggCoordinator
    from megatron.core.inference.disaggregation.kv_router import DecodeTarget, RequestInfo
    from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout

    targets = [
        DecodeTarget("decode0", [KVShardLayout(8, 8, 1, 0, 1, 0, 2)]),
        DecodeTarget("decode1", [KVShardLayout(8, 8, 2, 0, 1, 0, 3),
                                 KVShardLayout(8, 8, 2, 1, 1, 0, 4)]),
    ]

    def coord_for(replica_id):
        # bypass __init__ (which needs a transport backend); we only test
        # the pure router-replay assignment logic.
        c = DisaggCoordinator.__new__(DisaggCoordinator)
        c.role = "decode"
        c.router_name = "sticky"
        c.decode_targets = targets
        c.replica_id = replica_id
        return c

    infos = [RequestInfo(i) for i in range(20)]
    a = coord_for("decode0").assigned_request_ids(infos)
    b = coord_for("decode1").assigned_request_ids(infos)
    assert set(a).isdisjoint(b)
    assert sorted(a + b) == list(range(20))          # every request covered once
    assert coord_for("decode0").assigned_request_ids(infos) == a   # deterministic


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
