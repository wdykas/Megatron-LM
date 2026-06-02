# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shard spec parsing (incl. role + dp), role-layout validation, the
pg_collection->layout factory, and deterministic decode assignment.

The pure parsing/partition logic needs no runtime. The layout factory
reads sizes/ranks off a ProcessGroupCollection via get_pg_size/
get_pg_rank; we fake a collection and monkeypatch those helpers (no
torch.distributed init) to assert the layout maps each group to the
right KVShardLayout field.
"""

import pytest

from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    normalize_shard_specs,
    parse_inference_shards_spec,
)


def test_shard_spec_objects_match_string_parsing():
    objs = [InferenceShardSpec(tp=2, role="prefill"),
            InferenceShardSpec(tp=1, dp=2, role="decode")]
    assert normalize_shard_specs(objs, 4) == parse_inference_shards_spec(
        "tp=2,role=prefill+tp=1,dp=2,role=decode", 4
    )
    # expt_tp defaults to tp; raw dicts also accepted
    assert InferenceShardSpec(tp=4).to_dict()["expt_tp"] == 4
    assert normalize_shard_specs([{"tp": 1, "role": "prefill"}, {"tp": 1, "role": "decode"}], 2)
    # bad role rejected at construction
    with pytest.raises(ValueError):
        InferenceShardSpec(tp=1, role="both")


# --------------------------------------------------------------------------
# spec parsing
# --------------------------------------------------------------------------
def test_parse_defaults_and_dp_and_role():
    specs = parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=4)
    # parser returns InferenceShardSpec with defaults filled (expt_tp -> tp).
    assert specs[0] == InferenceShardSpec(tp=2, role="prefill")
    assert specs[1] == InferenceShardSpec(tp=1, dp=2, role="decode")
    # dict form (serialization / external consumers) carries the resolved keys.
    assert specs[0].to_dict() == {"tp": 2, "pp": 1, "ep": 1, "dp": 1, "expt_tp": 2, "role": "prefill"}


def test_parse_partitions_world_with_dp():
    # world must equal sum(tp*pp*dp): 2 + (1*1*2) = 4
    parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=4)
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=2,role=prefill+tp=1,dp=2,role=decode", world_size=5)


def test_parse_rejects_bad_role_and_unknown_key():
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=1,role=both", world_size=1)
    with pytest.raises(AssertionError):
        parse_inference_shards_spec("tp=1,foo=2", world_size=1)


def test_plus_and_semicolon_separators_equivalent():
    a = parse_inference_shards_spec("tp=2,role=prefill+tp=1,role=decode", world_size=3)
    b = parse_inference_shards_spec("tp=2,role=prefill;tp=1,role=decode", world_size=3)
    assert a == b


# --------------------------------------------------------------------------
# role-layout validation + decode-instance count
# --------------------------------------------------------------------------
def test_validate_specs_counts_decode_instances():
    from megatron.core.inference.disaggregation.orchestration import _validate_disagg_specs

    # one prefill + a decode pool: a dp=2 shard is 2 instances, plus 1 more = 3
    specs = parse_inference_shards_spec(
        "tp=2,role=prefill+tp=1,dp=2,role=decode+tp=2,role=decode", world_size=6
    )
    assert _validate_disagg_specs(specs) == 3


def test_validate_specs_rejects_multi_instance_prefill_and_untagged():
    from megatron.core.inference.disaggregation.orchestration import _validate_disagg_specs

    with pytest.raises(AssertionError):  # dp>1 prefill = multiple prefill instances
        _validate_disagg_specs(parse_inference_shards_spec(
            "tp=1,dp=2,role=prefill+tp=1,role=decode", world_size=3))
    with pytest.raises(AssertionError):  # an untagged shard
        _validate_disagg_specs(parse_inference_shards_spec(
            "tp=1,role=prefill+tp=1", world_size=2))
    with pytest.raises(AssertionError):  # no decode
        _validate_disagg_specs(parse_inference_shards_spec(
            "tp=1,role=prefill", world_size=1))


# --------------------------------------------------------------------------
# pg_collection -> KVShardLayout
# --------------------------------------------------------------------------
class _FakeGroup:
    pass


class _FakePG:
    def __init__(self, tp, pp, ep, expt_tp):
        self.tp, self.pp, self.ep, self.expt_tp = tp, pp, ep, expt_tp


def test_layout_from_pg_collection(monkeypatch):
    import megatron.core.utils as mcu
    from megatron.core.inference.disaggregation.orchestration import layout_from_pg_collection

    sizes, ranks = {}, {}
    monkeypatch.setattr(mcu, "get_pg_size", lambda g=None: sizes.get(id(g), 1))
    monkeypatch.setattr(mcu, "get_pg_rank", lambda g=None: ranks.get(id(g), 0))
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 5)

    tp_g, pp_g, ep_g, etp_g = _FakeGroup(), _FakeGroup(), _FakeGroup(), _FakeGroup()
    for g, (s, r) in [(tp_g, (4, 1)), (pp_g, (2, 0)), (ep_g, (2, 1)), (etp_g, (1, 0))]:
        sizes[id(g)], ranks[id(g)] = s, r
    lay = layout_from_pg_collection(_FakePG(tp_g, pp_g, ep_g, etp_g), num_layers=8, num_heads=16)

    assert (lay.tp_size, lay.tp_rank) == (4, 1)
    assert (lay.pp_size, lay.pp_rank) == (2, 0)
    assert (lay.ep_size, lay.ep_rank) == (2, 1)
    assert (lay.etp_size, lay.etp_rank) == (1, 0)
    assert lay.global_rank == 5
    assert lay.head_range() == (4, 8)      # attention TP4 -> heads [4,8)
    assert lay.layer_range() == (0, 4)     # PP2 rank0 -> layers [0,4)
    assert lay.kv_shard_key() == (1, 0)    # EP/ETP are replica dims


# --------------------------------------------------------------------------
# decode assignment (deterministic, disjoint partition across instances)
# --------------------------------------------------------------------------
def test_decode_assignment_is_deterministic_and_partitions_the_pool():
    from megatron.core.inference.disaggregation.disagg_coordinator import DisaggCoordinator
    from megatron.core.inference.disaggregation.kv_router import DecodeTarget, RequestInfo
    from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout

    # two decode instances of one dp=2 shard (distinct replica_ids per dp rank)
    targets = [
        DecodeTarget("decode_s1_dp0", [KVShardLayout(8, 8, 1, 0, 1, 0, 2)]),
        DecodeTarget("decode_s1_dp1", [KVShardLayout(8, 8, 1, 0, 1, 0, 3)]),
    ]

    def coord_for(replica_id):
        c = DisaggCoordinator.__new__(DisaggCoordinator)  # skip __init__ (needs backend)
        c.role = "decode"
        c.router_name = "sticky"
        c.decode_targets = targets
        c.replica_id = replica_id
        return c

    infos = [RequestInfo(i) for i in range(20)]
    a = coord_for("decode_s1_dp0").assigned_request_ids(infos)
    b = coord_for("decode_s1_dp1").assigned_request_ids(infos)
    assert set(a).isdisjoint(b)
    assert sorted(a + b) == list(range(20))                       # every request covered
    assert coord_for("decode_s1_dp0").assigned_request_ids(infos) == a   # deterministic
