# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shard spec parsing/validation (CPU; no torch.distributed)."""

import pytest

from megatron.core.inference.shards_spec import (
    InferenceShardSpec,
    normalize_shard_specs,
    parse_inference_shards_spec,
)


def test_shard_spec_objects_match_string_parsing():
    objs = [InferenceShardSpec(tp=2, role="prefill"), InferenceShardSpec(tp=1, dp=2, role="decode")]
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
    assert specs[0].to_dict() == {
        "tp": 2,
        "pp": 1,
        "ep": 1,
        "dp": 1,
        "expt_tp": 2,
        "role": "prefill",
    }


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


def test_cp_accepted_only_when_one():
    # cp is a recognized key (clear error, not "unknown key") but must be 1:
    # inference shards don't context-parallelize.
    assert parse_inference_shards_spec("tp=2,cp=1", world_size=2) == [
        InferenceShardSpec(tp=2, cp=1)
    ]
    with pytest.raises(ValueError):
        InferenceShardSpec(tp=1, cp=2)
    with pytest.raises(ValueError):
        parse_inference_shards_spec("tp=1,cp=2", world_size=1)
# role-layout validation + decode-instance count
# --------------------------------------------------------------------------
def test_validate_specs_counts_decode_instances():
    from megatron.core.inference.disaggregation.coordinator_setup import _validate_disagg_specs

    # one prefill + a decode pool: a dp=2 shard is 2 instances, plus 1 more = 3
    specs = parse_inference_shards_spec(
        "tp=2,role=prefill+tp=1,dp=2,role=decode+tp=2,role=decode", world_size=6
    )
    assert _validate_disagg_specs(specs) == 3


def test_validate_specs_allows_multi_prefill_rejects_untagged_and_no_decode():
    from megatron.core.inference.disaggregation.coordinator_setup import _validate_disagg_specs

    # Multiple prefill instances are allowed (dp>1 prefill = a prefill pool).
    assert _validate_disagg_specs(parse_inference_shards_spec(
        "tp=1,dp=2,role=prefill+tp=1,role=decode", world_size=3)) == 1
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
    import megatron.core.inference.disaggregation.coordinator_setup as orch
    from megatron.core.inference.disaggregation.coordinator_setup import layout_from_pg_collection

    sizes, ranks = {}, {}
    # Patch the names where layout_from_pg_collection looks them up (it imports
    # get_pg_size/get_pg_rank/dist at module top).
    monkeypatch.setattr(orch, "get_pg_size", lambda g=None: sizes.get(id(g), 1))
    monkeypatch.setattr(orch, "get_pg_rank", lambda g=None: ranks.get(id(g), 0))
    monkeypatch.setattr(orch.dist, "get_rank", lambda *a, **k: 5)

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
