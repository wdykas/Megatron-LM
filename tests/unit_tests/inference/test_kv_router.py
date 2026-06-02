# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pluggable decode-router policies + composition with resharding."""

import pytest
import torch

from megatron.core.inference.kv_shard_layout import KVShardLayout
from megatron.core.inference.kv_router import (
    DecodeRouter,
    DecodeTarget,
    LeastLoadedRouter,
    PromptLengthTieredRouter,
    RequestInfo,
    RoundRobinRouter,
    StickyHashRouter,
    make_router,
    register_router,
    route_and_plan,
)

L, Hh = 8, 8


def _layouts(tp, pp, base_rank):
    out, r = [], base_rank
    for p in range(pp):
        for t in range(tp):
            out.append(KVShardLayout(L, Hh, tp, t, pp, p, r))
            r += 1
    return out


def _targets():
    # heterogeneous decode pool: a small TP2 replica and a big TP4 replica
    return [
        DecodeTarget("small", _layouts(2, 1, 100), max_prompt_tokens=1024),
        DecodeTarget("big", _layouts(4, 2, 200), max_prompt_tokens=None),
    ]


def test_round_robin_cycles():
    r = RoundRobinRouter(_targets())
    ids = [r.select(RequestInfo(i)).target_id for i in range(4)]
    assert ids == ["small", "big", "small", "big"]


def test_sticky_is_deterministic_and_spreads():
    r = StickyHashRouter(_targets())
    # same request -> same replica (prefix-cache affinity)
    assert r.select(RequestInfo(42)).target_id == r.select(RequestInfo(42)).target_id
    # both replicas get used across many ids
    seen = {r.select(RequestInfo(i)).target_id for i in range(50)}
    assert seen == {"small", "big"}


def test_least_loaded_tracks_inflight():
    r = LeastLoadedRouter(_targets())
    a = r.select(RequestInfo(0)); r.on_admit(RequestInfo(0), a)   # load small=1
    b = r.select(RequestInfo(1)); r.on_admit(RequestInfo(1), b)   # -> big=1
    assert {a.target_id, b.target_id} == {"small", "big"}
    # both loaded equally -> tie goes to first (small); admit it
    c = r.select(RequestInfo(2)); r.on_admit(RequestInfo(2), c)
    assert c.target_id == "small"  # small=2, big=1 now
    d = r.select(RequestInfo(3))
    assert d.target_id == "big"    # least loaded
    # completing a small frees capacity
    r.on_complete(RequestInfo(0), a)   # small back to 1
    # now small(1) and big(1) tie -> small
    assert r.select(RequestInfo(4)).target_id == "small"


def test_prompt_length_tiered():
    r = PromptLengthTieredRouter(_targets())
    assert r.select(RequestInfo(0, num_prompt_tokens=500)).target_id == "small"   # fits 1024
    assert r.select(RequestInfo(1, num_prompt_tokens=4096)).target_id == "big"    # overflow -> unbounded tier


def test_registry_and_custom_router():
    r = make_router("round_robin", _targets())
    assert isinstance(r, RoundRobinRouter)
    with pytest.raises(KeyError):
        make_router("nope", _targets())

    class AlwaysBig(DecodeRouter):
        def select(self, request):
            return self._by_id["big"]

    register_router("always_big", AlwaysBig)
    rb = make_router("always_big", _targets())
    assert rb.select(RequestInfo(7)).target_id == "big"


def test_route_and_plan_reshards_to_selected_target():
    # prefill TP2 x PP2; route to the big replica and confirm the reshard
    # plan covers exactly the big replica's ranks.
    src = _layouts(2, 2, 0)
    r = make_router("round_robin", _targets())
    target, transfers = route_and_plan(r, RequestInfo(0), src)  # -> "small" first
    assert target.target_id == "small"
    dst_ranks = {lay.global_rank for lay in target.layouts}
    assert transfers, "expected a non-empty reshard plan"
    assert {t.dst_rank for t in transfers} <= dst_ranks
    # every dst rank of the selected replica is covered
    assert {t.dst_rank for t in transfers} == dst_ranks
