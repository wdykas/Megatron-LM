# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hetero-TP routing between disagg shards.

The dispatcher's ``tp_pair_routing`` builds the
``(src_tp_offset, dst_tp_offset)`` exchange table for one cross-shard
hop. Matched TP collapses to 1-to-1; divisible hetero-TP fans out (when
dst > src) or strides (when src > dst); non-divisible TPs are rejected.
"""

import pytest

from megatron.core.inference.route_dispatcher import (
    RouteDispatcher,
    tp_pair_routing,
)
from megatron.rl.inference.route_planner import Route, RouteHop


def test_matched_tp_is_one_to_one():
    """Matched TP collapses to identity: rank k sends to rank k."""
    assert tp_pair_routing(4, 4) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_dst_larger_fans_out():
    """Each src rank fans out to dst_tp/src_tp dst ranks so every dst
    rank receives. src_tp=2, dst_tp=4: src 0 → dst {0,1}; src 1 → dst {2,3}."""
    pairs = tp_pair_routing(src_tp=2, dst_tp=4)
    assert pairs == [(0, 0), (0, 1), (1, 2), (1, 3)]
    # Every dst rank appears exactly once.
    assert sorted(d for _, d in pairs) == [0, 1, 2, 3]


def test_src_larger_strides():
    """Only every (src_tp/dst_tp)-th src rank sends; multiple src
    ranks holding the same TP-replicated hidden state, so we pick a
    representative. src_tp=4, dst_tp=2: src 0 → dst 0; src 2 → dst 1."""
    pairs = tp_pair_routing(src_tp=4, dst_tp=2)
    assert pairs == [(0, 0), (2, 1)]
    # Every dst rank receives.
    assert sorted(d for _, d in pairs) == [0, 1]
    # Src ranks 1 and 3 don't send (no entry in the table).
    assert 1 not in [s for s, _ in pairs]
    assert 3 not in [s for s, _ in pairs]


def test_non_divisible_tp_rejected():
    """tp=3 → tp=2 has no integer fanout / stride; reject."""
    with pytest.raises(AssertionError, match="divisibility"):
        tp_pair_routing(src_tp=3, dst_tp=2)
    with pytest.raises(AssertionError, match="divisibility"):
        tp_pair_routing(src_tp=2, dst_tp=3)


def _build_dispatcher(my_shard_idx, my_tp_offset, shard_tp, shard_rank_offset, route):
    """Build the dispatcher's plan via the module-level helper. We
    don't need a full dispatcher (no NVSHMEM init) — the plan is what
    we're inspecting."""
    from megatron.core.inference.route_dispatcher import _build_layer_plan

    plan = _build_layer_plan(
        route=route,
        my_shard_idx=my_shard_idx,
        my_tp_offset=my_tp_offset,
        shard_tp=list(shard_tp),
        shard_rank_offset=list(shard_rank_offset),
    )
    # Return a thin shim with the same `._plan` shape the rest of the
    # tests read.
    shim = type("PlanShim", (), {})()
    shim._plan = plan
    return shim


def test_dispatcher_plan_dst_larger_fans_out():
    """src shard tp=2, dst shard tp=4. Each src rank's plan should fan
    out to 2 dst peers on the hop's exit layer."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    # Shard 0 at offset 0 (tp=2), shard 1 at offset 2 (tp=4).
    # Each src rank (my_tp_offset=0,1) on shard 0:
    disp_src0 = _build_dispatcher(
        my_shard_idx=0,
        my_tp_offset=0,
        shard_tp=[2, 4],
        shard_rank_offset=[0, 2],
        route=route,
    )
    plan = disp_src0._plan[0]
    # src_tp=2, dst_tp=4, pairs = [(0,0), (0,1), (1,2), (1,3)].
    # my_tp_offset=0 sends to dst offsets {0, 1} → PEs {2, 3}.
    assert plan.send_to_pes == (2, 3)

    disp_src1 = _build_dispatcher(
        my_shard_idx=0,
        my_tp_offset=1,
        shard_tp=[2, 4],
        shard_rank_offset=[0, 2],
        route=route,
    )
    # my_tp_offset=1 sends to dst offsets {2, 3} → PEs {4, 5}.
    assert disp_src1._plan[0].send_to_pes == (4, 5)


def test_dispatcher_plan_src_larger_strides():
    """src shard tp=4, dst shard tp=2. Only stride-aligned src ranks
    send (src 0 → dst 0, src 2 → dst 1); src 1 and src 3 send nothing."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    # src offsets 0, 1, 2, 3 on a 4-rank shard:
    sends_by_offset = {}
    for off in range(4):
        disp = _build_dispatcher(
            my_shard_idx=0,
            my_tp_offset=off,
            shard_tp=[4, 2],
            shard_rank_offset=[0, 4],
            route=route,
        )
        sends_by_offset[off] = disp._plan[0].send_to_pes
    # src_tp=4, dst_tp=2, pairs = [(0,0), (2,1)].
    # src 0 → dst offset 0 → PE 4; src 2 → dst offset 1 → PE 5.
    # src 1 and src 3 are not in the table — send_to_pes is ().
    assert sends_by_offset == {0: (4,), 1: (), 2: (5,), 3: ()}


def test_dispatcher_plan_receive_dst_larger():
    """src shard tp=2, dst shard tp=4. Each dst rank receives from
    exactly one src rank (the fanout sender to it)."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    # dst offsets 0,1,2,3 on shard 1 (rank_offset 2):
    receives_by_offset = {}
    for off in range(4):
        disp = _build_dispatcher(
            my_shard_idx=1,
            my_tp_offset=off,
            shard_tp=[2, 4],
            shard_rank_offset=[0, 2],
            route=route,
        )
        receives_by_offset[off] = disp._plan[1].receive_from_pe
    # pairs = [(0,0), (0,1), (1,2), (1,3)]. dst 0,1 from src 0 (PE 0);
    # dst 2,3 from src 1 (PE 1).
    assert receives_by_offset == {0: 0, 1: 0, 2: 1, 3: 1}


def test_dispatcher_plan_matched_tp_one_to_one():
    """Matched TP: every rank sends to and receives from the same TP
    offset on the peer shard."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    for off in range(4):
        disp_src = _build_dispatcher(
            my_shard_idx=0,
            my_tp_offset=off,
            shard_tp=[4, 4],
            shard_rank_offset=[0, 4],
            route=route,
        )
        assert disp_src._plan[0].send_to_pes == (4 + off,)
        disp_dst = _build_dispatcher(
            my_shard_idx=1,
            my_tp_offset=off,
            shard_tp=[4, 4],
            shard_rank_offset=[0, 4],
            route=route,
        )
        assert disp_dst._plan[1].receive_from_pe == off
