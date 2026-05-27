# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-distributed unit tests for nvshmem_migration bookkeeping.

The NVSHMEM put/wait/ack primitives require a real GPU + multi-PE world
and are covered by a separate integration test. These tests cover the
pure-Python state that determines flag-slot routing correctness — lane
encoding and the per-(src_pe, dst_pe) FlagArena counter discipline. If
the counters drift between src and dst, the ack handshake aliases
between unrelated pairs and the whole transport silently corrupts data,
so getting this right matters more than the NVSHMEM call ergonomics.
"""

import pytest

from megatron.core.inference import nvshmem_migration as nv


@pytest.fixture(autouse=True)
def _nvshmem_migration_test_state():
    """Each test gets a freshly-initialized in-memory state. Reset on
    teardown so tests don't leak counters into one another."""
    nv._reset_state_for_test()
    nv._init_state_for_test(n_pes=4, ops_per_pair=8)
    yield
    nv._reset_state_for_test()


# ---- lane_for -------------------------------------------------------------


def test_lane_encoding_is_deterministic():
    """Same (src, dst) PE pair → same lane index, every call."""
    assert nv.lane_for(0, 1) == nv.lane_for(0, 1)
    assert nv.lane_for(2, 3) == nv.lane_for(2, 3)


def test_lane_encoding_distinct_pairs():
    """Different (src, dst) pairs map to distinct lanes. Otherwise an
    ack from one pair would land on another pair's slot."""
    seen = set()
    for src in range(4):
        for dst in range(4):
            lane = nv.lane_for(src, dst)
            assert lane not in seen, (
                f"lane collision: (src={src},dst={dst}) → {lane} already seen"
            )
            seen.add(lane)


def test_lane_encoding_directional():
    """Lane(A→B) is distinct from lane(B→A): migrations flow one way."""
    assert nv.lane_for(0, 1) != nv.lane_for(1, 0)


def test_lane_encoding_rejects_out_of_range():
    """PE indices outside ``[0, n_pes)`` are caller bugs, not silent
    wraps — the assertion catches them at construction time."""
    with pytest.raises(AssertionError):
        nv.lane_for(0, 4)  # n_pes=4 → max valid is 3
    with pytest.raises(AssertionError):
        nv.lane_for(-1, 0)


# ---- FlagArena ------------------------------------------------------------


def test_flag_arena_per_pair_counter_independent():
    """Counters advance per (src, dst) pair, independently. Two ops on
    the same pair pick consecutive slots; two ops on different pairs
    pick from different lanes entirely."""
    arena = nv.FlagArena()
    a1 = arena.take(0, 1)
    a2 = arena.take(0, 1)
    b1 = arena.take(0, 2)
    a3 = arena.take(0, 1)

    # (0,1) ops use consecutive offsets within their lane.
    assert a2 == a1 + 1
    assert a3 == a1 + 2

    # (0,2) op is in a different lane (different (src, dst)).
    assert b1 != a1 and b1 != a2 and b1 != a3
    # And it picks lane offset 0, since it's the first op on (0, 2).
    assert b1 % nv._ops_per_pair == 0


def test_flag_arena_two_arenas_independent():
    """Two FlagArena instances don't share counters — each migration
    builds its own. Both start from offset 0 within each lane."""
    arena_a = nv.FlagArena()
    arena_b = nv.FlagArena()
    assert arena_a.take(0, 1) == arena_b.take(0, 1)
    assert arena_a.take(2, 3) == arena_b.take(2, 3)


def test_flag_arena_indices_fall_in_lane_range():
    """Every flag index from a (src, dst) pair lands in that lane's
    contiguous slot range. Otherwise dst's flag/ack pool slots would
    collide with another pair's."""
    arena = nv.FlagArena()
    lane = nv.lane_for(1, 3)
    lo = lane * nv._ops_per_pair
    hi = lo + nv._ops_per_pair
    for _ in range(nv._ops_per_pair):
        idx = arena.take(1, 3)
        assert lo <= idx < hi, f"index {idx} not in lane [{lo}, {hi})"


def test_flag_arena_wraps_within_lane():
    """Within a single migration, ops on a (src, dst) pair beyond
    ``ops_per_pair`` wrap to the start of that lane. The ack handshake
    in ``put_slot_with_signal`` is what makes the wrap safe."""
    arena = nv.FlagArena()
    first = arena.take(0, 1)
    for _ in range(nv._ops_per_pair - 1):
        arena.take(0, 1)
    # ``_ops_per_pair``th call wraps back to the lane's first index.
    wrapped = arena.take(0, 1)
    assert wrapped == first


def test_flag_arena_walked_in_same_order_picks_same_indices():
    """Both src and dst build their op meta by walking the migration
    plan in identical order. Two FlagArena instances called with the
    same sequence of ``take(src, dst)`` arguments produce identical
    flag-index sequences — the invariant that lets the two sides
    agree on which flag refers to which op without coordination."""
    plan = [
        (0, 2),
        (0, 3),
        (1, 2),
        (0, 2),
        (1, 3),
        (1, 2),
        (0, 3),
        (0, 2),
    ]
    src_arena = nv.FlagArena()
    dst_arena = nv.FlagArena()
    src_indices = [src_arena.take(s, d) for s, d in plan]
    dst_indices = [dst_arena.take(s, d) for s, d in plan]
    assert src_indices == dst_indices


def test_flag_arena_indices_disjoint_across_pairs_within_one_migration():
    """Within a single migration, every (src, dst, op_offset) op
    picks a distinct flag index. Two ops on the same dst PE from
    different sources land on different flag slots (because their
    lanes differ), so dst's signal_wait on flag X never has to
    disambiguate between sources."""
    arena = nv.FlagArena()
    indices = []
    # Mix several pairs targeting dst PE 2; ensure each (src, op_idx)
    # combination gets its own flag index.
    for _ in range(3):
        indices.append(arena.take(0, 2))
        indices.append(arena.take(1, 2))
        indices.append(arena.take(3, 2))
    assert len(set(indices)) == len(indices), (
        f"duplicate flag indices on shared dst: {indices}"
    )


# ---- Cross-migration safety expectations ----------------------------------


def test_flag_arena_consecutive_migrations_reuse_same_indices():
    """Two back-to-back migrations both start with arena offset 0 on
    every pair. The ack handshake is what makes this safe: in
    production, src waits for dst's ack on flag F before reissuing on
    F. The arena itself doesn't try to avoid the collision — it
    delegates to acks. This test pins that contract: starting fresh
    arenas IS expected to reuse indices."""
    m1 = nv.FlagArena()
    m2 = nv.FlagArena()
    a = m1.take(0, 1)
    b = m2.take(0, 1)
    assert a == b, (
        "consecutive migrations should reuse flag indices; cross-migration "
        "safety comes from ack-based recycling, not arena isolation"
    )
