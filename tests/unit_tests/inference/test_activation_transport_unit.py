# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-distributed unit tests for activation_transport bookkeeping.

The NVSHMEM put/wait/ack primitives require a real GPU + multi-PE world
and are covered by a separate integration test. These tests cover the
pure-Python state that determines slot routing correctness — lane
encoding and per-lane send/recv counters. If the counters drift between
src and dst, every activation lands in the wrong slot, so getting this
right matters more than the NVSHMEM call ergonomics.
"""

import pytest

from megatron.core.inference import activation_transport as at


@pytest.fixture(autouse=True)
def _activation_transport_test_state():
    """Each test gets a freshly-initialized in-memory state. Reset on teardown
    so tests don't leak counters into one another."""
    at._reset_state_for_test()
    at._init_state_for_test(n_pes=4, pool_depth=4)
    yield
    at._reset_state_for_test()


def test_lane_encoding_is_deterministic():
    """Same (src, dst) PE pair → same lane index, every call."""
    assert at.lane_for(0, 1) == at.lane_for(0, 1)
    assert at.lane_for(2, 3) == at.lane_for(2, 3)


def test_lane_encoding_distinct_pairs():
    """Different (src, dst) pairs map to distinct lanes."""
    seen = set()
    for src in range(4):
        for dst in range(4):
            lane = at.lane_for(src, dst)
            assert lane not in seen, (
                f"lane collision: (src={src},dst={dst}) → {lane} "
                f"already seen"
            )
            seen.add(lane)


def test_lane_encoding_directional():
    """Lane(A→B) is distinct from lane(B→A): activations are one-way."""
    assert at.lane_for(0, 1) != at.lane_for(1, 0)


def test_lane_for_rejects_unregistered_pair():
    """``lane_for`` of a pair that wasn't registered raises. In
    production this typically means a route is using a PE pair whose
    layout route didn't pre-register it."""
    with pytest.raises(RuntimeError, match="not registered"):
        at.lane_for(99, 0)
    with pytest.raises(RuntimeError, match="not registered"):
        at.lane_for(0, 99)


def test_send_counter_advances_per_call():
    lane = at.lane_for(0, 1)
    s0 = at.next_send_slot(lane)
    s1 = at.next_send_slot(lane)
    s2 = at.next_send_slot(lane)
    assert (s0, s1, s2) == (lane * 4 + 0, lane * 4 + 1, lane * 4 + 2)


def test_send_counter_wraps_at_pool_depth():
    """The 4th send on a depth-4 lane returns to slot 0 (ring buffer)."""
    lane = at.lane_for(0, 1)
    slots = [at.next_send_slot(lane) for _ in range(6)]
    base = lane * 4
    assert slots == [base + 0, base + 1, base + 2, base + 3, base + 0, base + 1]


def test_recv_counter_independent_of_send_counter():
    """Send and recv counters increment independently. Both sides walk
    the same model order, but the slot indices come from different
    counters within the lane."""
    lane = at.lane_for(2, 3)
    base = lane * 4
    # Sender side advances 3.
    at.next_send_slot(lane)
    at.next_send_slot(lane)
    at.next_send_slot(lane)
    # Receiver still at 0.
    assert at.next_recv_slot(lane) == base + 0
    assert at.next_recv_slot(lane) == base + 1


def test_send_recv_lockstep_after_full_cycle():
    """When src and dst have both processed N activations on a lane,
    their counters land at the same slot — invariant we rely on for
    correctness."""
    lane = at.lane_for(1, 2)
    base = lane * 4
    sent = [at.next_send_slot(lane) for _ in range(10)]
    recvd = [at.next_recv_slot(lane) for _ in range(10)]
    assert sent == recvd, (
        f"counters drifted: sent={sent} recvd={recvd}; recv must mirror "
        f"send slot-by-slot for the ack-flag handshake to land on the "
        f"right slot."
    )
    # And the cycle wraps as expected.
    expected = [base + (i % 4) for i in range(10)]
    assert sent == expected


def test_reset_lane_counters_clears_both():
    lane = at.lane_for(0, 0)
    at.next_send_slot(lane)
    at.next_send_slot(lane)
    at.next_recv_slot(lane)
    at.reset_lane_counters(lane)
    assert at.next_send_slot(lane) == lane * 4 + 0
    assert at.next_recv_slot(lane) == lane * 4 + 0


def test_independent_lanes_have_independent_counters():
    """Counter advancement on lane A does NOT advance lane B."""
    lane_a = at.lane_for(0, 1)
    lane_b = at.lane_for(0, 2)
    at.next_send_slot(lane_a)
    at.next_send_slot(lane_a)
    # lane_b is still at counter 0.
    assert at.next_send_slot(lane_b) == lane_b * 4 + 0


def test_pool_stats_snapshot():
    """``pool_stats()`` returns the live sizing + counter state."""
    lane = at.lane_for(0, 1)
    at.next_send_slot(lane)
    at.next_send_slot(lane)
    stats = at.pool_stats()
    assert stats["active_pairs"] == 16
    assert stats["pool_depth"] == 4
    assert stats["send_counters"][lane] == 2
    assert stats["recv_counters"][lane] == 0


def test_is_initialized_flag():
    """The initialized flag tracks the lifecycle."""
    assert at.is_initialized() is True
    at._reset_state_for_test()
    assert at.is_initialized() is False
    # Re-init for the autouse fixture teardown.
    at._init_state_for_test(n_pes=4, pool_depth=4)


# ---- Active-pair registration -------------------------------------------


def test_register_activation_pair_idempotent():
    """Registering the same pair twice leaves the registry unchanged —
    important because a route helper may register a pair for multiple
    hops that share endpoints."""
    at._reset_state_for_test()
    at._init_state_for_test(n_pes=4, pool_depth=4, register_all_pairs=False, realize=False)
    at.register_activation_pair(0, 1)
    n = len(at._active_pairs)
    at.register_activation_pair(0, 1)
    assert len(at._active_pairs) == n


def test_register_activation_pair_rejects_after_realize():
    """Pool size is fixed at realize time — late registrations would
    have nowhere to go and we raise instead of silently misbehaving."""
    at._reset_state_for_test()
    at._init_state_for_test(n_pes=4, pool_depth=4, register_all_pairs=False, realize=False)
    at.register_activation_pair(0, 1)
    at._pools_realized = True
    with pytest.raises(RuntimeError, match="already realized"):
        at.register_activation_pair(0, 2)


def test_pool_sizing_scales_with_active_pairs():
    """Active-pair count bounds pool size, not ``n_pes²``. The whole
    point of explicit registration is bounding the pool by actual
    route topology."""
    at._reset_state_for_test()
    at._init_state_for_test(n_pes=8, pool_depth=4, register_all_pairs=False, realize=False)
    # Only register 3 of the 64 possible pairs.
    at.register_activation_pair(0, 4)
    at.register_activation_pair(0, 5)
    at.register_activation_pair(1, 4)
    assert len(at._active_pairs) == 3
    # Lane indices for the registered pairs are 0..2; lane_for of an
    # unregistered pair raises.
    assert at.lane_for(0, 4) == 0
    assert at.lane_for(0, 5) == 1
    assert at.lane_for(1, 4) == 2
    with pytest.raises(RuntimeError, match="not registered"):
        at.lane_for(2, 6)


def test_register_activation_shard_pair_takes_cross_product():
    """The convenience helper registers every ``(s, d)`` in
    ``src_pes × dst_pes`` for a hop transition where any rank in the
    src shard might send to any rank in the dst shard."""
    at._reset_state_for_test()
    at._init_state_for_test(n_pes=8, pool_depth=4, register_all_pairs=False, realize=False)
    at.register_activation_shard_pair(src_pes=[0, 1, 2, 3], dst_pes=[4, 5, 6, 7])
    assert len(at._active_pairs) == 16
    lanes = {at.lane_for(s, d) for s in [0, 1, 2, 3] for d in [4, 5, 6, 7]}
    assert len(lanes) == 16
