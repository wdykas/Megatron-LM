# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Non-distributed unit tests for :class:`InferenceShard`.

The collective builders (:func:`build_inference_pg_collection`,
:func:`build_inference_pg_collections_for_shards`) require a real
``torch.distributed`` world and are covered by the multi-GPU
integration test elsewhere; these tests exercise the standalone
``InferenceShard`` dataclass API (``ranks()`` / ``owns_rank(...)``)
without any distributed setup.
"""

from megatron.core.inference.shards import InferenceShard


def _shard(index: int, rank_offset: int, world_size: int) -> InferenceShard:
    return InferenceShard(
        index=index,
        spec={"tp": world_size, "pp": 1, "ep": 1, "expt_tp": world_size, "dp": 1},
        rank_offset=rank_offset,
        world_size=world_size,
        pg_collection=None,
    )


def test_shard_ranks_enumerates_member_window():
    shard = _shard(index=0, rank_offset=4, world_size=3)
    assert shard.ranks() == [4, 5, 6]


def test_shard_owns_rank_uses_explicit_rank():
    shard = _shard(index=1, rank_offset=4, world_size=3)
    # Inside the window.
    assert shard.owns_rank(4) is True
    assert shard.owns_rank(6) is True
    # Boundaries: rank_offset + world_size is the first non-member rank.
    assert shard.owns_rank(7) is False
    # Below the window.
    assert shard.owns_rank(3) is False
    assert shard.owns_rank(0) is False


def test_shard_default_pg_collection_is_none_for_non_members():
    """An InferenceShard built for a non-member rank carries pg_collection=None;
    only the dataclass invariants matter here (the collective builder is what
    sets this in practice)."""
    shard = _shard(index=2, rank_offset=8, world_size=2)
    assert shard.pg_collection is None
    assert shard.http_url is None
