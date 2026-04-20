# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the request-migration bundle + plan-builder.

Runs non-distributed. Exercises:

- msgpack-compatible round-trip of :class:`RequestMigrationBundle`.
- Plan-builder invariants for matched TP (src_tp == dst_tp), reshape-up
  (src_tp=1 → dst_tp=4), reshape-down (src_tp=4 → dst_tp=1), partial overlap
  (src_tp=2 → dst_tp=4).
- Each (src_rank, dst_rank) pair appears at most once per block; head
  ranges are contiguous; no gaps in total destination coverage.
- MLA path emits full-tensor ops per rank pair.
"""
from megatron.core.inference.engines.request_migration import (
    KVLayout,
    RequestMigrationBundle,
    build_kv_migration_plan,
    deserialize_bundle,
    serialize_bundle,
)


def _sample_bundle(
    src_tp: int = 2,
    dst_tp: int = 1,
    num_kv_heads: int = 8,
    num_layers: int = 4,
    num_blocks: int = 3,
    is_mla: bool = False,
) -> RequestMigrationBundle:
    layout_kwargs = dict(
        pp_size=1,
        num_layers_total=num_layers,
        num_kv_heads_total=num_kv_heads,
        head_dim=64,
        block_size_tokens=16,
        is_mla=is_mla,
        kv_reduced_dim=128 if is_mla else None,
    )
    return RequestMigrationBundle(
        request_id=42,
        prompt_tokens=[1, 2, 3, 4],
        generated_tokens=[5, 6, 7],
        sampling_params={"temperature": 0.8, "top_p": 0.95},
        generated_log_probs=[-0.1, -0.2, -0.3],
        generated_top_n_logprobs=None,
        kv_cache_epoch=[(0, 0), (4, 1)],
        num_kv_blocks=num_blocks,
        last_block_offset=7,
        src_block_ids=list(range(100, 100 + num_blocks)),
        src_layout=KVLayout(tp_size=src_tp, **layout_kwargs),
        dst_layout=KVLayout(tp_size=dst_tp, **layout_kwargs),
    )


def test_bundle_roundtrip():
    """Bundle → dict → bundle preserves every field."""
    bundle = _sample_bundle()
    restored = deserialize_bundle(serialize_bundle(bundle))
    assert restored.request_id == bundle.request_id
    assert restored.prompt_tokens == bundle.prompt_tokens
    assert restored.generated_tokens == bundle.generated_tokens
    assert restored.sampling_params == bundle.sampling_params
    assert restored.generated_log_probs == bundle.generated_log_probs
    assert restored.kv_cache_epoch == bundle.kv_cache_epoch
    assert restored.num_kv_blocks == bundle.num_kv_blocks
    assert restored.last_block_offset == bundle.last_block_offset
    assert restored.src_block_ids == bundle.src_block_ids
    assert restored.src_layout == bundle.src_layout
    assert restored.dst_layout == bundle.dst_layout


def _check_plan_coverage(ops, bundle):
    """Every unique dst_rank's head ownership is fully tiled by the ops reaching it."""
    dst_heads_per_rank = bundle.dst_layout.heads_per_tp
    ops_by_dst: dict = {}
    for op in ops:
        ops_by_dst.setdefault(op.dst_rank, []).append(op)
    for dst_rank, dst_ops in ops_by_dst.items():
        spans = sorted(op.head_range for op in dst_ops)
        # Spans must be non-overlapping and collectively sized to
        # heads_per_tp for a valid tile.
        total = sum(hi - lo for lo, hi in spans)
        assert total == dst_heads_per_rank, (
            f"destination rank {dst_rank} expected {dst_heads_per_rank} heads covered, "
            f"got {total} via {spans}"
        )
        for (lo1, hi1), (lo2, _) in zip(spans, spans[1:]):
            assert hi1 <= lo2, f"overlapping head spans for dst {dst_rank}: {spans}"


def test_plan_matched_tp():
    """src_tp == dst_tp → one op per rank, no reshape."""
    bundle = _sample_bundle(src_tp=4, dst_tp=4, num_kv_heads=8)
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: tp,  # shard 0 at [0..4)
        dst_global_rank_of=lambda tp, pp: 4 + tp,  # shard 1 at [4..8)
        dst_block_ids=[200, 201, 202],
    )
    assert len(ops) == 4
    for sr in range(4):
        matching = [o for o in ops if o.src_rank == sr]
        assert len(matching) == 1
        op = matching[0]
        assert op.dst_rank == 4 + sr
        assert op.head_range == (sr * 2, (sr + 1) * 2)
        assert op.src_block_ids == [100, 101, 102]
        assert op.dst_block_ids == [200, 201, 202]


def test_plan_reshape_up():
    """src_tp=1 → dst_tp=4 (scatter): one src feeds four dst ranks."""
    bundle = _sample_bundle(src_tp=1, dst_tp=4, num_kv_heads=8)
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: 0,
        dst_global_rank_of=lambda tp, pp: 1 + tp,
        dst_block_ids=[200, 201, 202],
    )
    assert len(ops) == 4  # one per destination rank
    for dr in range(4):
        matching = [o for o in ops if o.dst_rank == 1 + dr]
        assert len(matching) == 1
        op = matching[0]
        assert op.src_rank == 0
        assert op.head_range == (dr * 2, (dr + 1) * 2)
    _check_plan_coverage(ops, bundle)


def test_plan_reshape_down():
    """src_tp=4 → dst_tp=1 (gather): four src ranks feed one dst rank."""
    bundle = _sample_bundle(src_tp=4, dst_tp=1, num_kv_heads=8)
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 4,
        dst_block_ids=[200, 201, 202],
    )
    assert len(ops) == 4
    dst_rank = 4
    seen_spans = []
    for sr in range(4):
        matching = [o for o in ops if o.src_rank == sr and o.dst_rank == dst_rank]
        assert len(matching) == 1
        seen_spans.append(matching[0].head_range)
    assert sorted(seen_spans) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    _check_plan_coverage(ops, bundle)


def test_plan_partial_overlap():
    """src_tp=2 → dst_tp=4: each src rank feeds two dst ranks."""
    bundle = _sample_bundle(src_tp=2, dst_tp=4, num_kv_heads=8)
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 2 + tp,
        dst_block_ids=[200, 201, 202],
    )
    # Each of 2 src ranks contributes to 2 dst ranks = 4 ops total.
    assert len(ops) == 4
    # src 0 owns heads [0,4) → dst 0 ([0,2)) + dst 1 ([2,4))
    # src 1 owns heads [4,8) → dst 2 ([4,6)) + dst 3 ([6,8))
    pairs = sorted((o.src_rank, o.dst_rank, o.head_range) for o in ops)
    assert pairs == [
        (0, 2, (0, 2)),
        (0, 3, (2, 4)),
        (1, 4, (4, 6)),
        (1, 5, (6, 8)),
    ]
    _check_plan_coverage(ops, bundle)


def test_plan_mla():
    """MLA: head dim collapses; every src-rank × dst-rank pair moves the full reduced dim."""
    bundle = _sample_bundle(src_tp=2, dst_tp=1, is_mla=True)
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 2,
        dst_block_ids=[200, 201, 202],
    )
    # 2 src ranks × 1 dst rank = 2 ops; each carries the full reduced dim.
    assert len(ops) == 2
    for op in ops:
        assert op.head_range == (0, bundle.src_layout.kv_reduced_dim)
        assert op.dst_rank == 2


def test_plan_rejects_pp():
    """PP>1 is unsupported in v0."""
    bundle = _sample_bundle()
    bundle.src_layout.pp_size = 2
    bundle.src_layout.num_layers_total = 4  # still divisible
    try:
        build_kv_migration_plan(
            bundle,
            src_global_rank_of=lambda tp, pp: 0,
            dst_global_rank_of=lambda tp, pp: 1,
            dst_block_ids=[200, 201, 202],
        )
    except AssertionError as e:
        assert "PP=1" in str(e)
    else:
        raise AssertionError("expected PP>1 to be rejected")


def test_plan_rejects_head_count_mismatch():
    """Migration cannot change model shape."""
    bundle = _sample_bundle()
    bundle.dst_layout.num_kv_heads_total = 16  # different model
    try:
        build_kv_migration_plan(
            bundle,
            src_global_rank_of=lambda tp, pp: 0,
            dst_global_rank_of=lambda tp, pp: 1,
            dst_block_ids=[200, 201, 202],
        )
    except AssertionError as e:
        assert "num_kv_heads_total" in str(e)
    else:
        raise AssertionError("expected head-count mismatch to be rejected")
