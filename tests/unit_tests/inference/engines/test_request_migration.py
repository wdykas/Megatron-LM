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
    MambaBlockRange,
    MambaLayout,
    RequestMigrationBundle,
    build_kv_migration_plan,
    build_mamba_migration_plan,
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
        kv_cache_epoch=[(0, 0), (4, 1)],
        num_kv_blocks=num_blocks,
        last_block_offset=7,
        src_block_ids=list(range(100, 100 + num_blocks)),
        src_layout=KVLayout(tp_size=src_tp, **layout_kwargs),
        dst_layout=KVLayout(tp_size=dst_tp, **layout_kwargs),
    )


def test_bundle_roundtrip():
    """Bundle → dict → bundle preserves every wire field.

    ``src_layout`` / ``dst_layout`` are intentionally not on the wire
    (the receiving handler restamps them from ``_migration_meta``), so
    they round-trip as ``None``.
    """
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
    assert restored.src_layout is None
    assert restored.dst_layout is None


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


def test_plan_matched_pp():
    """src_pp == dst_pp > 1: one op per pp stage per (tp pair).

    With PP=2, TP=1 on both sides and 4 layers, stage 0 owns layers
    [0, 2) and stage 1 owns [2, 4). Every (src_pp, dst_pp) diagonal
    pair emits one op covering that stage's layers; off-diagonal pairs
    have no layer overlap and emit nothing.
    """
    bundle = _sample_bundle(src_tp=1, dst_tp=1, num_kv_heads=4, num_layers=4)
    bundle.src_layout.pp_size = 2
    bundle.dst_layout.pp_size = 2
    # Ranks: src shard [0,1] with (tp=0,pp=0)=0, (tp=0,pp=1)=1;
    # dst shard [2,3] with (tp=0,pp=0)=2, (tp=0,pp=1)=3.
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: pp,  # tp=1 so pp is the only stride
        dst_global_rank_of=lambda tp, pp: 2 + pp,
        dst_block_ids=[200, 201, 202],
    )
    # Diagonals only: (src_pp=0,dst_pp=0) + (src_pp=1,dst_pp=1).
    assert len(ops) == 2
    pairs = sorted((o.src_rank, o.dst_rank, o.layer_range) for o in ops)
    assert pairs == [(0, 2, (0, 2)), (1, 3, (2, 4))]


def test_plan_mismatched_pp_reshape():
    """Reshape across PP: src_pp=1, dst_pp=2, 4 layers.

    Source rank 0 owns all 4 layers; destination ranks 1 and 2 own
    layers [0,2) and [2,4) respectively. Plan emits one op per dst
    stage, each carrying just the layers that stage owns.
    """
    bundle = _sample_bundle(src_tp=1, dst_tp=1, num_kv_heads=4, num_layers=4)
    bundle.src_layout.pp_size = 1
    bundle.dst_layout.pp_size = 2
    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=lambda tp, pp: 0,
        dst_global_rank_of=lambda tp, pp: 1 + pp,
        dst_block_ids=[200, 201, 202],
    )
    assert len(ops) == 2
    pairs = sorted((o.src_rank, o.dst_rank, o.layer_range) for o in ops)
    assert pairs == [(0, 1, (0, 2)), (0, 2, (2, 4))]


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


# ---- Mamba migration plan ---------------------------------------------------


def _mamba_bundle() -> RequestMigrationBundle:
    """Minimal bundle for mamba-plan tests. Only ``request_id`` is read
    by the plan-builder; other fields can be defaults."""
    return RequestMigrationBundle(
        request_id=7,
        prompt_tokens=[1, 2],
        generated_tokens=[3],
        sampling_params={},
    )


def _mamba_layouts(
    src_tp: int,
    dst_tp: int,
    pp_size: int = 1,
    *,
    src_pp_size: int = None,
    dst_pp_size: int = None,
    layer_type_list: tuple = (),
    num_mamba_layers: int = 8,
):
    """Build (src, dst) MambaLayout pair for tests.

    Fixed model dims (model invariants — same on both sides):
      d_inner_total=64, nheads_total=16, ngroups_total=8, d_state=16,
      headdim=4, d_conv=4. Block global sizes:
      - conv_x: d_inner_total = 64
      - conv_B / conv_C: ngroups_total * d_state = 8 * 16 = 128
      - ssm: nheads_total = 16
    """
    src_pp = src_pp_size if src_pp_size is not None else pp_size
    dst_pp = dst_pp_size if dst_pp_size is not None else pp_size
    base = dict(
        num_layers_total=num_mamba_layers,
        d_inner_total=64,
        nheads_total=16,
        ngroups_total=8,
        d_conv=4,
        headdim=4,
        d_state=16,
        layer_type_list=layer_type_list,
    )
    return (
        MambaLayout(tp_size=src_tp, pp_size=src_pp, **base),
        MambaLayout(tp_size=dst_tp, pp_size=dst_pp, **base),
    )


def _assert_kind_tiles_per_rank(ops, src_layout, dst_layout):
    """Per-kind sanity: for each dst rank, the union of dst_local_range
    across all blocks of that kind (across every op reaching the
    rank) exactly tiles that rank's local block
    (size = block_local_size(kind)). Verifies the plan doesn't leave
    any rank's block under- or double-covered. Each op may carry up
    to 4 blocks (one per kind), so iterate ``op.blocks``.
    """
    expected_local = {k: dst_layout.block_local_size(k) for k in dst_layout.BLOCK_KINDS}
    by_rank_kind: dict = {}
    for op in ops:
        for block in op.blocks:
            by_rank_kind.setdefault((op.dst_rank, block.kind), []).append(block)
    for (dst_rank, kind), kind_blocks in by_rank_kind.items():
        spans = sorted(b.dst_local_range for b in kind_blocks)
        # Translate to local-block coords (subtract block_local_offset).
        off = dst_layout.block_local_offset(kind)
        local = [(lo - off, hi - off) for lo, hi in spans]
        total = sum(hi - lo for lo, hi in local)
        assert total == expected_local[kind], (
            f"dst rank {dst_rank} kind {kind} covered {total} != "
            f"expected {expected_local[kind]} (spans {local})"
        )
        for (lo1, hi1), (lo2, _) in zip(local, local[1:]):
            assert hi1 <= lo2, (
                f"dst rank {dst_rank} kind {kind} overlapping ranges {local}"
            )


def _kinds_per_overlap(ops, dst_rank):
    """All kinds emitted to a given dst rank (across every op's blocks)."""
    kinds = set()
    for op in ops:
        if op.dst_rank != dst_rank:
            continue
        for block in op.blocks:
            kinds.add(block.kind)
    return sorted(kinds)


def test_mamba_plan_matched_tp():
    """src_tp == dst_tp: one op per matched rank pair carrying every
    block kind packed together. All blocks fully covered on src and
    dst."""
    bundle = _mamba_bundle()
    src_layout, dst_layout = _mamba_layouts(src_tp=4, dst_tp=4)
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 100 + tp,
    )
    # 4 matched-tp pairs × 1 pp pair = 4 ops; each carries all 4
    # block kinds.
    assert len(ops) == 4
    for op in ops:
        # Matched TP → diagonal rank pairing.
        assert op.src_rank + 100 == op.dst_rank
        # Every op should bundle all four kinds, in BLOCK_KINDS order.
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
        for b in op.blocks:
            # Matched TP also means src and dst have identical local
            # offsets and block sizes, so the ranges coincide.
            assert b.src_local_range == b.dst_local_range
    for r in range(100, 104):
        assert _kinds_per_overlap(ops, r) == sorted(src_layout.BLOCK_KINDS)
    _assert_kind_tiles_per_rank(ops, src_layout, dst_layout)


def test_mamba_plan_reshape_up_per_block():
    """src_tp=1 → dst_tp=4: each block's global range fans out across
    4 dst tp ranks. One op per dst rank — 4 ops — each carrying all
    4 block kinds, all from src rank 0."""
    bundle = _mamba_bundle()
    src_layout, dst_layout = _mamba_layouts(src_tp=1, dst_tp=4)
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: 0,
        dst_global_rank_of=lambda tp, pp: 100 + tp,
    )
    assert len(ops) == 4
    assert all(op.src_rank == 0 for op in ops)
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    for r in range(100, 104):
        assert _kinds_per_overlap(ops, r) == sorted(src_layout.BLOCK_KINDS)
    _assert_kind_tiles_per_rank(ops, src_layout, dst_layout)


def test_mamba_plan_reshape_down_per_block():
    """src_tp=4 → dst_tp=1: 4 src ranks merge into 1 dst rank. One op
    per src rank — 4 ops — each carrying all 4 block kinds; their
    union tiles the dst's full local block per kind."""
    bundle = _mamba_bundle()
    src_layout, dst_layout = _mamba_layouts(src_tp=4, dst_tp=1)
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 100,
    )
    assert len(ops) == 4  # one per src tp rank
    assert all(op.dst_rank == 100 for op in ops)
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    _assert_kind_tiles_per_rank(ops, src_layout, dst_layout)


def test_mamba_plan_partial_overlap_per_block():
    """src_tp=2 → dst_tp=4: each src rank covers 2 dst ranks per
    block. 2 src × 2 dst per src = 4 ops total, each carrying all 4
    block kinds."""
    bundle = _mamba_bundle()
    src_layout, dst_layout = _mamba_layouts(src_tp=2, dst_tp=4)
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 100 + tp,
    )
    assert len(ops) == 4
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    # src rank 0 → dst {100, 101}, src rank 1 → dst {102, 103}.
    src_to_dsts: dict = {}
    for op in ops:
        src_to_dsts.setdefault(op.src_rank, set()).add(op.dst_rank)
    assert src_to_dsts == {0: {100, 101}, 1: {102, 103}}
    _assert_kind_tiles_per_rank(ops, src_layout, dst_layout)


def test_mamba_plan_hetero_pp_reshape_up():
    """src_pp=1 → dst_pp=2: src has 1 PP rank owning all mamba layers,
    dst splits across 2 PP ranks. Each dst PP rank receives ops with
    its own mamba subset."""
    bundle = _mamba_bundle()
    # 8 transformer layers, 4 mamba (M at indices 0, 2, 4, 6).
    layer_types = ("M", "E", "M", "E", "M", "E", "M", "E")
    src_layout, dst_layout = _mamba_layouts(
        src_tp=2,
        dst_tp=2,
        src_pp_size=1,
        dst_pp_size=2,
        layer_type_list=layer_types,
        num_mamba_layers=4,
    )
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 100 + pp * 2 + tp,
    )
    # 1 src_pp × 2 dst_pp × 2 matched-tp = 4 ops; each packs 4 kinds.
    assert len(ops) == 4
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    # Dst PP=0 ranks get mamba indices (0, 1); dst PP=1 ranks get (2, 3).
    by_dst: dict = {}
    for op in ops:
        by_dst.setdefault(op.dst_rank, []).append(op)
    for r in (100, 101):
        assert all(op.mamba_layer_indices == (0, 1) for op in by_dst[r])
    for r in (102, 103):
        assert all(op.mamba_layer_indices == (2, 3) for op in by_dst[r])


def test_mamba_plan_hetero_pp_reshape_down():
    """src_pp=2 → dst_pp=1: src splits, dst owns all. Each src PP rank
    contributes its mamba subset to the (single) dst PP rank."""
    bundle = _mamba_bundle()
    layer_types = ("E", "M", "E", "M", "M", "E", "E", "M")
    src_layout, dst_layout = _mamba_layouts(
        src_tp=2,
        dst_tp=2,
        src_pp_size=2,
        dst_pp_size=1,
        layer_type_list=layer_types,
        num_mamba_layers=4,
    )
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: pp * 2 + tp,
        dst_global_rank_of=lambda tp, pp: 100 + tp,
    )
    # 2 src_pp × 1 dst_pp × 2 matched-tp = 4 ops; each packs 4 kinds.
    assert len(ops) == 4
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    by_src: dict = {}
    for op in ops:
        by_src.setdefault(op.src_rank, []).append(op)
    # Src PP=0 ranks carry mamba indices (0, 1).
    for r in (0, 1):
        assert all(op.mamba_layer_indices == (0, 1) for op in by_src[r])
    # Src PP=1 ranks carry mamba indices (2, 3).
    for r in (2, 3):
        assert all(op.mamba_layer_indices == (2, 3) for op in by_src[r])


def test_mamba_plan_hetero_tp_block_local_offsets():
    """src_tp=2 → dst_tp=1, hybrid model with conv state packing
    (x, B, C): verify each block routes to its own local offset on
    both sides, and the offsets DIFFER between src and dst because
    block-local sizes differ per TP. With kinds packed into one op
    per (src_rank, dst_rank) pair, each op carries 4
    :class:`MambaBlockRange` entries — one per kind — that must each
    address the right packed-conv offset."""
    bundle = _mamba_bundle()
    src_layout, dst_layout = _mamba_layouts(src_tp=2, dst_tp=1)

    # Sanity: block_local_offset differs between src (tp=2) and dst (tp=1).
    # conv_x at offset 0 on both. conv_B at d_inner_local_tp = 32 (src) vs 64 (dst).
    assert src_layout.block_local_offset("conv_x") == 0
    assert src_layout.block_local_offset("conv_B") == 32
    assert dst_layout.block_local_offset("conv_x") == 0
    assert dst_layout.block_local_offset("conv_B") == 64
    assert dst_layout.block_local_offset("conv_C") == 64 + 128

    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: tp,
        dst_global_rank_of=lambda tp, pp: 100,
    )
    # 2 src_tp × 1 dst_tp = 2 ops; each carries 4 block kinds packed.
    assert len(ops) == 2
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
        for block in op.blocks:
            src_lo, _ = block.src_local_range
            dst_lo, _ = block.dst_local_range
            src_block_off = src_layout.block_local_offset(block.kind)
            dst_block_off = dst_layout.block_local_offset(block.kind)
            # Each src tp_rank views its own block-local at offset 0
            # within the kind's block; src_lo equals just the kind's
            # block_local_offset on the src side.
            assert src_lo == src_block_off
            if op.src_rank == 0:
                # First half → dst_lo = dst_block_off + 0
                assert dst_lo == dst_block_off
            else:
                # Second half → dst_lo = dst_block_off + src_block_size
                assert dst_lo == dst_block_off + src_layout.block_local_size(
                    block.kind
                )
    _assert_kind_tiles_per_rank(ops, src_layout, dst_layout)


def test_mamba_plan_hetero_tp_and_pp():
    """src tp=2,pp=2 + dst tp=1,pp=1: combined hetero TP and hetero PP."""
    bundle = _mamba_bundle()
    layer_types = ("M", "M", "E", "E", "M", "E", "M", "E")
    src_layout, dst_layout = _mamba_layouts(
        src_tp=2,
        dst_tp=1,
        src_pp_size=2,
        dst_pp_size=1,
        layer_type_list=layer_types,
        num_mamba_layers=4,
    )
    ops = build_mamba_migration_plan(
        bundle,
        src_layout,
        dst_layout,
        src_global_rank_of=lambda tp, pp: pp * 2 + tp,
        dst_global_rank_of=lambda tp, pp: 100,
    )
    # 2 src_pp × 1 dst_pp × 2 src_tp × 1 dst_tp = 4 ops; each packs 4 kinds.
    assert len(ops) == 4
    assert all(op.dst_rank == 100 for op in ops)
    for op in ops:
        assert tuple(b.kind for b in op.blocks) == src_layout.BLOCK_KINDS
    # PP-0 src ranks (0, 1) carry mamba indices (0, 1); PP-1 (2, 3) carry (2, 3).
    pp0_idx = {op.mamba_layer_indices for op in ops if op.src_rank in (0, 1)}
    pp1_idx = {op.mamba_layer_indices for op in ops if op.src_rank in (2, 3)}
    assert pp0_idx == {(0, 1)}
    assert pp1_idx == {(2, 3)}
