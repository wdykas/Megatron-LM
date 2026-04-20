# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""KV-transport tests for :func:`execute_kv_migration_plan`.

The transport is the bridge between the engine-internal
``snapshot_request`` / ``detach_request`` / ``inject_request`` surgery
and the cross-shard `build_cross_shard_group` process group. It takes
the ops emitted by :func:`build_kv_migration_plan`, stages contiguous
tensors, and routes them through :class:`NCCLCopyService` so the
destination engine's ``memory_buffer`` ends up holding the source's
KV slices at the destination-side block ids.

These tests exercise three shapes of transport:

- **local-rank copy** (non-distributed): a single rank acting as both
  src and dst — proves the gather → (local-copy via same-rank task_id
  matching) → scatter path for non-MLA shapes, MLA shapes, and a
  head-range subslice.
- **cross-rank transfer** (4-GPU NCCL): two "shards" on different
  ranks, execute a matched-TP plan, verify the destination's
  ``memory_buffer`` matches the source's original contents.
- **heterogeneous-TP transfer** (4-GPU NCCL): src_tp=2, dst_tp=1,
  verifies head-dim gather semantics end-to-end over the wire.
"""
from typing import List

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.engines.request_migration import (
    KVLayout,
    RequestMigrationBundle,
    build_kv_migration_plan,
    execute_kv_migration_plan,
)


def _alloc_memory_buffer(
    *,
    num_layers_pp: int,
    total_blocks: int,
    block_size: int,
    heads_per_tp: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cuda"),
    is_mla: bool = False,
    kv_reduced_dim: int = 0,
) -> torch.Tensor:
    if is_mla:
        return torch.empty(
            (num_layers_pp, total_blocks, block_size, kv_reduced_dim),
            dtype=dtype,
            device=device,
        )
    return torch.empty(
        (2, num_layers_pp, total_blocks, block_size, heads_per_tp, head_dim),
        dtype=dtype,
        device=device,
    )


def _build_bundle(
    *,
    src_tp: int,
    dst_tp: int,
    num_kv_heads: int = 8,
    num_layers: int = 2,
    num_blocks: int = 2,
    block_size: int = 4,
    head_dim: int = 8,
    is_mla: bool = False,
    kv_reduced_dim: int = 0,
    src_block_ids: List[int] = (0, 1),
) -> RequestMigrationBundle:
    layout_kwargs = dict(
        pp_size=1,
        num_layers_total=num_layers,
        num_kv_heads_total=num_kv_heads,
        head_dim=head_dim,
        block_size_tokens=block_size,
        is_mla=is_mla,
        kv_reduced_dim=kv_reduced_dim if is_mla else None,
    )
    return RequestMigrationBundle(
        request_id=0,
        prompt_tokens=[0],
        generated_tokens=[],
        sampling_params={},
        num_kv_blocks=len(src_block_ids),
        last_block_offset=0,
        src_block_ids=list(src_block_ids),
        src_layout=KVLayout(tp_size=src_tp, **layout_kwargs),
        dst_layout=KVLayout(tp_size=dst_tp, **layout_kwargs),
    )


class TestTransportLocalCopy:
    """Single-rank sanity tests — no distributed setup required.

    When ``src_rank == dst_rank == current_rank`` the transport uses
    :class:`NCCLCopyService`'s task_id-matched local-copy path; no
    network traffic, just a device-local copy on its dedicated stream.
    These tests still need ``torch.distributed`` initialised because
    :class:`NCCLCopyService` queries ``dist.get_rank()`` / group size,
    so we spin up a 1-rank gloo group on the fly.
    """

    @classmethod
    def setup_class(cls):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory_buffer tensors")
        # The local-copy path uses src_rank=dst_rank=0 with
        # NCCLCopyService's task_id-matched same-rank shortcut. Under a
        # multi-rank torchrun harness every rank would need its own
        # memory_buffer + collective participation — out of scope for a
        # unit test (the cross-rank tests below cover it). Skip if the
        # harness is already multi-rank. Otherwise stand up a 1-rank
        # gloo world so ``NCCLCopyService.__init__`` can query
        # ``dist.get_rank()``.
        import os

        if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
            pytest.skip("local-copy test is non-distributed; see TestTransportCrossRank")
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29530")
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            dist.init_process_group(backend="gloo", init_method="env://")

    def _roundtrip_buffer(self, *, is_mla: bool, head_subslice: bool):
        """Common helper — copies from src_blocks to dst_blocks on the
        same rank via the local-copy path and compares."""
        num_kv_heads = 8
        num_layers = 2
        block_size = 4
        head_dim = 8
        kv_reduced_dim = 16
        bundle = _build_bundle(
            src_tp=1 if is_mla else 2,
            dst_tp=1 if is_mla else 2,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            block_size=block_size,
            head_dim=head_dim,
            is_mla=is_mla,
            kv_reduced_dim=kv_reduced_dim,
            src_block_ids=[0, 1],
        )
        layout = bundle.src_layout

        # For the local-copy path both src and dst live on the same
        # rank and share the same memory buffer. We therefore allocate
        # a *global-sized* buffer (all heads) and set both head offsets
        # to 0, so every op's global head_range translates 1:1 into
        # local indices.
        buf = _alloc_memory_buffer(
            num_layers_pp=num_layers,
            total_blocks=4,  # room for 2 src + 2 dst
            block_size=block_size,
            heads_per_tp=num_kv_heads,
            head_dim=head_dim,
            is_mla=is_mla,
            kv_reduced_dim=kv_reduced_dim,
        )
        # Deterministic fill so we can compare.
        with torch.no_grad():
            buf.copy_(torch.arange(buf.numel(), dtype=buf.dtype, device=buf.device).view(buf.shape))

        ops = build_kv_migration_plan(
            bundle,
            src_global_rank_of=lambda tp, pp: 0,
            dst_global_rank_of=lambda tp, pp: 0,
            dst_block_ids=[2, 3],
        )
        if head_subslice:
            # Narrow every op's head_range to the first half. The plan
            # builder normally spits out full-owner ranges for same-TP;
            # we manually clip here to prove _gather/_scatter respect
            # partial ranges.
            for op in ops:
                lo, hi = op.head_range
                new_hi = lo + (hi - lo) // 2 if hi - lo > 1 else hi
                op.head_range = (lo, new_hi)

        # Snapshot expected slices before running.
        expected_slices = []
        for op in ops:
            src_t = torch.tensor(op.src_block_ids, dtype=torch.long, device=buf.device)
            if is_mla:
                slc = buf[
                    op.layer_range[0] : op.layer_range[1],
                    src_t,
                    :,
                    op.head_range[0] : op.head_range[1],
                ].clone()
            else:
                slc = buf[
                    :,
                    op.layer_range[0] : op.layer_range[1],
                    src_t,
                    :,
                    op.head_range[0] : op.head_range[1],
                    :,
                ].clone()
            expected_slices.append(slc)

        execute_kv_migration_plan(ops, buf, layout, group=None)

        # Destination slices should match the source slices we copied.
        for op, expected in zip(ops, expected_slices):
            dst_t = torch.tensor(op.dst_block_ids, dtype=torch.long, device=buf.device)
            if is_mla:
                got = buf[
                    op.layer_range[0] : op.layer_range[1],
                    dst_t,
                    :,
                    op.head_range[0] : op.head_range[1],
                ]
            else:
                got = buf[
                    :,
                    op.layer_range[0] : op.layer_range[1],
                    dst_t,
                    :,
                    op.head_range[0] : op.head_range[1],
                    :,
                ]
            assert torch.equal(got, expected), f"mismatch for op {op}"

    def test_local_roundtrip_non_mla(self):
        self._roundtrip_buffer(is_mla=False, head_subslice=False)

    def test_local_roundtrip_non_mla_head_subslice(self):
        self._roundtrip_buffer(is_mla=False, head_subslice=True)

    def test_local_roundtrip_mla(self):
        self._roundtrip_buffer(is_mla=True, head_subslice=False)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="need 4 GPUs for cross-rank transport test",
)
class TestTransportCrossRank:
    """Cross-rank NCCL transport tests — must be launched under torchrun."""

    @classmethod
    def setup_class(cls):
        # Torchrun sets the env variables but leaves ``dist.init_process_group``
        # to the user. Bring up the default NCCL world if a harness hasn't
        # done it already.
        if not dist.is_initialized():
            import os

            if "RANK" not in os.environ:
                pytest.skip("cross-rank test requires torchrun to set RANK/WORLD_SIZE")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl")
        if dist.get_world_size() < 4:
            pytest.skip("cross-rank test requires world_size >= 4")
    """Real NCCL transport tests.

    Models two shards that occupy disjoint rank windows:
    - shard 0: ranks [0, 1] with TP=2
    - shard 1: ranks [2, 3] with TP=2 (matched TP → one op per rank pair)

    The test fills the source shard's ``memory_buffer`` with rank-
    identifiable values, runs the plan, and verifies the destination
    shard's buffer now holds the source's bytes at the destination
    block ids.

    Second case: src_tp=2 → dst_tp=1 heterogeneous reshape (gather).
    Both source ranks send their half-head slice to the single
    destination rank.
    """

    def _run_cross_rank_case(
        self,
        *,
        src_tp: int,
        dst_tp: int,
        num_kv_heads: int,
    ):
        """Shared body: every rank builds its shard-local buffer, the
        plan says who sends what to whom, and we verify after the run."""
        rank = dist.get_rank()
        world = dist.get_world_size()
        assert world == 4
        device = torch.device(f"cuda:{rank}")

        # Shard layout: shard 0 = ranks [0..src_tp), shard 1 = ranks
        # [src_tp..src_tp+dst_tp). Every rank is in exactly one shard.
        src_ranks = list(range(src_tp))
        dst_ranks = list(range(src_tp, src_tp + dst_tp))
        in_src = rank in src_ranks
        in_dst = rank in dst_ranks

        num_layers = 2
        block_size = 4
        head_dim = 8
        num_blocks = 2
        src_heads_per_tp = num_kv_heads // src_tp
        dst_heads_per_tp = num_kv_heads // dst_tp

        # Allocate this rank's shard-local buffer (right heads count).
        if in_src:
            heads_per_tp = src_heads_per_tp
            my_src_offset = (rank - src_ranks[0]) * src_heads_per_tp
            my_dst_offset = 0  # not used
        elif in_dst:
            heads_per_tp = dst_heads_per_tp
            my_src_offset = 0
            my_dst_offset = (rank - dst_ranks[0]) * dst_heads_per_tp
        else:
            heads_per_tp = max(src_heads_per_tp, dst_heads_per_tp)
            my_src_offset = 0
            my_dst_offset = 0

        buf = _alloc_memory_buffer(
            num_layers_pp=num_layers,
            total_blocks=max(num_blocks, 2),
            block_size=block_size,
            heads_per_tp=heads_per_tp,
            head_dim=head_dim,
        )
        # Fill src-side with rank-identifiable content (rank * 1000 + position).
        if in_src:
            base = torch.arange(buf.numel(), dtype=buf.dtype, device=device).view(buf.shape)
            buf.copy_(rank * 1000.0 + base)

        # Build the plan. Rank lambdas match the shard layout above.
        bundle = _build_bundle(
            src_tp=src_tp,
            dst_tp=dst_tp,
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            block_size=block_size,
            head_dim=head_dim,
            src_block_ids=[0, 1],
        )
        ops = build_kv_migration_plan(
            bundle,
            src_global_rank_of=lambda tp, pp, _start=src_ranks[0]: _start + tp,
            dst_global_rank_of=lambda tp, pp, _start=dst_ranks[0]: _start + tp,
            dst_block_ids=[0, 1],
        )

        # Build the cross-shard group. `dist.new_group` is
        # world-collective, so every rank must enter the call even if
        # it isn't in the group (we already are, given world=4 and the
        # union is ranks 0-3, but keep the pattern for clarity).
        group_ranks = sorted(set(src_ranks + dst_ranks))
        group = dist.new_group(ranks=group_ranks)

        # Snapshot src-side contents so we can predict what the dst
        # should receive without needing a reference engine.
        src_ranks_to_heads = {
            r: (
                (r - src_ranks[0]) * src_heads_per_tp,
                (r - src_ranks[0] + 1) * src_heads_per_tp,
            )
            for r in src_ranks
        }

        # Ranks outside src_ranks ∪ dst_ranks are not in the cross-shard
        # process group (``dist.new_group`` returns a NON_GROUP_MEMBER
        # sentinel for them) — they must not enter the transport.
        if in_src or in_dst:
            execute_kv_migration_plan(
                ops,
                buf,
                bundle.src_layout if in_src else bundle.dst_layout,
                group,
                my_src_head_offset=my_src_offset,
                my_dst_head_offset=my_dst_offset,
            )

        # After the run, every dst-side op's head range should hold
        # the source rank's original values. We reconstruct the
        # expectation algebraically (no collective needed).
        if in_dst:
            for op in ops:
                if op.dst_rank != rank:
                    continue
                # Figure out which src rank sent this op and what base
                # value it used when filling its buffer.
                sr = op.src_rank
                src_rank_lo, src_rank_hi = src_ranks_to_heads[sr]
                # The sender's local head offset for this op's global
                # head_range:
                send_local_lo = op.head_range[0] - src_rank_lo
                send_local_hi = op.head_range[1] - src_rank_lo
                # Build the expected contiguous tensor that would have
                # been produced by the sender's _gather_kv_slice on its
                # rank-identifiable buffer. Sender buffer shape:
                sender_buf_shape = (
                    2,
                    num_layers,
                    max(num_blocks, 2),
                    block_size,
                    src_heads_per_tp,
                    head_dim,
                )
                base_flat = torch.arange(
                    torch.Size(sender_buf_shape).numel(),
                    dtype=buf.dtype,
                    device=device,
                )
                sender_buf = sr * 1000.0 + base_flat.view(sender_buf_shape)
                src_blocks = torch.tensor(op.src_block_ids, device=device, dtype=torch.long)
                expected = sender_buf[
                    :,
                    op.layer_range[0] : op.layer_range[1],
                    src_blocks,
                    :,
                    send_local_lo:send_local_hi,
                    :,
                ].contiguous()

                dst_blocks = torch.tensor(op.dst_block_ids, device=device, dtype=torch.long)
                recv_local_lo = op.head_range[0] - my_dst_offset
                recv_local_hi = op.head_range[1] - my_dst_offset
                got = buf[
                    :,
                    op.layer_range[0] : op.layer_range[1],
                    dst_blocks,
                    :,
                    recv_local_lo:recv_local_hi,
                    :,
                ]
                assert torch.equal(got, expected), (
                    f"rank {rank} dst op mismatch: src={op.src_rank} "
                    f"heads={op.head_range}"
                )

        if in_src or in_dst:
            dist.destroy_process_group(group)

    def test_matched_tp_transfer(self):
        """src_tp=2 → dst_tp=2 (matched): one op per rank pair."""
        self._run_cross_rank_case(src_tp=2, dst_tp=2, num_kv_heads=8)

    def test_heterogeneous_tp2_to_tp1_transfer(self):
        """src_tp=2 → dst_tp=1: two source ranks gather into one dst rank.

        Layout on a 4-GPU world:
          - shard 0 (src): ranks [0, 1], each holding half the heads
          - shard 1 (dst): rank [2] alone, holding all heads
          - rank 3 is a bystander (not in the cross-shard group)

        Proves the head-dim reshape path on real NCCL: the dst rank
        receives two tensors (one from each src) whose head ranges
        concatenate to cover the full head dim.
        """
        self._run_cross_rank_case(src_tp=2, dst_tp=1, num_kv_heads=4)
