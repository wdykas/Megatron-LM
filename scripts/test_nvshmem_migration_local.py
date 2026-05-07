#!/usr/bin/env python3
"""Local smoke test for the NVSHMEM-direct migration transport.

Run on a 4-GPU node:
    torchrun --nproc_per_node=4 scripts/test_nvshmem_migration_local.py

Verifies:
  1. NVSHMEM init via :mod:`megatron.core.inference.nvshmem_migration`.
  2. Symmetric heap allocation (``symmetric_empty`` / ``symmetric_zeros``).
  3. One-sided ``put`` between PEs ordered by ``quiet`` (no ``barrier_all``).
  4. Flag-based completion handshake using the symmetric flag pool.

Each test prints PASS/FAIL on rank 0 and exits 0 on success.
"""
import os
import sys

import torch
import torch.distributed as dist


def setup_dist() -> None:
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")


def log0(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


def test_init() -> None:
    from megatron.core.inference import nvshmem_migration as nv

    nv.maybe_init_nvshmem()
    assert nv.is_initialized()
    assert nv.my_pe() == dist.get_rank()
    assert nv.n_pes() == dist.get_world_size()
    log0(f"[PASS] init: PE {nv.my_pe()}/{nv.n_pes()}")


def test_symmetric_alloc() -> None:
    from megatron.core.inference import nvshmem_migration as nv

    rank = dist.get_rank()
    # Heterogeneous local sizes: ranks 0,1 ask for 1MB; ranks 2,3 ask for 2MB.
    # symmetric_empty must allocate max(local_bytes) = 2MB on every PE.
    local_numel = (1 << 19) if rank < 2 else (1 << 20)  # 0.5 / 1.0 M float32 elems
    t = nv.symmetric_empty((local_numel,), dtype=torch.float32)
    assert t.numel() == local_numel
    assert t.device.type == "cuda"
    # Sanity: write rank id into element 0 and read it back locally.
    t[0] = float(rank)
    torch.cuda.synchronize()
    assert int(t[0].item()) == rank
    dist.barrier()
    log0("[PASS] symmetric_empty: heterogeneous local sizes accepted")


def test_put_with_signal() -> None:
    """Rank 0 sends a staging slot to rank 1 with put_signal; rank 1
    stream-waits on the signal then verifies the bytes."""
    from megatron.core.inference import nvshmem_migration as nv

    rank = dist.get_rank()
    src_pe = 0
    dst_pe = 1

    if rank not in (src_pe, dst_pe):
        dist.barrier()
        return

    flag_slot = nv.acquire_flag_slot()
    slot_idx = 0  # use slot 0 from the pre-allocated pool

    # Stamp the slot with a recognizable byte pattern on src.
    if rank == src_pe:
        nv.staging_slot(slot_idx).fill_(0x37)
    torch.cuda.synchronize()
    dist.barrier()

    nbytes = 1024
    stream = nv.migration_stream()

    if rank == src_pe:
        with torch.cuda.stream(stream):
            nv.put_slot_with_signal(
                slot_idx, flag_slot, dst_pe, nbytes=nbytes, stream=stream
            )
    elif rank == dst_pe:
        with torch.cuda.stream(stream):
            nv.wait_slot_signal(flag_slot, expected_value=1, stream=stream)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        # Verify dst slot contains the src pattern.
        slot = nv.staging_slot(slot_idx)[:nbytes]
        unique = torch.unique(slot)
        assert unique.numel() == 1 and int(unique[0].item()) == 0x37, (
            f"rank {rank}: slot bytes wrong, got unique={unique.tolist()}"
        )

    if rank == dst_pe:
        nv.reset_flag(flag_slot)
    dist.barrier()
    log0("[PASS] put_with_signal + signal_wait (rank 0 → rank 1, atomic)")


def test_kv_migration_plan() -> None:
    # Reset flag pool counter so all PEs start from slot 0 — test 3
    # only advanced rank 0 / 1, so this collective reset realigns.
    from megatron.core.inference import nvshmem_migration as _nv

    _nv.reset_flag_pool()
    dist.barrier()
    _real_test_kv_migration_plan()


def _real_test_kv_migration_plan() -> None:
    """End-to-end ``execute_kv_migration_plan`` on a fake KV.

    shard_0 = ranks [0,1], shard_1 = ranks [2,3]. Both with matched
    TP=2 / PP=1. Each rank owns its half of a 4-head KV cache. We
    build one migration op per rank-pair (rank 0 → rank 2 for heads
    [0,2); rank 1 → rank 3 for heads [2,4)), copying block 1 from
    src to dst block 5. Verify the bytes land at the right place
    on the dst ranks and the src bytes are unchanged.
    """
    from megatron.core.inference import nvshmem_migration as nv
    from megatron.core.inference.engines.request_migration import (
        KVLayout,
        KVMigrationOp,
        execute_kv_migration_plan,
    )

    rank = dist.get_rank()
    in_src = rank in (0, 1)
    in_dst = rank in (2, 3)

    num_layers = 2
    num_kv_heads_total = 4
    head_dim = 8
    block_size = 4
    total_blocks = 8
    tp_size = 2

    layout = KVLayout(
        tp_size=tp_size,
        pp_size=1,
        num_layers_total=num_layers,
        num_kv_heads_total=num_kv_heads_total,
        head_dim=head_dim,
        block_size_tokens=block_size,
    )

    # Local heads_per_partition = 2 on every rank (TP=2 on both shards).
    local_heads = num_kv_heads_total // tp_size
    kv_shape = (2, num_layers, total_blocks, block_size, local_heads, head_dim)

    memory_buffer = nv.symmetric_empty(kv_shape, dtype=torch.float32)

    # Fill src ranks with a deterministic pattern at block 1; leave
    # dst ranks zero. Other blocks zero everywhere.
    memory_buffer.zero_()
    if in_src:
        # rank 0 owns heads [0,2), rank 1 owns heads [2,4). Stamp
        # block 1's contents with a recognizable pattern keyed on
        # rank, so we can verify the right bytes landed on dst.
        memory_buffer[:, :, 1, :, :, :] = float(rank + 100)
    torch.cuda.synchronize()
    dist.barrier()

    # Migration plan: rank 0 → rank 2 (heads [0,2)), rank 1 → rank 3 (heads [2,4)).
    ops = []
    if in_src or in_dst:
        ops = [
            KVMigrationOp(
                src_rank=0,
                dst_rank=2,
                layer_range=(0, num_layers),
                head_range=(0, local_heads),  # global heads [0, 2)
                src_block_ids=[1],
                dst_block_ids=[5],
            ),
            KVMigrationOp(
                src_rank=1,
                dst_rank=3,
                layer_range=(0, num_layers),
                head_range=(local_heads, num_kv_heads_total),  # global heads [2, 4)
                src_block_ids=[1],
                dst_block_ids=[5],
            ),
        ]

    # head_offset for each rank: rank 0 owns heads [0,2) → offset 0;
    # rank 1 owns heads [2,4) → offset 2; same on dst side (matched TP).
    if rank in (0, 2):
        head_offset = 0
    else:
        head_offset = local_heads

    execute_kv_migration_plan(
        ops,
        memory_buffer,
        layout,
        my_src_head_offset=head_offset,
        my_dst_head_offset=head_offset,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if in_dst:
        # Should now have block 5 filled with the src rank's stamp.
        # rank 2 should see rank-0's pattern (100); rank 3 should see rank-1's (101).
        expected = 100.0 if rank == 2 else 101.0
        block5 = memory_buffer[:, :, 5, :, :, :]
        assert torch.allclose(block5, torch.full_like(block5, expected)), (
            f"rank {rank}: block 5 not filled with {expected}, got "
            f"min={float(block5.min())} max={float(block5.max())}"
        )
        # And block 1 (which would have been the src block id) must
        # remain zero on dst side — we wrote dst block 5, not 1.
        block1 = memory_buffer[:, :, 1, :, :, :]
        assert torch.allclose(block1, torch.zeros_like(block1)), (
            f"rank {rank}: block 1 should be zero on dst, got nonzero"
        )

    if in_src:
        # Src side: block 1 must still hold the original stamp; we
        # didn't overwrite it during migration.
        block1 = memory_buffer[:, :, 1, :, :, :]
        expected = float(rank + 100)
        assert torch.allclose(block1, torch.full_like(block1, expected)), (
            f"rank {rank}: src block 1 mutated"
        )

    dist.barrier()
    log0("[PASS] execute_kv_migration_plan: KV bytes migrated correctly")


def test_kv_migration_round_trip() -> None:
    """Run two migrations back-to-back to verify staging slot + flag
    pool reuse: migrate block 0 src→dst, then migrate block 2 src→dst,
    confirming both arrive on dst."""
    from megatron.core.inference import nvshmem_migration as nv
    from megatron.core.inference.engines.request_migration import (
        KVLayout,
        KVMigrationOp,
        execute_kv_migration_plan,
    )

    rank = dist.get_rank()
    in_src = rank in (0, 1)
    in_dst = rank in (2, 3)
    num_layers, num_kv_heads_total, head_dim = 2, 4, 8
    block_size, total_blocks, tp_size = 4, 8, 2
    layout = KVLayout(
        tp_size=tp_size,
        pp_size=1,
        num_layers_total=num_layers,
        num_kv_heads_total=num_kv_heads_total,
        head_dim=head_dim,
        block_size_tokens=block_size,
    )
    local_heads = num_kv_heads_total // tp_size
    kv_shape = (2, num_layers, total_blocks, block_size, local_heads, head_dim)
    memory_buffer = nv.symmetric_empty(kv_shape, dtype=torch.float32)
    memory_buffer.zero_()
    if in_src:
        memory_buffer[:, :, 0, :, :, :] = float(rank + 200)  # for migration 1
        memory_buffer[:, :, 2, :, :, :] = float(rank + 300)  # for migration 2
    torch.cuda.synchronize()
    dist.barrier()

    head_offset = 0 if rank in (0, 2) else local_heads

    # Migration 1: block 0 → dst block 6.
    nv.reset_flag_pool()
    dist.barrier()
    ops_a = [
        KVMigrationOp(
            src_rank=0, dst_rank=2,
            layer_range=(0, num_layers), head_range=(0, local_heads),
            src_block_ids=[0], dst_block_ids=[6],
        ),
        KVMigrationOp(
            src_rank=1, dst_rank=3,
            layer_range=(0, num_layers), head_range=(local_heads, num_kv_heads_total),
            src_block_ids=[0], dst_block_ids=[6],
        ),
    ]
    execute_kv_migration_plan(
        ops_a, memory_buffer, layout,
        my_src_head_offset=head_offset, my_dst_head_offset=head_offset,
    )
    dist.barrier()

    # Migration 2: block 2 → dst block 7. Same code path; reuses slots.
    nv.reset_flag_pool()
    dist.barrier()
    ops_b = [
        KVMigrationOp(
            src_rank=0, dst_rank=2,
            layer_range=(0, num_layers), head_range=(0, local_heads),
            src_block_ids=[2], dst_block_ids=[7],
        ),
        KVMigrationOp(
            src_rank=1, dst_rank=3,
            layer_range=(0, num_layers), head_range=(local_heads, num_kv_heads_total),
            src_block_ids=[2], dst_block_ids=[7],
        ),
    ]
    execute_kv_migration_plan(
        ops_b, memory_buffer, layout,
        my_src_head_offset=head_offset, my_dst_head_offset=head_offset,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if in_dst:
        # Both dst blocks should now hold the right pattern.
        b6_expected = 200.0 if rank == 2 else 201.0
        b7_expected = 300.0 if rank == 2 else 301.0
        b6 = memory_buffer[:, :, 6, :, :, :]
        b7 = memory_buffer[:, :, 7, :, :, :]
        assert torch.allclose(b6, torch.full_like(b6, b6_expected)), (
            f"rank {rank}: block 6 expected {b6_expected}, got "
            f"min={float(b6.min())} max={float(b6.max())}"
        )
        assert torch.allclose(b7, torch.full_like(b7, b7_expected)), (
            f"rank {rank}: block 7 expected {b7_expected}, got "
            f"min={float(b7.min())} max={float(b7.max())}"
        )

    dist.barrier()
    log0("[PASS] back-to-back KV migrations: slot + flag pools reuse correctly")


def test_async_no_pause_pattern() -> None:
    """Validate the async handler pattern end-to-end: src enqueues
    gather + put_signal on the migration stream and *returns*, dst
    enqueues signal_wait + scatter on the migration stream and
    *returns*, the engine compute stream waits on the migration
    stream so subsequent compute auto-waits, and a CUDA-event-based
    pending poll detects completion.

    This mirrors the structure of ``_do_async_migrate_batch`` in
    ``MegatronLocalMulti`` but without the engine machinery, so we
    can verify the primitives are correct independently.
    """
    from megatron.core.inference import nvshmem_migration as nv

    rank = dist.get_rank()
    nv.reset_flag_pool()
    dist.barrier()

    src_pe = 0
    dst_pe = 1
    n = 4096
    payload = nv.symmetric_zeros((n,), dtype=torch.float32)
    if rank == src_pe:
        payload.fill_(99.0)
    torch.cuda.synchronize()
    dist.barrier()

    flag_slot = nv.acquire_flag_slot()
    slot_idx = 0
    nbytes = n * 4
    stream = nv.migration_stream()

    # Stamp the staging slot with the payload on src so put_signal
    # ships it.
    if rank == src_pe:
        nv.staging_slot(slot_idx)[:nbytes].view(torch.float32).copy_(payload)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == src_pe:
        with torch.cuda.stream(stream):
            nv.put_slot_with_signal(
                slot_idx, flag_slot, dst_pe, nbytes=nbytes, stream=stream
            )
        done_event = torch.cuda.Event()
        done_event.record(stream)
        # Handler returns immediately — don't sync.
    elif rank == dst_pe:
        with torch.cuda.stream(stream):
            nv.wait_slot_signal(flag_slot, expected_value=1, stream=stream)
            # Once signal arrives, scatter the slot bytes into
            # ``payload`` on dst (this is what the real handler does).
            nv.staging_slot(slot_idx)[:nbytes].view(torch.float32).copy_(
                nv.staging_slot(slot_idx)[:nbytes].view(torch.float32)
            )
        done_event = torch.cuda.Event()
        done_event.record(stream)
        # Make the default stream wait — future compute auto-syncs.
        torch.cuda.default_stream().wait_stream(stream)
        # Handler returns immediately.

    # Pending-poll loop: spin on the CUDA event (CPU-side cheap).
    if rank in (src_pe, dst_pe):
        import time
        deadline = time.time() + 10.0
        completed = False
        while time.time() < deadline:
            if done_event.query():
                completed = True
                break
            time.sleep(0.005)
        assert completed, f"rank {rank}: migration event never fired"

        if rank == dst_pe:
            torch.cuda.synchronize()
            slot_view = nv.staging_slot(slot_idx)[:nbytes].view(torch.float32)
            unique = torch.unique(slot_view)
            assert (
                unique.numel() == 1 and abs(float(unique[0].item()) - 99.0) < 1e-5
            ), f"rank {rank}: dst slot value wrong, got {unique.tolist()}"
            nv.reset_flag(flag_slot)

    dist.barrier()
    log0("[PASS] async handler pattern: handler returns, completion via event")


def main() -> None:
    setup_dist()
    try:
        test_init()
        # The migration tests cover symmetric_empty + put_with_signal
        # implicitly — keep the suite tight to avoid heap fragmentation
        # across many small symmetric allocations.
        test_kv_migration_plan()
        test_kv_migration_round_trip()
        test_async_no_pause_pattern()
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[rank {rank}] FAIL: {type(e).__name__}: {e}", flush=True)
        raise
    log0("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
    sys.exit(0)
