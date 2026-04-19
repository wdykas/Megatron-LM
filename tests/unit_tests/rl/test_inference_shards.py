# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for heterogeneous inference shards: different DP replicas with
different (TP, PP, EP) parallelisms for RL inference.

Run with:
    uv run python -m torch.distributed.run --nproc-per-node=4 \
        -m pytest tests/unit_tests/rl/test_inference_shards.py -v
"""
import copy
import gc
from typing import List, Optional

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.shards import InferenceShard, build_inference_pg_collections_for_shards
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.resharding.refit import clear_all_caches, swap_model_weights
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.rl.parallel_utils import (
    build_cross_shard_group,
    get_inference_shards,
    get_my_inference_shard,
    set_inference_shards,
    swap_weights_across_shards,
)
from tests.unit_tests.test_utilities import Utils


def _base_cfg(num_layers=2):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=32,
        num_attention_heads=8,
        num_query_groups=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def _build_gpt(cfg, pg_collection, vocab_size=128, seq_len=8):
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    return GPTModel(
        config=cfg,
        transformer_layer_spec=layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=(pp_rank == 0),
        post_process=(pp_rank == pp_size - 1),
        fp16_lm_cross_entropy=False,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
    )


def _find_my_shard(shards: List[InferenceShard]) -> Optional[InferenceShard]:
    rank = dist.get_rank()
    for s in shards:
        if s.rank_offset <= rank < s.rank_offset + s.world_size:
            return s
    return None


def test_parse_rl_inference_shards_string():
    """Non-distributed: parser in validate_args turns the CLI string into a list
    of dict shard specs with sensible defaults."""
    # Reproduce the parsing logic inline to avoid pulling in full arg parsing.
    # If this test diverges from megatron/training/arguments.py, update both.
    s = "tp=2,pp=1,ep=1,dp=1;tp=4,dp=2"
    parsed = []
    for shard_str in s.split(";"):
        spec = {}
        for kv in shard_str.split(","):
            k, v = kv.split("=")
            spec[k] = int(v)
        spec.setdefault("tp", 1)
        spec.setdefault("pp", 1)
        spec.setdefault("ep", 1)
        spec.setdefault("dp", 1)
        spec.setdefault("expt_tp", spec["tp"])
        parsed.append(spec)

    assert len(parsed) == 2
    assert parsed[0] == {"tp": 2, "pp": 1, "ep": 1, "dp": 1, "expt_tp": 2}
    assert parsed[1] == {"tp": 4, "pp": 1, "ep": 1, "dp": 2, "expt_tp": 4}


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for heterogeneous shard test"
)
def test_build_shards_basic():
    """Two shards with different TP: (TP=2,DP=1) + (TP=1,DP=2) over 4 ranks."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=2),
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )

        # Rank layout:
        #   ranks [0,1] -> shard 0
        #   ranks [2,3] -> shard 1
        assert len(shards) == 2
        assert shards[0].rank_offset == 0 and shards[0].world_size == 2
        assert shards[1].rank_offset == 2 and shards[1].world_size == 2

        my = _find_my_shard(shards)
        assert my is not None, "every rank should belong to exactly one shard here"
        assert shards[my.index].pg_collection is not None
        # Other shards must see pg_collection=None on this rank
        for s in shards:
            if s.index != my.index:
                assert s.pg_collection is None

        # Sanity-check group sizes
        pgc = shards[my.index].pg_collection
        spec = shards[my.index].spec
        assert dist.get_world_size(pgc.tp) == spec["tp"]
        assert dist.get_world_size(pgc.pp) == spec["pp"]
        assert dist.get_world_size(pgc.dp) == spec["dp"]
        # EP is part of expert grid; verify expert grid too
        assert dist.get_world_size(pgc.ep) == spec["ep"]
        assert dist.get_world_size(pgc.expt_tp) == spec["expt_tp"]
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for heterogeneous shard test"
)
def test_shard_rank_partition_explicit():
    """Check exact rank membership for the shard this rank belongs to."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        rank = dist.get_rank()
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),  # ranks 0,1
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=2),  # ranks 2,3
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )
        if rank in (0, 1):
            assert shards[0].pg_collection is not None
            # TP group should contain exactly {0,1} because tp=2
            tp_ranks = dist.get_process_group_ranks(shards[0].pg_collection.tp)
            assert sorted(tp_ranks) == [0, 1]
        else:
            assert shards[1].pg_collection is not None
            # TP=1 -> singleton group containing only this rank
            tp_ranks = dist.get_process_group_ranks(shards[1].pg_collection.tp)
            assert tp_ranks == [rank]
            # DP=2 -> group is {2,3}
            dp_ranks = dist.get_process_group_ranks(shards[1].pg_collection.dp)
            assert sorted(dp_ranks) == [2, 3]
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for idle-rank shard test"
)
def test_shards_with_idle_ranks():
    """When shards don't cover the full world, ranks outside all shards get
    ``pg_collection=None`` on every shard — i.e. they are idle."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        rank = dist.get_rank()
        # Shards only consume 3 of 4 ranks; rank 3 is idle.
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),  # ranks 0,1
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=1),  # rank 2
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )
        my = _find_my_shard(shards)
        if rank < 3:
            assert my is not None, f"rank {rank} expected to own a shard"
            assert shards[my.index].pg_collection is not None
            # The shards that aren't this rank's own must be None
            for s in shards:
                if s.index != my.index:
                    assert s.pg_collection is None
        else:
            # rank 3 is idle: no shard owns it
            assert my is None
            for s in shards:
                assert s.pg_collection is None
    finally:
        Utils.destroy_model_parallel()


def _training_pg_collection_from_mpu():
    """Build a ProcessGroupCollection from mpu's current training groups.

    Only populates the fields refit actually reads; everything else is left
    unset on the dataclass (init=False fields).
    """
    import megatron.core.parallel_state as mpu_ps
    from megatron.core.process_groups_config import ProcessGroupCollection

    pgs = ProcessGroupCollection()
    pgs.tp = mpu_ps.get_tensor_model_parallel_group()
    pgs.cp = mpu_ps.get_context_parallel_group()
    pgs.pp = mpu_ps.get_pipeline_model_parallel_group()
    pgs.ep = mpu_ps.get_expert_model_parallel_group()
    pgs.embd = mpu_ps.get_embedding_group()
    pgs.pos_embd = mpu_ps.get_position_embedding_group()
    pgs.dp = mpu_ps.get_data_parallel_group()
    pgs.tp_cp = mpu_ps.get_tensor_and_context_parallel_group()
    pgs.mp = mpu_ps.get_model_parallel_group()
    pgs.expt_tp = mpu_ps.get_expert_tensor_parallel_group()
    pgs.expt_dp = mpu_ps.get_expert_data_parallel_group()
    pgs.tp_ep = mpu_ps.get_expert_tensor_and_model_parallel_group()
    pgs.tp_ep_pp = mpu_ps.get_expert_tensor_model_pipeline_parallel_group()
    pgs.dp_cp = mpu_ps.get_data_parallel_group(with_context_parallel=True)
    pgs.tp_dp_cp = mpu_ps.get_tensor_and_data_parallel_group(with_context_parallel=True)
    return pgs


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for cross-shard group test"
)
def test_cross_shard_group_broadcast():
    """Verify that :func:`build_cross_shard_group` makes DP replicas reachable
    to each other via ordinary torch collectives."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        rank = dist.get_rank()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),  # ranks 0,1
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=2),  # ranks 2,3
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )
        set_inference_shards(shards)
        # Group spanning BOTH shards — i.e. all 4 ranks. Broadcasting a value
        # from shard 0's rank 0 to shard 1's ranks validates that cross-shard
        # communication is reachable through plain torch.distributed.
        both = build_cross_shard_group([0, 1])
        assert both is not None
        val = torch.tensor([42.0 if rank == 0 else 0.0], device=device)
        dist.broadcast(val, src=0, group=both)
        assert val.item() == 42.0

        # Also: group spanning only shard 1, verified from outside.
        only_shard1 = build_cross_shard_group([1])
        if rank in (2, 3):
            assert only_shard1 is not None
            members = dist.get_process_group_ranks(only_shard1)
            assert sorted(members) == [2, 3]
        else:
            assert only_shard1 is None
    finally:
        set_inference_shards(None)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for shard-url exchange test"
)
def test_shard_url_exchange_logic():
    """Validate the ``all_gather_object``-based URL/address exchange pattern
    used by ``MegatronLocalMulti.launch``. We don't actually spin up engines
    here — that requires a full RL setup — but the address-exchange bookkeeping
    is the non-obvious bit and is standalone-testable.
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    try:
        rank = dist.get_rank()
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),  # ranks 0,1
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=2),  # ranks 2,3
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )
        set_inference_shards(shards)

        # Simulate each shard's rank_offset holding the authoritative dp_addr;
        # every other rank contributes empty string.
        my_shard = get_my_inference_shard()
        assert my_shard is not None
        my_dp_addr = (
            f"tcp://host-shard{my_shard.index}:7{my_shard.index}"
            if rank == my_shard.rank_offset
            else ""
        )

        # This is the exact logic MegatronLocalMulti.launch runs.
        world_size = dist.get_world_size()
        all_addrs: List[Optional[str]] = [None] * world_size
        dist.all_gather_object(all_addrs, my_dp_addr)
        for s in shards:
            addr = all_addrs[s.rank_offset] or None
            s.coordinator_addr = addr

        # Every rank should now see every shard's authoritative address.
        assert shards[0].coordinator_addr == "tcp://host-shard0:70"
        assert shards[1].coordinator_addr == "tcp://host-shard1:71"
    finally:
        set_inference_shards(None)
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="need >=4 GPUs for heterogeneous refit"
)
def test_heterogeneous_refit_end_to_end():
    """Build a TP=4 training model, two heterogeneous inference shards
    (TP=2 on ranks 0-1, TP=1 on ranks 2-3), refit through
    ``swap_weights_across_shards``, and verify that each shard's destination
    model produces the same logits as the training model on the same input."""
    try:
        import transformer_engine  # noqa: F401
    except Exception:
        pytest.skip("Transformer Engine not available")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=4, pipeline_model_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        vocab, seq_len, batch = 128, 8, 2
        cfg = _base_cfg()

        src_pgs = _training_pg_collection_from_mpu()
        src_cfg = copy.deepcopy(cfg)
        src_cfg.tensor_model_parallel_size = 4
        src_model = _build_gpt(src_cfg, src_pgs, vocab, seq_len).to(device).eval()

        # --- Two heterogeneous inference shards ---
        specs = [
            dict(tp=2, pp=1, ep=1, expt_tp=2, dp=1),  # ranks 0,1
            dict(tp=1, pp=1, ep=1, expt_tp=1, dp=2),  # ranks 2,3
        ]
        shards = build_inference_pg_collections_for_shards(
            total_world_size=dist.get_world_size(), shards=specs
        )
        set_inference_shards(shards)

        my = _find_my_shard(shards)
        dst_cfg = copy.deepcopy(cfg)
        dst_cfg.tensor_model_parallel_size = my.spec["tp"]
        dst_model = _build_gpt(dst_cfg, my.pg_collection, vocab, seq_len).to(device).eval()

        # --- Refit: drives swap_model_weights once per shard, collectively ---
        swap_weights_across_shards([src_model], [dst_model], refit_method="nccl")

        tokens = torch.randint(
            0, vocab, size=(batch, seq_len), device=device, dtype=torch.long
        )
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        mask = torch.ones((batch, 1, seq_len, seq_len), device=device, dtype=torch.bool)

        with torch.no_grad():
            src_out = src_model(tokens, pos_ids, mask)
            dst_out = dst_model(tokens, pos_ids, mask)

        assert torch.isfinite(src_out).all(), "source forward non-finite"
        assert torch.isfinite(dst_out).all(), "dst forward non-finite"

        max_diff = (src_out - dst_out).abs().max().item()
        assert torch.allclose(src_out, dst_out, atol=5e-4, rtol=5e-4), (
            f"shard {my.index} (tp={my.spec['tp']}): refit outputs differ "
            f"(max_diff={max_diff:.6f})"
        )
    finally:
        set_inference_shards(None)
        clear_all_caches()
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
