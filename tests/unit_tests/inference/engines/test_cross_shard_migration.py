# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end cross-shard request-migration test (Step 4+5).

Brings up TWO real GPT engines on a 4-GPU job:

  - shard 0 engine on ranks [0, 1] with TP=2
  - shard 1 engine on ranks [2, 3] with TP=2

Runs a request on the shard 0 engine until it has some decode tokens +
KV blocks, then calls :func:`migrate_request_cross_shard` to move it
to the shard 1 engine. The test asserts:

  - the source engine no longer tracks the request (detach worked)
  - the destination engine has the request registered with the fresh
    block ids and the injected tokens
  - the destination engine can step and generate a new token on the
    migrated request (decode resumes from the migrated KV)
"""
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.engines.request_migration import (
    KVLayout,
    migrate_request_cross_shard,
)


@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="cross-shard migration test requires 4 GPUs",
)
class TestCrossShardMigration:
    """Real-engine migration across two shards on a 4-GPU node."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                pytest.skip("requires torchrun")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl")
        if dist.get_world_size() != 4:
            pytest.skip("requires world_size == 4")

    @torch.inference_mode()
    def test_migrate_one_request_shard0_to_shard1(self):
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            DynamicEngineTestConfig,
        )
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            TestDynamicInferenceEngine as _EngineScaffold,
        )

        rank = dist.get_rank()
        in_shard0 = rank in (0, 1)
        in_shard1 = rank in (2, 3)
        src_ranks = [0, 1]
        dst_ranks = [2, 3]

        # Each rank only builds *its* shard's engine. Both shards use
        # TP=2; the scaffold's `_build_test_env` calls
        # ``Utils.initialize_model_parallel(tp, pp)`` which sets the
        # training TP group to include all ranks in the current group.
        # Since torchrun already has 4 ranks, TP=2 partitions them into
        # [0,1] and [2,3] — exactly our two shards.
        # Two pre-populated requests per shard (one to migrate away, one
        # to keep busy). The dst shard also gets one so its engine is
        # warmed up with real batch state before we inject — a fully
        # cold dst engine would fail on decode metadata. Context
        # capacity of 4 leaves room for the injected 3rd slot on dst.
        config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=32,
            num_gap_steps=2,
            model_provider="gpt",
            tensor_model_parallel_size=2,
            context_max_requests=4,
        )
        env = _EngineScaffold._build_test_env(config)

        # Both shards add their own requests and step so every engine
        # is warm and has at least one request in the active batch
        # that is distinct from anything involved in the migration.
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        # Confirm the source shard has something to migrate.
        if in_shard0:
            _, src_blocks = env.engine.snapshot_request(env.requests[0].request_id)
            assert src_blocks.numel() > 0, "expected source request to have KV blocks"

        # Describe both shards' KV layouts. Both shards have TP=2 here
        # (matched TP migration) so the plan will emit one op per
        # rank pair; stretch goals can swap in a mismatched layout.
        model_config = env.engine.controller.inference_wrapped_model.model.config
        ctx = env.engine.context
        num_kv_heads = model_config.num_query_groups or model_config.num_attention_heads
        layout_common = dict(
            pp_size=1,
            num_layers_total=model_config.num_layers,
            num_kv_heads_total=num_kv_heads,
            head_dim=ctx.hidden_size_per_attention_head,
            block_size_tokens=ctx.block_size_tokens,
            is_mla=False,
        )
        src_layout = KVLayout(tp_size=2, **layout_common)
        dst_layout = KVLayout(tp_size=2, **layout_common)

        # Head offset for this rank in its shard's TP group.
        my_src_head_offset = 0
        my_dst_head_offset = 0
        if in_shard0:
            my_src_head_offset = (rank - src_ranks[0]) * (num_kv_heads // 2)
        if in_shard1:
            my_dst_head_offset = (rank - dst_ranks[0]) * (num_kv_heads // 2)

        cross_shard_group = dist.new_group(ranks=sorted(src_ranks + dst_ranks))

        # Role + engine handle for this rank.
        if in_shard0:
            role = "src"
            engine = env.engine
            request_id_src = env.requests[0].request_id
        elif in_shard1:
            role = "dst"
            engine = env.engine
            request_id_src = None
        else:
            role = "bystander"
            engine = None
            request_id_src = None

        # Unique dst request id so the orchestrator can later find the
        # migrated request without colliding with any existing id.
        migrated_request_id = 12345

        returned = migrate_request_cross_shard(
            role=role,
            engine=engine,
            request_id_src=request_id_src,
            src_layout=src_layout,
            dst_layout=dst_layout,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            cross_shard_group=cross_shard_group,
            my_src_head_offset=my_src_head_offset,
            my_dst_head_offset=my_dst_head_offset,
            request_id_dst=migrated_request_id,
        )

        # Post-migration invariants.
        if role == "src":
            # Source engine dropped the migrated request but still
            # tracks the one that stayed behind.
            assert env.requests[0].request_id not in env.engine.requests
            assert env.requests[1].request_id in env.engine.requests

        if role == "dst":
            assert returned == migrated_request_id
            assert migrated_request_id in env.engine.requests
            injected = env.engine.get_request(migrated_request_id)
            # The request carries the migrated prompt + generated history.
            assert len(injected.prompt_tokens) == config.min_prompt_length
            assert len(injected.generated_tokens) >= 1, (
                "migration should have carried at least one decode token"
            )
            # The injected slot should have the fresh block ids the
            # allocator handed out, with positive block count.
            ctx = env.engine.context
            matches = (ctx.request_ids == migrated_request_id).nonzero(
                as_tuple=False
            ).flatten()
            assert matches.numel() == 1
            slot_idx = int(matches.item())
            injected_block_count = int(ctx.request_kv_block_counts[slot_idx].item())
            assert injected_block_count >= 1, (
                "injected request should have at least one KV block"
            )
            # Decode-state slot: not in prefill, query_length=1.
            assert int(ctx.request_in_prefill_status_tensor[slot_idx].item()) == 0
            assert int(ctx.request_query_lengths[slot_idx].item()) == 1
            # kv_length_offsets must match the post-decode invariant:
            # one fewer than prompt + generated so the last generated
            # token is the pending query for the next forward pass.
            expected_kvlo = len(injected.prompt_tokens) + len(injected.generated_tokens) - 1
            assert int(ctx.request_kv_length_offsets[slot_idx].item()) == expected_kvlo

            # Continuation: the engine must be able to step and emit a
            # new token on the migrated request.
            tokens_before = len(injected.generated_tokens)
            _EngineScaffold._run_step(env)
            _EngineScaffold._run_step(env)
            updated = env.engine.get_request(migrated_request_id)
            tokens_after = len(updated.generated_tokens)
            assert tokens_after > tokens_before, (
                "migrated request did not produce new tokens after resume"
            )

        dist.destroy_process_group(cross_shard_group)

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()
