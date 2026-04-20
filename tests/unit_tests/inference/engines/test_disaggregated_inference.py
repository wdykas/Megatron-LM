# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end test for disaggregated inference.

Prefill runs on one shard, decode continues on another. The KV cache
moves between them through the existing migration transport
(:func:`migrate_request_cross_shard` → :class:`NCCLCopyService`).

This is the request-migration primitive applied at the
prefill→decode boundary rather than mid-decode — same machinery,
different timing.

Layout on a 4-GPU world:
  - shard 0 (prefill): ranks [0, 1] with TP=2
  - shard 1 (decode):  ranks [2, 3] with TP=2
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
    reason="disaggregated inference test requires 4 GPUs",
)
class TestDisaggregatedInference:
    """Real-engine prefill-on-shard-A / decode-on-shard-B test."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                pytest.skip("requires torchrun")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl")
        if dist.get_world_size() != 4:
            pytest.skip("requires world_size == 4")

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    @torch.inference_mode()
    def test_prefill_shard0_decode_shard1(self):
        """Disaggregate: submit to shard 0, run prefill + one decode
        step on shard 0, migrate to shard 1, verify decode continues."""
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            DynamicEngineTestConfig,
        )
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            TestDynamicInferenceEngine as _EngineScaffold,
        )

        rank = dist.get_rank()
        in_src = rank in (0, 1)
        in_dst = rank in (2, 3)
        src_ranks = [0, 1]
        dst_ranks = [2, 3]

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

        # Warm both shards with their own request so the dst engine's
        # decode metadata is initialised.
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        # Now simulate the "disaggregation moment": add a fresh request
        # to the prefill shard and run exactly ONE prefill step — which
        # produces the first decode token (generated_tokens has len=1).
        # This is the earliest point at which a request can be migrated
        # (``inject_request`` asserts ``gen_len >= 1``).
        DISAGG_REQUEST_ID = 1_234_567
        if in_src:
            import types

            from megatron.core.inference.inference_request import (
                DynamicInferenceRequest,
            )
            from megatron.core.inference.sampling_params import SamplingParams

            prompt_tokens = torch.randint(
                0,
                config.vocab_size - 1,
                (config.min_prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            sampling = SamplingParams(
                num_tokens_to_generate=16,
                termination_id=-1,
                return_log_probs=False,
                skip_prompt_log_probs=True,
            )
            disagg_request = DynamicInferenceRequest(
                request_id=DISAGG_REQUEST_ID,
                prompt_tokens=prompt_tokens,
                sampling_params=sampling,
            )
            env.engine._add_request(disagg_request)
            # One step: processes prefill + samples first decode token.
            _EngineScaffold._run_step(env)
            engine_req = env.engine.get_request(DISAGG_REQUEST_ID)
            gen_len = (
                engine_req.generated_tokens.numel()
                if isinstance(engine_req.generated_tokens, torch.Tensor)
                else len(engine_req.generated_tokens)
            )
            assert gen_len >= 1, (
                f"prefill should have produced ≥1 decode token, got {gen_len}"
            )

        # Describe the KV layouts for the two shards. Matched TP=2 here.
        model_config = env.engine.controller.inference_wrapped_model.model.config
        ctx = env.engine.context
        num_kv_heads = (
            model_config.num_query_groups or model_config.num_attention_heads
        )
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

        my_src_head_offset = (
            (rank - src_ranks[0]) * (num_kv_heads // 2) if in_src else 0
        )
        my_dst_head_offset = (
            (rank - dst_ranks[0]) * (num_kv_heads // 2) if in_dst else 0
        )

        cross_shard_group = dist.new_group(ranks=sorted(src_ranks + dst_ranks))

        role = "src" if in_src else ("dst" if in_dst else "bystander")
        engine_for_role = env.engine if (in_src or in_dst) else None

        migrated_id = migrate_request_cross_shard(
            role=role,
            engine=engine_for_role,
            request_id_src=DISAGG_REQUEST_ID if in_src else None,
            src_layout=src_layout,
            dst_layout=dst_layout,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            cross_shard_group=cross_shard_group,
            my_src_head_offset=my_src_head_offset,
            my_dst_head_offset=my_dst_head_offset,
            request_id_dst=DISAGG_REQUEST_ID,
        )

        # --- Post-handoff invariants on both sides ---
        if role == "src":
            # Prefill shard released the request after transport.
            assert DISAGG_REQUEST_ID not in env.engine.requests
            # Unrelated requests stay.
            for r in env.requests:
                assert r.request_id in env.engine.requests

        if role == "dst":
            assert migrated_id == DISAGG_REQUEST_ID
            assert DISAGG_REQUEST_ID in env.engine.requests
            injected = env.engine.get_request(DISAGG_REQUEST_ID)
            # Handoff happened at prefill→decode boundary: exactly one
            # generated token carried across, and the slot must be in
            # DECODE state.
            ctx_dst = env.engine.context
            matches = (
                (ctx_dst.request_ids == DISAGG_REQUEST_ID)
                .nonzero(as_tuple=False)
                .flatten()
            )
            assert matches.numel() == 1
            slot_idx = int(matches.item())
            assert int(ctx_dst.request_in_prefill_status_tensor[slot_idx].item()) == 0
            assert int(ctx_dst.request_query_lengths[slot_idx].item()) == 1
            # Decode continues cleanly — step a few times, more tokens.
            tokens_before = len(injected.generated_tokens)
            _EngineScaffold._run_step(env)
            _EngineScaffold._run_step(env)
            tokens_after = len(
                env.engine.get_request(DISAGG_REQUEST_ID).generated_tokens
            )
            assert tokens_after > tokens_before, (
                "decode shard did not continue generating after handoff"
            )

        dist.destroy_process_group(cross_shard_group)
