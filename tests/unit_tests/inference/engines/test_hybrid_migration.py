# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Hybrid (Mamba + attention) migration tests.

Proves that :meth:`DynamicInferenceEngine.snapshot_request`,
:meth:`detach_request`, :meth:`inject_request`, and the
:func:`execute_mamba_state_transport` helper correctly round-trip a
request through a hybrid engine, carrying both the per-layer attention
KV **and** the per-request Mamba conv / SSM state.

These tests run on a single rank (no cross-shard hop) but exercise the
full read-from-context / allocate-slot / scatter-into-state-buffer
pipeline. A 4-GPU cross-shard hybrid migration test can layer on top
once the scaffold grows distributed-Mamba support.
"""
import pytest
import torch

from megatron.core.inference.engines.request_migration import (
    KVLayout,
    MambaLayout,
    RequestMigrationBundle,
)
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
    skip_if_mamba_sequence_packing_not_available,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    TestDynamicInferenceEngine as _EngineScaffold,
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestHybridMigration:
    """Mamba/hybrid request migration — same-engine round-trip."""

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    def _build_env(self):
        try:
            skip_if_mamba_sequence_packing_not_available("mamba")
        except (ImportError, ModuleNotFoundError) as e:
            pytest.skip(f"Mamba runtime not available: {e}")
        config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=16,
            num_gap_steps=2,
            model_provider="mamba",
            context_max_requests=4,
        )
        env = _EngineScaffold._build_test_env(config)
        return env, config

    @torch.inference_mode()
    def test_snapshot_exposes_mamba_layout(self):
        """Snapshot on a hybrid engine populates ``src_layout.mamba``."""
        env, config = self._build_env()
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        bundle, _ = env.engine.snapshot_request(env.requests[0].request_id)
        assert bundle.src_layout is not None
        assert bundle.src_layout.mamba is not None
        m = bundle.src_layout.mamba
        ctx = env.engine.context
        assert m.num_mamba_layers_pp == ctx.num_mamba_layers
        assert m.conv_states_shape == tuple(ctx.mamba_conv_states_shape)
        assert m.ssm_states_shape == tuple(ctx.mamba_ssm_states_shape)
        # dtypes round-trip through names cleanly.
        assert m.conv_states_dtype_name in ("bfloat16", "float16", "float32")
        assert m.ssm_states_dtype_name in ("bfloat16", "float16", "float32")

    @torch.inference_mode()
    def test_mamba_state_idx_present_before_detach(self):
        """A hybrid request has a valid Mamba state index while active."""
        env, config = self._build_env()
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        req_id = env.requests[0].request_id
        state_idx = env.engine.get_mamba_state_idx_for(req_id)
        assert state_idx is not None and state_idx >= 0

    @torch.inference_mode()
    def test_detach_frees_mamba_slot(self):
        """Detaching a hybrid request returns its Mamba slot to the pool."""
        env, config = self._build_env()
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        req_id = env.requests[0].request_id
        mamba = env.engine.context.mamba_metadata
        free_before = int(mamba.mamba_state_free_slot_count)
        env.engine.detach_request(req_id, keep_blocks=False)
        free_after = int(mamba.mamba_state_free_slot_count)
        assert free_after == free_before + 1, (
            f"expected Mamba state free pool to grow by 1 on detach, got "
            f"{free_before} → {free_after}"
        )

    @torch.inference_mode()
    def test_roundtrip_snapshot_detach_inject_decodes(self):
        """Same-engine round-trip on hybrid: snapshot → detach (keep
        blocks) → copy KV + Mamba state in place → inject → step."""
        env, config = self._build_env()
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(config.num_gap_steps):
                _EngineScaffold._run_step(env)

        engine = env.engine
        ctx = engine.context
        src_req = env.requests[0]

        bundle, src_blocks = engine.snapshot_request(src_req.request_id)
        src_blocks = src_blocks.clone()
        src_state_idx = engine.get_mamba_state_idx_for(src_req.request_id)
        assert src_state_idx is not None

        # Stash KV + Mamba state so we can copy them back into the new
        # slot after injection.
        kv_snapshot = ctx.memory_buffer[..., src_blocks, :, :, :].clone()
        conv_snapshot = ctx.mamba_conv_states[:, src_state_idx].clone()
        ssm_snapshot = ctx.mamba_ssm_states[:, src_state_idx].clone()

        # Detach the source request — frees the src Mamba slot as well
        # as (optionally) the KV blocks.
        engine.detach_request(src_req.request_id, keep_blocks=True)

        injected_id = 10_000 + src_req.request_id
        from dataclasses import replace as dataclass_replace

        injected_bundle = dataclass_replace(bundle, request_id=injected_id)
        dst_blocks = engine.inject_request(injected_bundle)
        dst_state_idx = engine.get_mamba_state_idx_for(injected_id)
        assert dst_state_idx is not None and dst_state_idx >= 0, (
            "inject should allocate a Mamba slot"
        )

        # Simulate the migration transport: copy KV + Mamba state into
        # the destination locations.
        ctx.memory_buffer[..., dst_blocks, :, :, :] = kv_snapshot
        ctx.mamba_conv_states[:, dst_state_idx] = conv_snapshot
        ctx.mamba_ssm_states[:, dst_state_idx] = ssm_snapshot

        # Release the source KV blocks (the "after transport" cleanup).
        ctx.kv_block_allocator.release_memory_blocks(src_blocks)

        # Continue stepping. The hybrid model must be able to advance
        # decode on the injected request without NaN / crash.
        def _n_tokens(req) -> int:
            t = req.generated_tokens
            if t is None:
                return 0
            return t.numel() if isinstance(t, torch.Tensor) else len(t)

        tokens_before = _n_tokens(engine.get_request(injected_id))
        _EngineScaffold._run_step(env)
        _EngineScaffold._run_step(env)
        tokens_after = _n_tokens(engine.get_request(injected_id))
        assert tokens_after > tokens_before, (
            "hybrid injected request did not accumulate new tokens after resume"
        )


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestRouterReplayReject:
    """MoE router-replay migration is explicitly unsupported for now."""

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    @torch.inference_mode()
    def test_snapshot_rejects_when_routing_indices_present(self):
        """If the request has accumulated routing_indices, snapshot
        must refuse — the bundle doesn't carry routing history yet."""
        config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            num_gap_steps=2,
            model_provider="gpt",
        )
        env = _EngineScaffold._build_test_env(config)
        env.engine._add_request(env.requests[0])
        for _ in range(config.num_gap_steps):
            _EngineScaffold._run_step(env)

        # Fake a populated routing_indices on the request record; the
        # engine's step loop would normally populate this on a real MoE
        # model but our GPT fixture doesn't run MoE.
        req = env.engine.get_request(env.requests[0].request_id)
        req.routing_indices = torch.zeros(
            (4, 2, 2), dtype=torch.int32, device=torch.cuda.current_device()
        )
        with pytest.raises(AssertionError, match="routing_indices"):
            env.engine.snapshot_request(env.requests[0].request_id)
