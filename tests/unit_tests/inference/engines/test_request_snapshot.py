# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engine-level tests for the migration-snapshot API.

:meth:`DynamicInferenceEngine.snapshot_request` is the read-only half of
the inter-engine request migration path (Step 2a). It must:

- produce a :class:`RequestMigrationBundle` whose tokens / sampling params /
  KV-cache epoch match the engine's live request record,
- return block ids that match what the context's KV allocator handed out
  (``request_to_kv_block_ids[slot][:block_count]``),
- derive a valid :class:`KVLayout` from the engine's model + context,
- reject requests that are not in the active batch.

The heavy engine scaffold (model, context, wrapper, controller) is
re-used from the existing ``test_dynamic_engine.py`` infrastructure so we
exercise ``snapshot_request`` against a real engine with real KV blocks
rather than a mock.
"""
import pytest
import torch

from megatron.core.inference.engines.request_migration import (
    KVLayout,
    RequestMigrationBundle,
)
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    DynamicEngineTestConfig,
)
from tests.unit_tests.inference.engines.test_dynamic_engine import (
    TestDynamicInferenceEngine as _EngineScaffold,
)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestSnapshotRequest:
    """Tests for :meth:`DynamicInferenceEngine.snapshot_request`."""

    @classmethod
    def teardown_class(cls):
        # Keep the engine fixture's rounder reset.
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    def _drive_requests_into_active_batch(self, num_gap_steps: int = 3):
        """Build an engine with two GPT requests and step until both have
        completed prefill and generated at least one decode token."""
        config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=16,
            num_gap_steps=num_gap_steps,
            model_provider="gpt",
        )
        env = _EngineScaffold._build_test_env(config)

        # Register both requests. Step after each add so they land in the
        # active batch and generate at least one decode token.
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(num_gap_steps):
                _EngineScaffold._run_step(env)

        return env

    def test_snapshot_matches_engine_state(self):
        """Bundle fields track the engine's view of the request."""
        env = self._drive_requests_into_active_batch()
        engine = env.engine

        req = env.requests[0]
        bundle, block_ids = engine.snapshot_request(req.request_id)

        assert isinstance(bundle, RequestMigrationBundle)
        assert bundle.request_id == req.request_id

        # Tokens round-trip as plain lists.
        expected_prompt = (
            req.prompt_tokens.tolist()
            if isinstance(req.prompt_tokens, torch.Tensor)
            else list(req.prompt_tokens)
        )
        assert bundle.prompt_tokens == expected_prompt
        # generated_tokens may be None/empty prior to first decode, but
        # with 3 gap steps we should have at least one.
        assert len(bundle.generated_tokens) >= 1

        # Sampling params are msgpack-compatible plain dict.
        assert isinstance(bundle.sampling_params, dict)
        assert "num_tokens_to_generate" in bundle.sampling_params

        # kv_cache_epoch is serialisable (list of int tuples).
        for epoch_marker in bundle.kv_cache_epoch:
            assert isinstance(epoch_marker, tuple)
            assert len(epoch_marker) == 2

    def test_snapshot_block_ids_match_context(self):
        """Returned block ids equal `request_to_kv_block_ids[slot][:count]`."""
        env = self._drive_requests_into_active_batch()
        engine = env.engine
        ctx = engine.context

        req = env.requests[0]
        bundle, block_ids = engine.snapshot_request(req.request_id)

        # Re-derive the slot by searching the context tensor — same as the
        # implementation does internally.
        matches = (ctx.request_ids == req.request_id).nonzero(as_tuple=False).flatten()
        assert matches.numel() == 1
        slot_idx = int(matches.item())
        count = int(ctx.request_kv_block_counts[slot_idx].item())

        assert bundle.num_kv_blocks == count
        assert block_ids.numel() == count
        assert torch.equal(
            block_ids.to(ctx.request_to_kv_block_ids.dtype),
            ctx.request_to_kv_block_ids[slot_idx, :count],
        )

        # src_block_ids on the bundle is the Python-list mirror of the tensor.
        assert bundle.src_block_ids == block_ids.tolist()

        # last_block_offset matches the context tensor.
        assert bundle.last_block_offset == int(
            ctx.request_last_kv_block_offset[slot_idx].item()
        )

    def test_snapshot_layout_reflects_engine_config(self):
        """The KVLayout descriptor reflects the engine's model + context."""
        env = self._drive_requests_into_active_batch()
        engine = env.engine
        ctx = engine.context
        model_config = engine.controller.inference_wrapped_model.model.config

        bundle, _ = engine.snapshot_request(env.requests[0].request_id)
        layout = bundle.src_layout
        assert isinstance(layout, KVLayout)

        # TP / PP sizes come from the context's pg_collection — when the
        # engine is built without one (as in the unit-test scaffold) we
        # fall back to 1.
        assert layout.tp_size >= 1
        assert layout.pp_size >= 1
        # num_kv_heads_total = num_query_groups for GQA, else num_attention_heads.
        expected_kv_heads = (
            model_config.num_query_groups or model_config.num_attention_heads
        )
        assert layout.num_kv_heads_total == expected_kv_heads
        assert layout.num_layers_total == model_config.num_layers
        assert layout.head_dim == ctx.hidden_size_per_attention_head
        assert layout.block_size_tokens == ctx.block_size_tokens
        # dst_layout must be filled in by the caller (routing layer) before
        # building a migration plan.
        assert bundle.dst_layout is None

    def test_snapshot_rejects_unknown_request(self):
        """An unknown request_id fails loudly."""
        env = self._drive_requests_into_active_batch()
        with pytest.raises(AssertionError, match="not tracked"):
            env.engine.snapshot_request(999_999)

    def test_snapshot_rejects_non_active_request(self):
        """A request that hasn't entered the active batch is rejected."""
        # Build env but don't step — the request sits in waiting_request_ids
        # without a context slot.
        config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=4,
            num_gap_steps=0,
            model_provider="gpt",
        )
        env = _EngineScaffold._build_test_env(config)
        env.engine._add_request(env.requests[0])
        with pytest.raises(AssertionError, match="slots in the active batch"):
            env.engine.snapshot_request(env.requests[0].request_id)


@pytest.mark.internal
@pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
)
class TestDetachInjectRoundtrip:
    """Tests for detach_request + inject_request (Step 2b).

    Exercises the same-engine round-trip that the inter-engine transport
    will use in Step 3: snapshot a live request, detach it while keeping
    its KV blocks, inject a copy under a fresh id, copy KV block contents
    in place (simulating the transport), then keep stepping. The
    injected request must continue generating tokens without the engine
    crashing or NaN-ing.
    """

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    def _drive_into_active_batch(self, num_gap_steps: int = 3):
        config = DynamicEngineTestConfig(
            num_requests=2,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=16,
            num_gap_steps=num_gap_steps,
            model_provider="gpt",
        )
        env = _EngineScaffold._build_test_env(config)
        for request in env.requests:
            env.engine._add_request(request)
            for _ in range(num_gap_steps):
                _EngineScaffold._run_step(env)
        return env

    @torch.inference_mode()
    def test_detach_removes_from_active_batch_and_keeps_blocks(self):
        """detach(keep_blocks=True) should remove the request from the
        context's active region and hand back the block ids, and the
        blocks must *not* be returned to the free pool."""
        env = self._drive_into_active_batch()
        engine = env.engine
        ctx = engine.context
        allocator = ctx.kv_block_allocator

        req = env.requests[0]
        active_before = ctx.total_request_count - ctx.paused_request_count
        free_before = allocator.total_avail

        _, block_ids = engine.snapshot_request(req.request_id)
        block_ids = block_ids.clone()

        returned = engine.detach_request(req.request_id, keep_blocks=True)

        # Block ids we got back match the snapshot view exactly.
        assert torch.equal(returned, block_ids)
        # Active-batch count dropped by one.
        assert ctx.total_request_count - ctx.paused_request_count == active_before - 1
        # Engine's request dict no longer tracks it.
        assert req.request_id not in engine.requests
        # Allocator free count did NOT grow (blocks kept).
        assert allocator.total_avail == free_before
        # The detached request's id is no longer resident in the slot table
        # active region.
        resident_ids = ctx.request_ids[
            ctx.paused_request_count : ctx.total_request_count
        ].tolist()
        assert req.request_id not in resident_ids

        # Release the held blocks manually (this is what a migration
        # orchestrator would do after the transport completes).
        allocator.release_memory_blocks(returned)
        assert allocator.total_avail == free_before + returned.numel()

    @torch.inference_mode()
    def test_detach_releases_blocks_when_keep_is_false(self):
        """detach(keep_blocks=False) is a fire-and-forget abort."""
        env = self._drive_into_active_batch()
        allocator = env.engine.context.kv_block_allocator
        free_before = allocator.total_avail

        _, block_ids = env.engine.snapshot_request(env.requests[0].request_id)
        n_blocks = block_ids.numel()

        env.engine.detach_request(env.requests[0].request_id, keep_blocks=False)

        assert allocator.total_avail == free_before + n_blocks

    @torch.inference_mode()
    def test_roundtrip_snapshot_detach_inject_steps_cleanly(self):
        """Same-engine round-trip: after snapshot/detach/KV-copy/inject
        the engine can step without crashing and the injected request
        generates at least one more token."""
        env = self._drive_into_active_batch()
        engine = env.engine
        ctx = engine.context

        src_req = env.requests[0]
        bundle, src_blocks = engine.snapshot_request(src_req.request_id)
        src_blocks = src_blocks.clone()

        # Stash the source KV contents before detaching so we can copy
        # them into the destination blocks after injection.
        source_kv_snapshot = ctx.memory_buffer[..., src_blocks, :, :, :].clone()

        # Detach the source (blocks stay allocated so we can read KV).
        engine.detach_request(src_req.request_id, keep_blocks=True)

        # Inject a copy under a fresh id so self.requests doesn't collide.
        injected_id = 10_000 + src_req.request_id
        injected_bundle = dataclass_replace(bundle, request_id=injected_id)
        dst_blocks = engine.inject_request(injected_bundle)
        assert dst_blocks.numel() == bundle.num_kv_blocks
        assert injected_id in engine.requests

        # Simulate the transport: copy KV block contents from source to
        # the newly-allocated destination blocks in place.
        ctx.memory_buffer[..., dst_blocks, :, :, :] = source_kv_snapshot

        # Release the source blocks now that the "transport" completed.
        ctx.kv_block_allocator.release_memory_blocks(src_blocks)

        def _n_tokens(req) -> int:
            t = req.generated_tokens
            if t is None:
                return 0
            return t.numel() if isinstance(t, torch.Tensor) else len(t)

        # Step the engine — the injected request should be picked up in
        # DECODE and generate a new token without NaNs/crashes.
        tokens_before = _n_tokens(engine.get_request(injected_id))
        _EngineScaffold._run_step(env)
        _EngineScaffold._run_step(env)
        tokens_after = _n_tokens(engine.get_request(injected_id))
        assert tokens_after > tokens_before, (
            "injected request did not accumulate any new tokens"
        )


# Imported lazily so the module still imports on Python installs without
# dataclasses.replace (3.12 always has it; kept for style).
from dataclasses import replace as dataclass_replace  # noqa: E402
