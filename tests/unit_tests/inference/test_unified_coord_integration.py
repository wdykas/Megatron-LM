# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end integration tests for the unified coord + disagg HTTP path.

Two tests in layers:

1. **Single-shard HTTP round-trip.** Spawn a real coord subprocess, build
   one real engine, wire it up via ``start_listening_to_data_parallel_coordinator``,
   submit via :class:`InferenceClient` (the same code path the HTTP
   server uses), and await the completion future. Proves the
   client→coord→engine→ENGINE_REPLY→client future loop actually closes
   when driven through the real ZMQ pipes. This is the traffic-path
   backstop that previous unit tests didn't cover.

2. **Cross-shard rank rewire.** Covered by the raw-ZMQ test in
   ``test_unified_coord.py::test_update_request_rank_redispatches_reply``
   — not repeated here because spinning up two real engines with
   proper shard-local process groups requires the full multi-shard
   pg-collection setup that lives in ``MegatronLocalMulti.launch`` and
   depends on training-arg globals. The raw-ZMQ test proves the
   coord-rewire half of the path; this file proves the engine half.
"""
import asyncio
import multiprocessing
import os

import pytest
import torch
import torch.distributed as dist


class _PicklableTokenizer:
    """Spawn-pickleable tokenizer stub for the coord subprocess.

    The engine-test scaffold uses a SimpleNamespace+lambda tokenizer
    that can't cross a ``spawn`` process boundary. The coord only ever
    calls ``detokenize`` to render finished-request payloads, so any
    fixed string suffices for an integration test that asserts on
    tokens, not text.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def detokenize(self, tokens):
        return "tokenized_prompt"


@pytest.mark.skipif(
    torch.cuda.device_count() < 1,
    reason="integration test requires at least one GPU",
)
class TestUnifiedCoordIntegration:
    """Real coord + real engine + real client over a single rank."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                pytest.skip("requires torchrun")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend="nccl")
        if dist.get_world_size() != 1:
            pytest.skip("requires world_size == 1 (run with --nproc_per_node=1)")

    @classmethod
    def teardown_class(cls):
        from tests.unit_tests.inference.engines.test_dynamic_engine import set_rounder
        from tests.unit_tests.test_utilities import Utils

        set_rounder(64)
        Utils.destroy_model_parallel()

    def test_client_submit_reply_roundtrip(self):
        """HTTP-equivalent client submits; completion future resolves
        with the engine's generated tokens — end-to-end through the
        unified coordinator.
        """
        # torch.inference_mode is set inside the async coroutine because
        # asyncio.run recreates the running task context and mode is
        # thread/task-local; wrapping the sync call here is not enough.
        asyncio.run(self._run_scenario())

    async def _run_scenario(self):
        # The engine's add_request path does in-place updates on
        # inference-mode tensors; those require inference_mode set on
        # the running asyncio task. Wrap the whole scenario.
        with torch.inference_mode():
            await self._run_scenario_inner()

    async def _run_scenario_inner(self):
        from megatron.core.inference.data_parallel_inference_coordinator import (
            DataParallelInferenceCoordinator,
        )
        from megatron.core.inference.inference_client import InferenceClient
        from megatron.core.inference.sampling_params import SamplingParams
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            DynamicEngineTestConfig,
        )
        from tests.unit_tests.inference.engines.test_dynamic_engine import (
            TestDynamicInferenceEngine as _EngineScaffold,
        )

        config = DynamicEngineTestConfig(
            num_requests=1,
            min_prompt_length=8,
            max_prompt_length=8,
            num_tokens_to_generate=16,
            num_gap_steps=0,
            model_provider="gpt",
            tensor_model_parallel_size=1,
            context_max_requests=4,
        )
        env = _EngineScaffold._build_test_env(config)
        engine = env.engine

        # Spawn the unified coord in-process (it runs as a subprocess,
        # but management is local to this rank since world_size=1).
        spawn_ctx = multiprocessing.get_context("spawn")
        pipe_parent, pipe_child = spawn_ctx.Pipe()
        ready_event = spawn_ctx.Event()
        coord_proc = spawn_ctx.Process(
            target=DataParallelInferenceCoordinator.entrypoint,
            kwargs={
                "pipe_connection": pipe_child,
                "ready_event": ready_event,
                "data_parallel_size": 1,
                "tokenizer": _PicklableTokenizer(config.vocab_size),
                "max_requests": engine.context.max_requests,
                "inference_coordinator_port": None,
                "deterministic_mode": True,
                "block_size_tokens": engine.context.block_size_tokens,
                "enable_prefix_caching": False,
                "hostname": "127.0.0.1",
            },
        )
        coord_proc.start()
        coord_addr = pipe_parent.recv()
        pipe_parent.close()

        client = None
        try:
            await engine.start_listening_to_data_parallel_coordinator(
                launch_inference_coordinator=False,
                coordinator_addr=coord_addr,
                shard_index=0,
                hostname="127.0.0.1",
            )

            client = InferenceClient(inference_coordinator_address=coord_addr)
            client.start()

            prompt_tokens = list(range(1, config.min_prompt_length + 1))
            sampling = SamplingParams(
                num_tokens_to_generate=16,
                termination_id=-1,
                return_log_probs=False,
                skip_prompt_log_probs=True,
            )

            # target_shard_index=0 is the shard tag we registered above;
            # covers the scoped-routing code path while still being a
            # one-shard test.
            completion_future = client.add_request(
                prompt_tokens, sampling, target_shard_index=0
            )

            # Server-side id round-trip — exercises the SUBMIT_REQUEST_ACK
            # path that the auto-disagg driver relies on.
            server_id = await asyncio.wait_for(
                client.wait_for_server_id(0), timeout=10.0
            )
            assert server_id >= 0

            reply = await asyncio.wait_for(completion_future, timeout=60.0)
            gen_tokens = reply.get("generated_tokens") or []
            assert len(gen_tokens) >= 1, (
                f"completion future resolved with no generated tokens; "
                f"coord/engine reply path is broken. reply={reply!r}"
            )
            assert reply.get("request_id") == server_id, (
                f"reply carries server_id={reply.get('request_id')} but the "
                f"coord acked {server_id}"
            )
        finally:
            if client is not None:
                client.stop_engines()
                client.shutdown_coordinator()
                client.stop()
            coord_proc.join(timeout=10)
            if coord_proc.is_alive():
                coord_proc.terminate()
                coord_proc.join(timeout=5)
