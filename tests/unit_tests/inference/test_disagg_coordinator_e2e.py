# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end native (coordinator-driven) prefill->decode disaggregation.

Run with 2 ranks (``torchrun --nproc-per-node 2``): rank 0 = prefill shard,
rank 1 = decode shard. Reuses the coordinator-test ``DummyEngine`` harness
(real ZMQ coordinator + engine loop + InferenceClient) and overrides only the
disagg bits, so it exercises the actual new control plane: REGISTER_ROLE
registration, 2-hop routing (SUBMIT->prefill, PREFILL_DONE->SEND_KV/RECV_KV),
the engine's SEND_KV/RECV_KV handlers + real KV transport, and the client
round-trip. The LLM forward is stubbed; KV export/import are fakes (the real
reshard/transport is covered by test_disagg_e2e).
"""

import asyncio
import multiprocessing
import os

import pytest
import torch

from tests.unit_tests.test_utilities import Utils

zmq = pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.inference_request import Status
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.test_data_parallel_inference_coordinator import (
    DummyContext,
    DummyEngine,
    DummyTokenizer,
    cleanup_engine,
)

# global model dims for the fake KV. prompt len 4 / BS 4 => block_count 1.
L, H, BS, HD, BC = 2, 4, 4, 6, 1
PORT = 46733


def _layout(global_rank):
    return dict(num_layers=L, num_heads=H, tp_size=1, tp_rank=0, pp_size=1, pp_rank=0,
                global_rank=global_rank, ep_size=1, ep_rank=0, etp_size=1, etp_rank=0)


class _FakeKVContext(DummyContext):
    """DummyContext + the KV staging hooks + the schema attrs
    derive_decode_schema reads. Tensors are CUDA (the world group is NCCL)."""

    def __init__(self):
        super().__init__()
        self._dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.cache_mla_latent = False
        self.is_hybrid_model = False
        self.block_size_tokens = BS
        # memory_buffer shape drives derive_decode_schema (local dims; TP1 -> H).
        self.memory_buffer = torch.zeros(2, L, BC, BS, H, HD, device=self._dev)
        self.imported = None

    def export_request_kv(self, request_id):
        return {
            "layout": "std_attn_v1", "block_count": BC, "block_size_tokens": BS,
            "num_layers": L, "num_heads_per_partition": H, "hidden_per_head": HD,
            "block_hashes": [],
            "staging_tensor": torch.full((BC, 2, L, BS, H, HD), 1.0, device=self._dev),
        }

    def import_request_kv(self, payload):
        self.imported = payload["staging_tensor"]
        return {"block_ids": list(range(payload["block_count"])), "ok": True}


class DisaggDummyEngine(DummyEngine):
    """DummyEngine + disagg behavior: prefill emits PREFILL_DONE (staging KV)
    instead of ENGINE_REPLY; decode behaves normally (RECV_KV path admits the
    request via the real schedule_requests handler)."""

    def __init__(self):
        super().__init__()
        self.context = _FakeKVContext()
        self._disagg_staged_kv = {}
        self._disagg_backend = None

    async def async_step(self, *, verbose=False):
        from collections import deque as _dq

        result = {"active_request_ids": [], "finished_request_records": [],
                  "step_time": 0.01, "cuda_graph_request_count": 1}
        # activate queued
        while self.waiting_request_ids:
            rid = self.waiting_request_ids.popleft()
            self.requests[rid].record[-1].status = Status.ACTIVE_AND_GENERATING_TOKENS
            self.context.active_cnt += 1
            result["active_request_ids"].append(rid)
        await asyncio.sleep(0)
        # finish active
        to_remove = []
        for rid, entry in self.requests.items():
            req = entry.record[-1]
            if req.status != Status.ACTIVE_AND_GENERATING_TOKENS:
                continue
            req.status = Status.COMPLETED
            self.context.active_cnt -= 1
            to_remove.append(rid)
            if self.disagg_role == "prefill":
                # prefill-stop: stage KV + tell the coordinator (no client reply)
                self._disagg_staged_kv[rid] = self.context.export_request_kv(rid)
                if self.is_mp_coordinator:
                    self.socket_for_receiving_requests.send(
                        msgpack.packb([Headers.PREFILL_DONE.value, rid], use_bin_type=True)
                    )
            else:
                result["finished_request_records"].append(entry.record)
                entry.future.set_result(entry.record)
                if self.is_mp_coordinator:
                    self.socket_for_receiving_requests.send(
                        msgpack.packb(
                            [Headers.ENGINE_REPLY.value, [entry.record.merge().serialize()]],
                            use_bin_type=True,
                        )
                    )
        for rid in to_remove:
            del self.requests[rid]
        return result


@pytest.fixture
def init_mp(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@pytest.fixture
def disagg_coordinator():
    """Rank 0 spawns the disaggregated coordinator (2 instances: prefill+decode)."""
    rank = int(os.environ.get("RANK", "0"))
    proc = None
    if rank == 0:
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        ready = ctx.Event()
        proc = ctx.Process(
            target=DataParallelInferenceCoordinator.entrypoint,
            kwargs=dict(pipe_connection=child, ready_event=ready, data_parallel_size=0,
                        tokenizer=DummyTokenizer(), max_requests=16,
                        inference_coordinator_port=PORT, disaggregated=True),
        )
        proc.start()
        while not parent.poll(timeout=0.1):
            assert proc.is_alive()
        dp_addr = parent.recv()
        parent.close()
    else:
        dp_addr = f"tcp://localhost:{PORT}"
    yield dp_addr
    if rank == 0 and proc is not None and proc.is_alive():
        proc.terminate()
        proc.join(timeout=10.0)


@pytest.mark.internal
@pytest.mark.skip(
    reason="WIP: native coordinator-driven disagg e2e. The control plane is "
    "unit-tested (test_disagg_coordinator_routing / _2hop); this multi-GPU "
    "harness still needs robust coordinator addressing/cleanup before it's "
    "reliable in CI."
)
def test_native_disagg_prefill_to_decode(init_mp, disagg_coordinator):
    if Utils.world_size != 2:
        pytest.skip("native disagg e2e needs exactly 2 ranks (prefill + decode)")
    asyncio.run(_run_disagg_e2e(disagg_coordinator))


async def _run_disagg_e2e(dp_addr):
    rank = torch.distributed.get_rank()
    role = "prefill" if rank == 0 else "decode"

    def trace(msg):
        line = f"[rank{rank}/{role}] {msg}\n"
        print(line, end="", flush=True)
        with open(f"/tmp/dtrace_{rank}.log", "a") as _f:
            _f.write(line)
            _f.flush()

    trace("building engine")
    engine = DisaggDummyEngine()
    engine.set_disaggregation_config(
        role=role,
        instance_layouts=[_layout(rank)],
        identity="prefill" if role == "prefill" else "decode_s1_dp0",
        total_instances=2,
        world_group=None,
        spawn_coordinator=False,  # the fixture already spawned it
    )
    trace("start_listening...")
    await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=PORT, launch_inference_coordinator=False
    )
    trace("registered (start_listening done)")

    # Both engines have sent REGISTER_ROLE; let the coordinator process them
    # before the client submits (avoids racing decode registration).
    torch.distributed.barrier()
    await asyncio.sleep(1.0)

    client = None
    try:
        if rank == 0:
            client = InferenceClient(dp_addr)
            client.start()
            trace("submitting request")
            fut = client.add_request(
                prompt=[1, 2, 3, 4],
                sampling_params=SamplingParams(num_tokens_to_generate=2),
            )
            result = await asyncio.wait_for(fut, timeout=30.0)
            trace(f"got result: {result.get('status')}")
            assert result["status"] == Status.COMPLETED.name, result
        # Keep the decode engine's loop alive (asyncio task runs concurrently
        # with this sleep) until the prefill rank's request has completed.
        await asyncio.sleep(2.0 if rank == 0 else 25.0)
    finally:
        trace("cleanup")
        await cleanup_engine(engine, client)
