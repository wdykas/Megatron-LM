# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Round-trip test for the prefill->decode KV transfer.

Runs two gloo ranks on CPU (CI-friendly, no CUDA / NVSHMEM needed):
rank 0 = prefill, rank 1 = decode. A minimal fake engine context
provides ``export_request_kv`` / ``import_request_kv`` (the real ones
are exercised on the inference-context tests). The fake context is
config-driven so the decode side can derive the schema the same way
the real :func:`derive_decode_schema` does.

We cover the default **header-free** path (decode derives shapes from
config + prompt, no metadata on the wire) for the attention-only and
hybrid (Mamba + snapshots) layouts, plus the object-metadata fallback.
"""

import os

import pytest
import torch

mp = torch.multiprocessing

# Shared config so prefill export and decode derivation agree.
BLOCK_SIZE = 4
PROMPT_LEN = 8           # -> block_count = 2
NUM_LAYERS = 3
HEADS = 2
HIDDEN = 5
TOTAL_BLOCKS = 8
NUM_MAMBA = 2
CONV_DIM = 7
SSM_DIM = 9
BLOCK_COUNT = (PROMPT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
N_SNAP = PROMPT_LEN // BLOCK_SIZE
PROMPT = list(range(PROMPT_LEN))


class _FakeCtx:
    def __init__(self, hybrid):
        self.cache_mla_latent = False
        self.block_size_tokens = BLOCK_SIZE
        self.memory_buffer = torch.zeros(
            2, NUM_LAYERS, TOTAL_BLOCKS, BLOCK_SIZE, HEADS, HIDDEN
        )
        self.is_hybrid_model = hybrid
        if hybrid:
            self.mamba_conv_states = torch.zeros(NUM_MAMBA, 16, CONV_DIM)
            self.mamba_ssm_states = torch.zeros(NUM_MAMBA, 16, SSM_DIM)
            self.mamba_slot_allocator = object()
        else:
            self.mamba_conv_states = None
            self.mamba_ssm_states = None
            self.mamba_slot_allocator = None
        self.imported = None

    def export_request_kv(self, request_id):
        base = 1000.0 if request_id == "hyb" else 0.0
        d = {
            "layout": "std_attn_v1",
            "block_count": BLOCK_COUNT,
            "block_size_tokens": BLOCK_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads_per_partition": HEADS,
            "hidden_per_head": HIDDEN,
            "block_hashes": [11, 22],
            "staging_tensor": (
                torch.arange(BLOCK_COUNT * 2 * NUM_LAYERS * BLOCK_SIZE * HEADS * HIDDEN,
                             dtype=torch.float32)
                .reshape(BLOCK_COUNT, 2, NUM_LAYERS, BLOCK_SIZE, HEADS, HIDDEN)
                + base
            ),
        }
        if request_id == "hyb":
            d["layout"] = "hybrid_v1"
            d["mamba_payload"] = {
                "num_mamba_layers": NUM_MAMBA,
                "conv_states_tensor": torch.full((NUM_MAMBA, CONV_DIM), 3.0),
                "ssm_states_tensor": torch.full((NUM_MAMBA, SSM_DIM), 4.0),
                "snapshots": {
                    "block_hashes": [11, 22],
                    "conv_states_tensor": torch.stack(
                        [torch.full((NUM_MAMBA, CONV_DIM), 5.0 + i) for i in range(N_SNAP)]
                    ),
                    "ssm_states_tensor": torch.stack(
                        [torch.full((NUM_MAMBA, SSM_DIM), 7.0 + i) for i in range(N_SNAP)]
                    ),
                },
            }
        return d

    def import_request_kv(self, payload):
        self.imported = payload
        return {"block_ids": list(range(payload["block_count"])), "ok": True}


class _FakeEngine:
    def __init__(self, hybrid):
        self.context = _FakeCtx(hybrid)


def _summarize(payload):
    s = {"layout": payload["layout"], "attn_sum": float(payload["staging_tensor"].sum())}
    mp_ = payload.get("mamba_payload")
    if mp_:
        s["conv_sum"] = float(mp_["conv_states_tensor"].sum())
        s["ssm_sum"] = float(mp_["ssm_states_tensor"].sum())
        if mp_.get("snapshots"):
            s["snap_conv_sum"] = float(mp_["snapshots"]["conv_states_tensor"].sum())
            s["snap_ssm_sum"] = float(mp_["snapshots"]["ssm_states_tensor"].sum())
    return s


def _worker(rank, world, port, rid, header_free, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core.inference.disaggregation import kv_transfer as h
    from megatron.core.inference.disaggregation.transfer_backends.base import (
        set_kv_transport_backend,
    )
    from megatron.core.inference.disaggregation.transfer_backends.nccl import NcclTransportBackend

    backend = NcclTransportBackend()
    backend.init()
    set_kv_transport_backend(backend)
    eng = _FakeEngine(hybrid=(rid == "hyb"))

    if rank == 0:  # prefill
        ho = h.send_request_kv(eng, rid, dst=1, backend=backend, header_free=header_free)
        if ho is not None:
            ho.wait()
        q.put(("prefill", _summarize(eng.context.export_request_kv(rid))))
    else:  # decode
        res = h.recv_request_kv(
            eng, src=0, backend=backend,
            header_free=header_free, prompt_token_ids=PROMPT,
        )
        q.put(("decode", None if res is None else _summarize(eng.context.imported)))
    dist.destroy_process_group()


def _run(rid, header_free):
    port = 29630 + (abs(hash((rid, header_free))) % 2000)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, rid, header_free, q)) for r in range(2)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(2):
        role, val = q.get(timeout=60)
        out[role] = val
    for p in procs:
        p.join(timeout=60)
    return out


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_header_free_attention_only():
    out = _run("attn", header_free=True)
    assert out["decode"]["attn_sum"] == out["prefill"]["attn_sum"]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_header_free_hybrid_mamba_snapshots():
    out = _run("hyb", header_free=True)
    p, d = out["prefill"], out["decode"]
    for key in ["attn_sum", "conv_sum", "ssm_sum", "snap_conv_sum", "snap_ssm_sum"]:
        assert d[key] == p[key], f"{key} mismatch: {d[key]} != {p[key]}"


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_object_metadata_fallback():
    out = _run("hyb", header_free=False)
    p, d = out["prefill"], out["decode"]
    for key in ["attn_sum", "conv_sum", "ssm_sum", "snap_conv_sum", "snap_ssm_sum"]:
        assert d[key] == p[key]
