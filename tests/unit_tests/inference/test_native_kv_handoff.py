# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Round-trip test for the native prefill->decode KV handoff.

Runs two gloo ranks on CPU (CI-friendly, no CUDA / NVSHMEM needed):
rank 0 = prefill, rank 1 = decode. A minimal fake engine context
provides ``export_request_kv`` / ``import_request_kv`` (the real ones
are exercised on the inference-context tests); here we verify the
transport plumbing -- metadata control plane + tensor data plane --
moves the KV bytes intact for both the attention-only and the
hybrid (Mamba + per-block snapshots) layouts.
"""

import os

import pytest
import torch

mp = torch.multiprocessing


# ---- minimal fake engine: just enough surface for the handoff ----
class _FakeCtx:
    def __init__(self):
        self.memory_buffer = torch.zeros(1)  # device probe (cpu)
        self.imported = None

    def export_request_kv(self, request_id):
        if request_id == "empty":
            return None
        d = {
            "layout": "std_attn_v1",
            "block_count": 2,
            "block_size_tokens": 4,
            "num_layers": 3,
            "num_heads_per_partition": 2,
            "hidden_per_head": 5,
            "block_hashes": [11, 22],
            "staging_tensor": torch.arange(2 * 2 * 3 * 4 * 2 * 5, dtype=torch.float32).reshape(
                2, 2, 3, 4, 2, 5
            )
            + (1000 if request_id == "hyb" else 0),
        }
        if request_id == "hyb":
            d["layout"] = "hybrid_v1"
            d["mamba_payload"] = {
                "num_mamba_layers": 2,
                "conv_states_tensor": torch.full((2, 7), 3.0),
                "ssm_states_tensor": torch.full((2, 9), 4.0),
                "snapshots": {
                    "block_hashes": [11, 22],
                    "conv_states_tensor": torch.stack([torch.full((2, 7), 5.0), torch.full((2, 7), 6.0)]),
                    "ssm_states_tensor": torch.stack([torch.full((2, 9), 7.0), torch.full((2, 9), 8.0)]),
                },
            }
        return d

    def import_request_kv(self, payload):
        self.imported = payload
        return {"block_ids": list(range(payload["block_count"])), "ok": True}


class _FakeEngine:
    def __init__(self):
        self.context = _FakeCtx()


def _worker(rank, world, port, rid, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core.inference.kv_transport_backend import (
        NcclTransportBackend,
        set_kv_transport_backend,
    )
    from megatron.core.inference import native_kv_handoff as h

    backend = NcclTransportBackend()
    backend.init()
    set_kv_transport_backend(backend)
    eng = _FakeEngine()

    if rank == 0:  # prefill
        ho = h.send_request_kv(eng, rid, dst=1, backend=backend)
        if ho is not None:
            ho.wait()
        # echo what we sent so the parent can compare
        exported = eng.context.export_request_kv(rid)
        q.put(("prefill", None if exported is None else _summarize(exported)))
    else:  # decode
        res = h.recv_request_kv(eng, src=0, backend=backend)
        imp = eng.context.imported
        q.put(("decode", res if res is None else _summarize_imported(imp)))
    dist.destroy_process_group()


def _summarize(payload):
    s = {"layout": payload["layout"], "attn_sum": float(payload["staging_tensor"].sum())}
    if payload.get("mamba_payload"):
        mp = payload["mamba_payload"]
        s["conv_sum"] = float(mp["conv_states_tensor"].sum())
        s["ssm_sum"] = float(mp["ssm_states_tensor"].sum())
        if mp.get("snapshots"):
            s["snap_conv_sum"] = float(mp["snapshots"]["conv_states_tensor"].sum())
            s["snap_ssm_sum"] = float(mp["snapshots"]["ssm_states_tensor"].sum())
    return s


def _summarize_imported(payload):
    return _summarize(payload)


def _run(rid):
    port = 29610 + (abs(hash(rid)) % 2000)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, port, rid, q)) for r in range(2)]
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
def test_attention_only_roundtrip():
    out = _run("attn")
    assert out["prefill"] is not None and out["decode"] is not None
    assert out["prefill"]["layout"] == "std_attn_v1"
    assert out["decode"]["attn_sum"] == out["prefill"]["attn_sum"]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_hybrid_mamba_snapshots_roundtrip():
    out = _run("hyb")
    p, d = out["prefill"], out["decode"]
    assert p["layout"] == "hybrid_v1" and d["layout"] == "hybrid_v1"
    for key in ["attn_sum", "conv_sum", "ssm_sum", "snap_conv_sum", "snap_ssm_sum"]:
        assert d[key] == p[key], f"{key} mismatch: {d[key]} != {p[key]}"


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_empty_export_signals_none():
    out = _run("empty")
    assert out["prefill"] is None
    assert out["decode"] is None
