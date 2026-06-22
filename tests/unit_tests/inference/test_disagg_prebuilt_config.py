# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""configure_prebuilt_disagg_engine derives each rank's disagg role/identity/
layouts from a shard layout + live process groups (gloo, CPU; incl. dp>1 decode)."""

import os

import pytest
import torch

mp = torch.multiprocessing

L, H = 4, 8  # global model dims (read off the fake engine's model_config)


class _FakeModelConfig:
    num_layers = L
    num_query_groups = H
    num_attention_heads = H


class _FakeController:
    model_config = _FakeModelConfig()


class _RecordingEngine:
    """Captures the set_disaggregation_config kwargs for assertions."""

    def __init__(self):
        self.controller = _FakeController()
        self.cfg = None

    def set_disaggregation_config(self, **kwargs):
        self.cfg = kwargs


def _worker(rank, world, spec_str, port, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world)
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from megatron.core.inference.disaggregation.coordinator_setup import (
            configure_prebuilt_disagg_engine,
        )
        from megatron.core.inference.shards import build_inference_pg_collections_for_shards
        from megatron.core.inference.shards_spec import parse_inference_shards_spec

        specs = parse_inference_shards_spec(spec_str, world)
        shards = build_inference_pg_collections_for_shards(world, specs)
        my = next(s for s in shards if s.pg_collection is not None)

        engine = _RecordingEngine()
        setup = configure_prebuilt_disagg_engine(engine, my.pg_collection, specs)
        cfg = engine.cfg

        q.put(
            (
                rank,
                {
                    "role": setup.role,
                    "replica_id": setup.replica_id,
                    "is_primary": setup.is_primary,
                    "total_instances": setup.total_instances,
                    "identity": cfg["identity"],
                    "layout_ranks": sorted(d["global_rank"] for d in cfg["instance_layouts"]),
                    "spawn": cfg["spawn_coordinator"],
                },
            )
        )
        # Rendezvous before any rank tears down: rank 0 hosts the TCPStore, so
        # if it exits while a slower rank is still in a collective the latter
        # hits a "Broken pipe". The barrier keeps the group alive until all
        # ranks have finished (only reached on the success path).
        dist.barrier()
    except Exception:
        import traceback

        q.put((rank, {"ERROR": traceback.format_exc()}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port():
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(spec_str, world, port=None):
    port = port or _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, world, spec_str, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    out = {}
    for _ in range(world):
        rank, payload = q.get(timeout=180)
        out[rank] = payload
    for p in procs:
        p.join(timeout=60)
    errs = {r: v["ERROR"] for r, v in out.items() if "ERROR" in v}
    assert not errs, errs
    return out


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_prefill_decode_single_instance():
    out = _run("tp=1,role=prefill+tp=1,role=decode", world=2)
    # rank 0 -> prefill (spawns coordinator, owns the client)
    assert out[0]["role"] == "prefill"
    assert out[0]["identity"] == "prefill_s0_dp0"
    assert out[0]["is_primary"] and out[0]["spawn"]
    assert out[0]["layout_ranks"] == [0]
    # rank 1 -> the single decode instance
    assert out[1]["role"] == "decode"
    assert out[1]["replica_id"] == "decode_s1_dp0"
    assert out[1]["identity"] == "decode_s1_dp0"
    assert not out[1]["is_primary"] and not out[1]["spawn"]
    assert out[1]["layout_ranks"] == [1]
    for r in (0, 1):
        assert out[r]["total_instances"] == 2


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_decode_dp_is_independent_instances():
    # prefill {0}; decode dp=2 -> two independent decode instances {1}, {2}.
    out = _run("tp=1,role=prefill+tp=1,dp=2,role=decode", world=3)
    assert out[0]["role"] == "prefill" and out[0]["identity"] == "prefill_s0_dp0"
    assert out[1]["replica_id"] == "decode_s1_dp0" and out[1]["layout_ranks"] == [1]
    assert out[2]["replica_id"] == "decode_s1_dp1" and out[2]["layout_ranks"] == [2]
    # three instances total: one prefill + two decode.
    for r in (0, 1, 2):
        assert out[r]["total_instances"] == 3


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_prefill_dp_is_independent_instances():
    # prefill dp=2 -> two independent prefill instances {0}, {1}; decode {2}.
    out = _run("tp=1,dp=2,role=prefill+tp=1,role=decode", world=3)
    assert out[0]["role"] == "prefill" and out[0]["identity"] == "prefill_s0_dp0"
    assert out[1]["role"] == "prefill" and out[1]["identity"] == "prefill_s0_dp1"
    assert out[0]["layout_ranks"] == [0] and out[1]["layout_ranks"] == [1]
    assert out[2]["role"] == "decode" and out[2]["identity"] == "decode_s1_dp0"
    # three instances total: two prefill + one decode.
    for r in (0, 1, 2):
        assert out[r]["total_instances"] == 3


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_prefill_tp2_instance_layout_gathered():
    # prefill tp=2 {0,1} -> decode tp=1 {2}. The prefill instance layout must
    # gather both of its ranks over pg.mp.
    out = _run("tp=2,role=prefill+tp=1,role=decode", world=3)
    assert out[0]["role"] == "prefill" and out[1]["role"] == "prefill"
    assert out[0]["layout_ranks"] == [0, 1] and out[1]["layout_ranks"] == [0, 1]
    assert out[2]["role"] == "decode" and out[2]["layout_ranks"] == [2]
