# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end multi-GPU test exercising the FULL producer→handler path.

The test plays both sides of the wire:

  1. Build a Route from a 2-shard layout and ``plan_route`` (planner side).
  2. ``serialize_route`` to the wire form (what the coord would send).
  3. On each rank, ``deserialize_route`` (what the engine receives).
  4. Build a ``RouteDispatcher`` via the exact construction the handler
     does (mimicking ``_on_route_request_signal``).
  5. Run a real forward pass driven by ``dispatch_layer``, with hidden
     states flowing over NVSHMEM between ranks.
  6. Assert the exit rank's output matches a single-rank baseline.

This pairs with ``test_disagg_e2e_protocol.py`` (which validates the
coord's fan-out logic in-process) — together they cover the full path
without needing a real model or a subprocess coord.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import activation_transport as at
from megatron.core.inference.route_dispatcher import RouteDispatcher
from megatron.core.inference.shards import InferenceShard
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    deserialize_route,
    plan_route,
    serialize_route,
)


@pytest.fixture(scope="module")
def _dist_world():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("e2e disagg needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_dispatcher_from_wire_route(
    wire_route, shards, my_shard_idx, my_pe, hidden_shape, hidden_dtype
):
    """Reconstruct a ``RouteDispatcher`` from the wire-form route the
    coord would deliver. Mirrors
    ``MegatronLocalMulti._on_route_request_signal`` so the test
    exercises the same construction path the engine handler uses.
    """
    route = deserialize_route(wire_route)
    my_shard = shards[my_shard_idx]
    tp_offset = my_pe - my_shard.rank_offset
    shard_tp = [int(s.spec["tp"]) for s in shards]
    shard_rank_offset = [s.rank_offset for s in shards]
    return RouteDispatcher(
        route=route,
        my_shard_idx=my_shard_idx,
        my_pe=my_pe,
        my_tp_offset=tp_offset,
        shard_tp=shard_tp,
        shard_rank_offset=shard_rank_offset,
        hidden_shape=hidden_shape,
        hidden_dtype=hidden_dtype,
    )


def test_full_producer_to_transport_e2e(_dist_world):
    """Full path: plan → serialize → (coord wire) → deserialize → build
    dispatcher → run forward → activations flow via NVSHMEM → exit
    rank's output matches the single-rank baseline."""
    rank = _dist_world
    if dist.get_world_size() < 2:
        pytest.skip("needs >=2 ranks")

    at._reset_state_for_test()
    at.maybe_init_activation_transport(
        num_lanes=4, pool_depth=4, slot_bytes=64 * 1024, max_pes=2
    )

    hidden_dim = 64
    batch = 2
    num_layers = 4

    # Layout: 2 shards, tp=1 each, alternating layer-kind ownership.
    # Shard 0 owns even-index layers (M-like); shard 1 owns odd (* / E).
    layer_type_list = ("M", "*", "M", "*")
    shards = [
        InferenceShard(
            index=0,
            spec={"tp": 1, "pp": 1, "ep": 1, "expt_tp": 1, "dp": 1, "kinds": ("M",)},
            rank_offset=0,
            world_size=1,
            pg_collection=None,
            kinds=("M",),
            layer_indices=(0, 2),
        ),
        InferenceShard(
            index=1,
            spec={"tp": 1, "pp": 1, "ep": 1, "expt_tp": 1, "dp": 1, "kinds": ("*",)},
            rank_offset=1,
            world_size=1,
            pg_collection=None,
            kinds=("*",),
            layer_indices=(1, 3),
        ),
    ]

    # === 1. Planner side: build Route from layout. ===
    route = plan_route(shards, layer_type_list=layer_type_list)
    assert route.entry_shard == 0
    assert route.exit_shard == 1
    assert len(route.hops) == num_layers  # full alternation

    # === 2. Coord side: serialize for the wire. ===
    wire_route = serialize_route(route)

    # === 3. Engine side: rank reconstructs the dispatcher from the
    # wire form (this is what _on_route_request_signal does). ===
    dispatcher = _build_dispatcher_from_wire_route(
        wire_route=wire_route,
        shards=shards,
        my_shard_idx=rank,
        my_pe=rank,
        hidden_shape=(batch, hidden_dim),
        hidden_dtype=torch.float32,
    )

    # === 4. Build a toy model. ===
    g = torch.Generator(device="cpu").manual_seed(7)
    initial_hidden = torch.empty(batch, hidden_dim)
    initial_hidden.normal_(generator=g)
    initial_hidden = initial_hidden.cuda()
    gw = torch.Generator(device="cpu").manual_seed(42)
    weights = []
    for _ in range(num_layers):
        w = torch.empty(hidden_dim, hidden_dim)
        w.normal_(generator=gw)
        weights.append(w.cuda())

    def apply_layer(layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        return torch.relu(h @ weights[layer_idx])

    # Single-rank baseline.
    expected = initial_hidden
    for i in range(num_layers):
        expected = apply_layer(i, expected)

    # === 5. Drive forward via dispatch_layer; hidden flows over NVSHMEM. ===
    hidden = initial_hidden if dispatcher.is_entry_shard() else None
    for li in range(num_layers):
        hidden, _ = dispatcher.dispatch_layer(
            li, hidden, lambda h, i=li: apply_layer(i, h)
        )

    at.activation_stream().synchronize()

    # === 6. Exit shard's output should match the single-rank baseline. ===
    if dispatcher.is_exit_shard():
        max_diff = (hidden - expected).abs().max().item()
        assert torch.allclose(hidden, expected, atol=1e-4, rtol=1e-4), (
            f"e2e disagg forward diverged from baseline: "
            f"max_diff={max_diff:.6f}"
        )

    dist.barrier()
