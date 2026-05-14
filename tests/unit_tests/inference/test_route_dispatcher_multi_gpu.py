# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end disagg forward pass via the high-level RouteDispatcher.

Mirrors the hand-rolled smoke test but uses :class:`RouteDispatcher`
to drive the layer loop — proves the dispatcher abstraction is what
will eventually wire into ``DynamicInferenceEngine.step()``.
"""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import activation_transport as at
from megatron.core.inference.route_dispatcher import RouteDispatcher
from megatron.core.inference.route_dispatcher import LayerAction
from megatron.rl.inference.route_planner import Route, RouteHop


@pytest.fixture(scope="module")
def _dist_world():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("dispatcher smoke needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    if dist.is_initialized():
        dist.destroy_process_group()


def test_dispatcher_drives_disagg_forward(_dist_world):
    """4-layer toy model with shard 0 = layers {0, 2}, shard 1 =
    layers {1, 3}. Dispatcher walks the model on each shard; the exit
    shard's final hidden must match a single-rank baseline."""
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

    # Baseline (serial).
    expected = initial_hidden
    for i in range(num_layers):
        expected = apply_layer(i, expected)

    # Disagg route: shard 0 → 1 → 0 → 1.
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
            RouteHop(shard_idx=0, layer_indices=(2,)),
            RouteHop(shard_idx=1, layer_indices=(3,)),
        ),
    )

    dispatcher = RouteDispatcher(
        route=route,
        my_shard_idx=rank,
        my_pe=rank,
        my_tp_offset=0,         # 2 ranks, 2 shards, tp=1 each
        shard_tp=[1, 1],
        shard_rank_offset=[0, 1],
        hidden_shape=(batch, hidden_dim),
        hidden_dtype=torch.float32,
    )

    # Forward pass driven by the dispatcher. The loop runs every
    # layer; the dispatcher decides per-layer whether to run locally,
    # send + clear hidden, or thread None through NOT_MY_REQUEST until
    # the next RECEIVE.
    hidden = initial_hidden if dispatcher.is_entry_shard() else None
    for li in range(num_layers):
        hidden, _ = dispatcher.dispatch_layer(
            li, hidden, lambda h, i=li: apply_layer(i, h)
        )

    at.activation_stream().synchronize()

    if dispatcher.is_exit_shard():
        max_diff = (hidden - expected).abs().max().item()
        assert torch.allclose(hidden, expected, atol=1e-4, rtol=1e-4), (
            f"dispatcher forward diverged from baseline: "
            f"max_diff={max_diff:.6f}"
        )

    dist.barrier()
