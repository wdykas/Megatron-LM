# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end 2-GPU disagg smoke test.

Hand-rolls a forward pass that walks through a 2-shard layer-kind-
disaggregated layout, transferring real activation tensors between
shards via :mod:`activation_transport` and dispatching layer compute
via :mod:`route_walker`. Verifies the disagg pipeline reproduces the
same output as a single-shard baseline running every layer locally.

Run with::

    python -m torch.distributed.run --nproc-per-node=2 \
        -m pytest tests/unit_tests/inference/test_disagg_smoke_multi_gpu.py -v

This is the minimum proof that the primitives compose into a real
end-to-end forward pass — without yet touching ``DynamicInferenceEngine``.
The engine-side integration in task 96 (route-walker wiring into
``step()``) is the next piece; once that lands, this test shape moves
into the engine test suite as the canonical disagg smoke.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.core.inference import activation_transport as at
from megatron.core.inference import nvshmem_migration as _nv
from megatron.core.inference.route_walker import LayerAction, RouteWalker
from megatron.rl.inference.route_planner import Route, RouteHop


@pytest.fixture(scope="module")
def _dist_world():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("disagg smoke needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_layer_weights(num_layers: int, hidden: int, seed: int) -> list:
    """Deterministic per-layer weights so both shards agree on the
    transformation. Each layer is a simple linear+ReLU on the hidden
    dim; the key property is that the composition is reproducible
    given the same seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    weights = []
    for i in range(num_layers):
        w = torch.empty(hidden, hidden)
        w.normal_(generator=g)
        weights.append(w.cuda())
    return weights


def _apply_layer(hidden: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Forward pass for one layer in the toy model."""
    return torch.relu(hidden @ w)


def test_disagg_forward_pass_two_shards_matches_baseline(_dist_world):
    """Two shards (shard 0 owns layers [0, 2], shard 1 owns [1, 3]) ⇒
    activations flow 0 → 1 → 0 → 1 over the route. The final hidden
    state on shard 1 must equal the same chain applied locally to the
    same input."""
    rank = _dist_world
    if dist.get_world_size() < 2:
        pytest.skip("needs >=2 ranks")

    # Set up activation transport (idempotent across tests within the
    # module).
    at._reset_state_for_test()
    at.maybe_init_activation_transport(
        num_lanes=4, pool_depth=4, slot_bytes=64 * 1024, max_pes=2
    )

    hidden_dim = 64
    batch = 2
    num_layers = 4  # shard 0 owns even layers, shard 1 owns odd
    payload_nbytes = batch * hidden_dim * 4  # float32

    # Deterministic input + identical weights on both ranks. The disagg
    # plan transfers activations across; the math is identical to the
    # serial baseline.
    g = torch.Generator(device="cpu").manual_seed(7)
    initial_hidden = torch.empty(batch, hidden_dim)
    initial_hidden.normal_(generator=g)
    initial_hidden = initial_hidden.cuda()
    weights = _build_layer_weights(num_layers, hidden_dim, seed=42)

    # Baseline (computed on both ranks identically for the assertion).
    expected = initial_hidden
    for w in weights:
        expected = _apply_layer(expected, w)

    # Disagg route: shard 0 runs layers 0 and 2; shard 1 runs 1 and 3.
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,), src_shard=None),
            RouteHop(shard_idx=1, layer_indices=(1,), src_shard=0),
            RouteHop(shard_idx=0, layer_indices=(2,), src_shard=1),
            RouteHop(shard_idx=1, layer_indices=(3,), src_shard=0),
        ),
        entry_shard=0,
        exit_shard=1,
    )

    my_shard = rank  # 0 → shard 0, 1 → shard 1
    walker = RouteWalker(route, my_shard_idx=my_shard)

    stream = at.activation_stream()
    hidden: torch.Tensor = initial_hidden  # entry shard starts here

    def _send_to(dst_shard: int, h: torch.Tensor) -> None:
        """Gather the hidden tensor into a slot, put + signal, sync."""
        lane = at.lane_for(rank, dst_shard)
        slot = at.next_send_slot(lane)
        sym = at.activation_slot(slot)
        with torch.cuda.stream(stream):
            view = sym[:payload_nbytes].view(torch.float32).reshape(h.shape)
            view.copy_(h)
        at.put_activation(slot, dst_pe=dst_shard, nbytes=payload_nbytes, stream=stream)

    def _recv_from(src_shard: int) -> torch.Tensor:
        """Wait for an activation on the lane from src_shard, scatter
        into a fresh tensor, ack back."""
        lane = at.lane_for(src_shard, rank)
        slot = at.next_recv_slot(lane)
        at.wait_activation(slot, stream=stream)
        stream.synchronize()
        sym = at.activation_slot(slot)
        view = sym[:payload_nbytes].view(torch.float32).reshape(batch, hidden_dim)
        out = view.clone()  # detach from the symmetric slot
        at.ack_activation(slot, src_pe=src_shard, stream=stream)
        return out

    # Walk the model layer-by-layer; the walker tells us what to do at
    # each step.
    for li in range(num_layers):
        dec = walker.before_layer(li)
        if dec.action is LayerAction.LOCAL:
            hidden = _apply_layer(hidden, weights[li])
        elif dec.action is LayerAction.RECEIVE:
            hidden = _recv_from(dec.peer_shard)
            walker.after_receive()
            # Re-issue: next decision must be LOCAL on this same layer.
            dec2 = walker.before_layer(li)
            assert dec2.action is LayerAction.LOCAL
            hidden = _apply_layer(hidden, weights[li])
        elif dec.action is LayerAction.SEND:
            _send_to(dec.peer_shard, hidden)
            walker.after_send()
            # Done sending; nothing more to compute locally until next
            # RECEIVE or DONE. If we still have hops, the loop continues
            # and the next layer will be NOT_MY_REQUEST until we hit
            # RECEIVE.
        elif dec.action is LayerAction.NOT_MY_REQUEST:
            pass  # not my hop, skip layer
        elif dec.action is LayerAction.DONE:
            break

    # Drain the walker — depending on the final hop's position, the
    # entry shard may have a pending SEND at layer 4 (past num_layers).
    dec = walker.before_layer(num_layers)
    if dec.action is LayerAction.SEND:
        _send_to(dec.peer_shard, hidden)
        walker.after_send()

    stream.synchronize()

    # Exit shard (shard 1) holds the final ``hidden``; entry shard
    # holds something stale. Verify on the exit shard.
    if my_shard == route.exit_shard:
        max_diff = (hidden - expected).abs().max().item()
        assert torch.allclose(hidden, expected, atol=1e-4, rtol=1e-4), (
            f"disagg forward pass diverged from baseline: max_diff={max_diff:.6f}"
        )

    dist.barrier()
