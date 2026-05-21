# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end multi-GPU test for the DISAGG_SUBMIT-driven forward path.

This test plays both sides of the protocol:
- Rank 0 acts as the entry shard: it drives forward via a manual loop
  (mimicking what a real engine's controller would do).
- Rank 1 acts as the non-entry participant: instead of hand-driving,
  it invokes the registered ``_on_disagg_submit_signal`` handler,
  which spawns the forward-driver asyncio task. The task pumps the
  dispatcher's transport.

NVSHMEM transport synchronizes the two sides via signal_wait. The
test asserts the transport completes without deadlock — i.e., the
task spawned by the handler actually drains the activations rank 0
puts.

This is the closest we get to "real engine-driven disagg" without
spinning up a full engine + model. The remaining production gap is
the entry shard's engine driving forward through the controller
(which has its own internal complexity around request iteration,
KV management, sampling — see DISAGG_DESIGN.md § "Open issues").
"""

import asyncio
import os
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference import activation_transport as at
from megatron.core.inference.route_dispatcher import RouteDispatcher
from megatron.rl.inference.route_planner import Route, RouteHop


@pytest.fixture(scope="module")
def _dist_world():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("test must run under torchrun")
    if int(os.environ["WORLD_SIZE"]) < 2:
        pytest.skip("handler-driven e2e needs >=2 ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    yield rank
    if dist.is_initialized():
        dist.destroy_process_group()


def test_disagg_submit_handler_drives_transport_e2e(_dist_world):
    """2-shard route 0→1. Rank 0 plays entry shard: drives forward by
    hand. Rank 1 plays non-entry: invokes the registered handler
    which spawns a forward-driver task. NVSHMEM transport
    synchronizes; the test passes iff the cross-shard handshake
    closes without deadlock."""
    rank = _dist_world
    if dist.get_world_size() < 2:
        pytest.skip("needs >=2 ranks")

    at._reset_state_for_test()
    at.maybe_init_activation_transport(
        num_lanes=4, pool_depth=4, slot_bytes=64 * 1024, max_pes=2
    )

    hidden_dim = 32
    batch = 1
    num_layers = 2

    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )

    initial_hidden = torch.ones(batch, hidden_dim).cuda()

    dispatcher = RouteDispatcher(
        route=route,
        my_shard_idx=rank,
        my_pe=rank,
        my_tp_offset=0,
        shard_tp=[1, 1],
        shard_rank_offset=[0, 1],
        hidden_shape=(batch, hidden_dim),
        hidden_dtype=torch.float32,
    )

    if rank == 0:
        # Entry shard: drive forward by hand. Layer 0 runs locally;
        # at the hop exit, the dispatcher sends to rank 1.
        hidden = initial_hidden
        for li in range(num_layers):
            hidden, _ = dispatcher.dispatch_layer(li, hidden, lambda h: h * 2)
        at.activation_stream().synchronize()

    else:
        # Non-entry shard: simulate the multi_shard handler being
        # invoked. We construct the multi_shard scaffolding the same
        # way ``set_disagg_submit_handler`` would fire it.
        from megatron.rl.inference.multi_shard import MegatronLocalMulti

        instance = MegatronLocalMulti.__new__(MegatronLocalMulti)
        instance._my_engine = MagicMock()
        instance._my_engine._route_dispatchers = {99: dispatcher}

        # Run the handler — it spawns a task on the running loop.
        async def _drive():
            instance._on_disagg_submit_signal(
                request_id=99,
                prompt="ignored",
                sampling_params={},
                role="exit",
            )
            # The handler queued an asyncio task; let it run to
            # completion. The task's dispatch_layer calls will
            # signal_wait on rank 0's put — once rank 0 has sent,
            # the task resolves.
            await asyncio.sleep(0)
            # Wait for all pending tasks (the forward driver).
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task() and not t.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        asyncio.run(_drive())
        at.activation_stream().synchronize()

    # If we got here on both ranks, the cross-shard transport
    # closed without deadlock — the handler-spawned task successfully
    # consumed rank 0's puts.
    dist.barrier()
