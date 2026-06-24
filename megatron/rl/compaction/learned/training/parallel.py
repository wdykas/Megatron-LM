# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parallel setup for online compactor training.

The compactor is **replicated** on every rank, and each rank trains it on its own
local KV slice (different KV heads across TP, different rollouts across DP). So for
the compactor *every rank is a data-parallel peer*: its data-parallel group is the
whole world (TP×PP×DP collapsed).

We express that with a ``ProcessGroupCollection`` whose dp group is the world and
whose tp/pp/ep groups are per-rank singletons, then use the **standard** Megatron
machinery — ``DistributedDataParallel`` + ``get_megatron_optimizer`` — with no
custom optimizer class. Training does the ordinary Megatron sequence
(``zero_grad_buffer`` → backward → ``finish_grad_sync`` → ``optimizer.step``); the
DDP all-reduces gradients across the world so replicas stay bit-identical
(validated under torchrun TP=2×DP=2 in scripts/_optc_smoke.py).
"""

from __future__ import annotations

import torch

from megatron.core.distributed import (
    DistributedDataParallel as DDP,
    DistributedDataParallelConfig,
)
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


def build_compactor_pg_collection() -> ProcessGroupCollection:
    """Process groups placing the compactor as a pure data-parallel model over the world.

    ``dp``/``dp_cp`` = the world group; ``tp``/``cp``/``pp``/``ep`` = per-rank
    singletons so the replicated compactor never shards. Passed both to the model's
    Megatron sub-modules (tp=1 → replicated) and to DDP (dp=world → grads averaged
    across every rank).
    """
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    world_group = torch.distributed.group.WORLD
    # Per-rank singleton group. new_group is collective: every rank must call it for
    # every member list, so loop over all ranks and keep our own.
    self_group = None
    for r in range(world):
        grp = torch.distributed.new_group(ranks=[r])
        if r == rank:
            self_group = grp
    return ProcessGroupCollection(
        dp=world_group,
        dp_cp=world_group,
        tp=self_group,
        cp=self_group,
        pp=self_group,
        ep=self_group,
        embd=None,
        pos_embd=None,
    )


def wrap_compactor_for_training(compactor, lr: float, pg_collection=None):
    """DDP-wrap the compactor and build its Megatron optimizer (the standard way).

    Returns ``(ddp_model, optimizer)``. ``ddp_model`` is an mcore
    ``DistributedDataParallel`` whose DP group is the world (when ``pg_collection``
    is the world-DP collection), so ``ddp_model.finish_grad_sync()`` averages
    gradients across every rank. The optimizer is a real ``get_megatron_optimizer``
    result (mixed precision: BF16 params + FP32 masters), checkpointable via its
    own ``sharded_state_dict``.
    """
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        bucket_size=None,
    )
    ddp_model = DDP(
        config=compactor.config,
        ddp_config=ddp_config,
        module=compactor,
        pg_collection=pg_collection,
    )
    # Start every replica identical (TE init uses per-rank RNG; no TP RNG tracker).
    for p in compactor.parameters():
        torch.distributed.broadcast(p.data, src=0)

    params_dtype = next(compactor.parameters()).dtype
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=lr,
            bf16=(params_dtype == torch.bfloat16),
            fp16=(params_dtype == torch.float16),
            params_dtype=params_dtype,
            use_distributed_optimizer=False,
            clip_grad=0.0,
            log_num_zeros_in_grad=False,
        ),
        [ddp_model],
    )
    return ddp_model, optimizer
