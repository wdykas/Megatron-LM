# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Option C: the online compactor as a first-class Megatron-parallel module.

The compactor is **fully replicated on every rank**, and each rank captures and
trains on a **different local KV slice** (different KV heads across TP, different
rollouts across DP). So for the compactor *every rank is a data-parallel peer* —
its data-parallel group is the **whole world** (TP×PP×DP collapsed). This is the
key insight that makes Option C correct under arbitrary parallelism: we do not
treat the compactor as TP-sharded (it is not) nor DP-only (TP ranks also hold
distinct data); we treat the entire world as one data-parallel group.

``CompactorParallelOptimizer`` wraps the replicated compactor in Megatron's
``DistributedDataParallel`` over that world group and drives it with a real
``get_megatron_optimizer``. Gradients are all-reduced across the world by
``finalize_model_grads`` on every ``step()``; replicas therefore stay
bit-identical (validated under torchrun TP=2×DP=2 in scripts/_optc_smoke.py:
param spread = 0). It exposes the same ``zero_grad()/step()/state`` duck-type as
the offline ``_CompactorOptimizer`` so ``train_compactor_trajectory`` is unchanged.

Checkpointing is idiomatic Megatron ``dist_checkpointing`` (see checkpoint.py) —
collective across all ranks, no bespoke rank-0 ``torch.save``.

This requires ``torch.distributed`` to be initialized — always true in the RL
loop. The offline trainer (scripts/train_still.py) runs single-process without
distributed and keeps ``_CompactorOptimizer``.
"""

from __future__ import annotations

import torch

from megatron.core.distributed import (
    DistributedDataParallel as DDP,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


def build_compactor_pg_collection() -> ProcessGroupCollection:
    """Process groups that make the compactor data-parallel across the WHOLE world.

    ``dp``/``dp_cp`` = the world group (every rank is a data-parallel peer holding
    a distinct KV slice). ``tp``/``pp``/``ep`` = per-rank singleton groups so no
    tensor/pipeline/expert sharding reductions ever fire for the replicated
    compactor. ``embd``/``pos_embd`` = None (the compactor has no tied embeddings).
    """
    world = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    world_group = torch.distributed.group.WORLD
    # Per-rank singleton group. new_group is collective: every rank must call it
    # for every member list, so loop over all ranks and keep our own group.
    self_group = None
    for r in range(world):
        grp = torch.distributed.new_group(ranks=[r])
        if r == rank:
            self_group = grp
    return ProcessGroupCollection(
        dp=world_group,
        dp_cp=world_group,
        tp=self_group,
        pp=self_group,
        ep=self_group,
        embd=None,
        pos_embd=None,
    )


class CompactorParallelOptimizer:
    """World-DP DDP + Megatron optimizer for the online compactor (Option C).

    Wraps the replicated compactor in Megatron ``DistributedDataParallel`` over
    the world group and a ``get_megatron_optimizer`` built on its grad buffers.
    Every ``step()`` runs ``finalize_model_grads`` to all-reduce gradients across
    the world (so all replicas stay identical) and then steps the optimizer.

    Duck-types the interface ``train_compactor_trajectory`` expects
    (``zero_grad`` / ``step`` / ``state``), so the training core is shared with
    the offline path unchanged.
    """

    def __init__(self, compactor, lr: float) -> None:
        config = compactor.config
        self.compactor = compactor
        self.pg_collection = build_compactor_pg_collection()

        ddp_cfg = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
            bucket_size=None,
        )
        self.ddp = DDP(
            config=config,
            ddp_config=ddp_cfg,
            module=compactor,
            pg_collection=self.pg_collection,
        )
        # Start every replica identical: TE init uses per-rank RNG and there is no
        # TP RNG tracker for the compactor, so broadcast weights from global rank 0.
        for p in compactor.parameters():
            torch.distributed.broadcast(p.data, src=0)

        _params_dtype = next(compactor.parameters()).dtype
        opt_cfg = OptimizerConfig(
            optimizer="adam",
            lr=lr,
            bf16=(_params_dtype == torch.bfloat16),
            fp16=(_params_dtype == torch.float16),
            params_dtype=_params_dtype,
            use_distributed_optimizer=False,
            clip_grad=0.0,                # grad-norm clip needs a world-wide reduce; off for now
            log_num_zeros_in_grad=False,
        )
        self.optimizer = get_megatron_optimizer(opt_cfg, [self.ddp])

    # --- duck-type interface used by train_compactor_trajectory ---

    @property
    def state(self):
        # Inner Adam state (keyed by FP32 masters) — used only for a logging count.
        return self.optimizer.optimizer.state

    def zero_grad(self):
        self.ddp.zero_grad_buffer()
        self.optimizer.zero_grad()

    def step(self):
        # All-reduce grads across the world group (DP+TP+PP all collapsed into the
        # world DP group), then step. Keeps every replica bit-identical.
        finalize_model_grads([self.ddp], pg_collection=self.pg_collection)
        self.optimizer.step()

    # --- checkpoint support (delegates to the Megatron optimizer) ---

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False):
        """Optimizer state as replicated ShardedTensors (idiomatic dist_checkpointing).

        Storing the FP32 masters / Adam moments as *sharded* (replicated) state — not
        as plain common state — is the correct representation: each tensor is written
        once and read on every rank, and it passes the cross-rank consistency check.
        """
        return self.optimizer.sharded_state_dict(model_sharded_state_dict, is_loading=is_loading)

    def load_state_dict(self, sd):
        self.optimizer.load_state_dict(sd)
