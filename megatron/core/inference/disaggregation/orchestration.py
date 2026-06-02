# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared prefill->decode disaggregation helpers: KV-shard layout from a shard's
process groups, role-layout validation, and global KV dims."""

from __future__ import annotations

from typing import List, Tuple

import torch.distributed as dist

from megatron.core.inference.disaggregation.kv_shard_layout import KVShardLayout
from megatron.core.inference.shards_spec import InferenceShardSpec
from megatron.core.utils import get_pg_rank, get_pg_size

PREFILL = "prefill"
DECODE = "decode"


def layout_from_pg_collection(pg, num_layers: int, num_heads: int) -> KVShardLayout:
    """Build a :class:`KVShardLayout` from a shard's ``ProcessGroupCollection``.

    Reads attention TP/PP (which shard the KV) and EP/ETP (KV-replica
    dimensions, used only for source dedup) from the collection's groups.
    """
    return KVShardLayout(
        num_layers=num_layers,
        num_heads=num_heads,
        tp_size=get_pg_size(pg.tp),
        tp_rank=get_pg_rank(pg.tp),
        pp_size=get_pg_size(pg.pp),
        pp_rank=get_pg_rank(pg.pp),
        global_rank=dist.get_rank(),
        ep_size=get_pg_size(getattr(pg, "ep", None)),
        ep_rank=get_pg_rank(getattr(pg, "ep", None)),
        etp_size=get_pg_size(getattr(pg, "expt_tp", None)),
        etp_rank=get_pg_rank(getattr(pg, "expt_tp", None)),
    )


def _validate_disagg_specs(specs: List[InferenceShardSpec]) -> int:
    """Check the role layout; return the number of decode instances."""
    prefill = [s for s in specs if s.role == PREFILL]
    decode = [s for s in specs if s.role == DECODE]
    untagged = [s for s in specs if s.role not in (PREFILL, DECODE)]
    assert not untagged, (
        f"every shard must declare role=prefill or role=decode for "
        f"disaggregation; {len(untagged)} shard(s) had none: {untagged}"
    )
    assert prefill and decode, (
        "disaggregation needs at least one prefill shard and one decode shard."
    )
    prefill_instances = sum(s.dp for s in prefill)
    assert len(prefill) == 1 and prefill_instances == 1, (
        "exactly one prefill instance is supported today (one prefill shard "
        f"with dp=1); got {len(prefill)} prefill shard(s) totalling "
        f"{prefill_instances} instance(s). Data-parallel prefill needs "
        "per-request prefill-instance scoping (a follow-up)."
    )
    return sum(s.dp for s in decode)


def _global_kv_dims(engine) -> Tuple[int, int]:
    """Global (num_layers, KV-head count) read off the built engine's model
    config. KV heads = num_query_groups for GQA, else num_attention_heads."""
    cfg = engine.controller.model_config
    num_heads = getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads
    return cfg.num_layers, num_heads
