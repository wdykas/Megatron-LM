# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configure a prefill/decode shard engine for the shared DP inference
coordinator (role + KV layouts + identity); called by ``MegatronAsyncLLM`` when
given ``inference_shards``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List

import torch.distributed as dist

from megatron.core.inference.disaggregation.orchestration import (
    PREFILL,
    _global_kv_dims,
    _validate_disagg_specs,
    layout_from_pg_collection,
)
from megatron.core.inference.shards_spec import InferenceShardSpec
from megatron.core.utils import get_pg_rank, get_pg_size


@dataclass
class DisaggCoordinatorSetup:
    """This rank's place in a coordinator-native disagg job."""

    role: str           # "prefill" / "decode"
    replica_id: str     # "prefill" / "decode_s{shard}_dp{dp}"
    engine: Any
    is_primary: bool    # global rank 0 -> owns the InferenceClient
    total_instances: int


def configure_prebuilt_disagg_engine(
    engine: Any, pg: Any, specs: List[InferenceShardSpec]
) -> DisaggCoordinatorSetup:
    """Configure an already-built engine for the shared coordinator.

    The caller built the model + engine against this rank's shard ``pg``
    outside (mirroring ``MegatronLLM(model=...)``); this only derives the disagg
    config and sets it on the engine. The per-rank KV layout is read from the
    live ``pg`` and the full per-instance layout is gathered over the instance's
    MP group (tp x pp), so it is correct for any tp/dp/pp rank ordering (no
    contiguity assumption).
    """
    total_instances = sum(s.dp for s in specs)
    _validate_disagg_specs(specs)  # role layout / single-prefill-instance checks
    rank = dist.get_rank()

    # Locate this rank's shard. Shard windows are contiguous (tp*pp*dp ranks
    # each) regardless of the intra-shard rank ordering.
    offset = 0
    my_index = None
    my_spec = None
    for i, s in enumerate(specs):
        if offset <= rank < offset + s.world_size:
            my_index, my_spec = i, s
            break
        offset += s.world_size
    assert my_spec is not None, f"rank {rank} not in any disagg shard window"
    role = my_spec.role

    num_layers, num_heads = _global_kv_dims(engine)
    dp_rank = get_pg_rank(pg.dp)
    my_layout = asdict(layout_from_pg_collection(pg, num_layers, num_heads))
    # Gather every rank of this instance (the MP group spans exactly tp x pp).
    layouts = [None] * get_pg_size(pg.mp)
    dist.all_gather_object(layouts, my_layout, group=pg.mp)

    replica_id = PREFILL if role == PREFILL else f"decode_s{my_index}_dp{dp_rank}"
    engine.set_disaggregation_config(
        role=role,
        instance_layouts=layouts,
        identity=replica_id,
        total_instances=total_instances,
        world_group=None,  # default world group for the cross-shard addr broadcast
        spawn_coordinator=(rank == 0),
    )
    return DisaggCoordinatorSetup(
        role=role, replica_id=replica_id, engine=engine,
        is_primary=(rank == 0), total_instances=total_instances,
    )
