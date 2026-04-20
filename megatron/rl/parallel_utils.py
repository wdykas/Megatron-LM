# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""RL-side convenience layer over :mod:`megatron.core.inference.shards`.

The process-group building primitives are framework-agnostic and live in
``megatron.core.inference.shards``. This module adds the Megatron-LM RL stack's
convenience layer on top:

- a module-level shard registry (``set_inference_shards`` / ``get_inference_shards``),
- a wrapper around :func:`megatron.core.inference.shards.build_cross_shard_group`
  that reads the registry,
- a wrapper around
  :func:`megatron.core.resharding.refit.swap_model_weights_across_shards` that
  likewise reads the registry.

Downstream frameworks (NeMo-RL, verl, etc.) should import directly from
``megatron.core.inference.shards`` and ``megatron.core.resharding.refit`` and
thread shard lists through their own state.

The symbols :class:`InferenceShard`, :func:`build_inference_pg_collection`, and
:func:`build_inference_pg_collections_for_shards` are re-exported here for
back-compat with code that already imports them from this module.
"""

from typing import List, Optional

import torch.distributed as dist

# Re-export the framework-agnostic primitives from core.
from megatron.core.inference.shards import InferenceShard
from megatron.core.inference.shards import build_cross_shard_group as _core_build_cross_shard_group
from megatron.core.inference.shards import (
    build_inference_pg_collection,
    build_inference_pg_collections_for_shards,
    clear_cross_shard_group_cache,
)
from megatron.core.resharding.refit import (
    swap_model_weights_across_shards as _core_swap_model_weights_across_shards,
)

__all__ = [
    "InferenceShard",
    "build_inference_pg_collection",
    "build_inference_pg_collections_for_shards",
    "build_cross_shard_group",
    "clear_cross_shard_group_cache",
    "set_inference_shards",
    "get_inference_shards",
    "get_my_inference_shard",
    "swap_weights_across_shards",
]


# ---- Registry ---------------------------------------------------------------
# A module-level registry lets refit and serving callsites see the shard layout
# built during training setup without threading it through every signature.
# Frameworks that want to avoid globals should call the core functions with
# explicit shard lists instead.

_INFERENCE_SHARDS: Optional[List[InferenceShard]] = None


def set_inference_shards(shards: Optional[List[InferenceShard]]) -> None:
    """Register the inference-shard layout constructed during setup.

    Passing ``None`` deregisters and also flushes the cross-shard group cache
    — process groups from a prior distributed world are not valid once torch
    distributed has been reinitialized.
    """
    global _INFERENCE_SHARDS
    _INFERENCE_SHARDS = shards
    if shards is None:
        clear_cross_shard_group_cache()


def get_inference_shards() -> Optional[List[InferenceShard]]:
    """Return the registered inference-shard layout, or None if not using shards."""
    return _INFERENCE_SHARDS


def get_my_inference_shard() -> Optional[InferenceShard]:
    """Return the shard the current rank belongs to, or None if idle / unset."""
    shards = get_inference_shards()
    if shards is None:
        return None
    for s in shards:
        if s.owns_rank():
            return s
    return None


# ---- Registry-driven wrappers ------------------------------------------------


def build_cross_shard_group(shard_indices: List[int]) -> Optional[dist.ProcessGroup]:
    """Build a cross-shard group using the registered shard layout.

    Thin wrapper around
    :func:`megatron.core.inference.shards.build_cross_shard_group` that reads
    the registry. Every rank must call this simultaneously.
    """
    shards = get_inference_shards()
    if shards is None:
        raise RuntimeError(
            "No inference shards registered. Call set_inference_shards first."
        )
    return _core_build_cross_shard_group(shards, shard_indices)


def swap_weights_across_shards(src_model, inference_model, refit_method: str) -> None:
    """Refit weights into every registered inference shard.

    Thin wrapper around
    :func:`megatron.core.resharding.refit.swap_model_weights_across_shards`
    that reads the registry. Falls back to a single
    :func:`swap_model_weights` call when no shards are registered.
    """
    from megatron.core.resharding.refit import swap_model_weights

    shards = get_inference_shards()
    if shards is None:
        swap_model_weights(src_model, inference_model, refit_method)
        return
    _core_swap_model_weights_across_shards(
        src_model, inference_model, shards, refit_method
    )
