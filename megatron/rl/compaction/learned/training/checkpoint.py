# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed checkpointing for trained compactors.

The compactor is a Megatron ``MegatronModule`` built from TransformerEngine
layers, so it is checkpointed the same way the main model is: via
``megatron.core.dist_checkpointing``. The model weights are stored as a
*sharded* state dict (which correctly handles TE ``_extra_state``), while the
model config, optimizer state, and step are stored as *common* (replicated)
state. A checkpoint is therefore a **directory**, not a ``.pt``.

This requires ``torch.distributed`` + Megatron model-parallel state to be
initialized (always true inside the RL training loop). The async file writer
also needs a working multiprocessing context — present under ``torchrun``.

Usage — save:
    save_checkpoint(compactor, "run/step_0001000", step=1000, optimizer=opt)

Usage — load:
    model, meta = load_checkpoint("run/step_0001000", map_location="cuda")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.core import dist_checkpointing
from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor, PerceiverConfig
from megatron.rl.compaction.learned.models.belief import GatedRecurrentUpdater, GatedUpdaterConfig


@dataclass
class CheckpointMeta:
    """Metadata stored (as common state) alongside the model in a checkpoint."""

    model_type: str
    config: dict
    step: int
    metadata: dict


def _model_type_and_config(model: nn.Module) -> tuple[str, dict]:
    if isinstance(model, PerceiverCompactor):
        return "perceiver", dataclasses.asdict(model.cfg)
    if isinstance(model, GatedRecurrentUpdater):
        return "gated_recurrent", dataclasses.asdict(model.cfg)
    raise TypeError(
        f"save_checkpoint only supports PerceiverCompactor and GatedRecurrentUpdater, "
        f"got {type(model).__name__}."
    )


def _build_model(model_type: str, config_dict: dict, params_dtype=torch.float32, pg_collection=None) -> nn.Module:
    if model_type == "perceiver":
        return PerceiverCompactor(PerceiverConfig(**config_dict), params_dtype=params_dtype,
                                  pg_collection=pg_collection)
    if model_type == "gated_recurrent":
        return GatedRecurrentUpdater(GatedUpdaterConfig(**config_dict), params_dtype=params_dtype,
                                     pg_collection=pg_collection)
    raise ValueError(f"Unknown model_type '{model_type}' in checkpoint.")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    step: int = 0,
    optimizer: Any | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a compactor checkpoint (a directory) via dist_checkpointing.

    The model is saved as a sharded state dict; config/optimizer/step are saved as
    common (replicated) state.
    """
    path = Path(path)
    model_type, config_dict = _model_type_and_config(model)

    sharded_sd: dict[str, Any] = {
        "model": model.sharded_state_dict(),
        # Common (replicated) state — plain objects, saved via the common store.
        "model_type": model_type,
        "config": config_dict,
        "step": step,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        if hasattr(optimizer, "sharded_state_dict"):
            # Online (Option C): the Megatron optimizer's state (FP32 masters + Adam
            # moments) is stored as replicated ShardedTensors keyed to the model's
            # sharded params — the idiomatic, consistency-checked representation.
            sharded_sd["optimizer"] = optimizer.sharded_state_dict(sharded_sd["model"])
        else:
            # Offline (plain torch optimizer, single-process): no sharded path; the
            # replicated tensors go in the common store.
            sharded_sd["optimizer"] = optimizer.state_dict()

    path.mkdir(parents=True, exist_ok=True)
    dist_checkpointing.save(sharded_sd, str(path))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cuda",
    params_dtype: torch.dtype = torch.float32,
    pg_collection=None,
) -> tuple[nn.Module, CheckpointMeta]:
    """Load a compactor from a dist-checkpoint directory.

    Reads common state first to recover the model config, rebuilds the model
    (replicated via ``pg_collection``), then loads the sharded weights into it.
    """
    path = str(path)
    common = dist_checkpointing.load_common_state_dict(path)
    model_type = common["model_type"]
    config_dict = common["config"]

    model = _build_model(model_type, config_dict, params_dtype, pg_collection).to(map_location)
    loaded = dist_checkpointing.load({"model": model.sharded_state_dict()}, path)
    model.load_state_dict(loaded["model"])

    meta = CheckpointMeta(
        model_type=model_type,
        config=config_dict,
        step=common.get("step", 0),
        metadata=common.get("metadata", {}),
    )
    return model, meta


def load_optimizer_state(
    path: str | Path,
    optimizer: Any,
    model_sharded_state_dict: dict | None = None,
    map_location: str | torch.device = "cuda",
) -> None:
    """Restore optimizer state from a dist-checkpoint directory.

    Online (Option C): the optimizer exposes ``sharded_state_dict`` and the state
    was saved as replicated ShardedTensors; build the loading template and read it
    via the sharded path. ``model_sharded_state_dict`` is the same model sharded
    dict used at save time (the optimizer keys its state to those params).

    Offline (plain torch optimizer): the state is in the common store.
    """
    if hasattr(optimizer, "sharded_state_dict") and model_sharded_state_dict is not None:
        opt_template = optimizer.sharded_state_dict(model_sharded_state_dict, is_loading=True)
        loaded = dist_checkpointing.load({"optimizer": opt_template}, str(path))
        optimizer.load_state_dict(loaded["optimizer"])
        return
    common = dist_checkpointing.load_common_state_dict(str(path))
    if "optimizer" not in common:
        raise KeyError(f"Checkpoint at {path} has no optimizer state.")
    optimizer.load_state_dict(common["optimizer"])
