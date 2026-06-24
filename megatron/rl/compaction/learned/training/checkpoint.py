# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint save/load for trained compactors.

Stores both the model config (so the architecture can be reconstructed without
out-of-band files) and the optimizer/scheduler state for full training resume.

Usage — save:
    from megatron.rl.compaction.learned.training.checkpoint import save_checkpoint
    save_checkpoint(compactor, "run/ckpt_step1000.pt", step=1000,
                    optimizer=opt, scheduler=curriculum,
                    metadata={"val_ppl": 14.2})

Usage — load:
    from megatron.rl.compaction.learned.training.checkpoint import load_checkpoint
    model, meta = load_checkpoint("run/ckpt_step1000.pt")
    # model is a PerceiverCompactor or GatedRecurrentUpdater, weights loaded
    # meta.step, meta.metadata carry the saved state
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor, PerceiverConfig
from megatron.rl.compaction.learned.models.belief import GatedRecurrentUpdater, GatedUpdaterConfig


# ---------------------------------------------------------------------------
# Metadata type
# ---------------------------------------------------------------------------

@dataclass
class CheckpointMeta:
    """Metadata stored alongside the model in a checkpoint file.

    Attributes
    ----------
    model_type: "perceiver" or "gated_recurrent".
    config:     Serialised model config dict (reconstructable).
    step:       Gradient step at save time.
    metadata:   Arbitrary dict for experiment tracking (loss values, etc.).
    """

    model_type: str
    config: dict
    step: int
    metadata: dict


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a compactor checkpoint with config, weights, and optional opt state.

    Parameters
    ----------
    model:     A PerceiverCompactor or GatedRecurrentUpdater (or any nn.Module
               with a .cfg dataclass attribute of one of the two config types).
    path:      File path.  Parent directories are created if needed.
    step:      Current gradient step (for resume).
    optimizer: If provided, its state_dict is saved too.
    scheduler: If provided and has a state_dict() method, saves scheduler state.
               Pass a CurriculumScheduler to enable curriculum resume.
    metadata:  Arbitrary dict (val loss, PPL, timestamp, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PerceiverCompactor):
        model_type = "perceiver"
        config_dict = dataclasses.asdict(model.cfg)
    elif isinstance(model, GatedRecurrentUpdater):
        model_type = "gated_recurrent"
        config_dict = dataclasses.asdict(model.cfg)
    else:
        raise TypeError(
            f"save_checkpoint only supports PerceiverCompactor and "
            f"GatedRecurrentUpdater, got {type(model).__name__}. "
            f"For other nn.Module types use torch.save directly."
        )

    payload: dict[str, Any] = {
        "model_type": model_type,
        "config": config_dict,
        "step": step,
        "metadata": metadata or {},
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, CheckpointMeta]:
    """Load a compactor from a checkpoint file.

    Reconstructs the model from the saved config (no need to know the
    architecture in advance), loads weights, and returns metadata.

    Parameters
    ----------
    path:         Checkpoint file path.
    map_location: Passed to torch.load.  Use "cuda" to load directly onto GPU.

    Returns
    -------
    (model, meta) where model has weights loaded (no mode set — caller must call
    .eval() or .train() as appropriate).
    meta.step and meta.metadata carry the saved training state.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model_type: str = payload["model_type"]
    config_dict: dict = payload["config"]
    step: int = payload.get("step", 0)
    meta_dict: dict = payload.get("metadata", {})

    if model_type == "perceiver":
        cfg = PerceiverConfig(**config_dict)
        model: nn.Module = PerceiverCompactor(cfg)
    elif model_type == "gated_recurrent":
        cfg = GatedUpdaterConfig(**config_dict)
        model = GatedRecurrentUpdater(cfg)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}' in checkpoint. "
            f"Expected 'perceiver' or 'gated_recurrent'."
        )

    model.load_state_dict(payload["model_state_dict"])

    meta = CheckpointMeta(
        model_type=model_type,
        config=config_dict,
        step=step,
        metadata=meta_dict,
    )
    return model, meta


def load_optimizer_state(
    path: str | Path,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device = "cpu",
) -> None:
    """Restore optimizer state from a checkpoint (in-place)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "optimizer_state_dict" not in payload:
        raise KeyError(f"Checkpoint at {path} has no optimizer state.")
    optimizer.load_state_dict(payload["optimizer_state_dict"])


def load_scheduler_state(
    path: str | Path,
    scheduler: Any,
    map_location: str | torch.device = "cpu",
) -> None:
    """Restore scheduler (e.g. CurriculumScheduler) state from a checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "scheduler_state_dict" not in payload:
        raise KeyError(f"Checkpoint at {path} has no scheduler state.")
    scheduler.load_state_dict(payload["scheduler_state_dict"])
