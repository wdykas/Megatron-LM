# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Model-specific component discovery for the generic refit view."""

from collections.abc import Mapping
from functools import singledispatch
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist

from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.utils import unwrap_model

from .model_view import _RefitComponent, _RefitModelView
from .utils import named_refit_tensors


@singledispatch
def _refit_components(model: torch.nn.Module) -> Mapping[str, _RefitComponent] | None:
    """Return canonical components, or None to retain ordinary model handling.

    Register model-specific discovery functions here. Return child modules, not
    the original model itself. Component names must match
    the destination model's tensor namespace; discovery does not copy tensors or
    change the public prepare/swap interface.
    """
    return None


@_refit_components.register(MimoModel)
def _mimo_refit_components(model: MimoModel) -> Mapping[str, _RefitComponent]:
    """Map MIMO's CLIP/projector/language modules to native LLaVA names."""
    if set(model.mimo_config.modality_submodules_spec) != {'images'}:
        raise ValueError(
            "MIMO refit supports one 'images' modality with CLIP and an input projector"
        )
    components = {}
    if model.language_model is not None:
        language = unwrap_model(model.language_model)
        components['language_model'] = _RefitComponent(language, language.pg_collection)
    for tower in model.modality_submodules.values():
        tower = unwrap_model(tower)
        if (
            not isinstance(tower, VisionModalitySubmodules)
            or len(tower.encoders) != 1
            or len(tower.input_projections) != 1
            or tower.decoders
            or tower.output_projections
        ):
            raise ValueError(
                "MIMO refit requires one vision encoder and one input projector, without decoders"
            )
        vision = unwrap_model(next(iter(tower.encoders.values())))
        projector = unwrap_model(tower.input_projections[0])
        if not isinstance(vision, CLIPViTModel) or not isinstance(projector, MultimodalProjector):
            raise ValueError("MIMO refit requires CLIPViTModel and MultimodalProjector components")
        pg = tower.pg_collection
        if (
            pg is None
            or pg.tp is None
            or tuple(dist.get_process_group_ranks(projector.tp_group))
            != tuple(dist.get_process_group_ranks(pg.tp))
        ):
            raise ValueError(
                "MIMO refit projector TP group must match its modality's process groups"
            )
        components['vision_model'] = _RefitComponent(vision, vision.pg_collection)
        components['vision_projection'] = _RefitComponent(projector, pg)
    return components


_model_view_cache: WeakKeyDictionary[torch.nn.Module, _RefitModelView] = WeakKeyDictionary()


def _clear_model_view_cache() -> None:
    _model_view_cache.clear()


def _as_refit_model(model: torch.nn.Module) -> torch.nn.Module:
    """Discover and cache a view while preserving ordinary models unchanged."""
    cached = _model_view_cache.get(model)
    if cached is not None:
        return cached
    components = _refit_components(model)
    if components is None:
        return model
    if any(unwrap_model(component.module) is model for component in components.values()):
        raise ValueError("Refit adapters must return child modules, not the original model")
    view = _RefitModelView(components)
    # Every adapter must cover the original model's entire persistent state.
    if {id(t) for _, t in named_refit_tensors(model)} != {
        id(t) for _, t in named_refit_tensors(view)
    }:
        raise ValueError(
            f"{type(model).__name__} refit adapter does not cover all model parameters "
            "and persistent buffers"
        )
    _model_view_cache[model] = view
    return view
