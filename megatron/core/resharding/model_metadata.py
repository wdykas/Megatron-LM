# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Ownership discovery and name matching for models without one root mesh."""

import re

import torch
import torch.distributed as dist

from megatron.core.utils import unwrap_model


def _refit_name_aliases(model):
    """Return only the namespace translation needed by the model pair."""
    from megatron.core.models.mimo import MimoModel

    if isinstance(model, MimoModel):
        return (
            (r'^language_model\.', 'language_model.'),
            (r'^modality_submodules\.images\.encoders\.[^.]+\.', 'vision_model.'),
            (r'^modality_submodules\.images\.input_projections\.0\.', 'vision_projection.'),
        )
    return ()


def _match_refit_name(name, aliases):
    if not aliases:
        return name
    for pattern, replacement in aliases:
        result, count = re.subn(pattern, replacement, name, count=1)
        if count:
            return result
    raise ValueError(f"No refit name mapping for {name!r}")


def _named_owned_tensors(model):
    """Yield storage name, logical name, tensor, groups and expert count.

    Ownership follows the nearest ancestor with a process-group collection.
    Containers inherit it; a child's collection replaces it as a whole. Only
    logical names omit DDP/precision wrappers and use global pipeline indices.
    Parameters are deduplicated like named_parameters(); persistent buffers
    retain their paths. Shared tensors must have the same ownership everywhere.
    """
    seen = {}
    seen_modules = set()
    group_keys = {}

    def group_key(pg):
        if id(pg) not in group_keys:
            group_keys[id(pg)] = tuple(
                (axis, tuple(dist.get_process_group_ranks(group)))
                for axis in ('tp', 'pp', 'dp', 'ep', 'expt_tp', 'gtp_remat', 'expt_gtp_remat')
                if (group := getattr(pg, axis, None)) is not None
            )
        return group_keys[id(pg)]

    def walk(module, storage_prefix='', logical_prefix='', pg=None, num_experts=None):
        repeated_module = id(module) in seen_modules
        seen_modules.add(id(module))
        pg = getattr(module, 'pg_collection', None) or pg
        config = getattr(module, 'config', None)
        if config is not None:
            num_experts = getattr(config, 'num_moe_experts', None)
            if getattr(config, 'fp8', None) or getattr(config, 'fp4', None):
                raise ValueError('Quantized composite refit is not supported')
        # Modules such as projectors retain only their TP group. Check that
        # inheriting the remaining groups does not silently use a different mesh.
        tp = getattr(module, 'tp_group', None)
        if tp is not None and pg is not None:
            axes = ('tp', 'expt_tp') if num_experts is not None else ('tp',)
            if not any(
                dist.get_process_group_ranks(tp) == dist.get_process_group_ranks(group)
                for axis in axes
                if (group := getattr(pg, axis, None)) is not None
            ):
                raise ValueError(f'{storage_prefix}TP group disagrees with inherited ownership')
        tensors = [(name, tensor, True) for name, tensor in module.named_parameters(recurse=False)]
        tensors.extend(
            (name, tensor, False)
            for name, tensor in module._buffers.items()
            if tensor is not None and name not in module._non_persistent_buffers_set
        )
        for name, tensor, is_parameter in tensors:
            storage_name = storage_prefix + name
            if pg is None or any(getattr(pg, axis, None) is None for axis in ('tp', 'pp', 'dp')):
                raise ValueError(f'{storage_name}: requires explicit tp, pp and dp ownership')
            if num_experts is not None and any(
                getattr(pg, axis, None) is None for axis in ('ep', 'expt_tp')
            ):
                raise ValueError(f'{storage_name}: MoE requires explicit ep and expt_tp groups')
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or (
                tensor.is_floating_point()
                and tensor.dtype
                not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            ):
                raise ValueError(f'{storage_name}: tensor format requires a refit transform')
            ownership = (group_key(pg), num_experts)
            previous = seen.get(id(tensor))
            if previous is not None:
                if previous != ownership:
                    raise ValueError(f'{storage_name}: shared tensor has conflicting ownership')
                if is_parameter or repeated_module:
                    continue
            seen[id(tensor)] = ownership
            yield storage_name, logical_prefix + name, tensor, pg, num_experts
        wrapped = unwrap_model(module) is not module
        for name, child in module.named_children():
            logical_name = name
            if wrapped and name == 'module':
                logical_name = ''
            elif name.isdigit() and isinstance(getattr(child, 'layer_number', None), int):
                logical_name = str(child.layer_number - 1)
            yield from walk(
                child,
                storage_prefix + name + '.',
                logical_prefix + (logical_name + '.' if logical_name else ''),
                pg,
                num_experts,
            )

    yield from walk(model)
