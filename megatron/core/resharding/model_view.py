# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Component views for refitting models with different module trees and meshes."""

from collections.abc import Mapping
from dataclasses import dataclass
from weakref import ref

import torch
import torch.distributed as dist

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import unwrap_model

from .utils import named_refit_tensors


@dataclass(frozen=True)
class _RefitComponent:
    """A local component and the process groups owning its tensors.

    Args:
        module: Local module, unwrapped from DDP/precision wrappers by the view.
        pg_collection: This component's groups, not the parent model's groups.
        num_experts: Global expert count for an MoE component. Defaults to the
            module config's num_moe_experts when available, otherwise None.
    """

    module: torch.nn.Module
    pg_collection: ProcessGroupCollection
    num_experts: int | None = None


class _RefitModelView(torch.nn.Module):
    """Expose existing component tensors under common names without copying them.

    Internal adapter: matching component names refer to equivalent modules,
    although their original paths and parallelism may differ. Components absent
    on a rank are omitted; an empty view is valid. Callers keep passing their
    original models to the public API. The view is not a forward-pass wrapper.

    Views currently support ordinary floating-point parameters, not automatic
    FP8/FP4 format conversion. Persistent buffers are included. A shared tensor
    may occur inside one component (tied embeddings), but cannot belong to two
    components with potentially different ownership.

    Construct once and reuse between optimizer steps. Do not change component
    membership or process groups in place. After changing either side's layout,
    all refit ranks must clear_plan_cache() and prepare again, including ranks
    whose local layout did not change.

    Args:
        components: Logical component names mapped to local module/group pairs.
    """

    def __init__(self, components: Mapping[str, _RefitComponent]) -> None:
        super().__init__()
        specifications = []
        tensor_owners = {}
        for name, component in sorted(components.items()):
            if not name or '.' in name:
                raise ValueError(f"Component name must be nonempty and contain no dots: {name!r}")
            module = unwrap_model(component.module)
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"Component {name!r} must contain a torch module")
            pg = component.pg_collection
            if pg is None or any(getattr(pg, axis, None) is None for axis in ('tp', 'pp', 'dp')):
                raise ValueError(f"Component {name!r} requires explicit tp, pp and dp groups")
            config = getattr(module, 'config', None)
            configured_experts = getattr(config, 'num_moe_experts', None)
            num_experts = component.num_experts
            if num_experts is None:
                num_experts = configured_experts
            elif configured_experts is not None and num_experts != configured_experts:
                raise ValueError(f"Component {name!r}: num_experts disagrees with its config")
            if num_experts is not None:
                if num_experts <= 0:
                    raise ValueError(f"Component {name!r}: num_experts must be positive")
                if any(getattr(pg, axis, None) is None for axis in ('ep', 'expt_tp')):
                    raise ValueError(
                        f"Component {name!r}: MoE requires explicit ep and expt_tp groups"
                    )
            if getattr(config, 'fp8', None) or getattr(config, 'fp4', None):
                raise ValueError(f"Component {name!r}: quantized refit views are not supported")
            for _, tensor in named_refit_tensors(module):
                if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                    raise ValueError(
                        f"Component {name!r}: tensor subclasses require a refit transform"
                    )
                if tensor.is_floating_point() and tensor.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ):
                    raise ValueError(f"Component {name!r}: unsupported tensor dtype {tensor.dtype}")
                owner = tensor_owners.setdefault(id(tensor), name)
                if owner != name:
                    raise ValueError(f"Components {owner!r} and {name!r} share a tensor")
            # Match native destination paths without copying the original tensors.
            self.add_module(name, module)
            specifications.append((name, _RefitComponent(module, pg, num_experts)))
        self._specifications = tuple(specifications)

    def local_components(self) -> tuple[tuple[str, _RefitComponent], ...]:
        """Return the immutable component specifications used for metadata."""
        return self._specifications

    def layout_key(self) -> tuple:
        """Describe component identities and actual group membership for caching."""
        return tuple(
            (
                name,
                # Preserve identity without keeping model tensors alive in the plan cache.
                # Unlike id(module), a dead weak reference cannot alias a new model.
                ref(component.module),
                component.num_experts,
                tuple(
                    (axis, tuple(dist.get_process_group_ranks(pg)))
                    for axis in ('tp', 'pp', 'dp', 'ep', 'expt_tp', 'gtp_remat', 'expt_gtp_remat')
                    if (pg := getattr(component.pg_collection, axis, None)) is not None
                ),
            )
            for name, component in self._specifications
        )
