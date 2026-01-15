# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Speed of Light (SOL) Estimation Module.

Core components:
    - device_specs: GPU specifications (H100, A100, B200) and data types
    - operation_registry: Operation types, tensor shapes, FLOP/memory calculation
    - roofline: Roofline model analysis (compute vs memory bound)
    - layer_hooks: Automatic shape capture via PyTorch module hooks
    - cuda_graph_tracker: Track CUDA graph captures/replays
    - optimizer_tracker: Track optimizer step time
    - phase_timer: Wall-clock time per phase
"""

from .device_specs import (
    DeviceSpec,
    DataType,
    get_current_device_spec,
    torch_dtype_to_datatype,
    DEVICE_SPECS,
    H100_SXM5_SPEC,
    A100_SXM4_SPEC,
    B200_SPEC,
)
from .operation_registry import (
    OpType,
    OpRecord,
    TensorShape,
)
from .roofline import (
    RooflineAnalyzer,
    RooflineResult,
    BoundType,
    estimate_matmul_sol,
)
from .layer_hooks import (
    LayerSOLHooks,
    CapturedOp,
    sol_profile,
)
from .cuda_graph_tracker import (
    CUDAGraphTracker,
    CapturedGraph,
    GraphReplayEvent,
    GraphState,
    get_graph_tracker,
)
from .optimizer_tracker import (
    OptimizerTracker,
    OptimizerStepEvent,
)
from .phase_timer import (
    PhaseTimer,
    PhaseTimeRecord,
)

__all__ = [
    # Device specs
    "DeviceSpec",
    "DataType",
    "get_current_device_spec",
    "torch_dtype_to_datatype",
    "DEVICE_SPECS",
    "H100_SXM5_SPEC",
    "A100_SXM4_SPEC",
    "B200_SPEC",
    # Operation registry
    "OpType",
    "OpRecord",
    "TensorShape",
    # Roofline
    "RooflineAnalyzer",
    "RooflineResult",
    "BoundType",
    "estimate_matmul_sol",
    # Layer hooks
    "LayerSOLHooks",
    "CapturedOp",
    "sol_profile",
    # CUDA graph tracker
    "CUDAGraphTracker",
    "CapturedGraph",
    "GraphReplayEvent",
    "GraphState",
    "get_graph_tracker",
    # Optimizer tracker
    "OptimizerTracker",
    "OptimizerStepEvent",
    # Phase timer
    "PhaseTimer",
    "PhaseTimeRecord",
]

__version__ = "0.8.0"
