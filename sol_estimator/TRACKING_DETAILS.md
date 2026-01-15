# SOL (Speed of Light) Tracking Implementation Details

This document describes how SOL estimation and tracking works in the Megatron-RL framework.

## Architecture Overview

The SOL tracking system uses a **hybrid approach** combining two capture mechanisms:

```
┌─────────────────────────────────────────────────────────────────┐
│                        SOLTracker                                │
│                   (sol_integration.py)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │   LayerSOLHooks     │      │    FunctionalPatcher        │   │
│  │   (layer_hooks.py)  │      │  (functional_patcher.py)    │   │
│  ├─────────────────────┤      ├─────────────────────────────┤   │
│  │ Captures nn.Module  │      │ Captures torch.* and        │   │
│  │ forward passes via  │      │ F.* function calls via      │   │
│  │ PyTorch hooks       │      │ monkey-patching             │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. SOLTracker (`megatron/rl/sol_integration.py`)

The main orchestrator that:
- Manages both `LayerSOLHooks` and `FunctionalPatcher` instances
- Tracks the current phase (NVTX range) stack
- Combines results from both capture mechanisms
- Logs to TensorBoard and WandB
- Prints detailed reports to console

**Key Methods:**
- `initialize(model, args)` - Attaches hooks to model and patches functions
- `push_phase(name)` / `pop_phase()` - Manages phase tracking
- `log_analysis(iteration, clear)` - Analyzes and logs all captured ops
- `clear()` - Clears captured operations for next iteration

### 2. LayerSOLHooks (`sol_estimator/layer_hooks.py`)

Captures operations at the `nn.Module` level using PyTorch's hook system.

**Supported Layer Types:**
| Layer Type | Op Type | Notes |
|------------|---------|-------|
| `Linear` | MATMUL | Weight shape inferred from module |
| `TELinear` | MATMUL | Transformer Engine linear |
| `TELayerNormLinear` | MATMUL + LAYERNORM | Fused TE layer |
| `TELayerNormMLP` | MATMUL + LAYERNORM | Fused TE MLP |
| `TEDotProductAttention` | ATTENTION | Flash attention |
| `Embedding` | EMBEDDING | Token embeddings |
| `LayerNorm` / `RMSNorm` | LAYERNORM | Normalization layers |

**Data Captured per Operation (`CapturedOp`):**
```python
@dataclass
class CapturedOp:
    layer_name: str           # Full module path (e.g., "model.layers.0.attn")
    layer_type: str           # Class name (e.g., "TEDotProductAttention")
    op_type: OpType           # MATMUL, ATTENTION, LAYERNORM, etc.
    input_shapes: List[Tuple] # Input tensor shapes
    output_shapes: List[Tuple]# Output tensor shapes
    dtype: DataType           # FP16, BF16, FP32, etc.
    measured_time_us: float   # Actual GPU time (via CUDA events)
    phase: str                # Current NVTX phase when captured
```

### 3. FunctionalPatcher (`sol_estimator/functional_patcher.py`)

Captures operations at the functional API level by monkey-patching PyTorch functions.

**Patched Functions:**

| Module | Functions |
|--------|-----------|
| `torch` | `matmul`, `bmm`, `mm`, `addmm`, `baddbmm`, `einsum` |
| `torch` | `softmax`, `layer_norm`, `embedding` |
| `torch.nn.functional` | `linear`, `embedding`, `layer_norm`, `softmax`, `log_softmax` |

**NOT Patched (to avoid conflicts with Megatron's fused kernels):**
- Activation functions: `gelu`, `silu`, `relu`, `sigmoid`, `tanh`
- `dropout`

**Data Captured per Operation (`CapturedFunctionalOp`):**
```python
@dataclass
class CapturedFunctionalOp:
    func_name: str            # Function name (e.g., "F.linear")
    op_type: OpType           # Inferred operation type
    input_shapes: List[Tuple] # Input tensor shapes
    output_shapes: List[Tuple]# Output tensor shapes  
    dtype: DataType           # Data type
    measured_time_us: float   # Actual GPU time
    phase: str                # Current NVTX phase
    module_context: str       # Parent nn.Module (if tracked)
```

## Phase Tracking

Phases correspond to NVTX ranges and allow grouping operations by training stage.

**Phase Hierarchy Example:**
```
rl_timing/prepare-data
├── rl_timing/prepare-data/group-stats
├── rl_timing/prepare-data/advantages
├── rl_timing/prepare-data/trajectories
└── rl_timing/compute-logprobs
    ├── rl_timing/compute-logprobs/old
    ├── rl_timing/compute-logprobs/ref
    └── rl_timing/compute-logprobs/batch/forward
rl_timing/grpo-loss
    ├── rl_timing/grpo-loss/ratios
    ├── rl_timing/grpo-loss/advantages
    └── rl_timing/grpo-loss/final
```

**Usage in Code:**
```python
from megatron.rl.sol_integration import sol_nvtx_range

with sol_nvtx_range("rl_timing/compute-logprobs/forward", log_level=2):
    logits = model(tokens, position_ids)
```

The `sol_nvtx_range` context manager:
1. Starts an NVTX range (for profiling)
2. Pushes the phase onto the SOL tracker's phase stack
3. All ops captured inside are tagged with this phase
4. Pops the phase and ends the NVTX range on exit

## Roofline Analysis

Each captured operation is analyzed against the GPU's theoretical limits.

**Device Specifications (`device_specs.py`):**
```python
# H100 SXM example
DeviceSpec(
    name="H100_SXM",
    memory_bandwidth_bytes_per_sec=3.35e12,  # 3.35 TB/s
    peak_flops={
        DataType.FP32: 67e12,    # 67 TFLOPS
        DataType.FP16: 1979e12,  # 1979 TFLOPS (with sparsity)
        DataType.BF16: 1979e12,
        DataType.FP8_E4M3: 3958e12,
    }
)
```

**Analysis Output (`RooflineResult`):**
```python
@dataclass
class RooflineResult:
    estimated_time_us: float  # Theoretical minimum time
    bound_type: BoundType     # COMPUTE_BOUND or MEMORY_BOUND
    arithmetic_intensity: float  # FLOPs per byte
    achieved_flops: float     # If measured, actual FLOPS achieved
    efficiency: float         # Ratio of theoretical to measured
```

## Data Flow

```
1. Model Forward Pass
   │
   ├─► LayerSOLHooks captures nn.Module ops
   │   └─► Creates CapturedOp with shapes, dtype, timing
   │
   └─► FunctionalPatcher captures torch.*/F.* calls
       └─► Creates CapturedFunctionalOp with shapes, dtype, timing

2. End of Iteration (log_training_sol called)
   │
   ├─► SOLTracker.log_analysis()
   │   │
   │   ├─► Convert CapturedOp/CapturedFunctionalOp to OpRecord
   │   │
   │   ├─► Run roofline analysis on each OpRecord
   │   │
   │   ├─► Aggregate by layer type, function, and phase
   │   │
   │   ├─► Log to TensorBoard (if enabled)
   │   │
   │   ├─► Log to WandB (if enabled)
   │   │   ├─► Basic metrics: sol/total_ops, sol/estimated_time_ms, etc.
   │   │   ├─► Per-layer: sol_layer/{type}/estimated_ms
   │   │   ├─► Per-function: sol_func/{name}/estimated_ms
   │   │   └─► Per-phase: sol_phase/{name}/estimated_ms
   │   │
   │   └─► Print report to console (at intervals)
   │
   └─► SOLTracker.clear() - Reset for next iteration
```

## WandB Metrics

**Summary Metrics (logged every iteration):**
| Metric | Description |
|--------|-------------|
| `sol/total_ops` | Total captured operations |
| `sol/total_tflops` | Total theoretical FLOPs |
| `sol/total_memory_gb` | Total memory traffic |
| `sol/estimated_time_ms` | Theoretical minimum time |
| `sol/measured_time_ms` | Actual GPU time |
| `sol/efficiency_pct` | Ratio (estimated/measured × 100) |

**Per-Layer Metrics:**
| Metric | Description |
|--------|-------------|
| `sol_layer/{type}/count` | Number of ops of this layer type |
| `sol_layer/{type}/estimated_ms` | Estimated time for this layer type |
| `sol_layer/{type}/measured_ms` | Measured time |
| `sol_layer/{type}/efficiency_pct` | Efficiency percentage |

**Per-Phase Metrics:**
| Metric | Description |
|--------|-------------|
| `sol_phase/{name}/count` | Operations in this phase |
| `sol_phase/{name}/estimated_ms` | Estimated time for phase |
| `sol_phase/{name}/measured_ms` | Measured time |
| `sol_phase/{name}/efficiency_pct` | Efficiency percentage |

**Tables (logged at report intervals):**
- `sol/by_layer_type` - Breakdown by layer type
- `sol/by_function` - Breakdown by function name
- `sol/by_phase` - Breakdown by phase
- `sol/top_operations` - Top 50 operations by estimated time

## Console Report Format

```
==========================================================================================
                           SOL (Speed of Light) Analysis
==========================================================================================
Device: H100_SXM | Ops: 450 | Total: 12.34 ms estimated
------------------------------------------------------------------------------------------
By Layer Type:
  TEDotProductAttention                         160 ops,     8.50 ms ( 68.9%) [meas: 10.2ms, eff: 83.3%]
  TELayerNormLinear                             120 ops,     2.80 ms ( 22.7%) [meas: 3.1ms, eff: 90.3%]
  TELayerNormMLP                                 80 ops,     0.85 ms (  6.9%) [meas: 0.9ms, eff: 94.4%]
  RMSNorm                                        90 ops,     0.19 ms (  1.5%) [meas: 0.2ms, eff: 95.0%]
------------------------------------------------------------------------------------------
By Function:
  F.embedding                                    20 ops,     0.05 ms (  0.4%) [meas: 0.06ms, eff: 83.3%]
  F.log_softmax                                  10 ops,     0.02 ms (  0.2%) [meas: 0.03ms, eff: 66.7%]
------------------------------------------------------------------------------------------
By Phase:
  rl_timing/compute-logprobs                    300 ops,     9.50 ms ( 77.0%) [meas: 11.2ms, eff: 84.8%]
  rl_timing/grpo-loss                           100 ops,     2.10 ms ( 17.0%) [meas: 2.5ms, eff: 84.0%]
  (no phase)                                     50 ops,     0.74 ms (  6.0%) [meas: 0.8ms, eff: 92.5%]
==========================================================================================
```

## Configuration

**Command-line Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--rl-sol-report-interval` | 100 | Iterations between detailed reports |
| `--rl-sol-device` | auto | Device spec to use (auto-detected) |

## Initialization Flow

```python
# In rl_utils.py prepare_data_for_update():
initialize_sol(model, args)  # Called before first forward pass
clear_sol_captures()         # Clear any stale data

# ... training code with sol_nvtx_range() annotations ...

# In training.py after train_step():
log_training_sol(iteration, clear=True)  # Log and clear for next iteration
```

## Limitations

1. **Fused Kernels**: Transformer Engine uses fused CUDA kernels that may not map 1:1 to PyTorch operations. The layer hooks capture these at the module level.

2. **torch.compile**: The functional patcher is compatible with `torch.compile` by skipping capture during tracing (uses `torch._dynamo.is_compiling()`).

3. **Timing Overhead**: CUDA event timing adds small overhead. Disable in production by setting `--timing-log-level 0`.

4. **Memory Estimates**: Memory bandwidth estimates assume optimal access patterns; actual memory traffic may be higher due to cache effects.

## Files

| File | Description |
|------|-------------|
| `megatron/rl/sol_integration.py` | Main integration module with SOLTracker |
| `sol_estimator/layer_hooks.py` | LayerSOLHooks for nn.Module capture |
| `sol_estimator/functional_patcher.py` | FunctionalPatcher for torch.*/F.* capture |
| `sol_estimator/operation_registry.py` | OpRecord, OpType, FLOP/memory calculations |
| `sol_estimator/roofline.py` | RooflineAnalyzer for theoretical analysis |
| `sol_estimator/device_specs.py` | GPU specifications (H100, A100, etc.) |
