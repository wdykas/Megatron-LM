# SOL (Speed of Light) Estimator

Estimates theoretical performance bounds for ML training operations using a hybrid approach: PyTorch module hooks for high-level layers and functional patching for low-level operations.

## Architecture Overview

The SOL estimator uses two complementary capture mechanisms:

1. **LayerSOLHooks**: Attaches forward hooks to `nn.Module` layers (Linear, Attention, Norm, etc.) to capture input/output shapes and measure execution time.

2. **FunctionalPatcher**: Patches `torch.*` and `torch.nn.functional.*` functions to capture all tensor operations, including those in loss calculations and non-module code.

This hybrid approach ensures comprehensive coverage across both Transformer layers and tensor math operations.

## Integration Flow (Megatron-RL)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Iteration Start                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. initialize_sol(model, args)                                             │
│     └── Attaches LayerSOLHooks + FunctionalPatcher to model                 │
│                                                                             │
│  2. prepare_data_for_update() [inference phase]                             │
│     ├── sol_nvtx_range("rl/compute-logprobs") ──► Captures ops by phase     │
│     │   ├── get_logprobs() forward passes                                   │
│     │   └── Hooks record: layer shapes, measured times                      │
│     └── sol_nvtx_range("rl/kl-estimation")                                  │
│                                                                             │
│  3. train_step() [training phase]                                           │
│     ├── sol_nvtx_range("rl/train/forward")                                  │
│     │   └── Forward pass with gradient tracking                             │
│     └── sol_nvtx_range("rl/train/grpo-loss")                                │
│         └── Loss calculation (captured by FunctionalPatcher)                │
│                                                                             │
│  4. log_training_sol(iteration)                                             │
│     ├── Combines captures from LayerSOLHooks + FunctionalPatcher            │
│     ├── Logs to TensorBoard/WandB (every iteration)                         │
│     ├── Prints report (at --rl-sol-report-interval)                         │
│     └── Clears captures for next iteration                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Standalone Usage

```python
from sol_estimator.layer_hooks import LayerSOLHooks, sol_profile
from sol_estimator.functional_patcher import FunctionalPatcher

# Option 1: Context manager for module hooks
with sol_profile(model) as hooks:
    output = model(input_tokens)
hooks.print_report()

# Option 2: Manual control with functional patcher
patcher = FunctionalPatcher()
patcher.patch()
patcher.register_model(model)  # For module context tracking

output = model(input_tokens)
loss = loss_fn(output, targets)  # Also captured

patcher.analyze()
print(patcher.get_summary())
patcher.unpatch()
```

### Megatron-RL Integration

Enable with command-line flags:

```bash
--rl-enable-sol-tracking              # Enable SOL tracking
--rl-sol-report-interval 100          # Print report every N iterations
```

## Module Structure

```
sol_estimator/
├── __init__.py           # Main exports
├── device_specs.py       # GPU specs (H100, A100, B200) with TFLOPS/bandwidth
├── operation_registry.py # OpType enum, TensorShape, OpRecord with FLOP calculation
├── roofline.py           # Roofline model: compute vs memory bound analysis
├── layer_hooks.py        # LayerSOLHooks: nn.Module hook-based capture
├── functional_patcher.py # FunctionalPatcher: torch.* function patching
├── README.md
└── examples/
    └── example_usage.py
```

## Captured Operations

### LayerSOLHooks (Module-based)
- `Linear`, `ColumnParallelLinear`, `RowParallelLinear`
- `TEColumnParallelLinear`, `TERowParallelLinear`
- `LayerNorm`, `RMSNorm`, `FusedRMSNorm`, `TENorm`
- `Softmax`
- `DotProductAttention`, `FlashAttention`, `TEDotProductAttention`
- `Embedding`, `VocabParallelEmbedding`

### FunctionalPatcher (Function-based)
- `torch.matmul`, `torch.bmm`, `torch.mm`
- `torch.nn.functional.linear`
- `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`
- `torch.nn.functional.layer_norm`
- `torch.sum`, `torch.mean`, `torch.max`, `torch.min`
- `torch.exp`, `torch.log`
- And more tensor math operations

### Not Currently Captured
- **Communication primitives** (all-reduce, all-gather, reduce-scatter): The infrastructure exists in `operation_registry.py` (`OpType.ALLREDUCE`, etc.) but patching `torch.distributed` is not yet implemented. These would need special handling for accurate inter-GPU communication timing.

## Output Example

```
==========================================================================================
SOL Analysis Report (Hybrid: Module Hooks + Functional Patches)
==========================================================================================
Device: H100-SXM5-80GB | Dtype: bf16

Summary (Combined):
  Total Ops: 4590 (layers: 4494, functions: 96)
  Total FLOPs: 380.5 TFLOPS
  Total Memory: 590.2 GB
  Estimated Time: 302.1 ms

By Phase:
  rl/compute-logprobs           2500 ops,   180.3 ms ( 59.7%)
  rl/kl-estimation               800 ops,    45.2 ms ( 15.0%)
  rl/train/forward              1100 ops,    70.1 ms ( 23.2%)
  rl/train/grpo-loss             190 ops,     6.5 ms (  2.1%)

By Layer Type (nn.Module hooks):
  TERowParallelLinear           2736 ops,   138.7 ms ( 45.9%)
  ColumnParallelLinear           454 ops,   156.4 ms ( 51.8%)
  VocabParallelEmbedding         454 ops,     0.9 ms (  0.3%)
  ...

By Function (torch.* patches):
  matmul                          32 ops,     2.1 ms ( 32.3%)
  softmax                         32 ops,     1.8 ms ( 27.7%)
  sum                             16 ops,     0.5 ms (  7.7%)
  ...

Top 10 Operations (by time):
Layer                           Type                 Est.Time   Meas.Time  Eff%   Bound
------------------------------- -------------------- ---------- ---------- ------ --------
module.output_layer             ColumnParallelLinear   3214.7us   3450.2us  93.2% compute
module.layers.0.mlp.fc1         TERowParallelLinear    1256.3us   1380.1us  91.0% compute
...
==========================================================================================
```

## Roofline Model

The estimator uses a roofline model to classify operations:

- **Compute-bound**: Limited by GPU TFLOPS (e.g., large matrix multiplies)
- **Memory-bound**: Limited by memory bandwidth (e.g., element-wise ops, small batches)

Arithmetic intensity = FLOPs / Bytes moved

If intensity > ridge point, operation is compute-bound.

## Device Specifications

Pre-configured specs for:
- **H100 SXM5 80GB**: 989 TF (BF16), 3.35 TB/s HBM3
- **A100 SXM4 80GB**: 312 TF (BF16), 2.04 TB/s HBM2e
- **B200**: 2250 TF (BF16), 8.0 TB/s HBM3e

Auto-detection via `get_current_device_spec()`.
