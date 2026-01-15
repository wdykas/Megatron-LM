# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Layer hooks for automatic shape capture from PyTorch modules.

This module provides hooks that attach to actual PyTorch layers (Linear, LayerNorm, etc.)
to capture real tensor shapes during forward AND backward passes, with optional timing measurement.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from contextlib import contextmanager
import weakref

from .device_specs import DataType, DeviceSpec, get_current_device_spec, torch_dtype_to_datatype
from .operation_registry import OpType, TensorShape, OpRecord
from .roofline import RooflineAnalyzer, RooflineResult


@dataclass
class CapturedOp:
    """A captured operation from a layer forward or backward pass."""
    layer_name: str
    layer_type: str
    op_type: OpType
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    dtype: DataType
    # Measured time in microseconds (filled if timing is enabled)
    measured_time_us: Optional[float] = None
    # Phase/NVTX range this op was captured in
    phase: Optional[str] = None
    # Whether this is a backward pass operation
    is_backward: bool = False
    
    def to_op_record(self) -> OpRecord:
        """Convert to OpRecord for roofline analysis."""
        inputs = [TensorShape(shape=s, dtype=self.dtype) for s in self.input_shapes]
        outputs = [TensorShape(shape=s, dtype=self.dtype) for s in self.output_shapes]
        
        # For Linear layers (MATMUL), we need to infer the weight matrix shape
        # Input: [batch, seq, in_features], Output: [batch, seq, out_features]
        # Weight: [in_features, out_features] (transposed in the actual matmul)
        if self.op_type == OpType.MATMUL and len(self.input_shapes) == 1 and len(self.output_shapes) >= 1:
            in_shape = self.input_shapes[0]
            out_shape = self.output_shapes[0]
            if len(in_shape) >= 1 and len(out_shape) >= 1:
                in_features = in_shape[-1]
                out_features = out_shape[-1]
                # Add inferred weight shape as second input
                weight_shape = (in_features, out_features)
                inputs.append(TensorShape(shape=weight_shape, dtype=self.dtype))
        
        # Backward pass typically has similar FLOPs to forward for most ops
        name = f"{self.layer_name}/backward" if self.is_backward else self.layer_name
        
        record = OpRecord(
            name=name,
            op_type=self.op_type,
            inputs=inputs,
            outputs=outputs,
        )
        record.compute_flops_and_memory()
        
        # For backward, multiply FLOPs by ~2 (grad_input + grad_weight for linear)
        if self.is_backward and self.op_type == OpType.MATMUL:
            record.flops = int(record.flops * 2)  # Two matmuls in backward
        
        return record


class LayerSOLHooks:
    """Automatic shape capture via PyTorch module hooks.
    
    Captures both forward and backward passes for comprehensive SOL analysis.
    
    Usage:
        hooks = LayerSOLHooks()
        
        # Register hooks on a model (includes forward + backward)
        hooks.register(model)
        
        # Run forward pass - shapes are automatically captured
        output = model(input)
        
        # Run backward pass - also captured!
        loss.backward()
        
        # Synchronize and analyze (required for timing)
        hooks.synchronize()
        results = hooks.analyze()
        hooks.print_report()
        
        # Clean up
        hooks.remove()
    """
    
    # Map layer class names to operation types - expanded for broader coverage
    LAYER_OP_MAP = {
        # Linear layers
        'Linear': OpType.MATMUL,
        'ColumnParallelLinear': OpType.MATMUL,
        'RowParallelLinear': OpType.MATMUL,
        'TEColumnParallelLinear': OpType.MATMUL,
        'TERowParallelLinear': OpType.MATMUL,
        # TE Linear variants
        'TELinear': OpType.MATMUL,
        'FusedLinear': OpType.MATMUL,
        
        # Normalization
        'LayerNorm': OpType.LAYERNORM,
        'RMSNorm': OpType.RMSNORM,
        'FusedRMSNorm': OpType.RMSNORM,
        'TENorm': OpType.LAYERNORM,
        'FusedLayerNorm': OpType.LAYERNORM,
        'MixedFusedRMSNorm': OpType.RMSNORM,
        'TritonRMSNorm': OpType.RMSNORM,
        
        # Softmax
        'Softmax': OpType.SOFTMAX,
        'FusedSoftmax': OpType.SOFTMAX,
        'FusedScaleMaskSoftmax': OpType.SOFTMAX,
        
        # Attention
        'DotProductAttention': OpType.ATTENTION,
        'FlashAttention': OpType.ATTENTION,
        'TEDotProductAttention': OpType.ATTENTION,
        'CoreAttention': OpType.ATTENTION,
        'SelfAttention': OpType.ATTENTION,
        'CrossAttention': OpType.ATTENTION,
        'MultiheadAttention': OpType.ATTENTION,
        'FlashSelfAttention': OpType.ATTENTION,
        'FusedAttention': OpType.ATTENTION,
        
        # Embedding
        'Embedding': OpType.EMBEDDING,
        'VocabParallelEmbedding': OpType.EMBEDDING,
        'ParallelEmbedding': OpType.EMBEDDING,
        
        # MLP / FFN components (often fused)
        'MLP': OpType.MATMUL,  # MLP blocks contain multiple matmuls
        'ParallelMLP': OpType.MATMUL,
        'SwitchMLP': OpType.MATMUL,
        'TELayerNormMLP': OpType.MATMUL,
        
        # Dropout (useful to track even if ~free)
        'Dropout': OpType.ELEMENTWISE,
        
        # Activation functions
        'GELU': OpType.ELEMENTWISE,
        'GeLU': OpType.ELEMENTWISE,
        'SiLU': OpType.ELEMENTWISE,
        'ReLU': OpType.ELEMENTWISE,
        'QuickGELU': OpType.ELEMENTWISE,
        'NewGELU': OpType.ELEMENTWISE,
        'FastGELU': OpType.ELEMENTWISE,
        
        # Transformer blocks (capture as attention for FLOPs estimation)
        'TransformerLayer': OpType.ATTENTION,
        'ParallelTransformerLayer': OpType.ATTENTION,
        'TETransformerLayer': OpType.ATTENTION,
        'MegatronModule': OpType.UNKNOWN,
    }
    
    def __init__(
        self,
        device_spec: Optional[DeviceSpec] = None,
        dtype: DataType = DataType.BF16,
        layer_types: Optional[Set[str]] = None,
        measure_time: bool = True,
        capture_backward: bool = True,
    ):
        """Initialize the hook manager.
        
        Args:
            device_spec: GPU specs for roofline analysis. Auto-detected if None.
            dtype: Default data type for operations.
            layer_types: Set of layer class names to hook. If None, hooks all known types.
            measure_time: If True, measure actual execution time using CUDA events.
            capture_backward: If True, also capture backward pass operations.
        """
        self.device_spec = device_spec or get_current_device_spec()
        self.dtype = dtype
        self.layer_types = layer_types or set(self.LAYER_OP_MAP.keys())
        self.measure_time = measure_time
        self.capture_backward = capture_backward
        
        self.analyzer = RooflineAnalyzer(self.device_spec, dtype)
        
        self._handles: List[Any] = []
        self._captured_ops: List[CapturedOp] = []
        self._results: List[RooflineResult] = []
        self._enabled = True
        
        # For timing: list of (start_event, end_event, op_index)
        self._pending_events: List[Tuple[Any, Any, int]] = []
        self._torch = None  # Lazy import
        
        # Phase tracking (for grouping by NVTX range)
        self._phase_stack: List[str] = []
    
    @property
    def current_phase(self) -> Optional[str]:
        """Get the current phase name (innermost NVTX range)."""
        return self._phase_stack[-1] if self._phase_stack else None
    
    @contextmanager
    def phase(self, name: str):
        """Context manager to mark operations as belonging to a phase."""
        self._phase_stack.append(name)
        try:
            yield
        finally:
            self._phase_stack.pop()
    
    def set_phase(self, name: Optional[str]):
        """Set the current phase directly."""
        if name is None:
            self._phase_stack.clear()
        else:
            self._phase_stack = [name]
    
    def push_phase(self, name: str):
        """Push a phase onto the stack."""
        self._phase_stack.append(name)
    
    def pop_phase(self):
        """Pop the current phase from the stack."""
        if self._phase_stack:
            self._phase_stack.pop()
    
    def _get_torch(self):
        """Lazy import torch."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch
    
    def enable(self):
        """Enable shape capture."""
        self._enabled = True
    
    def disable(self):
        """Disable shape capture."""
        self._enabled = False
    
    def clear(self):
        """Clear captured operations."""
        self._captured_ops.clear()
        self._results.clear()
        self._pending_events.clear()
    
    def _extract_shapes(self, tensors) -> Tuple[List[Tuple[int, ...]], DataType]:
        """Extract shapes from tensor inputs/outputs."""
        shapes = []
        detected_dtype = self.dtype
        
        if tensors is None:
            return shapes, detected_dtype
        
        if hasattr(tensors, 'shape'):
            shapes.append(tuple(tensors.shape))
            if hasattr(tensors, 'dtype'):
                detected_dtype = torch_dtype_to_datatype(tensors.dtype)
        elif isinstance(tensors, (tuple, list)):
            for item in tensors:
                if item is not None and hasattr(item, 'shape'):
                    shapes.append(tuple(item.shape))
                    if hasattr(item, 'dtype'):
                        detected_dtype = torch_dtype_to_datatype(item.dtype)
        
        return shapes, detected_dtype
    
    def _create_forward_hooks(self, layer_name: str, layer_type: str, op_type: OpType):
        """Create pre and post forward hooks for timing."""
        torch = self._get_torch()
        
        # Shared state for this layer instance
        state = {'start_event': None}
        
        def pre_hook(module, inputs):
            if not self._enabled:
                return
            if self.measure_time and torch.cuda.is_available():
                state['start_event'] = torch.cuda.Event(enable_timing=True)
                state['start_event'].record()
        
        def post_hook(module, inputs, outputs):
            if not self._enabled:
                return
            
            input_shapes, detected_dtype = self._extract_shapes(inputs)
            output_shapes, _ = self._extract_shapes(outputs)
            
            if input_shapes:
                op_index = len(self._captured_ops)
                self._captured_ops.append(CapturedOp(
                    layer_name=layer_name,
                    layer_type=layer_type,
                    op_type=op_type,
                    input_shapes=input_shapes,
                    output_shapes=output_shapes,
                    dtype=detected_dtype,
                    phase=self.current_phase,
                    is_backward=False,
                ))
                
                if self.measure_time and torch.cuda.is_available() and state['start_event'] is not None:
                    end_event = torch.cuda.Event(enable_timing=True)
                    end_event.record()
                    self._pending_events.append((state['start_event'], end_event, op_index))
                    state['start_event'] = None
        
        return pre_hook, post_hook
    
    def _create_backward_hook(self, layer_name: str, layer_type: str, op_type: OpType):
        """Create a backward hook for capturing backward pass operations."""
        torch = self._get_torch()
        
        # Use a local state dict that persists with this closure
        # This avoids KeyError issues when hooks are cleared/removed
        local_state = {'start_event': None}
        
        def backward_pre_hook(module, grad_output):
            """Called before backward computation of this module."""
            if not self._enabled:
                return
            if self.measure_time and torch.cuda.is_available():
                local_state['start_event'] = torch.cuda.Event(enable_timing=True)
                local_state['start_event'].record()
        
        def backward_hook(module, grad_input, grad_output):
            """Called after backward computation of this module."""
            if not self._enabled:
                return
            
            # Extract shapes from gradients
            input_shapes, detected_dtype = self._extract_shapes(grad_input)
            output_shapes, _ = self._extract_shapes(grad_output)
            
            # Use output shapes if input shapes not available
            if not input_shapes and output_shapes:
                input_shapes = output_shapes
            
            if input_shapes or output_shapes:
                # Add backward phase suffix if we have a current phase
                bwd_phase = f"{self.current_phase}/backward" if self.current_phase else "backward"
                
                op_index = len(self._captured_ops)
                self._captured_ops.append(CapturedOp(
                    layer_name=layer_name,
                    layer_type=layer_type,
                    op_type=op_type,
                    input_shapes=input_shapes or output_shapes,
                    output_shapes=output_shapes or input_shapes,
                    dtype=detected_dtype,
                    phase=bwd_phase,
                    is_backward=True,
                ))
                
                # Timing
                if self.measure_time and torch.cuda.is_available():
                    if local_state.get('start_event') is not None:
                        end_event = torch.cuda.Event(enable_timing=True)
                        end_event.record()
                        self._pending_events.append((local_state['start_event'], end_event, op_index))
                        local_state['start_event'] = None
        
        return backward_pre_hook, backward_hook
    
    def register(self, model, prefix: str = ""):
        """Register hooks on all matching layers in a model.
        
        Args:
            model: PyTorch module to hook
            prefix: Name prefix for layers
        """
        torch = self._get_torch()
        
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            
            if class_name in self.layer_types:
                full_name = f"{prefix}/{name}" if prefix else (name or class_name)
                op_type = self.LAYER_OP_MAP.get(class_name, OpType.UNKNOWN)
                
                # Forward hooks
                pre_hook, post_hook = self._create_forward_hooks(full_name, class_name, op_type)
                handle1 = module.register_forward_pre_hook(pre_hook)
                handle2 = module.register_forward_hook(post_hook)
                self._handles.extend([handle1, handle2])
                
                # Backward hooks (if enabled and module supports it)
                if self.capture_backward:
                    try:
                        bwd_pre_hook, bwd_hook = self._create_backward_hook(full_name, class_name, op_type)
                        # Use full_backward_pre_hook if available (PyTorch 2.0+)
                        if hasattr(module, 'register_full_backward_pre_hook'):
                            handle3 = module.register_full_backward_pre_hook(bwd_pre_hook)
                            self._handles.append(handle3)
                        handle4 = module.register_full_backward_hook(bwd_hook)
                        self._handles.append(handle4)
                    except Exception:
                        # Some modules may not support backward hooks
                        pass
    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def synchronize(self):
        """Synchronize CUDA and compute measured times."""
        if not self._pending_events:
            return
        
        torch = self._get_torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for start_event, end_event, op_index in self._pending_events:
            if op_index < len(self._captured_ops):
                try:
                    elapsed_ms = start_event.elapsed_time(end_event)
                    self._captured_ops[op_index].measured_time_us = elapsed_ms * 1000
                except Exception:
                    pass  # Timing failed, leave as None
        
        self._pending_events.clear()
    
    def analyze(self) -> List[RooflineResult]:
        """Analyze all captured operations."""
        self.synchronize()
        self._results.clear()
        
        for op in self._captured_ops:
            record = op.to_op_record()
            result = self.analyzer.analyze(record)
            self._results.append(result)
        
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._results:
            self.analyze()
        
        if not self._results:
            return {}
        
        total_flops = sum(r.record.flops for r in self._results)
        total_mem = sum(r.record.total_memory_bytes for r in self._results)
        total_estimated_time = sum(r.estimated_time_us for r in self._results)
        
        measured_times = [op.measured_time_us for op in self._captured_ops if op.measured_time_us is not None]
        total_measured_time = sum(measured_times) if measured_times else None
        
        # Count forward vs backward
        forward_ops = sum(1 for op in self._captured_ops if not op.is_backward)
        backward_ops = sum(1 for op in self._captured_ops if op.is_backward)
        
        # Group by layer type
        by_type = {}
        for op, result in zip(self._captured_ops, self._results):
            # Separate forward/backward
            key = f"{op.layer_type}{'(bwd)' if op.is_backward else ''}"
            if key not in by_type:
                by_type[key] = {"count": 0, "flops": 0, "estimated_time_us": 0, "measured_time_us": 0}
            by_type[key]["count"] += 1
            by_type[key]["flops"] += result.record.flops
            by_type[key]["estimated_time_us"] += result.estimated_time_us
            if op.measured_time_us is not None:
                by_type[key]["measured_time_us"] += op.measured_time_us
        
        # Group by layer instance
        by_layer_instance = {}
        for op, result in zip(self._captured_ops, self._results):
            name = op.layer_name
            if name not in by_layer_instance:
                by_layer_instance[name] = {
                    "layer_type": op.layer_type,
                    "count": 0, 
                    "flops": 0, 
                    "estimated_time_us": 0, 
                    "measured_time_us": 0,
                    "forward_count": 0,
                    "backward_count": 0,
                }
            by_layer_instance[name]["count"] += 1
            by_layer_instance[name]["flops"] += result.record.flops
            by_layer_instance[name]["estimated_time_us"] += result.estimated_time_us
            if op.is_backward:
                by_layer_instance[name]["backward_count"] += 1
            else:
                by_layer_instance[name]["forward_count"] += 1
            if op.measured_time_us is not None:
                by_layer_instance[name]["measured_time_us"] += op.measured_time_us
        
        # Group by phase
        by_phase = {}
        for op, result in zip(self._captured_ops, self._results):
            phase = op.phase or "(no phase)"
            if phase not in by_phase:
                by_phase[phase] = {"count": 0, "flops": 0, "estimated_time_us": 0, "measured_time_us": 0}
            by_phase[phase]["count"] += 1
            by_phase[phase]["flops"] += result.record.flops
            by_phase[phase]["estimated_time_us"] += result.estimated_time_us
            if op.measured_time_us is not None:
                by_phase[phase]["measured_time_us"] += op.measured_time_us
        
        # Add hierarchical aggregates
        parent_phases = set()
        for phase in by_phase:
            if phase == "(no phase)":
                continue
            parts = phase.split("/")
            for i in range(1, len(parts)):
                parent = "/".join(parts[:i])
                parent_phases.add(parent)
        
        by_phase_hierarchical = {}
        for parent in parent_phases:
            stats = {"count": 0, "flops": 0, "estimated_time_us": 0, "measured_time_us": 0, "is_aggregate": True}
            for phase, phase_stats in by_phase.items():
                if phase.startswith(parent + "/") or phase == parent:
                    stats["count"] += phase_stats["count"]
                    stats["flops"] += phase_stats["flops"]
                    stats["estimated_time_us"] += phase_stats["estimated_time_us"]
                    stats["measured_time_us"] += phase_stats["measured_time_us"]
            if stats["count"] > 0:
                by_phase_hierarchical[parent + " (total)"] = stats
        
        by_phase.update(by_phase_hierarchical)
        
        return {
            "total_ops": len(self._results),
            "forward_ops": forward_ops,
            "backward_ops": backward_ops,
            "total_flops": total_flops,
            "total_memory_bytes": total_mem,
            "total_estimated_time_us": total_estimated_time,
            "total_measured_time_us": total_measured_time,
            "by_layer_type": by_type,
            "by_layer_instance": by_layer_instance,
            "by_phase": by_phase,
        }
    
    def print_debug_shapes(self, filter_type: Optional[str] = None):
        """Print captured shapes for debugging."""
        print(f"\n{'='*80}")
        print(f"Captured Shapes Debug")
        print(f"{'='*80}")
        
        for i, op in enumerate(self._captured_ops):
            if filter_type and op.layer_type != filter_type:
                continue
            bwd_str = " [BACKWARD]" if op.is_backward else ""
            print(f"\n[{i}] {op.layer_name}{bwd_str}")
            print(f"    Type: {op.layer_type} -> {op.op_type.name}")
            print(f"    Inputs: {op.input_shapes}")
            print(f"    Outputs: {op.output_shapes}")
            print(f"    Phase: {op.phase}")
            if i < len(self._results):
                r = self._results[i]
                print(f"    FLOPs: {r.record.flops:,}")
                print(f"    Memory: {r.record.total_memory_bytes:,} bytes")
        
        print(f"{'='*80}\n")
    
    def print_report(self, top_n: int = 20):
        """Print a report of captured operations."""
        if not self._results:
            self.analyze()
        
        if not self._results:
            print("No operations captured.")
            return
        
        summary = self.get_summary()
        has_measured = summary.get("total_measured_time_us") is not None
        
        print(f"\n{'='*110}")
        print(f"Layer SOL Analysis Report")
        print(f"{'='*110}")
        print(f"Device: {self.device_spec.name}")
        print(f"Data Type: {self.dtype.value}")
        print(f"")
        
        print(f"Summary:")
        print(f"  Total layers captured: {summary['total_ops']} (forward: {summary['forward_ops']}, backward: {summary['backward_ops']})")
        print(f"  Total FLOPs: {summary['total_flops']/1e12:.2f} TFLOPS")
        print(f"  Total Memory: {summary['total_memory_bytes']/1e9:.2f} GB")
        print(f"  Estimated Time (SOL): {summary['total_estimated_time_us']/1000:.2f} ms")
        if has_measured:
            efficiency = summary['total_estimated_time_us'] / summary['total_measured_time_us'] * 100
            print(f"  Measured Time:        {summary['total_measured_time_us']/1000:.2f} ms")
            print(f"  Overall Efficiency:   {efficiency:.1f}% of SOL")
        print(f"")
        
        print(f"By Layer Type:")
        header = f"  {'Layer Type':<35} {'Count':>6} {'Est(ms)':>10} "
        if has_measured:
            header += f"{'Meas(ms)':>10} {'Eff%':>8}"
        header += f" {'%Total':>8}"
        print(header)
        print(f"  {'-'*35} {'-'*6} {'-'*10} " + (f"{'-'*10} {'-'*8}" if has_measured else "") + f" {'-'*8}")
        
        # Sort by estimated time
        sorted_types = sorted(summary["by_layer_type"].items(), key=lambda x: x[1]["estimated_time_us"], reverse=True)
        for layer_type, stats in sorted_types:
            pct = stats["estimated_time_us"] / summary["total_estimated_time_us"] * 100 if summary["total_estimated_time_us"] > 0 else 0
            line = f"  {layer_type:<35} {stats['count']:>6} {stats['estimated_time_us']/1000:>10.2f} "
            if has_measured and stats["measured_time_us"] > 0:
                eff = stats["estimated_time_us"] / stats["measured_time_us"] * 100
                line += f"{stats['measured_time_us']/1000:>10.2f} {eff:>7.1f}%"
            elif has_measured:
                line += f"{'N/A':>10} {'N/A':>8}"
            line += f" {pct:>7.1f}%"
            print(line)
        print(f"")
        
        # By Phase
        by_phase = summary.get("by_phase", {})
        if by_phase and not (len(by_phase) == 1 and "(no phase)" in by_phase):
            print(f"By Phase (NVTX Range):")
            header = f"  {'Phase':<50} {'Count':>6} {'Est(ms)':>10} "
            if has_measured:
                header += f"{'Meas(ms)':>10} {'Eff%':>8}"
            header += f" {'%Total':>8}"
            print(header)
            print(f"  {'-'*50} {'-'*6} {'-'*10} " + (f"{'-'*10} {'-'*8}" if has_measured else "") + f" {'-'*8}")
            
            sorted_phases = sorted(by_phase.items(), key=lambda x: x[1]["estimated_time_us"], reverse=True)
            
            for phase, stats in sorted_phases:
                if summary["total_estimated_time_us"] > 0:
                    pct = stats["estimated_time_us"] / summary["total_estimated_time_us"] * 100
                else:
                    pct = 0
                phase_display = phase[-50:] if len(phase) > 50 else phase
                line = f"  {phase_display:<50} {stats['count']:>6} {stats['estimated_time_us']/1000:>10.2f} "
                if has_measured and stats["measured_time_us"] > 0:
                    eff = stats["estimated_time_us"] / stats["measured_time_us"] * 100
                    line += f"{stats['measured_time_us']/1000:>10.2f} {eff:>7.1f}%"
                elif has_measured:
                    line += f"{'N/A':>10} {'N/A':>8}"
                line += f" {pct:>7.1f}%"
                print(line)
            print(f"")
        
        # Top operations
        sorted_results = sorted(
            zip(self._captured_ops, self._results),
            key=lambda x: x[1].estimated_time_us,
            reverse=True
        )
        
        print(f"Top {min(top_n, len(sorted_results))} Operations (by estimated time):")
        header = f"{'Layer':<40} {'Type':<15} {'Shapes':<18} {'Est(us)':>10}"
        if has_measured:
            header += f" {'Meas(us)':>10} {'Eff%':>7}"
        header += f" {'Bound':<8}"
        print(header)
        print(f"{'-'*40} {'-'*15} {'-'*18} {'-'*10}" + (f" {'-'*10} {'-'*7}" if has_measured else "") + f" {'-'*8}")
        
        for op, result in sorted_results[:top_n]:
            name = op.layer_name[-40:] if len(op.layer_name) > 40 else op.layer_name
            type_str = f"{op.layer_type[:12]}{'(B)' if op.is_backward else ''}"
            shapes = str(op.input_shapes[0]) if op.input_shapes else "?"
            if len(shapes) > 18:
                shapes = shapes[:15] + "..."
            
            line = f"{name:<40} {type_str:<15} {shapes:<18} {result.estimated_time_us:>10.1f}"
            if has_measured:
                if op.measured_time_us is not None and op.measured_time_us > 0:
                    eff = result.estimated_time_us / op.measured_time_us * 100
                    line += f" {op.measured_time_us:>10.1f} {eff:>6.1f}%"
                else:
                    line += f" {'N/A':>10} {'N/A':>7}"
            line += f" {result.bound_type.value:<8}"
            print(line)
        
        print(f"{'='*110}\n")


@contextmanager
def sol_profile(model, device_spec: Optional[DeviceSpec] = None, dtype: DataType = DataType.BF16, 
                measure_time: bool = True, capture_backward: bool = True):
    """Context manager for SOL profiling a forward (and optionally backward) pass.
    
    Usage:
        with sol_profile(model, capture_backward=True) as hooks:
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
        
        hooks.print_report()
    """
    hooks = LayerSOLHooks(device_spec, dtype, measure_time=measure_time, capture_backward=capture_backward)
    hooks.register(model)
    try:
        yield hooks
    finally:
        hooks.remove()
