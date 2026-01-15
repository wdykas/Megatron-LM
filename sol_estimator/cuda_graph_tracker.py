# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""CUDA Graph tracking for SOL estimation.

When using CUDA graphs, standard PyTorch hooks only fire during graph capture,
not during replay. This module tracks:
1. Operations captured during graph capture phase
2. Graph replay events (to multiply captured ops by replay count)
3. Graph execution time
"""

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class GraphState(Enum):
    """CUDA graph capture state."""
    NONE = "none"
    CAPTURING = "capturing"
    REPLAYING = "replaying"


@dataclass
class CapturedGraph:
    """Information about a captured CUDA graph."""
    graph_id: int
    name: str
    capture_phase: Optional[str] = None
    
    # Operations captured during this graph's capture phase
    captured_op_count: int = 0
    captured_flops: int = 0
    captured_memory_bytes: int = 0
    captured_estimated_time_us: float = 0.0
    
    # Replay tracking
    replay_count: int = 0
    total_replay_time_us: float = 0.0


@dataclass 
class GraphReplayEvent:
    """A single graph replay event."""
    graph_id: int
    graph_name: str
    measured_time_us: float
    phase: Optional[str] = None


class CUDAGraphTracker:
    """Tracks CUDA graph captures and replays for accurate SOL estimation.
    
    Usage:
        tracker = CUDAGraphTracker()
        
        # Option 1: Manual tracking with context managers
        with tracker.capture_mode("inference_graph"):
            # Graph capture happens here - hooks will fire normally
            cuda_graph.capture_begin()
            output = model(input)
            cuda_graph.capture_end()
        
        # During replay
        with tracker.replay_mode("inference_graph"):
            cuda_graph.replay()  # Hooks won't fire, but we track the replay
        
        # Option 2: Automatic via monkey-patching
        tracker.patch()  # Patches torch.cuda.CUDAGraph
        # ... normal code using CUDA graphs ...
        tracker.unpatch()
        
        # Get summary
        summary = tracker.get_summary()
    """
    
    def __init__(self, measure_time: bool = True):
        self._measure_time = measure_time
        self._graphs: Dict[int, CapturedGraph] = {}
        self._replay_events: List[GraphReplayEvent] = []
        self._current_state = GraphState.NONE
        self._current_graph_id: Optional[int] = None
        self._current_graph_name: Optional[str] = None
        self._phase_stack: List[str] = []
        self._lock = threading.Lock()
        
        # For monkey-patching
        self._original_funcs: Dict[str, Callable] = {}
        self._patched = False
        self._torch = None
        
        # Graph ID counter
        self._next_graph_id = 1
        
        # Timing for current replay
        self._replay_start_event = None
    
    def _get_torch(self):
        """Lazy import torch."""
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch
    
    @property
    def current_phase(self) -> Optional[str]:
        return self._phase_stack[-1] if self._phase_stack else None
    
    def push_phase(self, name: str):
        self._phase_stack.append(name)
    
    def pop_phase(self):
        if self._phase_stack:
            self._phase_stack.pop()
    
    @property
    def is_capturing(self) -> bool:
        """Check if currently in graph capture mode."""
        return self._current_state == GraphState.CAPTURING
    
    @property
    def is_replaying(self) -> bool:
        """Check if currently in graph replay mode."""
        return self._current_state == GraphState.REPLAYING
    
    @contextmanager
    def capture_mode(self, name: str = "unnamed_graph"):
        """Context manager to mark a CUDA graph capture phase.
        
        Use this around your graph capture code so SOL tracking can
        associate captured operations with this graph.
        
        Args:
            name: Human-readable name for this graph
        """
        graph_id = self._next_graph_id
        self._next_graph_id += 1
        
        with self._lock:
            self._current_state = GraphState.CAPTURING
            self._current_graph_id = graph_id
            self._current_graph_name = name
            
            self._graphs[graph_id] = CapturedGraph(
                graph_id=graph_id,
                name=name,
                capture_phase=self.current_phase,
            )
        
        logger.debug(f"CUDA graph capture started: {name} (id={graph_id})")
        
        try:
            yield graph_id
        finally:
            with self._lock:
                self._current_state = GraphState.NONE
                self._current_graph_id = None
                self._current_graph_name = None
            logger.debug(f"CUDA graph capture ended: {name}")
    
    @contextmanager
    def replay_mode(self, graph_id_or_name):
        """Context manager to mark a CUDA graph replay.
        
        Use this around graph.replay() calls so we can track replay events.
        
        Args:
            graph_id_or_name: Either the graph_id returned from capture_mode,
                            or the name string used during capture
        """
        torch = self._get_torch()
        
        # Find the graph
        graph = None
        if isinstance(graph_id_or_name, int):
            graph = self._graphs.get(graph_id_or_name)
        else:
            for g in self._graphs.values():
                if g.name == graph_id_or_name:
                    graph = g
                    break
        
        if graph is None:
            logger.warning(f"CUDA graph not found for replay: {graph_id_or_name}")
            yield
            return
        
        # Start timing
        start_event = None
        end_event = None
        if self._measure_time and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        with self._lock:
            self._current_state = GraphState.REPLAYING
            self._current_graph_id = graph.graph_id
        
        try:
            yield
        finally:
            # End timing
            measured_time_us = 0.0
            if start_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                measured_time_us = start_event.elapsed_time(end_event) * 1000
            
            with self._lock:
                graph.replay_count += 1
                graph.total_replay_time_us += measured_time_us
                
                self._replay_events.append(GraphReplayEvent(
                    graph_id=graph.graph_id,
                    graph_name=graph.name,
                    measured_time_us=measured_time_us,
                    phase=self.current_phase,
                ))
                
                self._current_state = GraphState.NONE
                self._current_graph_id = None
    
    def register_captured_ops(self, op_count: int, flops: int, 
                              memory_bytes: int, estimated_time_us: float):
        """Register operations captured during graph capture.
        
        Call this after analyzing hooks during a capture_mode context.
        This associates the captured operations with the current graph.
        
        Args:
            op_count: Number of operations captured
            flops: Total FLOPs of captured operations
            memory_bytes: Total memory of captured operations
            estimated_time_us: Estimated SOL time of captured operations
        """
        if self._current_graph_id is None:
            return
        
        with self._lock:
            graph = self._graphs.get(self._current_graph_id)
            if graph:
                graph.captured_op_count = op_count
                graph.captured_flops = flops
                graph.captured_memory_bytes = memory_bytes
                graph.captured_estimated_time_us = estimated_time_us
                logger.debug(f"Registered {op_count} ops for graph {graph.name}")
    
    def patch(self):
        """Patch torch.cuda.CUDAGraph to automatically track captures and replays.
        
        Also patches Megatron's is_graph_capturing() for TE-based graphs.
        """
        if self._patched:
            return
        
        torch = self._get_torch()
        tracker = self
        
        # 1. Patch torch.cuda.CUDAGraph
        if hasattr(torch.cuda, 'CUDAGraph'):
            original_class = torch.cuda.CUDAGraph
            
            class TrackedCUDAGraph(original_class):
                """CUDAGraph wrapper that tracks captures and replays."""
                
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._tracker_graph_id = None
                    self._tracker_name = f"graph_{id(self)}"
                
                def capture_begin(self, *args, **kwargs):
                    self._tracker_graph_id = tracker._next_graph_id
                    tracker._next_graph_id += 1
                    
                    with tracker._lock:
                        tracker._current_state = GraphState.CAPTURING
                        tracker._current_graph_id = self._tracker_graph_id
                        tracker._current_graph_name = self._tracker_name
                        
                        tracker._graphs[self._tracker_graph_id] = CapturedGraph(
                            graph_id=self._tracker_graph_id,
                            name=self._tracker_name,
                            capture_phase=tracker.current_phase,
                        )
                    
                    return super().capture_begin(*args, **kwargs)
                
                def capture_end(self, *args, **kwargs):
                    result = super().capture_end(*args, **kwargs)
                    
                    with tracker._lock:
                        tracker._current_state = GraphState.NONE
                        tracker._current_graph_id = None
                        tracker._current_graph_name = None
                    
                    return result
                
                def replay(self, *args, **kwargs):
                    graph = tracker._graphs.get(self._tracker_graph_id)
                    
                    start_event = None
                    end_event = None
                    if tracker._measure_time and torch.cuda.is_available():
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    
                    result = super().replay(*args, **kwargs)
                    
                    measured_time_us = 0.0
                    if start_event is not None:
                        end_event.record()
                        torch.cuda.synchronize()
                        measured_time_us = start_event.elapsed_time(end_event) * 1000
                    
                    if graph:
                        with tracker._lock:
                            graph.replay_count += 1
                            graph.total_replay_time_us += measured_time_us
                            
                            tracker._replay_events.append(GraphReplayEvent(
                                graph_id=graph.graph_id,
                                graph_name=graph.name,
                                measured_time_us=measured_time_us,
                                phase=tracker.current_phase,
                            ))
                    
                    return result
            
            self._original_funcs['CUDAGraph'] = original_class
            torch.cuda.CUDAGraph = TrackedCUDAGraph
            logger.debug("Patched torch.cuda.CUDAGraph")
        
        # 2. Try to patch Megatron's cuda_graphs module for TE-based graphs
        self._patch_megatron_cuda_graphs()
        
        self._patched = True
        logger.info("CUDA graph tracking enabled")
    
    def _patch_megatron_cuda_graphs(self):
        """Patch Megatron's CUDA graph system for TE-based graphs."""
        tracker = self
        torch = self._get_torch()
        
        try:
            from megatron.core.transformer import cuda_graphs as mcg
            
            # Save original functions
            original_set_capture_start = mcg._set_capture_start
            original_set_capture_end = mcg._set_capture_end
            
            def tracked_set_capture_start():
                original_set_capture_start()
                graph_id = tracker._next_graph_id
                tracker._next_graph_id += 1
                
                with tracker._lock:
                    tracker._current_state = GraphState.CAPTURING
                    tracker._current_graph_id = graph_id
                    tracker._current_graph_name = f"megatron_graph_{graph_id}"
                    
                    tracker._graphs[graph_id] = CapturedGraph(
                        graph_id=graph_id,
                        name=f"megatron_graph_{graph_id}",
                        capture_phase=tracker.current_phase,
                    )
                logger.debug(f"Megatron CUDA graph capture started: id={graph_id}")
            
            def tracked_set_capture_end():
                original_set_capture_end()
                with tracker._lock:
                    tracker._current_state = GraphState.NONE
                    tracker._current_graph_id = None
                    tracker._current_graph_name = None
                logger.debug("Megatron CUDA graph capture ended")
            
            mcg._set_capture_start = tracked_set_capture_start
            mcg._set_capture_end = tracked_set_capture_end
            
            self._original_funcs['mcg._set_capture_start'] = original_set_capture_start
            self._original_funcs['mcg._set_capture_end'] = original_set_capture_end
            
            # Patch the graph replay - note the class is _CudaGraphRunner (with underscore)
            if hasattr(mcg, '_CudaGraphRunner'):
                original_replay = mcg._CudaGraphRunner.replay_graph_capture
                
                def _estimate_module_ops(base_module, input_args):
                    """Estimate ops/FLOPs for a graphed module based on its type."""
                    # Count submodules to estimate ops
                    op_count = 0
                    flops = 0
                    
                    # Get input shape from args if available
                    batch_size = 1
                    seq_len = 1
                    hidden_size = 1
                    
                    if input_args and len(input_args) > 0:
                        first_arg = input_args[0]
                        if hasattr(first_arg, 'shape'):
                            shape = first_arg.shape
                            if len(shape) >= 2:
                                batch_size = shape[0]
                                seq_len = shape[1]
                            if len(shape) >= 3:
                                hidden_size = shape[2]
                    
                    # Count Linear layers (each = 1 matmul)
                    for name, module in base_module.named_modules():
                        module_type = type(module).__name__
                        if 'Linear' in module_type or 'ColumnParallel' in module_type or 'RowParallel' in module_type:
                            op_count += 1
                            # Estimate FLOPs: 2 * batch * seq * in_features * out_features
                            if hasattr(module, 'weight') and module.weight is not None:
                                in_f = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                                out_f = module.weight.shape[0]
                                flops += 2 * batch_size * seq_len * in_f * out_f
                        elif 'Attention' in module_type or 'DotProduct' in module_type:
                            # Attention has multiple matmuls: QKV projection + QK^T + softmax@V
                            op_count += 4  # Q, K, V projections + attention
                        elif 'LayerNorm' in module_type or 'RMSNorm' in module_type:
                            op_count += 1
                            # ~5 ops per element
                            flops += 5 * batch_size * seq_len * hidden_size
                    
                    return op_count, flops
                
                def tracked_replay(self_runner, is_first_microbatch, args, kwargs):
                    # Log that we're being called (first few times only for debugging)
                    runner_id = id(self_runner)
                    is_new = runner_id not in tracker._graphs
                    if is_new:
                        logger.info(f"CUDA graph replay_graph_capture called: runner_id={runner_id}")
                    
                    result = original_replay(self_runner, is_first_microbatch, args, kwargs)
                    
                    # Don't time individual replays to avoid sync overhead
                    measured_time_us = 0.0
                    
                    # Find or create a graph entry for this runner
                    runner_id = id(self_runner)
                    # Try to get a meaningful name from the runner's base_module
                    runner_name = None
                    base_module = getattr(self_runner, 'base_module', None)
                    if base_module is not None:
                        runner_name = getattr(base_module, '_name', None) or type(base_module).__name__
                    runner_name = runner_name or f"graph_{runner_id}"
                    
                    with tracker._lock:
                        if runner_id not in tracker._graphs:
                            # Estimate ops from the base module
                            est_ops = 0
                            est_flops = 0
                            if base_module is not None:
                                est_ops, est_flops = _estimate_module_ops(base_module, args)
                            
                            # Estimate time assuming ~1000 TFLOPS peak (H100 BF16)
                            est_time_us = (est_flops / 1e15) * 1e6 if est_flops > 0 else 0.0
                            tracker._graphs[runner_id] = CapturedGraph(
                                graph_id=runner_id,
                                name=runner_name,
                                capture_phase=tracker.current_phase,
                                captured_op_count=est_ops,
                                captured_flops=est_flops,
                                captured_estimated_time_us=est_time_us,
                            )
                            logger.info(f"CUDA graph replay detected: {runner_name}, estimated {est_ops} ops, {est_flops/1e9:.2f} GFLOPs")
                        
                        graph = tracker._graphs[runner_id]
                        graph.replay_count += 1
                        graph.total_replay_time_us += measured_time_us
                        
                        tracker._replay_events.append(GraphReplayEvent(
                            graph_id=runner_id,
                            graph_name=runner_name,
                            measured_time_us=measured_time_us,
                            phase=tracker.current_phase,
                        ))
                    
                    logger.debug(f"CUDA graph replay: {runner_name}, time={measured_time_us:.2f}us")
                    return result
                
                mcg._CudaGraphRunner.replay_graph_capture = tracked_replay
                self._original_funcs['_CudaGraphRunner.replay_graph_capture'] = original_replay
                logger.info("Patched _CudaGraphRunner.replay_graph_capture")
            
            logger.info(f"Patched Megatron CUDA graph functions (has _CudaGraphRunner: {hasattr(mcg, '_CudaGraphRunner')})")
            
        except ImportError as e:
            logger.debug(f"Megatron cuda_graphs module not available, skipping patch: {e}")
        except Exception as e:
            logger.warning(f"Failed to patch Megatron cuda_graphs: {e}")
            import traceback
            traceback.print_exc()
    
    def unpatch(self):
        """Restore original torch.cuda.CUDAGraph and Megatron functions."""
        if not self._patched:
            return
        
        torch = self._get_torch()
        
        # Restore torch.cuda.CUDAGraph
        if 'CUDAGraph' in self._original_funcs:
            torch.cuda.CUDAGraph = self._original_funcs['CUDAGraph']
        
        # Restore Megatron functions
        try:
            from megatron.core.transformer import cuda_graphs as mcg
            
            if 'mcg._set_capture_start' in self._original_funcs:
                mcg._set_capture_start = self._original_funcs['mcg._set_capture_start']
            if 'mcg._set_capture_end' in self._original_funcs:
                mcg._set_capture_end = self._original_funcs['mcg._set_capture_end']
            if '_CudaGraphRunner.replay_graph_capture' in self._original_funcs:
                mcg._CudaGraphRunner.replay_graph_capture = self._original_funcs['_CudaGraphRunner.replay_graph_capture']
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error unpatching Megatron functions: {e}")
        
        self._original_funcs.clear()
        self._patched = False
    
    def clear(self):
        """Clear replay events for next iteration, but keep graph info."""
        with self._lock:
            # Only clear replay events - keep graph capture info
            # (graphs are usually captured once at startup)
            self._replay_events.clear()
            # Reset replay counts on graphs for per-iteration tracking
            for graph in self._graphs.values():
                graph.replay_count = 0
                graph.total_replay_time_us = 0.0
    
    def clear_all(self):
        """Fully clear all tracked graphs and replay events."""
        with self._lock:
            self._graphs.clear()
            self._replay_events.clear()
            self._next_graph_id = 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of CUDA graph captures and replays."""
        with self._lock:
            if not self._graphs and not self._replay_events:
                return {}
            
            total_captures = len(self._graphs)
            total_replays = sum(g.replay_count for g in self._graphs.values())
            total_replay_time = sum(g.total_replay_time_us for g in self._graphs.values())
            
            # Calculate effective ops (captured ops * replay count)
            effective_ops = 0
            effective_flops = 0
            effective_estimated_time = 0.0
            
            for graph in self._graphs.values():
                replays = max(1, graph.replay_count)  # At least 1 for capture
                effective_ops += graph.captured_op_count * replays
                effective_flops += graph.captured_flops * replays
                effective_estimated_time += graph.captured_estimated_time_us * replays
            
            # By graph
            by_graph = {}
            for graph in self._graphs.values():
                by_graph[graph.name] = {
                    "graph_id": graph.graph_id,
                    "captured_ops": graph.captured_op_count,
                    "captured_flops": graph.captured_flops,
                    "captured_estimated_time_us": graph.captured_estimated_time_us,
                    "replay_count": graph.replay_count,
                    "total_replay_time_us": graph.total_replay_time_us,
                    "avg_replay_time_us": graph.total_replay_time_us / graph.replay_count if graph.replay_count > 0 else 0,
                    "capture_phase": graph.capture_phase,
                }
            
            # By phase
            by_phase = {}
            for event in self._replay_events:
                phase = event.phase or "(no phase)"
                if phase not in by_phase:
                    by_phase[phase] = {"count": 0, "measured_time_us": 0.0}
                by_phase[phase]["count"] += 1
                by_phase[phase]["measured_time_us"] += event.measured_time_us
            
            return {
                "total_graphs_captured": total_captures,
                "total_replays": total_replays,
                "total_replay_time_us": total_replay_time,
                "effective_ops": effective_ops,
                "effective_flops": effective_flops,
                "effective_estimated_time_us": effective_estimated_time,
                "by_graph": by_graph,
                "by_phase": by_phase,
            }
    
    def print_report(self):
        """Print a summary of CUDA graph activity."""
        summary = self.get_summary()
        
        if not summary:
            print("No CUDA graphs tracked.")
            return
        
        print(f"\n{'='*70}")
        print("CUDA Graph Tracking Report")
        print(f"{'='*70}")
        print(f"Graphs captured: {summary['total_graphs_captured']}")
        print(f"Total replays: {summary['total_replays']}")
        print(f"Total replay time: {summary['total_replay_time_us']/1000:.2f} ms")
        print(f"")
        print(f"Effective (replay-adjusted):")
        print(f"  Operations: {summary['effective_ops']}")
        print(f"  FLOPs: {summary['effective_flops']/1e12:.2f} TFLOPS")
        print(f"  Estimated Time: {summary['effective_estimated_time_us']/1000:.2f} ms")
        print(f"")
        
        # Individual graph details removed for brevity
        
        by_phase = summary.get("by_phase", {})
        if by_phase:
            print(f"\nBy Phase:")
            for phase, stats in by_phase.items():
                print(f"  {phase}: {stats['count']} replays, {stats['measured_time_us']/1000:.2f} ms")
        
        print(f"{'='*70}\n")


# Global instance for convenience
_global_graph_tracker: Optional[CUDAGraphTracker] = None


def get_graph_tracker() -> CUDAGraphTracker:
    """Get or create the global CUDA graph tracker."""
    global _global_graph_tracker
    if _global_graph_tracker is None:
        _global_graph_tracker = CUDAGraphTracker()
    return _global_graph_tracker
