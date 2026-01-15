# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Optimizer step tracking for SOL analysis.

Patches optimizer.step() to measure time spent in optimizer operations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch

logger = logging.getLogger(__name__)


@dataclass
class OptimizerStepEvent:
    """Record of a single optimizer step."""
    optimizer_type: str
    measured_time_us: float
    phase: Optional[str] = None
    num_params: int = 0


class OptimizerTracker:
    """Tracks optimizer step time using monkey-patching."""
    
    def __init__(self, measure_time: bool = True):
        self.measure_time = measure_time
        self._events: List[OptimizerStepEvent] = []
        self._patched_optimizers: Dict[int, Any] = {}
        self._phase_stack: List[str] = []
        self._pending_events: List[tuple] = []
        
    def register_optimizer(self, optimizer):
        """Register an optimizer to track its step() calls."""
        opt_id = id(optimizer)
        if opt_id in self._patched_optimizers:
            return
        
        optimizer_type = type(optimizer).__name__
        
        # Handle ChainedOptimizer
        if hasattr(optimizer, 'chained_optimizers'):
            for child_opt in optimizer.chained_optimizers:
                self.register_optimizer(child_opt)
        
        original_step = optimizer.step
        tracker = self
        
        # Count parameters
        num_params = 0
        if hasattr(optimizer, 'param_groups'):
            for group in optimizer.param_groups:
                for p in group['params']:
                    if hasattr(p, 'requires_grad') and p.requires_grad:
                        num_params += p.numel()
        
        def tracked_step(*args, **kwargs):
            phase = tracker._phase_stack[-1] if tracker._phase_stack else None
            
            if tracker.measure_time and torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                
            result = original_step(*args, **kwargs)
            
            if tracker.measure_time and torch.cuda.is_available():
                end_event.record()
                tracker._pending_events.append((start_event, end_event, optimizer_type, num_params, phase))
            else:
                tracker._events.append(OptimizerStepEvent(
                    optimizer_type=optimizer_type,
                    measured_time_us=0,
                    phase=phase,
                    num_params=num_params,
                ))
                
            return result
        
        optimizer.step = tracked_step
        self._patched_optimizers[opt_id] = (optimizer, original_step)
        logger.debug(f"Registered optimizer {optimizer_type} for SOL tracking")
        
    def unregister_optimizer(self, optimizer):
        """Restore original step() method."""
        opt_id = id(optimizer)
        if opt_id in self._patched_optimizers:
            opt, original_step = self._patched_optimizers[opt_id]
            opt.step = original_step
            del self._patched_optimizers[opt_id]
    
    def push_phase(self, phase: str):
        self._phase_stack.append(phase)
        
    def pop_phase(self) -> Optional[str]:
        if self._phase_stack:
            return self._phase_stack.pop()
        return None
    
    def synchronize(self):
        """Synchronize CUDA events and compute measured times."""
        if not self.measure_time or not torch.cuda.is_available():
            return
        if not self._pending_events:
            return
            
        torch.cuda.synchronize()
        
        for start_event, end_event, optimizer_type, num_params, phase in self._pending_events:
            try:
                elapsed_ms = start_event.elapsed_time(end_event)
                self._events.append(OptimizerStepEvent(
                    optimizer_type=optimizer_type,
                    measured_time_us=elapsed_ms * 1000,
                    phase=phase,
                    num_params=num_params,
                ))
            except Exception:
                pass
                
        self._pending_events.clear()
    
    def clear(self):
        """Clear captured events."""
        self._events.clear()
        self._pending_events.clear()
        
    def cleanup(self):
        """Remove all patches and restore original methods."""
        for opt_id, (opt, original_step) in list(self._patched_optimizers.items()):
            opt.step = original_step
        self._patched_optimizers.clear()
        self._events.clear()
        self._pending_events.clear()
        self._phase_stack.clear()
    
    def get_summary(self) -> Dict:
        """Get summary of optimizer step times."""
        # Process any remaining pending events
        if self._pending_events:
            self.synchronize()
        
        if not self._events:
            return {
                'total_steps': 0,
                'total_time_us': 0,
                'total_params': 0,
                'by_type': {},
                'by_phase': {},
            }
            
        total_time_us = sum(e.measured_time_us for e in self._events)
        total_params = max(e.num_params for e in self._events) if self._events else 0
        
        by_type = {}
        for event in self._events:
            if event.optimizer_type not in by_type:
                by_type[event.optimizer_type] = {
                    'count': 0,
                    'total_time_us': 0,
                    'num_params': event.num_params,
                }
            by_type[event.optimizer_type]['count'] += 1
            by_type[event.optimizer_type]['total_time_us'] += event.measured_time_us
        
        by_phase = {}
        for event in self._events:
            phase = event.phase or "(no phase)"
            if phase not in by_phase:
                by_phase[phase] = {'count': 0, 'total_time_us': 0}
            by_phase[phase]['count'] += 1
            by_phase[phase]['total_time_us'] += event.measured_time_us
            
        return {
            'total_steps': len(self._events),
            'total_time_us': total_time_us,
            'total_params': total_params,
            'by_type': by_type,
            'by_phase': by_phase,
        }




