# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""SOL (Speed of Light) integration for Megatron-RL."""

import importlib
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from megatron.core import mpu
from megatron.core.utils import log_single_rank
from megatron.training.global_vars import get_args, get_tensorboard_writer, get_wandb_writer
from megatron.training.utils import get_nvtx_range

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sol_estimator.layer_hooks import LayerSOLHooks

_SOL_IMPORT_ERROR = None
try:
    # sol_estimator is located at megatron-rl/sol_estimator/
    _sol_estimator_path = str(Path(__file__).parent.parent.parent)
    if _sol_estimator_path not in sys.path:
        sys.path.insert(0, _sol_estimator_path)
    from sol_estimator.layer_hooks import LayerSOLHooks
    from sol_estimator.cuda_graph_tracker import CUDAGraphTracker
    from sol_estimator.optimizer_tracker import OptimizerTracker
    from sol_estimator.phase_timer import PhaseTimer
    from sol_estimator import DataType
    SOL_ESTIMATOR_AVAILABLE = True
except Exception as e:
    SOL_ESTIMATOR_AVAILABLE = False
    _SOL_IMPORT_ERROR = e
    DataType = None
    CUDAGraphTracker = None
    OptimizerTracker = None
    PhaseTimer = None


class SOLTracker:
    """Tracks Speed of Light (SOL) metrics for model operations.
    
    Core trackers:
    - LayerSOLHooks: nn.Module-level captures (TE layers, Linear, Attention)
    - CUDAGraphTracker: CUDA graph captures and replays
    - OptimizerTracker: Optimizer step time
    - PhaseTimer: Wall-clock time per phase
    """
    
    def __init__(self):
        self.layer_hooks: "LayerSOLHooks" = None
        self.graph_tracker: "CUDAGraphTracker" = None
        self.optimizer_tracker: "OptimizerTracker" = None
        self.phase_timer: "PhaseTimer" = None
        self._extra_patchers = []
        self._phase_trackers = []
        self.initialized = False
    
    def initialize(self, model, args):
        """Initialize SOL tracking."""
        if not SOL_ESTIMATOR_AVAILABLE:
            if _SOL_IMPORT_ERROR:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"SOL estimator not available: {_SOL_IMPORT_ERROR}",
                )
            else:
                log_single_rank(logger, logging.WARNING, "SOL estimator not available")
            return False
        
        if not getattr(args, 'rl_enable_sol_tracking', False):
            return False
            
        if self.initialized:
            return True
            
        try:
            dtype = DataType.BF16 if args.bf16 else (DataType.FP16 if args.fp16 else DataType.FP32)
            
            # LayerSOLHooks: nn.Module layers (captures most compute)
            self.layer_hooks = LayerSOLHooks(dtype=dtype, measure_time=True)
            self.layer_hooks.register(model)
            
            # CUDAGraphTracker
            if getattr(args, 'rl_enable_sol_graph_tracking', False) and CUDAGraphTracker:
                self.graph_tracker = CUDAGraphTracker(measure_time=True)
                self.graph_tracker.patch()
            
            # PhaseTimer: wall-clock per phase
            if PhaseTimer:
                self.phase_timer = PhaseTimer()
            
            # OptimizerTracker
            if OptimizerTracker:
                self.optimizer_tracker = OptimizerTracker(measure_time=True)

            # Optional comm/sync tracking
            if getattr(args, 'rl_enable_sol_comm_tracking', False):
                self._setup_optional_patcher(
                    module_name="sol_estimator.collective_patcher",
                    class_candidates=("CollectivePatcher", "CollectiveTracker", "CollectiveHook"),
                    label="collective",
                )
            if getattr(args, 'rl_enable_sol_sync_tracking', False):
                self._setup_optional_patcher(
                    module_name="sol_estimator.sync_tracker",
                    class_candidates=("SyncTracker", "CudaSyncTracker", "SynchronizationTracker"),
                    label="sync",
                )
                self._setup_optional_patcher(
                    module_name="sol_estimator.functional_patcher",
                    class_candidates=("FunctionalPatcher", "TorchFunctionalPatcher"),
                    label="functional",
                )
            
            phase_candidates = [
                self.layer_hooks,
                self.graph_tracker,
                self.phase_timer,
                self.optimizer_tracker,
            ] + [p for _, p, _ in self._extra_patchers]
            self._phase_trackers = [
                t for t in phase_candidates
                if t is not None and hasattr(t, "push_phase") and hasattr(t, "pop_phase")
            ]
            self.initialized = True
            log_single_rank(logger, logging.INFO, 
                f"SOL tracking initialized: device={self.layer_hooks.device_spec.name}, dtype={dtype.value}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize SOL: {e}")
            import traceback
            traceback.print_exc()
            return False

    def clear(self):
        """Clear captured operations for next iteration."""
        for tracker in [self.layer_hooks, self.graph_tracker, self.phase_timer, self.optimizer_tracker]:
            if tracker is not None:
                tracker.clear()
        for _, patcher, _ in self._extra_patchers:
            if patcher is not None and hasattr(patcher, "clear"):
                patcher.clear()
    
    def cleanup(self):
        """Cleanup SOL tracking."""
        if self.layer_hooks:
            self.layer_hooks.remove()
        if self.graph_tracker:
            self.graph_tracker.unpatch()
        if self.phase_timer and hasattr(self.phase_timer, 'cleanup'):
            self.phase_timer.cleanup()
        if self.optimizer_tracker and hasattr(self.optimizer_tracker, 'cleanup'):
            self.optimizer_tracker.cleanup()
        for _, patcher, unpatch_fn in self._extra_patchers:
            if unpatch_fn is not None:
                try:
                    unpatch_fn()
                except Exception:
                    pass
        self._extra_patchers = []
        self._phase_trackers = []
        self.layer_hooks = self.graph_tracker = self.phase_timer = self.optimizer_tracker = None
        self.initialized = False

    @contextmanager
    def phase(self, name: str):
        """Context manager to mark operations as belonging to a phase."""
        for t in self._phase_trackers:
            if t is not None:
                t.push_phase(name)
        try:
            yield
        finally:
            for t in self._phase_trackers:
                if t is not None:
                    t.pop_phase()

    def register_optimizer(self, optimizer):
        """Register an optimizer to track."""
        if self.optimizer_tracker:
            self.optimizer_tracker.register_optimizer(optimizer)
            log_single_rank(logger, logging.INFO, f"SOL: Registered optimizer {type(optimizer).__name__}")

    def get_captured_count(self) -> int:
        """Get total number of captured operations."""
        if not self.layer_hooks:
            return 0
        # Try public API first, fall back to private attribute
        if hasattr(self.layer_hooks, 'get_captured_count'):
            return self.layer_hooks.get_captured_count()
        elif hasattr(self.layer_hooks, '_captured_ops'):
            return len(self.layer_hooks._captured_ops)
        return 0

    def _setup_optional_patcher(
        self,
        module_name: str,
        class_candidates: tuple[str, ...],
        label: str,
    ) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            logger.debug(f"SOL {label} tracker import failed: {e}")
            return

        def _first_callable(obj, names):
            for name in names:
                candidate = getattr(obj, name, None)
                if callable(candidate):
                    return candidate
            return None

        # Try class-based patchers first
        for class_name in class_candidates:
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            instance = None
            for kwargs in ({"measure_time": True}, {}):
                try:
                    instance = cls(**kwargs)
                    break
                except TypeError:
                    continue
                except Exception as e:
                    logger.debug(f"SOL {label} tracker init failed: {e}")
                    return
            if instance is None:
                continue
            patch_fn = _first_callable(instance, ("patch", "enable", "start"))
            if patch_fn is None:
                continue
            try:
                patch_fn()
            except Exception as e:
                logger.debug(f"SOL {label} tracker patch failed: {e}")
                return
            unpatch_fn = _first_callable(instance, ("unpatch", "disable", "stop"))
            self._extra_patchers.append((label, instance, unpatch_fn))
            return

        # Fallback to module-level patchers
        patch_fn = _first_callable(module, ("patch", "enable", "start"))
        if patch_fn is None:
            return
        try:
            patch_fn()
        except Exception as e:
            logger.debug(f"SOL {label} module patch failed: {e}")
            return
        unpatch_fn = _first_callable(module, ("unpatch", "disable", "stop"))
        self._extra_patchers.append((label, module, unpatch_fn))

    def _print_report(self, summary: dict, phase_times: dict = None, optimizer_summary: dict = None):
        """Print SOL report."""
        print("\n" + "=" * 80)
        print("SOL Analysis Report")
        print("=" * 80)
        
        if self.layer_hooks:
            print(f"Device: {self.layer_hooks.device_spec.name}, Dtype: {self.layer_hooks.dtype.value}")
        
        total_est = summary.get("total_estimated_time_us", 0)
        total_meas = summary.get("total_measured_time_us", 0)
        eff = (total_est / total_meas * 100) if total_meas > 0 else 0
        
        graph_summary = self.graph_tracker.get_summary() if self.graph_tracker else {}
        graph_eff_ops = graph_summary.get("effective_ops", 0)
        
        print(f"\nSummary:")
        print(f"  Operations: {summary.get('total_ops', 0)} (hooks) + {graph_eff_ops} (CUDA graphs)")
        print(f"  FLOPs: {summary.get('total_flops', 0) / 1e12:.2f} TFLOPS")
        print(f"  Estimated: {total_est / 1000:.2f} ms, Measured: {total_meas / 1000:.2f} ms")
        if total_meas > 0:
            print(f"  Efficiency: {eff:.1f}%")
        
        # Top layers
        by_layer = summary.get("by_layer_type", {})
        if by_layer:
            print(f"\nTop Layers:")
            for lt, stats in sorted(by_layer.items(), key=lambda x: x[1]["estimated_time_us"], reverse=True)[:10]:
                pct = (stats["estimated_time_us"] / total_est * 100) if total_est > 0 else 0
                meas_str = f" [meas: {stats['measured_time_us']/1000:.1f}ms]" if stats["measured_time_us"] > 0 else ""
                print(f"  {lt:35} {stats['count']:4} ops, {stats['estimated_time_us']/1000:7.1f} ms ({pct:4.1f}%){meas_str}")
        
        # Phases
        by_phase = summary.get("by_phase", {})
        if by_phase:
            print(f"\nBy Phase:")
            for phase, stats in sorted(by_phase.items(), key=lambda x: x[1]["estimated_time_us"], reverse=True)[:10]:
                pct = (stats["estimated_time_us"] / total_est * 100) if total_est > 0 else 0
                meas_str = f" [meas: {stats['measured_time_us']/1000:.1f}ms]" if stats["measured_time_us"] > 0 else ""
                print(f"  {phase:45} {stats['count']:4} ops, {stats['estimated_time_us']/1000:7.1f} ms ({pct:4.1f}%){meas_str}")
        
        # CUDA Graphs
        if graph_summary and graph_summary.get("total_replays", 0) > 0:
            print(f"\nCUDA Graphs: {graph_summary['total_graphs_captured']} captured, {graph_summary['total_replays']} replays")
            print(f"  Replay time: {graph_summary['total_replay_time_us']/1000:.1f} ms")
        
        # Optimizer
        if optimizer_summary and optimizer_summary.get("total_steps", 0) > 0:
            print(f"\nOptimizer: {optimizer_summary['total_steps']} steps, {optimizer_summary['total_time_us']/1000:.1f} ms")
        
        # Wall-clock
        if phase_times:
            wall_ms = phase_times.get("top_level_wall_time_us", 0) / 1000
            if wall_ms > 0:
                print(f"\nWall-clock: {wall_ms:.1f} ms (GPU measured: {total_meas/1000:.1f} ms)")
        
        print("=" * 80)

    def log_analysis(self, tb_writer=None, wandb_writer=None, iteration=0):
        """Analyze captured operations and log results."""
        if not self.layer_hooks:
            return
        
        num_ops = self.get_captured_count()
        if num_ops == 0:
            return
            
        try:
            self.layer_hooks.analyze()
            summary = self.layer_hooks.get_summary()
            
            for t in [self.layer_hooks, self.optimizer_tracker, self.graph_tracker]:
                if t and hasattr(t, 'synchronize'):
                    try:
                        t.synchronize()
                    except Exception:
                        pass
            
            phase_times = self.phase_timer.get_summary() if self.phase_timer else {}
            optimizer_summary = self.optimizer_tracker.get_summary() if self.optimizer_tracker else {}
            
            if not summary:
                return
            
            if tb_writer:
                tb_writer.add_scalar("sol/total_ops", summary["total_ops"], iteration)
                tb_writer.add_scalar("sol/total_tflops", summary["total_flops"] / 1e12, iteration)
                tb_writer.add_scalar("sol/estimated_time_ms", summary["total_estimated_time_us"] / 1000, iteration)
                if summary.get("total_measured_time_us"):
                    tb_writer.add_scalar("sol/measured_time_ms", summary["total_measured_time_us"] / 1000, iteration)
                    eff = summary["total_estimated_time_us"] / summary["total_measured_time_us"] * 100
                    tb_writer.add_scalar("sol/efficiency_pct", eff, iteration)
            
            if wandb_writer:
                self._log_to_wandb(wandb_writer, summary, iteration)
            
            args = get_args()
            report_interval = getattr(args, 'rl_sol_report_interval', 100)
            if (iteration % report_interval == 0) or (iteration == 1):
                if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                    self._print_report(summary, phase_times, optimizer_summary)
                    
        except Exception as e:
            logger.warning(f"SOL analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def _log_to_wandb(self, wandb_writer, summary, iteration):
        """Log SOL analysis to WandB."""
        try:
            metrics = {
                "sol/total_ops": summary["total_ops"],
                "sol/total_tflops": summary["total_flops"] / 1e12,
                "sol/estimated_time_ms": summary["total_estimated_time_us"] / 1000,
            }
            if summary.get("total_measured_time_us"):
                metrics["sol/measured_time_ms"] = summary["total_measured_time_us"] / 1000
                if summary["total_measured_time_us"] > 0:
                    metrics["sol/efficiency_pct"] = summary["total_estimated_time_us"] / summary["total_measured_time_us"] * 100
            wandb_writer.log(metrics, iteration)
        except Exception as e:
            logger.debug(f"WandB SOL logging failed: {e}")


_sol_tracker: Optional[SOLTracker] = None

def get_sol_tracker() -> SOLTracker:
    global _sol_tracker
    if _sol_tracker is None:
        _sol_tracker = SOLTracker()
    return _sol_tracker

@contextmanager
def sol_nvtx_range(name: str, log_level: int = 1):
    """Context manager that combines NVTX range with SOL phase tracking.
    
    Args:
        name: Name for the NVTX range and SOL phase
        log_level: Log level (kept for API compatibility, not used by nvtx_range)
    """
    nvtx_range = get_nvtx_range()
    sol_tracker = get_sol_tracker()
    if sol_tracker.initialized:
        with nvtx_range(name):
            with sol_tracker.phase(name):
                yield
    else:
        with nvtx_range(name):
            yield

def log_training_sol(iteration: int, clear: bool = True):
    sol_tracker = get_sol_tracker()
    if not sol_tracker.initialized:
        return
    if sol_tracker.get_captured_count() == 0:
        if clear:
            sol_tracker.clear()
        return
    sol_tracker.log_analysis(tb_writer=get_tensorboard_writer(), wandb_writer=get_wandb_writer(), iteration=iteration)
    if clear:
        sol_tracker.clear()

def initialize_sol(model, args) -> bool:
    return get_sol_tracker().initialize(model, args)

def cleanup_sol():
    get_sol_tracker().cleanup()

def clear_sol_captures():
    get_sol_tracker().clear()

def is_sol_available() -> bool:
    return SOL_ESTIMATOR_AVAILABLE

def register_sol_optimizer(optimizer):
    get_sol_tracker().register_optimizer(optimizer)
