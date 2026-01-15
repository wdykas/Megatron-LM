# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Roofline analysis for operations.

This module provides roofline model analysis to determine whether operations
are compute-bound or memory-bound, and estimate their theoretical performance.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .device_specs import DeviceSpec, DataType
from .operation_registry import OpRecord, OpType


class BoundType(Enum):
    """Classification of whether an operation is compute or memory bound."""
    COMPUTE_BOUND = "compute"
    MEMORY_BOUND = "memory"
    COMM_BOUND = "communication"


@dataclass
class RooflineResult:
    """Result of roofline analysis for an operation."""

    record: OpRecord
    bound_type: BoundType
    arithmetic_intensity: float  # FLOPS/byte

    # Theoretical performance
    peak_tflops: float  # Peak achievable TFLOPS
    achievable_tflops: float  # TFLOPS limited by roofline
    sol_percentage: float  # Achievable / Peak as percentage

    # Time estimates
    compute_time_us: float  # Time if compute-bound
    memory_time_us: float  # Time if memory-bound
    comm_time_us: float  # Time for communication
    estimated_time_us: float  # Max of the above (bottleneck)

    def __repr__(self) -> str:
        return (
            f"RooflineResult({self.record.name}: {self.bound_type.value}, "
            f"AI={self.arithmetic_intensity:.1f} FLOPS/byte, "
            f"SOL={self.sol_percentage:.1f}%, "
            f"time={self.estimated_time_us:.1f}us)"
        )


class RooflineAnalyzer:
    """Analyzer for roofline model performance estimation."""

    def __init__(
        self,
        device_spec: DeviceSpec,
        dtype: DataType = DataType.BF16,
        efficiency_factor: float = 0.8,
    ):
        """Initialize the roofline analyzer.

        Args:
            device_spec: Hardware specifications
            dtype: Default data type for analysis
            efficiency_factor: Fraction of theoretical peak achievable (0-1)
                              Accounts for real-world inefficiencies
        """
        self.device_spec = device_spec
        self.dtype = dtype
        self.efficiency_factor = efficiency_factor

    @property
    def peak_tflops(self) -> float:
        """Peak TFLOPS for the configured dtype."""
        return self.device_spec.get_peak_tflops(self.dtype) * self.efficiency_factor

    @property
    def memory_bandwidth_tb_s(self) -> float:
        """Effective memory bandwidth in TB/s."""
        return self.device_spec.hbm_bandwidth_tb_s * self.efficiency_factor

    def analyze(self, record: OpRecord) -> RooflineResult:
        """Analyze a single operation using the roofline model.

        Args:
            record: The operation record to analyze

        Returns:
            RooflineResult with analysis details
        """
        ai = record.arithmetic_intensity
        flops = record.flops
        mem_bytes = record.total_memory_bytes
        comm_bytes = record.comm_bytes

        # Calculate achievable performance
        # Roofline: perf = min(peak, bandwidth * AI)
        memory_limited_tflops = self.memory_bandwidth_tb_s * ai if ai != float("inf") else self.peak_tflops
        achievable_tflops = min(self.peak_tflops, memory_limited_tflops)

        # Calculate time estimates (in microseconds)
        compute_time_us = (flops / (self.peak_tflops * 1e12)) * 1e6 if self.peak_tflops > 0 else 0
        memory_time_us = (mem_bytes / (self.memory_bandwidth_tb_s * 1e12)) * 1e6 if self.memory_bandwidth_tb_s > 0 else 0

        # Communication time (using NVLink bandwidth if available)
        comm_time_us = 0.0
        if comm_bytes > 0 and self.device_spec.nvlink_bandwidth_tb_s:
            comm_bw = self.device_spec.nvlink_bandwidth_tb_s * self.efficiency_factor
            comm_time_us = (comm_bytes / (comm_bw * 1e12)) * 1e6

        # Determine bottleneck
        estimated_time_us = max(compute_time_us, memory_time_us, comm_time_us)

        # Classify bound type: whichever time is largest is the bottleneck
        if comm_time_us >= compute_time_us and comm_time_us >= memory_time_us:
            bound_type = BoundType.COMM_BOUND
        elif compute_time_us >= memory_time_us:
            bound_type = BoundType.COMPUTE_BOUND
        else:
            bound_type = BoundType.MEMORY_BOUND

        # SOL percentage
        if self.peak_tflops > 0:
            sol_percentage = (achievable_tflops / self.peak_tflops) * 100
        else:
            sol_percentage = 0.0

        return RooflineResult(
            record=record,
            bound_type=bound_type,
            arithmetic_intensity=ai,
            peak_tflops=self.peak_tflops,
            achievable_tflops=achievable_tflops,
            sol_percentage=sol_percentage,
            compute_time_us=compute_time_us,
            memory_time_us=memory_time_us,
            comm_time_us=comm_time_us,
            estimated_time_us=estimated_time_us,
        )

    def analyze_batch(self, records: List[OpRecord]) -> List[RooflineResult]:
        """Analyze a batch of operations.

        Args:
            records: List of operation records

        Returns:
            List of RooflineResult for each operation
        """
        return [self.analyze(r) for r in records]

    def summary(self, results: List[RooflineResult]) -> Dict:
        """Generate a summary of roofline analysis results.

        Args:
            results: List of RooflineResult from analyze_batch

        Returns:
            Summary dictionary with aggregated statistics
        """
        if not results:
            return {}

        total_compute_time = sum(r.compute_time_us for r in results)
        total_memory_time = sum(r.memory_time_us for r in results)
        total_comm_time = sum(r.comm_time_us for r in results)
        total_estimated_time = sum(r.estimated_time_us for r in results)

        # Count by bound type
        by_bound = {bt: [] for bt in BoundType}
        for r in results:
            by_bound[r.bound_type].append(r)

        # Find top bottlenecks
        sorted_by_time = sorted(results, key=lambda r: r.estimated_time_us, reverse=True)
        top_bottlenecks = sorted_by_time[:10]

        # Average SOL
        avg_sol = sum(r.sol_percentage for r in results) / len(results)

        return {
            "total_ops": len(results),
            "total_compute_time_us": total_compute_time,
            "total_memory_time_us": total_memory_time,
            "total_comm_time_us": total_comm_time,
            "total_estimated_time_us": total_estimated_time,
            "avg_sol_percentage": avg_sol,
            "by_bound_type": {
                bt.value: {
                    "count": len(ops),
                    "time_us": sum(r.estimated_time_us for r in ops),
                    "fraction": len(ops) / len(results) if results else 0,
                }
                for bt, ops in by_bound.items()
            },
            "top_bottlenecks": [
                {
                    "name": r.record.name,
                    "bound": r.bound_type.value,
                    "time_us": r.estimated_time_us,
                    "sol_pct": r.sol_percentage,
                }
                for r in top_bottlenecks
            ],
        }

    def print_roofline_info(self):
        """Print roofline model parameters for this configuration."""
        print(f"\n{'='*60}")
        print(f"Roofline Model Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device_spec.name}")
        print(f"Data Type: {self.dtype.value}")
        print(f"Efficiency Factor: {self.efficiency_factor:.0%}")
        print(f"")
        print(f"Peak Compute: {self.peak_tflops:.1f} TFLOPS")
        print(f"Peak Memory BW: {self.memory_bandwidth_tb_s:.2f} TB/s")
        print(f"Ridge Point: {self.ridge_point:.1f} FLOPS/byte")
        print(f"")
        print("Operation Classification:")
        print(f"  AI < {self.ridge_point*0.9:.1f}: Memory Bound")
        print(f"  AI > {self.ridge_point*1.1:.1f}: Compute Bound")
        print(f"  Between: Balanced (near knee)")
        print(f"{'='*60}\n")


def estimate_matmul_sol(
    m: int,
    n: int,
    k: int,
    device_spec: DeviceSpec,
    dtype: DataType = DataType.BF16,
    batch_size: int = 1,
) -> Tuple[float, BoundType, float]:
    """Quick estimate of SOL for a matrix multiplication.

    Args:
        m, n, k: Matrix dimensions for A[M,K] @ B[K,N] = C[M,N]
        device_spec: Hardware specifications
        dtype: Data type
        batch_size: Batch size for batched matmul

    Returns:
        Tuple of (SOL percentage, bound type, estimated time in us)
    """
    from .operation_registry import TensorShape, OpRecord

    a_shape = (batch_size, m, k) if batch_size > 1 else (m, k)
    b_shape = (batch_size, k, n) if batch_size > 1 else (k, n)

    inputs = [
        TensorShape(shape=a_shape, dtype=dtype),
        TensorShape(shape=b_shape, dtype=dtype),
    ]
    out_shape = (batch_size, m, n) if batch_size > 1 else (m, n)
    outputs = [TensorShape(shape=out_shape, dtype=dtype)]

    record = OpRecord(
        name="matmul",
        op_type=OpType.MATMUL,
        inputs=inputs,
        outputs=outputs,
    )
    record.compute_flops_and_memory()

    analyzer = RooflineAnalyzer(device_spec, dtype)
    result = analyzer.analyze(record)

    return result.sol_percentage, result.bound_type, result.estimated_time_us
