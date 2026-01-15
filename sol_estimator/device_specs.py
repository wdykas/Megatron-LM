# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Device specifications for SOL estimation.

This module contains hardware specifications for various GPU devices,
including compute throughput (FLOPS) and memory bandwidth characteristics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class DataType(Enum):
    """Supported data types for computation."""

    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"

    @property
    def bytes_per_element(self) -> int:
        """Return the number of bytes per element for this dtype."""
        bytes_map = {
            DataType.FP32: 4,
            DataType.TF32: 4,  # TF32 uses 4 bytes in memory
            DataType.FP16: 2,
            DataType.BF16: 2,
            DataType.FP8_E4M3: 1,
            DataType.FP8_E5M2: 1,
            DataType.INT8: 1,
        }
        return bytes_map.get(self, 4)


@dataclass
class DeviceSpec:
    """Specification of a GPU device's performance characteristics.

    All throughput values are in TFLOPS (10^12 FLOPS).
    All bandwidth values are in TB/s (10^12 bytes/second).
    """

    name: str

    # Peak compute throughput in TFLOPS for different data types
    # These are for tensor core operations (matrix multiply)
    peak_tflops: Dict[DataType, float] = field(default_factory=dict)

    # Peak non-tensor-core FLOPS (for element-wise operations)
    peak_scalar_tflops: Dict[DataType, float] = field(default_factory=dict)

    # HBM (main GPU memory) bandwidth in TB/s
    hbm_bandwidth_tb_s: float = 0.0

    # L2 cache bandwidth in TB/s (if known)
    l2_bandwidth_tb_s: Optional[float] = None

    # L1/shared memory bandwidth in TB/s (if known)
    l1_bandwidth_tb_s: Optional[float] = None

    # NVLink bandwidth per GPU in TB/s (bidirectional)
    nvlink_bandwidth_tb_s: Optional[float] = None

    # PCIe bandwidth in TB/s
    pcie_bandwidth_tb_s: Optional[float] = None

    # Memory capacity in GB
    memory_gb: float = 0.0

    # Number of SMs
    num_sms: int = 0

    def get_peak_tflops(self, dtype: DataType) -> float:
        """Get peak tensor TFLOPS for a given data type."""
        return self.peak_tflops.get(dtype, 0.0)

    def get_peak_scalar_tflops(self, dtype: DataType) -> float:
        """Get peak scalar (non-tensor-core) TFLOPS for a given data type."""
        return self.peak_scalar_tflops.get(dtype, 0.0)

    def arithmetic_intensity_threshold(self, dtype: DataType) -> float:
        """Calculate the arithmetic intensity threshold (FLOPS/byte) for roofline.

        Operations with arithmetic intensity above this threshold are compute-bound,
        those below are memory-bound.

        Returns:
            Threshold in FLOPS/byte
        """
        peak_tflops = self.get_peak_tflops(dtype)
        if peak_tflops == 0 or self.hbm_bandwidth_tb_s == 0:
            return 0.0
        # TFLOPS / (TB/s) = FLOPS/byte
        return peak_tflops / self.hbm_bandwidth_tb_s


# Pre-defined device specifications for common NVIDIA GPUs
# Values are theoretical peaks from NVIDIA specifications

H100_SXM5_SPEC = DeviceSpec(
    name="H100-SXM5-80GB",
    peak_tflops={
        DataType.FP32: 67.0,
        DataType.TF32: 989.0,
        DataType.FP16: 1979.0,
        DataType.BF16: 1979.0,
        DataType.FP8_E4M3: 3958.0,
        DataType.FP8_E5M2: 3958.0,
        DataType.INT8: 3958.0,
    },
    peak_scalar_tflops={
        DataType.FP32: 67.0,
        DataType.FP16: 134.0,
        DataType.BF16: 134.0,
    },
    hbm_bandwidth_tb_s=3.35,
    l2_bandwidth_tb_s=12.0,  # Approximate
    nvlink_bandwidth_tb_s=0.9,  # 900 GB/s per GPU
    pcie_bandwidth_tb_s=0.128,  # PCIe Gen5 x16
    memory_gb=80.0,
    num_sms=132,
)

H100_PCIE_SPEC = DeviceSpec(
    name="H100-PCIe-80GB",
    peak_tflops={
        DataType.FP32: 51.0,
        DataType.TF32: 756.0,
        DataType.FP16: 1513.0,
        DataType.BF16: 1513.0,
        DataType.FP8_E4M3: 3026.0,
        DataType.FP8_E5M2: 3026.0,
        DataType.INT8: 3026.0,
    },
    peak_scalar_tflops={
        DataType.FP32: 51.0,
        DataType.FP16: 102.0,
        DataType.BF16: 102.0,
    },
    hbm_bandwidth_tb_s=2.0,
    nvlink_bandwidth_tb_s=0.6,  # 600 GB/s
    pcie_bandwidth_tb_s=0.128,
    memory_gb=80.0,
    num_sms=114,
)

A100_SXM4_SPEC = DeviceSpec(
    name="A100-SXM4-80GB",
    peak_tflops={
        DataType.FP32: 19.5,
        DataType.TF32: 156.0,
        DataType.FP16: 312.0,
        DataType.BF16: 312.0,
        DataType.INT8: 624.0,
    },
    peak_scalar_tflops={
        DataType.FP32: 19.5,
        DataType.FP16: 78.0,
        DataType.BF16: 78.0,
    },
    hbm_bandwidth_tb_s=2.039,
    l2_bandwidth_tb_s=6.0,  # Approximate
    nvlink_bandwidth_tb_s=0.6,  # 600 GB/s
    pcie_bandwidth_tb_s=0.064,  # PCIe Gen4 x16
    memory_gb=80.0,
    num_sms=108,
)

A100_PCIE_SPEC = DeviceSpec(
    name="A100-PCIe-80GB",
    peak_tflops={
        DataType.FP32: 19.5,
        DataType.TF32: 156.0,
        DataType.FP16: 312.0,
        DataType.BF16: 312.0,
        DataType.INT8: 624.0,
    },
    peak_scalar_tflops={
        DataType.FP32: 19.5,
        DataType.FP16: 78.0,
        DataType.BF16: 78.0,
    },
    hbm_bandwidth_tb_s=2.039,
    pcie_bandwidth_tb_s=0.064,
    memory_gb=80.0,
    num_sms=108,
)

B200_SPEC = DeviceSpec(
    name="B200-192GB",
    peak_tflops={
        DataType.FP32: 90.0,  # Estimated
        DataType.TF32: 2250.0,
        DataType.FP16: 4500.0,
        DataType.BF16: 4500.0,
        DataType.FP8_E4M3: 9000.0,
        DataType.FP8_E5M2: 9000.0,
        DataType.INT8: 9000.0,
    },
    peak_scalar_tflops={
        DataType.FP32: 90.0,
        DataType.FP16: 180.0,
        DataType.BF16: 180.0,
    },
    hbm_bandwidth_tb_s=8.0,
    nvlink_bandwidth_tb_s=1.8,  # 1.8 TB/s per GPU
    memory_gb=192.0,
    num_sms=160,  # Estimated
)

# Mapping of device names to specs
DEVICE_SPECS: Dict[str, DeviceSpec] = {
    "H100-SXM5-80GB": H100_SXM5_SPEC,
    "H100-SXM": H100_SXM5_SPEC,
    "H100": H100_SXM5_SPEC,
    "H100-PCIe-80GB": H100_PCIE_SPEC,
    "H100-PCIE": H100_PCIE_SPEC,
    "A100-SXM4-80GB": A100_SXM4_SPEC,
    "A100-SXM": A100_SXM4_SPEC,
    "A100": A100_SXM4_SPEC,
    "A100-PCIe-80GB": A100_PCIE_SPEC,
    "A100-PCIE": A100_PCIE_SPEC,
    "B200": B200_SPEC,
    "B200-192GB": B200_SPEC,
}


def torch_dtype_to_datatype(dtype) -> DataType:
    """Convert PyTorch dtype to our DataType enum.

    Args:
        dtype: A torch.dtype or string representation

    Returns:
        Corresponding DataType enum value
    """
    # Handle string inputs
    if isinstance(dtype, str):
        dtype_str = dtype.lower()
        str_map = {
            "float32": DataType.FP32,
            "float": DataType.FP32,
            "float16": DataType.FP16,
            "half": DataType.FP16,
            "bfloat16": DataType.BF16,
            "bf16": DataType.BF16,
            "int8": DataType.INT8,
            "fp8": DataType.FP8_E4M3,
            "fp8_e4m3": DataType.FP8_E4M3,
            "fp8_e5m2": DataType.FP8_E5M2,
            "tf32": DataType.TF32,
        }
        return str_map.get(dtype_str, DataType.FP32)

    # Handle torch.dtype
    try:
        import torch

        dtype_map = {
            torch.float32: DataType.FP32,
            torch.float16: DataType.FP16,
            torch.bfloat16: DataType.BF16,
            torch.int8: DataType.INT8,
        }
        # Handle FP8 types if available (PyTorch 2.1+)
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map[torch.float8_e4m3fn] = DataType.FP8_E4M3
        if hasattr(torch, "float8_e5m2"):
            dtype_map[torch.float8_e5m2] = DataType.FP8_E5M2

        return dtype_map.get(dtype, DataType.FP32)
    except ImportError:
        return DataType.FP32


def get_current_device_spec(device=None) -> DeviceSpec:
    """Detect and return the spec for the current GPU device.

    Args:
        device: Optional torch device or device name string.
                If None, uses current CUDA device.

    Returns:
        DeviceSpec for the detected device, or a generic spec if unknown.
    """
    # If a string device name is passed, look it up directly
    if isinstance(device, str):
        for key, spec in DEVICE_SPECS.items():
            if key.upper() in device.upper() or device.upper() in key.upper():
                return spec
        # Return a generic spec
        return DeviceSpec(name=device, peak_tflops={DataType.FP32: 10.0}, hbm_bandwidth_tb_s=1.0)

    try:
        import torch

        if not torch.cuda.is_available():
            # Return a minimal spec for CPU (mainly for testing)
            return DeviceSpec(
                name="CPU",
                peak_tflops={DataType.FP32: 0.1},  # Placeholder
                hbm_bandwidth_tb_s=0.05,  # DDR bandwidth estimate
            )

        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")

        gpu_name = torch.cuda.get_device_name(device)

        # Try to match against known devices
        for key, spec in DEVICE_SPECS.items():
            if key.upper() in gpu_name.upper() or gpu_name.upper() in key.upper():
                return spec

        # If not found, try to extract basic info and create a generic spec
        props = torch.cuda.get_device_properties(device)

        # Estimate TFLOPS based on SM count and clock speed
        # This is a rough approximation
        estimated_fp32_tflops = (
            props.multi_processor_count
            * 128  # FP32 cores per SM (approximate)
            * 2  # FMA = 2 ops
            * props.clock_rate  # in kHz
            / 1e9  # Convert to TFLOPS
        )

        return DeviceSpec(
            name=gpu_name,
            peak_tflops={
                DataType.FP32: estimated_fp32_tflops,
                DataType.FP16: estimated_fp32_tflops * 2,  # Rough estimate
                DataType.BF16: estimated_fp32_tflops * 2,
            },
            hbm_bandwidth_tb_s=props.total_memory / (1024**4),  # Very rough estimate
            memory_gb=props.total_memory / (1024**3),
            num_sms=props.multi_processor_count,
        )
    except ImportError:
        return DeviceSpec(
            name="Unknown",
            peak_tflops={DataType.FP32: 10.0},
            hbm_bandwidth_tb_s=1.0,
        )
