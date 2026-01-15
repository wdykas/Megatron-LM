# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for device_specs.py - GPU specifications."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.device_specs import (
    DataType,
    DeviceSpec,
    get_device_spec,
    get_current_device_spec,
    torch_dtype_to_datatype,
    DTYPE_BYTES,
)


class TestDataType:
    """Tests for DataType enum."""
    
    def test_dtype_bytes(self):
        """Test dtype byte sizes are correct."""
        assert DTYPE_BYTES[DataType.FP32] == 4
        assert DTYPE_BYTES[DataType.FP16] == 2
        assert DTYPE_BYTES[DataType.BF16] == 2
        assert DTYPE_BYTES[DataType.INT8] == 1
        assert DTYPE_BYTES[DataType.FP8_E4M3] == 1
        assert DTYPE_BYTES[DataType.FP8_E5M2] == 1


class TestTorchDtypeConversion:
    """Tests for torch dtype conversion."""
    
    def test_convert_fp32(self):
        torch = pytest.importorskip("torch")
        assert torch_dtype_to_datatype(torch.float32) == DataType.FP32
    
    def test_convert_fp16(self):
        torch = pytest.importorskip("torch")
        assert torch_dtype_to_datatype(torch.float16) == DataType.FP16
    
    def test_convert_bf16(self):
        torch = pytest.importorskip("torch")
        assert torch_dtype_to_datatype(torch.bfloat16) == DataType.BF16
    
    def test_convert_int64(self):
        torch = pytest.importorskip("torch")
        assert torch_dtype_to_datatype(torch.int64) == DataType.INT64
    
    def test_unknown_defaults_to_fp16(self):
        # Non-torch type should default to FP16
        assert torch_dtype_to_datatype("unknown") == DataType.FP16


class TestDeviceSpec:
    """Tests for DeviceSpec class."""
    
    def test_h100_sxm_peak_flops(self):
        """Test H100 SXM peak FLOPS are reasonable."""
        spec = get_device_spec("H100-SXM5-80GB")
        
        # FP32: ~67 TFLOPS
        fp32_flops = spec.get_peak_flops(DataType.FP32)
        assert fp32_flops > 50e12
        assert fp32_flops < 80e12
        
        # FP16 (tensor cores): ~1000 TFLOPS
        fp16_flops = spec.get_peak_flops(DataType.FP16)
        assert fp16_flops > 800e12
        assert fp16_flops < 1200e12
        
        # BF16 should be same as FP16
        bf16_flops = spec.get_peak_flops(DataType.BF16)
        assert bf16_flops == fp16_flops
    
    def test_h100_memory_bandwidth(self):
        """Test H100 memory bandwidth is reasonable."""
        spec = get_device_spec("H100-SXM5-80GB")
        
        # H100 SXM: ~3.35 TB/s
        assert spec.memory_bandwidth > 3e12
        assert spec.memory_bandwidth < 4e12
    
    def test_a100_sxm_specs(self):
        """Test A100 SXM specifications."""
        spec = get_device_spec("A100-SXM4-80GB")
        
        # A100 FP16: ~312 TFLOPS
        fp16_flops = spec.get_peak_flops(DataType.FP16)
        assert fp16_flops > 250e12
        assert fp16_flops < 400e12
        
        # A100 memory: ~2 TB/s
        assert spec.memory_bandwidth > 1.5e12
        assert spec.memory_bandwidth < 2.5e12
    
    def test_unknown_device_returns_default(self):
        """Test unknown device returns H100 as default."""
        spec = get_device_spec("NonExistentGPU")
        assert spec is not None
        assert "H100" in spec.name


class TestGetCurrentDeviceSpec:
    """Tests for get_current_device_spec."""
    
    def test_returns_spec(self):
        """Test it returns a valid spec."""
        spec = get_current_device_spec()
        assert spec is not None
        assert isinstance(spec, DeviceSpec)
        assert spec.memory_bandwidth > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
