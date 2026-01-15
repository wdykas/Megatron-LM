# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for roofline.py - Roofline analysis and time estimation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.roofline import RooflineAnalyzer, BoundType
from sol_estimator.device_specs import DeviceSpec, DataType, get_device_spec
from sol_estimator.operation_registry import OpType, OpRecord, TensorShape


class TestRooflineAnalyzerBasic:
    """Basic tests for RooflineAnalyzer."""
    
    def test_init_with_default(self):
        """Test initialization with default device."""
        analyzer = RooflineAnalyzer()
        assert analyzer.device_spec is not None
    
    def test_init_with_spec(self):
        """Test initialization with specific device spec."""
        spec = get_device_spec("A100-SXM4-80GB")
        analyzer = RooflineAnalyzer(device_spec=spec)
        assert analyzer.device_spec.name == "A100-SXM4-80GB"
    
    def test_ridge_point(self):
        """Test ridge point calculation."""
        spec = get_device_spec("H100-SXM5-80GB")
        analyzer = RooflineAnalyzer(device_spec=spec)
        
        # Ridge point = peak_flops / memory_bandwidth
        # H100: ~1000 TFLOPS FP16, ~3 TB/s = ~333 FLOP/byte
        # Should be in reasonable range
        ridge = spec.get_peak_flops(DataType.FP16) / spec.memory_bandwidth
        assert ridge > 100
        assert ridge < 1000


class TestRooflineAnalyzerBound:
    """Tests for compute/memory bound classification."""
    
    @pytest.fixture
    def h100_analyzer(self):
        """H100 analyzer fixture."""
        spec = get_device_spec("H100-SXM5-80GB")
        return RooflineAnalyzer(device_spec=spec)
    
    def test_large_matmul_compute_bound(self, h100_analyzer):
        """Test large matmul is compute-bound."""
        record = OpRecord(
            name="large_mm",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        assert result.bound_type == BoundType.COMPUTE
    
    def test_small_matmul_memory_bound(self, h100_analyzer):
        """Test small matmul is memory-bound."""
        record = OpRecord(
            name="small_mm",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(64, 64), dtype=DataType.FP16),
                TensorShape(shape=(64, 64), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(64, 64), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        assert result.bound_type == BoundType.MEMORY
    
    def test_embedding_memory_bound(self, h100_analyzer):
        """Test embedding is always memory-bound."""
        record = OpRecord(
            name="embed",
            op_type=OpType.EMBEDDING,
            inputs=[
                TensorShape(shape=(8, 1024), dtype=DataType.INT64),
                TensorShape(shape=(50000, 768), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(8, 1024, 768), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        assert result.bound_type == BoundType.MEMORY


class TestRooflineAnalyzerTimeEstimation:
    """Tests for time estimation."""
    
    @pytest.fixture
    def h100_analyzer(self):
        """H100 analyzer fixture."""
        spec = get_device_spec("H100-SXM5-80GB")
        return RooflineAnalyzer(device_spec=spec)
    
    def test_estimate_time_compute_bound(self, h100_analyzer):
        """Test time estimation for compute-bound operation."""
        record = OpRecord(
            name="test",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(4096, 4096), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        
        # FLOPs = 2 * 4096^3 = ~137 TFLOPs
        # H100 FP16 = ~1000 TFLOPS
        # Time ~ 137 / 1000 = ~0.14 ms = 140 us
        assert result.estimated_time_us > 50
        assert result.estimated_time_us < 500
    
    def test_estimate_time_memory_bound(self, h100_analyzer):
        """Test time estimation for memory-bound operation."""
        record = OpRecord(
            name="test",
            op_type=OpType.EMBEDDING,
            inputs=[
                TensorShape(shape=(8, 1024), dtype=DataType.INT64),
                TensorShape(shape=(50000, 768), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(8, 1024, 768), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        
        # Time should be based on memory bandwidth
        assert result.estimated_time_us > 0
    
    def test_result_has_all_fields(self, h100_analyzer):
        """Test that RooflineResult has all expected fields."""
        record = OpRecord(
            name="test",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(512, 512), dtype=DataType.FP16),
                TensorShape(shape=(512, 512), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(512, 512), dtype=DataType.FP16),
            ],
        )
        record.compute_flops_and_memory()
        
        result = h100_analyzer.analyze(record)
        
        assert hasattr(result, 'record')
        assert hasattr(result, 'bound_type')
        assert hasattr(result, 'estimated_time_us')
        assert hasattr(result, 'peak_achievable_flops')
        assert hasattr(result, 'arithmetic_intensity')


class TestRooflineAnalyzerDifferentDtypes:
    """Test roofline analysis with different data types."""
    
    def test_fp32_slower_than_fp16(self):
        """Test FP32 is slower than FP16 for same size."""
        spec = get_device_spec("H100-SXM5-80GB")
        analyzer_fp16 = RooflineAnalyzer(device_spec=spec, dtype=DataType.FP16)
        analyzer_fp32 = RooflineAnalyzer(device_spec=spec, dtype=DataType.FP32)
        
        record_fp16 = OpRecord(
            name="test",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(1024, 1024), dtype=DataType.FP16),
                TensorShape(shape=(1024, 1024), dtype=DataType.FP16),
            ],
            outputs=[
                TensorShape(shape=(1024, 1024), dtype=DataType.FP16),
            ],
        )
        record_fp16.compute_flops_and_memory()
        
        record_fp32 = OpRecord(
            name="test",
            op_type=OpType.MATMUL,
            inputs=[
                TensorShape(shape=(1024, 1024), dtype=DataType.FP32),
                TensorShape(shape=(1024, 1024), dtype=DataType.FP32),
            ],
            outputs=[
                TensorShape(shape=(1024, 1024), dtype=DataType.FP32),
            ],
        )
        record_fp32.compute_flops_and_memory()
        
        result_fp16 = analyzer_fp16.analyze(record_fp16)
        result_fp32 = analyzer_fp32.analyze(record_fp32)
        
        # FP16 should be faster (higher peak TFLOPS)
        assert result_fp16.estimated_time_us < result_fp32.estimated_time_us


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
