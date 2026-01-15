# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for functional_patcher.py - torch.* and F.* operation capture."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from sol_estimator.functional_patcher import FunctionalPatcher
from sol_estimator.device_specs import DataType


class TestFunctionalPatcherBasic:
    """Basic tests for FunctionalPatcher."""
    
    def test_init(self):
        """Test initialization."""
        patcher = FunctionalPatcher(dtype=DataType.FP16)
        assert patcher.dtype == DataType.FP16
        assert len(patcher._captured_ops) == 0
        assert not patcher._patched
    
    def test_patch_unpatch(self):
        """Test patching and unpatching."""
        patcher = FunctionalPatcher()
        
        patcher.patch()
        assert patcher._patched
        assert len(patcher._original_funcs) > 0
        
        patcher.unpatch()
        assert not patcher._patched


class TestFunctionalPatcherCapture:
    """Tests for operation capture."""
    
    @pytest.fixture
    def patcher(self):
        """Set up patcher."""
        p = FunctionalPatcher(dtype=DataType.FP32, measure_time=False)
        p.patch()
        yield p
        p.unpatch()
    
    def test_capture_matmul(self, patcher):
        """Test capturing torch.matmul."""
        a = torch.randn(8, 64, 128)
        b = torch.randn(8, 128, 256)
        
        _ = torch.matmul(a, b)
        
        assert len(patcher._captured_ops) == 1
        op = patcher._captured_ops[0]
        assert "matmul" in op.name.lower()
        assert op.input_shapes[0] == (8, 64, 128)
    
    def test_capture_linear(self, patcher):
        """Test capturing F.linear."""
        x = torch.randn(8, 256)
        weight = torch.randn(512, 256)
        
        _ = F.linear(x, weight)
        
        assert len(patcher._captured_ops) == 1
        assert "linear" in patcher._captured_ops[0].name.lower()
    
    def test_capture_embedding(self, patcher):
        """Test capturing F.embedding."""
        indices = torch.randint(0, 1000, (8, 64))
        weight = torch.randn(1000, 256)
        
        _ = F.embedding(indices, weight)
        
        assert len(patcher._captured_ops) == 1
        assert "embedding" in patcher._captured_ops[0].name.lower()
    
    def test_capture_softmax(self, patcher):
        """Test capturing F.softmax."""
        x = torch.randn(8, 16, 1024)
        
        _ = F.softmax(x, dim=-1)
        
        assert len(patcher._captured_ops) == 1
        assert "softmax" in patcher._captured_ops[0].name.lower()
    
    def test_capture_log_softmax(self, patcher):
        """Test capturing F.log_softmax."""
        x = torch.randn(8, 16, 1024)
        
        _ = F.log_softmax(x, dim=-1)
        
        assert len(patcher._captured_ops) == 1
        assert "log_softmax" in patcher._captured_ops[0].name.lower()
    
    def test_capture_layer_norm(self, patcher):
        """Test capturing F.layer_norm."""
        x = torch.randn(8, 1024, 768)
        
        _ = F.layer_norm(x, (768,))
        
        assert len(patcher._captured_ops) == 1
        assert "layer_norm" in patcher._captured_ops[0].name.lower()
    
    def test_clear(self, patcher):
        """Test clearing captured operations."""
        a = torch.randn(8, 64)
        b = torch.randn(64, 128)
        _ = torch.matmul(a, b)
        
        assert len(patcher._captured_ops) > 0
        patcher.clear()
        assert len(patcher._captured_ops) == 0


class TestFunctionalPatcherPhases:
    """Tests for phase tracking."""
    
    def test_phase_tracking(self):
        """Test that phases are tracked correctly."""
        patcher = FunctionalPatcher(measure_time=False)
        patcher.patch()
        
        patcher.push_phase("test/compute")
        a = torch.randn(8, 64)
        b = torch.randn(64, 128)
        _ = torch.matmul(a, b)
        patcher.pop_phase()
        
        assert len(patcher._captured_ops) == 1
        assert patcher._captured_ops[0].phase == "test/compute"
        
        patcher.unpatch()


class TestFunctionalPatcherAnalysis:
    """Tests for analysis."""
    
    def test_analyze(self):
        """Test analyze returns results."""
        patcher = FunctionalPatcher(measure_time=False)
        patcher.patch()
        
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)
        _ = torch.matmul(a, b)
        
        results = patcher.analyze()
        assert len(results) == 1
        assert results[0].estimated_time_us > 0
        
        patcher.unpatch()
    
    def test_get_summary(self):
        """Test get_summary returns correct structure."""
        patcher = FunctionalPatcher(measure_time=False)
        patcher.patch()
        
        # Do a few operations
        a = torch.randn(64, 128)
        b = torch.randn(128, 256)
        _ = torch.matmul(a, b)
        _ = F.softmax(a, dim=-1)
        
        patcher.analyze()
        summary = patcher.get_summary()
        
        assert "total_ops" in summary
        assert "by_function" in summary
        assert "by_phase" in summary
        assert summary["total_ops"] == 2
        
        patcher.unpatch()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
