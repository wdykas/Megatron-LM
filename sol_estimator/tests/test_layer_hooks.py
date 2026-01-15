# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for layer_hooks.py - nn.Module operation capture."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")
from sol_estimator.layer_hooks import LayerSOLHooks, CapturedOp
from sol_estimator.device_specs import DataType
from sol_estimator.operation_registry import OpType


class SimpleLinear(torch.nn.Module):
    """Simple model for testing."""
    def __init__(self, in_features=256, out_features=512):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)


class MultiLayerModel(torch.nn.Module):
    """Model with multiple layers."""
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(256, 512)
        self.norm = torch.nn.LayerNorm(512)
        self.linear2 = torch.nn.Linear(512, 256)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class TestLayerSOLHooksBasic:
    """Basic tests for LayerSOLHooks."""
    
    def test_init(self):
        """Test initialization."""
        hooks = LayerSOLHooks(dtype=DataType.FP16)
        assert hooks.dtype == DataType.FP16
        assert len(hooks._captured_ops) == 0
    
    def test_register_and_remove(self):
        """Test hook registration and removal."""
        model = SimpleLinear()
        hooks = LayerSOLHooks()
        
        hooks.register(model)
        assert len(hooks._hook_handles) > 0
        
        hooks.remove()
        assert len(hooks._hook_handles) == 0


class TestLayerSOLHooksCapture:
    """Tests for operation capture."""
    
    @pytest.fixture
    def setup_model(self):
        """Set up model and hooks."""
        model = SimpleLinear()
        hooks = LayerSOLHooks(dtype=DataType.FP32, measure_time=False)
        hooks.register(model)
        yield model, hooks
        hooks.remove()
    
    def test_capture_linear(self, setup_model):
        """Test capturing Linear layer operations."""
        model, hooks = setup_model
        x = torch.randn(8, 256)
        
        with torch.no_grad():
            _ = model(x)
        
        assert len(hooks._captured_ops) == 1
        op = hooks._captured_ops[0]
        assert op.layer_type == "Linear"
        assert op.op_type == OpType.MATMUL
        assert op.input_shapes[0] == (8, 256)
    
    def test_capture_multiple_layers(self):
        """Test capturing multiple layer types."""
        model = MultiLayerModel()
        hooks = LayerSOLHooks(dtype=DataType.FP32, measure_time=False)
        hooks.register(model)
        
        x = torch.randn(8, 256)
        with torch.no_grad():
            _ = model(x)
        
        hooks.remove()
        
        # Should capture: linear1, norm, linear2
        assert len(hooks._captured_ops) >= 3
        
        layer_types = [op.layer_type for op in hooks._captured_ops]
        assert "Linear" in layer_types
        assert "LayerNorm" in layer_types
    
    def test_clear(self):
        """Test clearing captured operations."""
        model = SimpleLinear()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(8, 256)
        with torch.no_grad():
            _ = model(x)
        
        assert len(hooks._captured_ops) > 0
        hooks.clear()
        assert len(hooks._captured_ops) == 0
        
        hooks.remove()


class TestLayerSOLHooksPhases:
    """Tests for phase tracking."""
    
    def test_phase_tracking(self):
        """Test that phases are tracked correctly."""
        model = SimpleLinear()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(8, 256)
        
        hooks.push_phase("test/forward")
        with torch.no_grad():
            _ = model(x)
        hooks.pop_phase()
        
        assert len(hooks._captured_ops) == 1
        assert hooks._captured_ops[0].phase == "test/forward"
        
        hooks.remove()
    
    def test_nested_phases(self):
        """Test nested phase tracking."""
        model = MultiLayerModel()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(4, 256)
        
        hooks.push_phase("outer")
        hooks.push_phase("outer/inner")
        with torch.no_grad():
            _ = model(x)
        hooks.pop_phase()
        hooks.pop_phase()
        
        # All ops should be tagged with innermost phase
        for op in hooks._captured_ops:
            assert op.phase == "outer/inner"
        
        hooks.remove()


class TestLayerSOLHooksAnalysis:
    """Tests for analysis and summary."""
    
    def test_analyze(self):
        """Test analyze returns results."""
        model = SimpleLinear()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(8, 256)
        with torch.no_grad():
            _ = model(x)
        
        results = hooks.analyze()
        assert len(results) == 1
        assert results[0].estimated_time_us > 0
        
        hooks.remove()
    
    def test_get_summary(self):
        """Test get_summary returns correct structure."""
        model = MultiLayerModel()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(8, 256)
        with torch.no_grad():
            _ = model(x)
        
        hooks.analyze()
        summary = hooks.get_summary()
        
        assert "total_ops" in summary
        assert "total_flops" in summary
        assert "by_layer_type" in summary
        assert "by_layer_instance" in summary
        assert "by_phase" in summary
        
        assert summary["total_ops"] >= 3
        assert "Linear" in summary["by_layer_type"]
        
        hooks.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
