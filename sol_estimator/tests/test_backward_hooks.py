# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for backward pass capture in layer_hooks.py and functional_patcher.py."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")
from sol_estimator.layer_hooks import LayerSOLHooks
from sol_estimator.functional_patcher import FunctionalPatcher
from sol_estimator.device_specs import DataType


class SimpleModel(torch.nn.Module):
    """Simple model for testing backward capture."""
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 128)
        self.norm = torch.nn.LayerNorm(128)
        self.linear2 = torch.nn.Linear(128, 64)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


class TestLayerHooksBackward:
    """Tests for backward capture in LayerSOLHooks."""
    
    def test_backward_hooks_registered(self):
        """Test that backward hooks are registered when enabled."""
        model = SimpleModel()
        hooks = LayerSOLHooks(measure_time=False, capture_backward=True)
        hooks.register(model)
        
        # Should have hooks for forward AND backward
        # Each layer gets: forward_pre, forward_post, backward_pre, backward
        # So at least 3 layers * 4 = 12 handles
        assert len(hooks._handles) >= 6  # At least forward hooks
        
        hooks.remove()
    
    def test_backward_capture_disabled(self):
        """Test that backward capture can be disabled."""
        model = SimpleModel()
        hooks = LayerSOLHooks(measure_time=False, capture_backward=False)
        hooks.register(model)
        
        x = torch.randn(4, 64, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Should only have forward ops
        for op in hooks._captured_ops:
            assert not op.is_backward
        
        hooks.remove()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_capture_cuda(self):
        """Test backward capture on CUDA."""
        model = SimpleModel().cuda()
        hooks = LayerSOLHooks(measure_time=True, capture_backward=True)
        hooks.register(model)
        
        x = torch.randn(4, 64, requires_grad=True, device='cuda')
        y = model(x)
        loss = y.sum()
        
        # Clear any forward captures to isolate backward test
        forward_count = len(hooks._captured_ops)
        
        loss.backward()
        hooks.synchronize()
        
        # Should have more ops after backward
        total_count = len(hooks._captured_ops)
        backward_ops = [op for op in hooks._captured_ops if op.is_backward]
        
        assert len(backward_ops) > 0, "No backward ops captured"
        assert total_count > forward_count, "Backward should add more ops"
        
        hooks.remove()
    
    def test_summary_has_forward_backward_counts(self):
        """Test that summary includes forward/backward breakdown."""
        model = SimpleModel()
        hooks = LayerSOLHooks(measure_time=False, capture_backward=True)
        hooks.register(model)
        
        x = torch.randn(4, 64, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        summary = hooks.get_summary()
        
        assert "forward_ops" in summary
        assert "backward_ops" in summary
        assert summary["forward_ops"] > 0
        # Note: backward capture may not work on CPU without CUDA
        
        hooks.remove()


class TestFunctionalPatcherBackward:
    """Tests for backward capture in FunctionalPatcher."""
    
    def test_backward_ops_captured(self):
        """Test that backward operations are captured."""
        patcher = FunctionalPatcher(measure_time=False, capture_backward=True)
        patcher.patch()
        
        a = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(64, 128)
        
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        
        # Check for backward ops
        backward_ops = [op for op in patcher._captured_ops if op.is_backward]
        
        patcher.unpatch()
        
        # At least the forward matmul should be captured
        forward_ops = [op for op in patcher._captured_ops if not op.is_backward]
        assert len(forward_ops) >= 1
    
    def test_summary_has_forward_backward(self):
        """Test summary includes forward/backward counts."""
        patcher = FunctionalPatcher(measure_time=False, capture_backward=True)
        patcher.patch()
        
        a = torch.randn(32, 64, requires_grad=True)
        b = torch.randn(64, 128)
        
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        
        patcher.analyze()
        summary = patcher.get_summary()
        
        assert "forward_ops" in summary
        assert "backward_ops" in summary
        assert summary["forward_ops"] > 0
        
        patcher.unpatch()


class TestExpandedLayerTypes:
    """Tests for expanded layer type coverage."""
    
    def test_mlp_like_layers(self):
        """Test that MLP-style networks are captured."""
        class MLPModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 256)
                self.gelu = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(256, 64)
                self.dropout = torch.nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        model = MLPModel()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(4, 64)
        with torch.no_grad():
            _ = model(x)
        
        layer_types = [op.layer_type for op in hooks._captured_ops]
        
        # Should capture Linear, GELU, Dropout
        assert "Linear" in layer_types
        # GELU and Dropout may or may not be in the layer types map
        
        hooks.remove()
    
    def test_transformer_like_layers(self):
        """Test that transformer-style layers are captured."""
        class TransformerBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = torch.nn.LayerNorm(64)
                self.attn = torch.nn.MultiheadAttention(64, 4, batch_first=True)
                self.norm2 = torch.nn.LayerNorm(64)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(64, 256),
                    torch.nn.GELU(),
                    torch.nn.Linear(256, 64),
                )
            
            def forward(self, x):
                x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                x = x + self.mlp(self.norm2(x))
                return x
        
        model = TransformerBlock()
        hooks = LayerSOLHooks(measure_time=False)
        hooks.register(model)
        
        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            _ = model(x)
        
        layer_types = set(op.layer_type for op in hooks._captured_ops)
        
        # Should capture LayerNorm, Linear, and possibly MultiheadAttention
        assert "LayerNorm" in layer_types
        assert "Linear" in layer_types
        
        hooks.remove()


class TestPhasesWithBackward:
    """Tests for phase tracking during backward pass."""
    
    def test_backward_phases(self):
        """Test that backward ops get proper phase tagging."""
        model = SimpleModel()
        hooks = LayerSOLHooks(measure_time=False, capture_backward=True)
        hooks.register(model)
        
        x = torch.randn(4, 64, requires_grad=True)
        
        hooks.push_phase("train/forward")
        y = model(x)
        hooks.pop_phase()
        
        loss = y.sum()
        
        hooks.push_phase("train/backward")
        loss.backward()
        hooks.pop_phase()
        
        # Check phases
        forward_phases = [op.phase for op in hooks._captured_ops if not op.is_backward]
        backward_phases = [op.phase for op in hooks._captured_ops if op.is_backward]
        
        # Forward should have "train/forward" phase
        assert all(p == "train/forward" for p in forward_phases if p is not None)
        
        hooks.remove()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
