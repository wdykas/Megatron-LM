# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for collective_patcher.py - torch.distributed operation capture."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")

from sol_estimator.collective_patcher import CollectivePatcher, CapturedCollective
from sol_estimator.network_specs import CollectiveType


class TestCollectivePatcherBasic:
    """Basic tests for CollectivePatcher."""
    
    def test_init(self):
        """Test initialization."""
        patcher = CollectivePatcher()
        assert len(patcher._captured_ops) == 0
        assert not patcher._patched
        assert patcher.network_spec is not None
    
    def test_patch_unpatch_without_dist(self):
        """Test patching works even without distributed initialized."""
        patcher = CollectivePatcher()
        
        # Should not crash if distributed is not initialized
        patcher.patch()
        patcher.unpatch()


class TestCollectivePatcherCapture:
    """Tests for operation capture (requires distributed to be initialized)."""
    
    @pytest.fixture
    def mock_dist(self, monkeypatch):
        """Mock torch.distributed for testing."""
        class MockGroup:
            pass
        
        class MockDist:
            def __init__(self):
                self._world_size = 4
                self._all_reduce_called = False
                
            def is_initialized(self):
                return True
            
            def get_world_size(self, group=None):
                return self._world_size
            
            def all_reduce(self, tensor, op=None, group=None, async_op=False):
                self._all_reduce_called = True
                return None
            
            def broadcast(self, tensor, src, group=None, async_op=False):
                return None
            
            def barrier(self, group=None, async_op=False):
                return None
        
        mock = MockDist()
        if hasattr(torch, 'distributed'):
            # Monkeypatch individual functions
            monkeypatch.setattr(torch.distributed, 'is_initialized', mock.is_initialized)
            monkeypatch.setattr(torch.distributed, 'get_world_size', mock.get_world_size)
        
        return mock
    
    def test_captured_collective_name(self):
        """Test CapturedCollective name property."""
        captured = CapturedCollective(
            op_type=CollectiveType.ALL_REDUCE,
            tensor_size_bytes=1024,
            world_size=4,
            group_name="dp",
            async_op=False,
        )
        assert captured.name == "all_reduce(dp)"
    
    def test_phase_tracking(self):
        """Test phase tracking."""
        patcher = CollectivePatcher()
        
        patcher.push_phase("train/backward")
        assert patcher._get_current_phase() == "train/backward"
        
        patcher.push_phase("train/backward/grad_sync")
        assert patcher._get_current_phase() == "train/backward/grad_sync"
        
        patcher.pop_phase()
        assert patcher._get_current_phase() == "train/backward"
        
        patcher.pop_phase()
        assert patcher._get_current_phase() is None
    
    def test_register_group(self):
        """Test registering process groups."""
        patcher = CollectivePatcher()
        
        mock_group = object()
        patcher.register_group(mock_group, "dp")
        
        assert patcher._get_group_name(mock_group) == "dp"
        assert patcher._get_group_name(None) == "world"
    
    def test_clear(self):
        """Test clearing captured operations."""
        patcher = CollectivePatcher()
        
        # Manually add a captured op
        patcher._captured_ops.append(CapturedCollective(
            op_type=CollectiveType.ALL_REDUCE,
            tensor_size_bytes=1024,
            world_size=4,
            group_name="world",
            async_op=False,
        ))
        
        assert len(patcher._captured_ops) == 1
        patcher.clear()
        assert len(patcher._captured_ops) == 0
    
    def test_get_summary_empty(self):
        """Test get_summary with no operations."""
        patcher = CollectivePatcher()
        summary = patcher.get_summary()
        assert summary == {}
    
    def test_get_summary_with_ops(self):
        """Test get_summary with operations."""
        patcher = CollectivePatcher()
        
        # Add mock captured ops
        patcher._captured_ops.append(CapturedCollective(
            op_type=CollectiveType.ALL_REDUCE,
            tensor_size_bytes=1024 * 1024,
            world_size=4,
            group_name="dp",
            async_op=False,
            estimated_time_us=100.0,
            measured_time_us=120.0,
            phase="backward",
        ))
        patcher._captured_ops.append(CapturedCollective(
            op_type=CollectiveType.ALL_GATHER,
            tensor_size_bytes=512 * 1024,
            world_size=4,
            group_name="tp",
            async_op=False,
            estimated_time_us=50.0,
            measured_time_us=60.0,
            phase="forward",
        ))
        
        summary = patcher.get_summary()
        
        assert summary["total_ops"] == 2
        assert summary["total_bytes"] == 1024 * 1024 + 512 * 1024
        assert "by_collective_type" in summary
        assert "by_group" in summary
        assert "by_phase" in summary
        
        assert "all_reduce" in summary["by_collective_type"]
        assert "all_gather" in summary["by_collective_type"]
        assert "dp" in summary["by_group"]
        assert "tp" in summary["by_group"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
