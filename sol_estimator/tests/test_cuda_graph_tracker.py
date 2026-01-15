# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for cuda_graph_tracker.py - CUDA graph capture and replay tracking."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.cuda_graph_tracker import (
    CUDAGraphTracker,
    CapturedGraph,
    GraphReplayEvent,
    GraphState,
    get_graph_tracker,
)


class TestCUDAGraphTrackerBasic:
    """Basic tests for CUDAGraphTracker."""
    
    def test_init(self):
        """Test initialization."""
        tracker = CUDAGraphTracker()
        assert len(tracker._graphs) == 0
        assert len(tracker._replay_events) == 0
        assert tracker._current_state == GraphState.NONE
    
    def test_capture_mode(self):
        """Test capture mode context manager."""
        tracker = CUDAGraphTracker()
        
        with tracker.capture_mode("test_graph") as graph_id:
            assert tracker.is_capturing
            assert not tracker.is_replaying
            assert tracker._current_graph_name == "test_graph"
        
        assert not tracker.is_capturing
        assert graph_id in tracker._graphs
        assert tracker._graphs[graph_id].name == "test_graph"
    
    def test_replay_mode(self):
        """Test replay mode context manager."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        # First capture a graph
        with tracker.capture_mode("test_graph") as graph_id:
            pass
        
        # Then replay it
        with tracker.replay_mode(graph_id):
            assert tracker.is_replaying
        
        graph = tracker._graphs[graph_id]
        assert graph.replay_count == 1
    
    def test_replay_by_name(self):
        """Test replay by graph name."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        with tracker.capture_mode("my_inference_graph") as graph_id:
            pass
        
        # Replay using name instead of ID
        with tracker.replay_mode("my_inference_graph"):
            pass
        
        assert tracker._graphs[graph_id].replay_count == 1
    
    def test_multiple_replays(self):
        """Test counting multiple replays."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        with tracker.capture_mode("graph") as graph_id:
            pass
        
        for _ in range(5):
            with tracker.replay_mode(graph_id):
                pass
        
        assert tracker._graphs[graph_id].replay_count == 5
    
    def test_phase_tracking(self):
        """Test phase tracking during captures and replays."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        tracker.push_phase("inference")
        with tracker.capture_mode("graph") as graph_id:
            pass
        tracker.pop_phase()
        
        graph = tracker._graphs[graph_id]
        assert graph.capture_phase == "inference"
        
        # Replay in a different phase
        tracker.push_phase("generation")
        with tracker.replay_mode(graph_id):
            pass
        tracker.pop_phase()
        
        assert tracker._replay_events[0].phase == "generation"


class TestCUDAGraphTrackerOps:
    """Tests for registering captured ops with graphs."""
    
    def test_register_captured_ops(self):
        """Test registering ops captured during graph capture."""
        tracker = CUDAGraphTracker()
        
        with tracker.capture_mode("compute_graph"):
            # Simulate that hooks captured some ops during this graph capture
            tracker.register_captured_ops(
                op_count=100,
                flops=int(1e12),
                memory_bytes=int(1e9),
                estimated_time_us=1000.0
            )
        
        graph = list(tracker._graphs.values())[0]
        assert graph.captured_op_count == 100
        assert graph.captured_flops == int(1e12)
        assert graph.captured_memory_bytes == int(1e9)
        assert graph.captured_estimated_time_us == 1000.0


class TestCUDAGraphTrackerSummary:
    """Tests for summary generation."""
    
    def test_empty_summary(self):
        """Test summary when no graphs tracked."""
        tracker = CUDAGraphTracker()
        summary = tracker.get_summary()
        assert summary == {}
    
    def test_summary_with_graphs(self):
        """Test summary with captured graphs and replays."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        with tracker.capture_mode("graph1") as g1:
            tracker.register_captured_ops(50, int(5e11), int(1e9), 500.0)
        
        with tracker.capture_mode("graph2") as g2:
            tracker.register_captured_ops(100, int(1e12), int(2e9), 1000.0)
        
        # Replay each a few times
        for _ in range(3):
            with tracker.replay_mode(g1):
                pass
        
        for _ in range(2):
            with tracker.replay_mode(g2):
                pass
        
        summary = tracker.get_summary()
        
        assert summary["total_graphs_captured"] == 2
        assert summary["total_replays"] == 5
        
        # Effective ops = 50*3 + 100*2 = 150 + 200 = 350
        assert summary["effective_ops"] == 350
        
        # Effective flops = 5e11*3 + 1e12*2 = 1.5e12 + 2e12 = 3.5e12
        assert summary["effective_flops"] == int(3.5e12)
    
    def test_clear(self):
        """Test clearing tracker state."""
        tracker = CUDAGraphTracker(measure_time=False)
        
        with tracker.capture_mode("graph"):
            pass
        
        assert len(tracker._graphs) == 1
        
        tracker.clear()
        
        assert len(tracker._graphs) == 0
        assert len(tracker._replay_events) == 0


class TestGlobalGraphTracker:
    """Tests for global graph tracker instance."""
    
    def test_get_graph_tracker(self):
        """Test getting global instance."""
        tracker = get_graph_tracker()
        assert tracker is not None
        
        # Same instance on subsequent calls
        tracker2 = get_graph_tracker()
        assert tracker is tracker2


class TestCUDAGraphTrackerIntegration:
    """Integration tests with actual CUDA graphs (require CUDA)."""
    
    @pytest.fixture
    def has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @pytest.mark.skipif(True, reason="Requires CUDA graphs")
    def test_actual_cuda_graph(self, has_cuda):
        """Test with actual CUDA graph capture/replay."""
        if not has_cuda:
            pytest.skip("CUDA not available")
        
        import torch
        
        tracker = CUDAGraphTracker()
        tracker.patch()
        
        try:
            # Create a simple graph
            static_input = torch.randn(32, 64, device='cuda')
            static_output = torch.empty(32, 64, device='cuda')
            
            graph = torch.cuda.CUDAGraph()
            
            # Warmup
            for _ in range(3):
                static_output = static_input * 2
            
            # Capture
            with torch.cuda.graph(graph):
                static_output = static_input * 2
            
            # Replay
            for _ in range(5):
                graph.replay()
            
            summary = tracker.get_summary()
            assert summary["total_replays"] >= 5
        finally:
            tracker.unpatch()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
