# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for sync_tracker.py - CUDA synchronization tracking."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

torch = pytest.importorskip("torch")

from sol_estimator.sync_tracker import SyncTracker, CapturedSync


class TestSyncTrackerBasic:
    """Basic tests for SyncTracker."""
    
    def test_init(self):
        """Test initialization."""
        tracker = SyncTracker()
        assert len(tracker._captured_syncs) == 0
        assert not tracker._patched
    
    def test_patch_unpatch_no_cuda(self):
        """Test patching works even without CUDA."""
        tracker = SyncTracker()
        tracker.patch()
        tracker.unpatch()


class TestSyncTrackerCapture:
    """Tests for sync capture."""
    
    @pytest.fixture
    def tracker(self):
        """Set up tracker."""
        t = SyncTracker()
        t.patch()
        yield t
        t.unpatch()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_capture_device_sync(self, tracker):
        """Test capturing torch.cuda.synchronize()."""
        # Do some GPU work
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        
        # Sync
        torch.cuda.synchronize()
        
        assert len(tracker._captured_syncs) == 1
        assert tracker._captured_syncs[0].sync_type == 'device'
        assert tracker._captured_syncs[0].measured_time_us >= 0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_capture_stream_sync(self, tracker):
        """Test capturing stream.synchronize()."""
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            x = torch.randn(100, 100, device='cuda')
        
        stream.synchronize()
        
        # Should capture the stream sync
        stream_syncs = [s for s in tracker._captured_syncs if s.sync_type == 'stream']
        assert len(stream_syncs) >= 1
    
    def test_phase_tracking(self):
        """Test phase tracking."""
        tracker = SyncTracker()
        
        tracker.push_phase("train")
        assert tracker._get_current_phase() == "train"
        
        tracker.pop_phase()
        assert tracker._get_current_phase() is None
    
    def test_clear(self):
        """Test clearing captured syncs."""
        tracker = SyncTracker()
        
        # Add mock sync
        tracker._captured_syncs.append(CapturedSync(
            sync_type='device',
            measured_time_us=10.0,
        ))
        
        assert len(tracker._captured_syncs) == 1
        tracker.clear()
        assert len(tracker._captured_syncs) == 0
    
    def test_get_summary_empty(self):
        """Test get_summary with no syncs."""
        tracker = SyncTracker()
        summary = tracker.get_summary()
        assert summary == {}
    
    def test_get_summary_with_syncs(self):
        """Test get_summary with syncs."""
        tracker = SyncTracker()
        
        # Add mock syncs
        tracker._captured_syncs.append(CapturedSync(
            sync_type='device',
            measured_time_us=100.0,
            phase='forward',
        ))
        tracker._captured_syncs.append(CapturedSync(
            sync_type='stream',
            measured_time_us=50.0,
            phase='backward',
        ))
        tracker._captured_syncs.append(CapturedSync(
            sync_type='device',
            measured_time_us=75.0,
            phase='forward',
        ))
        
        summary = tracker.get_summary()
        
        assert summary["total_syncs"] == 3
        assert summary["total_time_us"] == 225.0
        
        assert "by_type" in summary
        assert summary["by_type"]["device"]["count"] == 2
        assert summary["by_type"]["device"]["measured_time_us"] == 175.0
        assert summary["by_type"]["stream"]["count"] == 1
        
        assert "by_phase" in summary
        assert summary["by_phase"]["forward"]["count"] == 2
        assert summary["by_phase"]["backward"]["count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
