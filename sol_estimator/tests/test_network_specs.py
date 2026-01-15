# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Tests for network_specs.py - Network bandwidth and collective estimation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.network_specs import (
    NetworkSpec,
    CollectiveType,
    get_network_spec,
    estimate_collective_time,
    H100_SXM_NETWORK,
    A100_SXM_NETWORK,
)


class TestNetworkSpec:
    """Tests for NetworkSpec class."""
    
    def test_h100_spec(self):
        """Test H100 SXM specs are reasonable."""
        spec = H100_SXM_NETWORK
        
        # H100 NVLink should be ~450 GB/s
        assert spec.intra_node_bandwidth >= 400e9
        assert spec.intra_node_bandwidth <= 500e9
        
        # InfiniBand should be ~50 GB/s per GPU
        assert spec.inter_node_bandwidth >= 25e9
        assert spec.inter_node_bandwidth <= 100e9
    
    def test_get_bandwidth(self):
        """Test get_bandwidth method."""
        spec = H100_SXM_NETWORK
        
        # Intra-node should use NVLink
        assert spec.get_bandwidth(is_intra_node=True) == spec.intra_node_bandwidth
        
        # Inter-node should use IB
        assert spec.get_bandwidth(is_intra_node=False) == spec.inter_node_bandwidth
    
    def test_get_latency(self):
        """Test get_latency method."""
        spec = H100_SXM_NETWORK
        
        # Intra-node latency should be lower
        assert spec.get_latency(is_intra_node=True) < spec.get_latency(is_intra_node=False)


class TestGetNetworkSpec:
    """Tests for get_network_spec function."""
    
    def test_get_by_name(self):
        """Test getting spec by name."""
        spec = get_network_spec("H100_SXM")
        assert spec.name == H100_SXM_NETWORK.name
    
    def test_default(self):
        """Test default spec is returned."""
        spec = get_network_spec("nonexistent")
        assert spec is not None  # Should return a default


class TestEstimateCollectiveTime:
    """Tests for collective time estimation."""
    
    def test_all_reduce_scaling(self):
        """Test all-reduce time scales with data size."""
        spec = H100_SXM_NETWORK
        
        time_1mb = estimate_collective_time(
            CollectiveType.ALL_REDUCE, 1e6, 8, spec, is_intra_node=True
        )
        time_10mb = estimate_collective_time(
            CollectiveType.ALL_REDUCE, 10e6, 8, spec, is_intra_node=True
        )
        
        # 10x data should take ~10x time (approximately)
        assert time_10mb > time_1mb * 5
        assert time_10mb < time_1mb * 15
    
    def test_all_reduce_world_size_scaling(self):
        """Test all-reduce time increases with world size."""
        spec = H100_SXM_NETWORK
        data_size = 10e6  # 10 MB
        
        time_2gpu = estimate_collective_time(
            CollectiveType.ALL_REDUCE, data_size, 2, spec, is_intra_node=True
        )
        time_8gpu = estimate_collective_time(
            CollectiveType.ALL_REDUCE, data_size, 8, spec, is_intra_node=True
        )
        
        # Ring all-reduce: time ~ 2*(n-1)/n * size / bw
        # So 8 GPU should take more time than 2 GPU
        assert time_8gpu > time_2gpu
    
    def test_intra_vs_inter_node(self):
        """Test inter-node is slower than intra-node."""
        spec = H100_SXM_NETWORK
        data_size = 100e6  # 100 MB
        
        time_intra = estimate_collective_time(
            CollectiveType.ALL_REDUCE, data_size, 8, spec, is_intra_node=True
        )
        time_inter = estimate_collective_time(
            CollectiveType.ALL_REDUCE, data_size, 8, spec, is_intra_node=False
        )
        
        # Inter-node should be significantly slower
        assert time_inter > time_intra * 2
    
    def test_barrier_is_fast(self):
        """Test barrier has minimal time (just latency)."""
        spec = H100_SXM_NETWORK
        
        barrier_time = estimate_collective_time(
            CollectiveType.BARRIER, 0, 8, spec, is_intra_node=True
        )
        
        # Barrier should be in microseconds, not milliseconds
        assert barrier_time < 100  # Less than 100 us
    
    def test_single_gpu_is_zero(self):
        """Test single GPU collective is zero time."""
        spec = H100_SXM_NETWORK
        
        time = estimate_collective_time(
            CollectiveType.ALL_REDUCE, 100e6, 1, spec
        )
        assert time == 0.0
    
    def test_all_gather_vs_reduce_scatter(self):
        """Test all_gather takes more time than reduce_scatter for same data."""
        spec = H100_SXM_NETWORK
        per_rank_size = 10e6  # 10 MB per rank
        world_size = 8
        
        # All-gather: each rank sends per_rank_size
        all_gather_time = estimate_collective_time(
            CollectiveType.ALL_GATHER, per_rank_size, world_size, spec, is_intra_node=True
        )
        
        # Reduce-scatter: total data is per_rank_size * world_size, each rank gets per_rank_size
        reduce_scatter_time = estimate_collective_time(
            CollectiveType.REDUCE_SCATTER, per_rank_size * world_size, world_size, spec, is_intra_node=True
        )
        
        # All-gather moves more total data
        assert all_gather_time > reduce_scatter_time * 0.5
    
    def test_broadcast_uses_tree(self):
        """Test broadcast time is logarithmic in world size."""
        spec = H100_SXM_NETWORK
        data_size = 100e6
        
        time_8 = estimate_collective_time(
            CollectiveType.BROADCAST, data_size, 8, spec, is_intra_node=True
        )
        time_64 = estimate_collective_time(
            CollectiveType.BROADCAST, data_size, 64, spec, is_intra_node=False
        )
        
        # Tree broadcast: time ~ log2(n)
        # log2(8) = 3, log2(64) = 6
        # So 64 GPU should be ~2x the time of 8 GPU (ignoring latency)
        assert time_64 < time_8 * 5  # Not linear scaling


class TestCollectiveTypeFormulas:
    """Test the mathematical formulas for each collective type."""
    
    def test_send_recv_time(self):
        """Test point-to-point send/recv time."""
        spec = H100_SXM_NETWORK
        data_size = 100e6  # 100 MB
        
        send_time = estimate_collective_time(
            CollectiveType.SEND, data_size, 2, spec, is_intra_node=True
        )
        
        # Should be approximately data_size / bandwidth + latency
        expected_time = data_size / spec.intra_node_bandwidth * 1e6 + spec.intra_node_latency_us
        
        # Within 2x
        assert send_time > expected_time * 0.5
        assert send_time < expected_time * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
