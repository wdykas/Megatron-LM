# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for the collaborative memory module.
"""

import pytest
import torch

from megatron.core.inference.contexts.collaborative_memory import (
    CollaborativeMemoryConfig,
    CollaborativeMemory,
)


class TestCollaborativeMemoryConfig:
    """Tests for CollaborativeMemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CollaborativeMemoryConfig()
        assert config.hidden_size == 4096
        assert config.memory_dim == 1024
        assert config.dropout == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CollaborativeMemoryConfig(
            hidden_size=2048,
            memory_dim=512,
        )
        assert config.hidden_size == 2048
        assert config.memory_dim == 512


class TestCollaborativeMemory:
    """Tests for CollaborativeMemory."""

    @pytest.fixture
    def config(self):
        return CollaborativeMemoryConfig(
            hidden_size=256,
            memory_dim=64,
        )

    @pytest.fixture
    def memory(self, config):
        return CollaborativeMemory(config)

    def test_register_and_clear_group(self, memory):
        """Test group registration and cleanup."""
        memory.register_group(group_id=0, request_ids=[0, 1, 2, 3])

        assert 0 in memory._active_groups
        assert memory._active_groups[0] == [0, 1, 2, 3]
        assert memory._request_to_group[0] == 0
        assert memory._request_to_group[1] == 0

        memory.clear_group(0)
        assert 0 not in memory._active_groups
        assert 0 not in memory._request_to_group

    def test_write_and_read(self, memory, config):
        """Test writing and reading from memory."""
        memory.register_group(group_id=0, request_ids=[0, 1, 2])

        hidden0 = torch.randn(config.hidden_size)
        memory.write(request_id=0, hidden_state=hidden0)

        hidden1 = torch.randn(config.hidden_size)
        memory.write(request_id=1, hidden_state=hidden1)

        hidden2 = torch.randn(config.hidden_size)
        collab_signal = memory.read(request_id=2, hidden_state=hidden2)

        assert collab_signal.shape == (config.hidden_size,)
        assert not torch.isnan(collab_signal).any()

    def test_read_no_peers(self, memory, config):
        """Test reading when no peers have written."""
        memory.register_group(group_id=0, request_ids=[0, 1])

        hidden = torch.randn(config.hidden_size)
        collab_signal = memory.read(request_id=0, hidden_state=hidden)

        assert collab_signal.shape == (config.hidden_size,)
        assert (collab_signal == 0).all()

    def test_forward_full(self, memory, config):
        """Test full forward pass."""
        memory.register_group(group_id=0, request_ids=[0, 1, 2])

        hidden0 = torch.randn(config.hidden_size)
        out0 = memory(request_id=0, hidden_state=hidden0)
        assert out0.shape == (config.hidden_size,)

        hidden1 = torch.randn(config.hidden_size)
        out1 = memory(request_id=1, hidden_state=hidden1)
        assert out1.shape == (config.hidden_size,)

        hidden2 = torch.randn(config.hidden_size)
        out2 = memory(request_id=2, hidden_state=hidden2)
        assert out2.shape == (config.hidden_size,)

    def test_isolation_between_groups(self, memory, config):
        """Test that different groups dont share state."""
        memory.register_group(group_id=0, request_ids=[0, 1])
        memory.register_group(group_id=1, request_ids=[2, 3])

        memory.write(request_id=0, hidden_state=torch.randn(config.hidden_size))

        hidden = torch.randn(config.hidden_size)
        collab_signal = memory.read(request_id=2, hidden_state=hidden)

        # Group 1 (request 2) should not see group 0's state
        assert (collab_signal == 0).all()

    def test_gradient_flow(self, memory, config):
        """Test that gradients flow through the module."""
        memory.register_group(group_id=0, request_ids=[0, 1])

        hidden0 = torch.randn(config.hidden_size, requires_grad=True)
        memory.write(request_id=0, hidden_state=hidden0)

        hidden1 = torch.randn(config.hidden_size, requires_grad=True)
        out = memory(request_id=1, hidden_state=hidden1)

        loss = out.sum()
        loss.backward()

        has_grad = False
        for param in memory.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to memory parameters"

    def test_num_params(self, memory):
        """Test parameter count."""
        num_params = memory.get_num_params()
        assert num_params > 0

    def test_clear_all(self, memory, config):
        """Test clearing all state."""
        memory.register_group(group_id=0, request_ids=[0, 1])
        memory.register_group(group_id=1, request_ids=[2, 3])
        memory.write(request_id=0, hidden_state=torch.randn(config.hidden_size))

        memory.clear_all()

        assert len(memory._active_groups) == 0
        assert len(memory._request_to_group) == 0
        assert len(memory._group_states) == 0

    def test_get_group_size(self, memory):
        """Test getting group size."""
        memory.register_group(group_id=0, request_ids=[0, 1, 2, 3])
        assert memory.get_group_size(0) == 4
        assert memory.get_group_size(1) == 4
        assert memory.get_group_size(99) == 0  # unregistered

    def test_get_num_peer_states(self, memory, config):
        """Test getting number of peer states."""
        memory.register_group(group_id=0, request_ids=[0, 1, 2])

        assert memory.get_num_peer_states(0) == 0

        memory.write(0, torch.randn(config.hidden_size))
        assert memory.get_num_peer_states(1) == 1

        memory.write(1, torch.randn(config.hidden_size))
        assert memory.get_num_peer_states(2) == 2


class TestCollaborativeMemoryGPU:
    """GPU-specific tests for CollaborativeMemory."""

    @pytest.fixture
    def config(self):
        return CollaborativeMemoryConfig(hidden_size=256, memory_dim=64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self, config):
        """Test forward pass on CUDA."""
        memory = CollaborativeMemory(config).cuda()
        memory.register_group(group_id=0, request_ids=[0, 1])

        hidden0 = torch.randn(config.hidden_size, device='cuda')
        out0 = memory(request_id=0, hidden_state=hidden0)
        assert out0.device.type == 'cuda'

        hidden1 = torch.randn(config.hidden_size, device='cuda')
        out1 = memory(request_id=1, hidden_state=hidden1)
        assert out1.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bf16(self, config):
        """Test with bfloat16 dtype."""
        memory = CollaborativeMemory(config).cuda().bfloat16()
        memory.register_group(group_id=0, request_ids=[0, 1])

        hidden = torch.randn(config.hidden_size, device='cuda', dtype=torch.bfloat16)
        out = memory(request_id=0, hidden_state=hidden)
        assert out.dtype == torch.bfloat16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

