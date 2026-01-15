# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Pytest configuration for SOL estimator tests."""

import pytest
import sys
from pathlib import Path

# Add sol_estimator to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    import torch
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 128)
            self.norm = torch.nn.LayerNorm(128)
            self.output = torch.nn.Linear(128, 64)
        
        def forward(self, x):
            x = self.linear(x)
            x = self.norm(x)
            x = self.output(x)
            return x
    
    return SimpleModel()
