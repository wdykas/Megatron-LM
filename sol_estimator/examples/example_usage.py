#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Example usage of the SOL estimator with PyTorch layer hooks.

This example demonstrates:
1. Creating a simple transformer model
2. Attaching SOL hooks to capture layer shapes
3. Running forward passes and analyzing performance
"""

import torch
import torch.nn as nn
from typing import Optional

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sol_estimator.layer_hooks import LayerSOLHooks, sol_profile
from sol_estimator import DataType


class SimpleTransformerBlock(nn.Module):
    """A simplified transformer block for demonstration."""
    
    def __init__(self, hidden_size: int, ffn_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self-attention projections
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # FFN
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        
        # Norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Simple attention (no masking for demo)
        batch, seq, _ = x.shape
        q = q.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq, self.hidden_size)
        
        x = residual + self.out_proj(out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.fc2(torch.relu(self.fc1(x)))
        x = residual + x
        
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstration."""
    
    def __init__(
        self,
        hidden_size: int = 4096,
        ffn_size: int = 14336,
        num_heads: int = 32,
        num_layers: int = 4,
        vocab_size: int = 32000,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, ffn_size, num_heads)
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def main():
    print("=" * 80)
    print("SOL Estimator - Layer Hooks Example")
    print("=" * 80)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(
        hidden_size=4096,
        ffn_size=14336,
        num_heads=32,
        num_layers=4,
    ).to(device=device, dtype=dtype)
    
    # Create sample input
    batch_size = 4
    seq_len = 2048
    tokens = torch.randint(0, 32000, (batch_size, seq_len), device=device)
    
    print(f"Input shape: {tokens.shape}")
    
    # Option 1: Context manager approach
    print("\n" + "-" * 40)
    print("Using context manager (sol_profile)")
    print("-" * 40)
    
    with sol_profile(model, dtype=DataType.BF16) as hooks:
        with torch.no_grad():
            output = model(tokens)
    
    print(f"Output shape: {output.shape}")
    hooks.print_report(top_n=10)
    
    # Option 2: Manual control
    print("\n" + "-" * 40)
    print("Using manual hook management")
    print("-" * 40)
    
    hooks = LayerSOLHooks(dtype=DataType.BF16)
    hooks.register(model)
    
    # Run a few forward passes
    for i in range(3):
        with torch.no_grad():
            output = model(tokens)
        print(f"Forward pass {i+1}: captured {len(hooks._captured_ops)} total ops")
    
    # Analyze all
    hooks.analyze()
    summary = hooks.get_summary()
    
    print(f"\nTotal ops across 3 passes: {summary['total_ops']}")
    print(f"Total estimated time: {summary['total_estimated_time_us']/1000:.2f} ms")
    
    # Clean up
    hooks.remove()
    
    # Option 3: Disable/enable hooks dynamically
    print("\n" + "-" * 40)
    print("Dynamic enable/disable")
    print("-" * 40)
    
    hooks = LayerSOLHooks(dtype=DataType.BF16)
    hooks.register(model)
    
    hooks.disable()  # Disable capture
    with torch.no_grad():
        output = model(tokens)
    print(f"While disabled: {len(hooks._captured_ops)} ops captured")
    
    hooks.enable()  # Re-enable
    with torch.no_grad():
        output = model(tokens)
    print(f"While enabled: {len(hooks._captured_ops)} ops captured")
    
    hooks.remove()
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
