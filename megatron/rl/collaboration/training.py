# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Simple interface for initializing collaborative memory with RL training.
"""

from typing import Optional

import torch

from megatron.core.inference.contexts.collaborative_memory import (
    CollaborativeMemoryConfig,
    CollaborativeMemory,
)


def create_collab_memory_from_args(args) -> Optional[CollaborativeMemory]:
    """
    Create CollaborativeMemory from command line args.
    
    Args:
        args: Parsed arguments
        
    Returns:
        CollaborativeMemory if collaboration enabled, else None
    """
    if not getattr(args, 'enable_collaborative_reasoning', False):
        return None

    config = CollaborativeMemoryConfig(
        hidden_size=args.hidden_size,
        memory_dim=getattr(args, 'collab_memory_dim', None) or args.hidden_size // 4,
        dropout=getattr(args, 'hidden_dropout', 0.1),
    )

    memory = CollaborativeMemory(config)

    # Move to GPU and match dtype
    memory = memory.cuda()
    if getattr(args, 'bf16', False):
        memory = memory.bfloat16()
    elif getattr(args, 'fp16', False):
        memory = memory.half()

    return memory


def get_collab_memory_param_group(memory: CollaborativeMemory, lr_mult: float = 10.0) -> dict:
    """
    Get parameter group for collaborative memory to add to optimizer.
    
    Args:
        memory: The collaborative memory module
        lr_mult: Learning rate multiplier
        
    Returns:
        Parameter group dict
    """
    return {
        'params': list(memory.parameters()),
        'lr_mult': lr_mult,
        'wd_mult': 1.0,
        'name': 'collab_memory',
    }
