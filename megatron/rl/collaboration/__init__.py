# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Collaborative Reasoning for Megatron-RL.

Enables parallel rollouts to share hidden states within groups.

Quick start:
    from megatron.rl.collaboration import create_collab_memory_from_args
    
    # In setup:
    collab_memory = create_collab_memory_from_args(args)
    
    # In generation:
    from megatron.rl.collaboration import CollaborativeGenerationState
    collab = CollaborativeGenerationState(collab_memory, sync_interval=8)
    collab.create_group(request_ids=[0, 1, 2, 3])
"""

from megatron.core.inference.contexts.collaborative_memory import (
    CollaborativeMemoryConfig,
    CollaborativeMemory,
)

from .training import create_collab_memory_from_args, get_collab_memory_param_group

from .engine_integration import CollaborativeGenerationState

from .collaborative_agent import CollaborativeRolloutGenerator, CollaborativeConfig

__all__ = [
    "CollaborativeMemoryConfig",
    "CollaborativeMemory",
    "create_collab_memory_from_args",
    "get_collab_memory_param_group",
    "CollaborativeGenerationState",
    "CollaborativeRolloutGenerator",
    "CollaborativeConfig",
]
