# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Collaborative Rollout Generator Agent.

Wraps an existing agent to coordinate collaborative memory.
The actual collaboration happens in the inference engine via engine_integration.py.
This agent just manages group lifecycle.
"""

from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from ..agent.api import GroupedRolloutGenerator, GroupedRolloutRequest, Rollout

from .engine_integration import CollaborativeGenerationState


@dataclass
class CollaborativeConfig:
    """Configuration for collaborative rollout generation."""
    sync_interval: int = 8
    warmup_tokens: int = 0


class CollaborativeRolloutGenerator(GroupedRolloutGenerator):
    """
    Wraps an agent to add collaborative reasoning between rollouts.
    
    The actual collaboration happens in the inference layer via
    CollaborativeGenerationState. This agent coordinates:
    - Creating groups when generating rollouts for a prompt
    - Cleaning up groups when done
    """

    def __init__(
        self,
        base_agent: GroupedRolloutGenerator,
        collab_state: Optional[CollaborativeGenerationState] = None,
        config: Optional[CollaborativeConfig] = None,
    ):
        self.base_agent = base_agent
        self.collab_state = collab_state
        self.config = config or CollaborativeConfig()

    async def group_rollout(
        self,
        request: GroupedRolloutRequest,
    ) -> List[Rollout]:
        """Generate a group of collaborative rollouts."""
        # If no collaboration, just use base agent
        if self.collab_state is None:
            return await self.base_agent.group_rollout(request)

        # Create group for these rollouts
        # Note: The actual request_ids come from the inference engine
        # This is just for coordination
        group_id = self.collab_state.create_group(
            list(range(request.rollouts_per_group))
        )

        try:
            return await self.base_agent.group_rollout(request)
        finally:
            # Cleanup happens via remove_request as each completes
            pass

    async def get_grouped_rollouts(
        self,
        request: GroupedRolloutRequest,
    ) -> AsyncIterator[List[Rollout]]:
        """Async generator for grouped rollouts."""
        async for group in self.base_agent.get_grouped_rollouts(request):
            yield group
