# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Integration of collaborative memory with the Dynamic Inference Engine.

This manages when and how rollouts collaborate during generation.
Key points:
- Collaboration only during decode phase (not prefill)
- Sync at configurable token intervals
- Only rollouts in same group can collaborate
"""

from typing import Dict, List, Set
import torch

from megatron.core.inference.contexts.collaborative_memory import CollaborativeMemory


class CollaborativeGenerationState:
    """
    Tracks collaborative state during generation.
    
    Attach to your inference engine to manage collaboration.
    
    Usage:
        collab = CollaborativeGenerationState(memory, sync_interval=8)
        
        # When starting rollouts for a prompt
        collab.create_group(request_ids=[0, 1, 2, 3])
        
        # After prefill completes for each request
        collab.mark_decode_start(request_id)
        
        # During each decode step
        collab.increment_token_count(request_id)
        if collab.should_sync(request_id):
            hidden = collab.apply_collaboration(request_id, hidden)
        
        # When request finishes
        collab.remove_request(request_id)
    """

    def __init__(
        self,
        collab_memory: CollaborativeMemory,
        sync_interval: int = 8,
    ):
        self.collab_memory = collab_memory
        self.sync_interval = sync_interval

        self._groups: Dict[int, Set[int]] = {}
        self._request_to_group: Dict[int, int] = {}
        self._token_counts: Dict[int, int] = {}
        self._in_decode: Set[int] = set()
        self._next_group_id = 0

    def create_group(self, request_ids: List[int]) -> int:
        """Create a collaborative group."""
        group_id = self._next_group_id
        self._next_group_id += 1

        self._groups[group_id] = set(request_ids)
        for rid in request_ids:
            self._request_to_group[rid] = group_id
            self._token_counts[rid] = 0

        self.collab_memory.register_group(group_id, request_ids)
        return group_id

    def mark_decode_start(self, request_id: int):
        """Mark request as starting decode phase."""
        self._in_decode.add(request_id)
        self._token_counts[request_id] = 0

    def increment_token_count(self, request_id: int):
        """Increment generated token count."""
        if request_id in self._token_counts:
            self._token_counts[request_id] += 1

    def should_sync(self, request_id: int) -> bool:
        """Check if should sync collaboration now."""
        if request_id not in self._in_decode:
            return False
        count = self._token_counts.get(request_id, 0)
        return count > 0 and count % self.sync_interval == 0

    def apply_collaboration(
        self,
        request_id: int,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply collaborative memory to hidden state."""
        return self.collab_memory(request_id, hidden_state)

    def remove_request(self, request_id: int):
        """Remove completed request."""
        group_id = self._request_to_group.pop(request_id, None)
        self._token_counts.pop(request_id, None)
        self._in_decode.discard(request_id)

        if group_id is not None and group_id in self._groups:
            self._groups[group_id].discard(request_id)
            if not self._groups[group_id]:
                del self._groups[group_id]
                self.collab_memory.clear_group(group_id)

    def cleanup(self):
        """Clean up all state."""
        self._groups.clear()
        self._request_to_group.clear()
        self._token_counts.clear()
        self._in_decode.clear()
        self.collab_memory.clear_all()
