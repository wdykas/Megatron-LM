# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Mixin for adding collaboration to inference contexts.

Usage:
    class MyContext(DynamicInferenceContext, CollaborativeContextMixin):
        def __init__(self, *args, collab_memory=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_collaboration(collab_memory, sync_interval=8)
"""

from typing import List, Optional, Set, Dict

import torch

from .collaborative_memory import CollaborativeMemory


class CollaborativeContextMixin:
    """
    Mixin that adds collaboration to an inference context.
    
    Tracks which requests are in which groups and when to sync.
    """

    def init_collaboration(
        self,
        collab_memory: Optional[CollaborativeMemory] = None,
        sync_interval: int = 8,
    ):
        """Initialize collaboration state."""
        self.collab_memory = collab_memory
        self.sync_interval = sync_interval
        self._collab_groups: Dict[int, Set[int]] = {}
        self._request_to_group: Dict[int, int] = {}
        self._token_counts: Dict[int, int] = {}
        self._in_decode: Set[int] = set()
        self._next_group_id = 0

    @property
    def collaboration_enabled(self) -> bool:
        return self.collab_memory is not None

    def create_collaborative_group(self, request_ids: List[int]) -> int:
        """Create a group of requests that can collaborate."""
        if not self.collaboration_enabled:
            return -1

        group_id = self._next_group_id
        self._next_group_id += 1

        self._collab_groups[group_id] = set(request_ids)
        for rid in request_ids:
            self._request_to_group[rid] = group_id
            self._token_counts[rid] = 0

        self.collab_memory.register_group(group_id, request_ids)
        return group_id

    def mark_decode_start(self, request_id: int):
        """Mark request as starting decode (collaboration only during decode)."""
        self._in_decode.add(request_id)
        self._token_counts[request_id] = 0

    def increment_decode_count(self, request_id: int):
        """Increment decode token count."""
        if request_id in self._token_counts:
            self._token_counts[request_id] += 1

    def should_sync(self, request_id: int) -> bool:
        """Check if should sync now."""
        if not self.collaboration_enabled:
            return False
        if request_id not in self._in_decode:
            return False
        count = self._token_counts.get(request_id, 0)
        return count > 0 and count % self.sync_interval == 0

    def apply_collaboration(
        self,
        request_id: int,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply collaboration to hidden state."""
        if not self.collaboration_enabled:
            return hidden_state
        return self.collab_memory(request_id, hidden_state)

    def remove_request(self, request_id: int):
        """Remove completed request."""
        group_id = self._request_to_group.pop(request_id, None)
        self._token_counts.pop(request_id, None)
        self._in_decode.discard(request_id)

        if group_id is not None and group_id in self._collab_groups:
            self._collab_groups[group_id].discard(request_id)
            if not self._collab_groups[group_id]:
                del self._collab_groups[group_id]
                if self.collab_memory:
                    self.collab_memory.clear_group(group_id)

    def cleanup_collaboration(self):
        """Clean up all state."""
        self._collab_groups.clear()
        self._request_to_group.clear()
        self._token_counts.clear()
        self._in_decode.clear()
        if self.collab_memory:
            self.collab_memory.clear_all()
