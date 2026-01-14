# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Collaborative Memory for Parallel Reasoning.

This module enables parallel rollouts to share reasoning states.
Each rollout can write its hidden state to group memory and read
from other rollouts in the same group.

The collaboration is group-based:
- Rollouts are organized into groups (one group per prompt)
- Only rollouts in the same group can see each other
- Each rollout writes its state and reads peers' states
- A learned gate controls how much to use peer information

Future work: Add cross-group pattern lookup using N-gram addressing
(see Engram paper) for sharing common reasoning patterns across prompts.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule


@dataclass
class CollaborativeMemoryConfig:
    """Configuration for collaborative memory."""

    hidden_size: int = 4096
    """Model hidden dimension."""

    memory_dim: int = 1024
    """Dimension for stored states (typically hidden_size // 4)."""

    num_heads: int = 4
    """Number of attention heads for peer aggregation."""

    dropout: float = 0.1
    """Dropout probability."""

    init_std: float = 0.02
    """Weight initialization standard deviation."""


class CollaborativeMemory(MegatronModule):
    """
    Enables collaboration between parallel rollouts via shared memory.
    
    Architecture:
    1. Write projection: hidden_state -> compressed memory_dim representation
    2. Query projection: for attending to peer states
    3. Read projection: memory_dim -> back to hidden_size
    4. Gate: learns when to use peer information
    
    Usage:
        memory = CollaborativeMemory(config)
        
        # Create group for rollouts of same prompt
        memory.register_group(group_id=0, request_ids=[0, 1, 2, 3])
        
        # Each rollout at sync step:
        updated_hidden = memory(request_id=0, hidden_state=current_hidden)
        
        # Cleanup when done
        memory.clear_group(group_id=0)
    """

    def __init__(self, config: CollaborativeMemoryConfig):
        super().__init__(config=None)
        self.config = config

        # Write projection: compress hidden state for storage
        self.write_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.memory_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Query projection: for attending to peers
        self.query_proj = nn.Linear(config.hidden_size, config.memory_dim)

        # Key projection: for peer states
        self.key_proj = nn.Linear(config.memory_dim, config.memory_dim)

        # Read projection: expand back to hidden size
        self.read_proj = nn.Sequential(
            nn.Linear(config.memory_dim, config.hidden_size),
            nn.Dropout(config.dropout),
        )

        # Gate: learn when to use peer information
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )

        # Runtime state
        self._group_states: Dict[int, Dict[int, torch.Tensor]] = {}  # group_id -> {req_id -> state}
        self._request_to_group: Dict[int, int] = {}
        self._active_groups: Dict[int, List[int]] = {}

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def register_group(self, group_id: int, request_ids: List[int]):
        """Register a collaborative group."""
        self._active_groups[group_id] = list(request_ids)
        self._group_states[group_id] = {}
        for req_id in request_ids:
            self._request_to_group[req_id] = group_id

    def clear_group(self, group_id: int):
        """Clear state for a completed group."""
        if group_id in self._active_groups:
            for req_id in self._active_groups[group_id]:
                self._request_to_group.pop(req_id, None)
            del self._active_groups[group_id]
        self._group_states.pop(group_id, None)

    def clear_all(self):
        """Clear all state."""
        self._group_states.clear()
        self._request_to_group.clear()
        self._active_groups.clear()

    def write(self, request_id: int, hidden_state: torch.Tensor):
        """
        Write rollout's hidden state to group memory.
        
        Args:
            request_id: The rollout ID
            hidden_state: [hidden_size] current hidden state
        """
        group_id = self._request_to_group.get(request_id)
        if group_id is None:
            return

        # Compress and store
        compressed = self.write_proj(hidden_state)
        
        if group_id not in self._group_states:
            self._group_states[group_id] = {}
        self._group_states[group_id][request_id] = compressed

    def read(self, request_id: int, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Read aggregated signal from peer rollouts.
        
        Uses attention to weight peer contributions.
        
        Args:
            request_id: The rollout ID
            hidden_state: [hidden_size] current hidden state
            
        Returns:
            [hidden_size] aggregated peer signal
        """
        group_id = self._request_to_group.get(request_id)
        
        # Collect peer states
        peer_states = []
        if group_id is not None and group_id in self._group_states:
            for rid, state in self._group_states[group_id].items():
                if rid != request_id:  # Exclude self
                    peer_states.append(state)

        if not peer_states:
            return torch.zeros(
                self.config.hidden_size,
                dtype=hidden_state.dtype,
                device=hidden_state.device,
            )

        # Stack peers: [num_peers, memory_dim]
        peers = torch.stack(peer_states)

        # Compute attention
        query = self.query_proj(hidden_state)  # [memory_dim]
        keys = self.key_proj(peers)  # [num_peers, memory_dim]

        # Attention scores
        scores = torch.matmul(query, keys.T) / (self.config.memory_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)  # [num_peers]

        # Weighted sum
        aggregated = torch.matmul(weights, peers)  # [memory_dim]

        # Expand back to hidden size
        signal = self.read_proj(aggregated)

        return signal

    def forward(
        self,
        request_id: int,
        hidden_state: torch.Tensor,
        do_write: bool = True,
        do_read: bool = True,
    ) -> torch.Tensor:
        """
        Full forward pass: write state, read from peers, apply gating.
        
        Args:
            request_id: The rollout ID
            hidden_state: [hidden_size] or [seq, hidden_size] current hidden state
            do_write: Whether to write to group memory
            do_read: Whether to read from peers
            
        Returns:
            Updated hidden state
        """
        # Handle sequence dimension - take last position
        if hidden_state.dim() == 2:
            working = hidden_state[-1]
            is_seq = True
        else:
            working = hidden_state
            is_seq = False

        if do_write:
            self.write(request_id, working)

        if do_read:
            peer_signal = self.read(request_id, working)

            # Apply gating
            gate_input = torch.cat([working, peer_signal], dim=-1)
            gate = self.gate(gate_input)
            working = working + gate * peer_signal

        # Restore shape
        if is_seq:
            result = hidden_state.clone()
            result[-1] = working
            return result
        return working

    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_group_size(self, request_id: int) -> int:
        """Get number of rollouts in this request's group."""
        group_id = self._request_to_group.get(request_id)
        if group_id is None:
            return 0
        return len(self._active_groups.get(group_id, []))

    def get_num_peer_states(self, request_id: int) -> int:
        """Get number of peer states available for this request."""
        group_id = self._request_to_group.get(request_id)
        if group_id is None or group_id not in self._group_states:
            return 0
        # Exclude self
        return len(self._group_states[group_id]) - (1 if request_id in self._group_states[group_id] else 0)

