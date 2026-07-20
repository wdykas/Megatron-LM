# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch-invariant helpers for training MoE permutation/unpermutation."""

from typing import Optional

import torch

from megatron.core import parallel_state
from megatron.core.transformer.cuda_graphs import is_graph_capturing


def build_inverse_permutation_map(
    routing_map: torch.Tensor,
    flat_sorted: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_out_tokens: int,
) -> torch.Tensor:
    """Build token/top-k -> permuted-row and expert-id map for BIK unpermute.

    The regular permutation map is row -> token. Batch-invariant unpermute needs
    the inverse ownership model so each output token can read its routed rows and
    add them in a fixed order.
    """
    num_tokens = routing_map.size(0)
    assert isinstance(
        num_out_tokens, int
    ), "batch-invariant graph unpermute requires static num_out_tokens"
    assert num_out_tokens % num_tokens == 0, (
        "batch-invariant graph unpermute expects fixed top-k per token"
    )

    topk = num_out_tokens // num_tokens
    row_ids = torch.arange(num_out_tokens, device=routing_map.device, dtype=torch.long)
    expert_ids = torch.div(flat_sorted, num_tokens, rounding_mode='floor').to(torch.long)
    token_ids = sorted_indices.to(torch.long)

    slots_by_token_expert = routing_map.bool().to(torch.long).cumsum(dim=1) - 1
    row_slots = slots_by_token_expert[token_ids, expert_ids]
    linear_slots = token_ids * topk + row_slots

    inverse_rows = torch.full((num_tokens, topk), -1, device=routing_map.device, dtype=torch.long)
    inverse_experts = torch.full((num_tokens, topk), -1, device=routing_map.device, dtype=torch.long)
    inverse_rows.view(-1).scatter_(0, linear_slots, row_ids)
    inverse_experts.view(-1).scatter_(0, linear_slots, expert_ids)
    return torch.stack((inverse_rows, inverse_experts), dim=0)


def unpermute(
    permuted_tokens: torch.Tensor,
    restore_shape: torch.Size,
    *,
    probs: Optional[torch.Tensor],
    routing_map: torch.Tensor,
    ep_rank_tree: bool,
    inverse_map: Optional[torch.Tensor],
) -> torch.Tensor:
    """Batch-invariant MoE unpermute.

    Accumulation is token-owned. With an inverse map, CUDA graph replay avoids
    data-dependent nonzero/item calls and adds contributions by EP rank then top-k
    slot, matching the inference NVLS rank-ordered combine.
    """
    assert routing_map is not None, "batch-invariant MoE unpermute requires routing_map"

    input_dtype = permuted_tokens.dtype
    output_tokens = torch.zeros(restore_shape, dtype=torch.float32, device=permuted_tokens.device)
    num_experts = routing_map.size(1)
    ep_size = 1
    experts_per_rank = num_experts
    if ep_rank_tree:
        ep_size = parallel_state.get_expert_model_parallel_world_size() or 1
        assert num_experts % ep_size == 0, "batch-invariant MoE expects contiguous EP shards"
        experts_per_rank = num_experts // ep_size

    if inverse_map is not None:
        _unpermute_from_inverse_map(
            output_tokens,
            permuted_tokens,
            probs=probs,
            inverse_map=inverse_map,
            ep_rank_tree=ep_rank_tree,
            ep_size=ep_size,
            experts_per_rank=experts_per_rank,
        )
        return output_tokens.to(dtype=input_dtype)

    assert not is_graph_capturing(), (
        "batch-invariant MoE unpermute requires batch_invariant_inverse_map "
        "during CUDA graph capture"
    )
    _unpermute_from_routing_map(
        output_tokens,
        permuted_tokens,
        probs=probs,
        routing_map=routing_map,
        ep_rank_tree=ep_rank_tree,
        ep_size=ep_size,
        experts_per_rank=experts_per_rank,
    )
    return output_tokens.to(dtype=input_dtype)


def _unpermute_from_inverse_map(
    output_tokens: torch.Tensor,
    permuted_tokens: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    inverse_map: torch.Tensor,
    ep_rank_tree: bool,
    ep_size: int,
    experts_per_rank: int,
) -> None:
    inverse_rows = inverse_map[0]
    inverse_experts = inverse_map[1]
    topk = inverse_rows.size(1)

    for ep_rank in range(ep_size):
        if ep_rank_tree:
            rank_partial = torch.zeros_like(output_tokens)
            start_expert = ep_rank * experts_per_rank
            end_expert = start_expert + experts_per_rank
        else:
            rank_partial = output_tokens

        for k in range(topk):
            row_ids = inverse_rows[:, k]
            expert_ids = inverse_experts[:, k]
            valid_mask = row_ids >= 0
            if ep_rank_tree:
                valid_mask = valid_mask & (expert_ids >= start_expert) & (expert_ids < end_expert)

            safe_rows = row_ids.clamp_min(0)
            chunk = permuted_tokens.index_select(0, safe_rows).to(torch.float32)
            if probs is not None:
                safe_experts = expert_ids.clamp_min(0)
                chunk = chunk * probs.gather(1, safe_experts.unsqueeze(1)).to(torch.float32)
            chunk.masked_fill_(~valid_mask.unsqueeze(-1), 0.0)
            rank_partial += chunk

        if ep_rank_tree:
            output_tokens += rank_partial


def _unpermute_from_routing_map(
    output_tokens: torch.Tensor,
    permuted_tokens: torch.Tensor,
    *,
    probs: Optional[torch.Tensor],
    routing_map: torch.Tensor,
    ep_rank_tree: bool,
    ep_size: int,
    experts_per_rank: int,
) -> None:
    cursor = 0
    for ep_rank in range(ep_size):
        rank_partial = torch.zeros_like(output_tokens) if ep_rank_tree else output_tokens
        start_expert = ep_rank * experts_per_rank
        end_expert = start_expert + experts_per_rank
        for expert_id in range(start_expert, end_expert):
            expert_mask = routing_map[:, expert_id].to(torch.bool)
            n_tokens = int(expert_mask.sum().item())
            if n_tokens == 0:
                continue
            token_ids = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)
            next_cursor = cursor + n_tokens

            chunk = permuted_tokens[cursor:next_cursor].to(torch.float32)
            if probs is not None:
                chunk = chunk * probs[token_ids, expert_id].to(torch.float32).unsqueeze(-1)
            rank_partial.index_add_(0, token_ids, chunk)
            cursor = next_cursor
        if ep_rank_tree:
            output_tokens += rank_partial
