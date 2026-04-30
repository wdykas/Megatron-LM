# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
CUDA-graph-compatible token dispatcher for inference.

This dispatcher is only used during CUDA-graphed inference iterations. It replaces
AlltoAll with AllGather/ReduceScatter for token exchange, keeping all metadata
GPU-resident to avoid host synchronizations that would break CUDA graph capture.

Supports latency-optimized NVLS collectives (multimem all-gather/reduce-scatter)
on Hopper+ GPUs with BF16, with automatic fallback to NCCL.
"""

from typing import List, Optional

import torch
import torch.distributed as dist

from megatron.core.inference.communication.torch_symm_triton import (
    are_tensors_nvls_eligible,
    multimem_all_gather_fused,
    multimem_all_reduce,
    multimem_reduce_scatter,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.moe.token_dispatcher import MoEAllGatherTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_pg_rank


class InferenceCUDAGraphTokenDispatcher(MoEAllGatherTokenDispatcher):
    """
    CUDA-graph-compatible AllGather token dispatcher for inference.

    Only used during CUDA-graphed inference iterations. Swapped in by
    MoELayer.set_inference_cuda_graphed_iteration() before graph capture
    and swapped out by MoELayer.unset_inference_cuda_graphed_iteration() after.

    Key features:
    - AllGather/ReduceScatter instead of AlltoAll for CUDA graph compatibility
    - GPU-resident metadata (no host synchronization)
    - NVLS collectives on Hopper+ with automatic NCCL fallback
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the InferenceCUDAGraphTokenDispatcher.

        Args:
            num_local_experts: Number of experts on this rank.
            local_expert_indices: Global indices of experts on this rank.
            config: Transformer configuration.
            pg_collection: Process group collection for distributed ops.
        """
        super().__init__(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=pg_collection,
        )
        self.topk = config.moe_router_topk

        self.triton_nvls_kernels_allowed = not self.config.inference_disable_triton_nvls_kernels

        # Variant-B runtime flag. Set to True by HybridStack when this MoE
        # layer sits inside an attention-bounded segment after another in-
        # segment layer. Tells token_dispatch the input is already gathered
        # ([global_tokens, hidden]) so it should skip the AG, and tells
        # token_combine to return the full global view rather than the
        # local slice.
        #
        # MTP / speculative-decoding note: under
        # ``num_speculative_tokens = K``, the leading dim of hidden_states
        # is ``G × (K + 1)`` instead of ``G``. Both AR-then-slice (default
        # combine) and AR-return-global (Variant B combine) operate on
        # the leading dim and divide by ``ep_size``, which holds because
        # ``(K + 1)`` cleanly divides each rank's slice when ``G %
        # ep_size == 0``. See docs/user-guide/features/attention_bounded_segments.md.
        self._segment_input_is_global = False
        # Variant-B Opt-1: stash for the local-slice shared-experts output.
        # The MoE layer sets these before combine; the combine path folds
        # the slice into the AR input at [start:end] so the all-reduce
        # naturally distributes shared additions across ranks. Cleared
        # after use so subsequent layers don't pick up stale state.
        self._shared_local_slice = None
        self._shared_local_start = 0
        self._shared_local_end = 0

    def _maybe_allocate_ag_buffers(
        self, routing_map: torch.Tensor, probs: torch.Tensor, hidden_states: torch.Tensor
    ) -> dict:
        """Allocate a single symmetric memory output buffer for fused all-gather.

        Creates one contiguous symmetric memory buffer sized for the gathered
        (global) routing_map, probs, and hidden_states, then returns sliced views
        into it. This allows a single fused NVLS all-gather kernel to write all
        three outputs in one launch.

        Args:
            routing_map (torch.Tensor): Local routing map, shape [local_tokens, topk].
                Boolean or integer tensor mapping each token to its selected experts.
            probs (torch.Tensor): Local routing probabilities, shape [local_tokens, topk].
                Normalized weights for each token's selected experts.
            hidden_states (torch.Tensor): Local hidden states, shape [local_tokens, hidden_dim].

        Returns:
            dict: A dictionary with the following keys:
                - "handle": Symmetric memory handle for NVLS ops, or None if
                  symmetric memory is unavailable.
                - "routing_map": Raw byte view for the gathered routing map output.
                - "routing_map_offset": Byte offset of routing_map within the buffer.
                - "probs": Raw byte view for the gathered probs output.
                - "probs_offset": Byte offset of probs within the buffer.
                - "hidden_states": Raw byte view for the gathered hidden states output.
                - "hidden_states_offset": Byte offset of hidden_states within the buffer.
                When allocation fails, all tensor views are None and offsets are 0.
        """
        _NONE = {
            "handle": None,
            "routing_map": None,
            "routing_map_offset": 0,
            "probs": None,
            "probs_offset": 0,
            "hidden_states": None,
            "hidden_states_offset": 0,
        }

        local_tokens = probs.size(0)
        global_tokens = local_tokens * self.ep_size
        topk = probs.size(-1)
        hidden_dim = hidden_states.size(-1)

        result = SymmetricMemoryManager.get_buffer(
            "ep", process_group=self.ep_group
        ).maybe_get_tensors(
            [
                (global_tokens * topk, routing_map.dtype),
                (global_tokens * topk, probs.dtype),
                (global_tokens * hidden_dim, hidden_states.dtype),
            ]
        )

        if result["handle"] is None:
            return _NONE

        (rm_buf, rm_off), (p_buf, p_off), (hs_buf, hs_off) = result["tensors"]
        return {
            "handle": result["handle"],
            "routing_map": rm_buf,
            "routing_map_offset": rm_off,
            "probs": p_buf,
            "probs_offset": p_off,
            "hidden_states": hs_buf,
            "hidden_states_offset": hs_off,
        }

    def _maybe_allocate_rs_buffer(self, x: torch.Tensor) -> dict:
        """Allocate a symmetric memory buffer for reduce-scatter input.

        The buffer has the same shape and dtype as x so that x can be copied
        into it before the NVLS reduce-scatter kernel.

        Args:
            x (torch.Tensor): The global hidden states to be reduce-scattered,
                shape [global_tokens, hidden_dim].

        Returns:
            dict: A dictionary with keys "handle" (symmetric memory handle, or
                None if unavailable) and "tensor" (the allocated buffer, or None).
        """
        symm_mem_buffer = SymmetricMemoryManager.get_buffer(
            "ep", process_group=self.ep_group
        ).maybe_get_tensor(list(x.size()), dtype=x.dtype)
        return symm_mem_buffer

    def token_dispatch(self, hidden_states, probs):
        """Gathers tokens from all EP ranks using AllGather.

        Performs all-gather on routing_map (stored in self.routing_map), probs,
        and hidden_states so that every rank holds the full global view.
        Uses latency-optimized fused NVLS multimem_all_gather on Hopper+ GPUs
        with BF16 when symmetric memory is available. Falls back to NCCL otherwise.

        Args:
            hidden_states (torch.Tensor): Local hidden states,
                shape [local_tokens, hidden_dim].
            probs (torch.Tensor): Local routing probabilities,
                shape [local_tokens, topk]. Normalized weights for each token's
                selected experts.

        Returns:
            tuple: (hidden_states, probs) gathered across all EP ranks.
                - hidden_states (torch.Tensor): Shape [global_tokens, hidden_dim].
                - probs (torch.Tensor): Shape [global_tokens, topk].
                Also updates self.routing_map in-place to the gathered
                shape [global_tokens, topk].
        """
        if self.ep_size == 1:
            return hidden_states, probs

        # Variant-B fast path. The previous in-segment MoE's combine left
        # ``hidden_states`` already in [global_tokens, hidden] form via
        # multimem all-reduce, so the standard all-gather is redundant.
        # We still need the routing-map and probs to be global. Since the
        # router was just run on the global hidden_states by every rank
        # (deterministically — same input → same router output), the
        # routing_map and probs are already global on every rank too, so
        # we can reuse them as-is.
        if self._segment_input_is_global:
            return hidden_states, probs

        # 1. Check inputs only: if inputs are 16-byte divisible,
        #  outputs (world_size * input) are too.
        nvls_eligible = self.triton_nvls_kernels_allowed and are_tensors_nvls_eligible(
            hidden_states, probs, self.routing_map
        )
        ag_buffers = None

        if nvls_eligible:
            # 2. Now attempt to allocate symmetric memory buffers for
            # all-gather outputs. If allocation fails, fallback to NCCL.
            ag_buffers = self._maybe_allocate_ag_buffers(self.routing_map, probs, hidden_states)

        # 3. Can use NVLS if eligible and buffers allocated successfully (handle is not None)
        can_use_nvls = nvls_eligible and ag_buffers["handle"] is not None

        if can_use_nvls:
            # Capture shapes for reshaping after all-gather
            # Output shape: [local_tokens * ep_size, dim]
            local_tokens = probs.size(0)
            global_tokens = local_tokens * self.ep_size
            topk = probs.size(1)
            hidden_dim = hidden_states.size(1)
            routing_map_dtype = self.routing_map.dtype
            probs_dtype = probs.dtype
            hidden_dtype = hidden_states.dtype

            # Fused NVLS all-gather: single kernel launch + single barrier for all 3 tensors
            multimem_all_gather_fused(
                ag_buffers["routing_map"].view(
                    torch.bfloat16
                ),  # .view does not change the underlying data
                self.routing_map.view(torch.bfloat16),
                ag_buffers["routing_map_offset"],
                ag_buffers["probs"].view(torch.bfloat16),
                probs.view(torch.bfloat16),
                ag_buffers["probs_offset"],
                ag_buffers["hidden_states"].view(torch.bfloat16),
                hidden_states.view(torch.bfloat16),
                ag_buffers["hidden_states_offset"],
                ag_buffers["handle"],
            )
            self.routing_map = (
                ag_buffers["routing_map"].view(routing_map_dtype).view(global_tokens, topk)
            )
            probs = ag_buffers["probs"].view(probs_dtype).view(global_tokens, topk)
            hidden_states = (
                ag_buffers["hidden_states"].view(hidden_dtype).view(global_tokens, hidden_dim)
            )
        else:
            # Fallback to NCCL for all tensors
            with torch.no_grad():
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )

        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """Pass-through: returns inputs directly without permutation.

        Unlike the training dispatcher, this does not permute tokens or compute
        tokens_per_expert. The downstream InferenceGroupedMLP (FlashInfer /
        CUTLASS fused MoE kernel) operates directly on the routing map stored
        in self.routing_map.

        Args:
            hidden_states (torch.Tensor): Gathered hidden states,
                shape [global_tokens, hidden_dim].
            probs (torch.Tensor): Gathered routing probabilities,
                shape [global_tokens, topk].

        Returns:
            tuple: (hidden_states, tokens_per_expert, probs) where
                tokens_per_expert is always None.
        """
        return hidden_states, None, probs

    def combine_preprocess(self, expert_output):
        """Pass-through: InferenceGroupedMLP already produces unpermuted output.

        No unpermutation is needed because dispatch_postprocess did not permute
        the tokens in the first place.

        Args:
            expert_output (torch.Tensor): Output from InferenceGroupedMLP,
                shape [global_tokens, hidden_dim].

        Returns:
            torch.Tensor: The input tensor unchanged.
        """
        return expert_output

    def token_combine(self, hidden_states):
        """Combines expert outputs across EP ranks using Reduce-Scatter.

        Reduces the global expert output (summing contributions from each rank)
        and scatters the result so each rank receives its local token slice.
        Uses latency-optimized NVLS multimem_reduce_scatter on Hopper+ GPUs
        with BF16 when symmetric memory is available. Falls back to NCCL otherwise.

        Args:
            hidden_states (torch.Tensor): Combined expert output after routing
                weights have been applied, shape [global_tokens, hidden_dim].

        Returns:
            torch.Tensor: Local slice of the reduced output,
                shape [local_tokens, hidden_dim] where
                local_tokens = global_tokens // ep_size.
        """
        if self.ep_size == 1:
            return hidden_states

        # Variant-B combine: when ABS+current_segment_owner is set AND
        # the dispatcher's _segment_input_is_global flag is set (managed
        # by HybridStack at segment entry), return the full global view.
        # Otherwise fall back to the AR-then-slice (bit-equivalent to
        # default) path.
        if (
            getattr(self.config, "enable_attention_bounded_segments", False)
            and getattr(self.config, "moe_combine_destination_policy", "original_owner")
            == "current_segment_owner"
        ):
            if self._segment_input_is_global:
                return self._token_combine_via_all_reduce_global(hidden_states)
            return self._token_combine_via_all_reduce(hidden_states)

        # Compute output shape first — check NVLS eligibility on the output,
        # since if the smaller output is 16-byte divisible, the input is too.
        output_shape = list(hidden_states.size())
        output_shape[0] = hidden_states.size(0) // self.ep_size
        output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # Check output only: if output is 16-byte divisible, input (world_size * output) is too.
        nvls_eligible = (
            self.triton_nvls_kernels_allowed
            and output.dtype in (torch.bfloat16, torch.float32)
            and are_tensors_nvls_eligible(output)
        )
        rs_buffer = None

        if nvls_eligible:
            rs_buffer = self._maybe_allocate_rs_buffer(hidden_states)

        can_use_nvls = nvls_eligible and rs_buffer["handle"] is not None

        if can_use_nvls:
            # Copy input to symmetric memory for reduce-scatter
            rs_buffer["tensor"].copy_(hidden_states)

            # Use latency-optimized NVLS reduce-scatter
            multimem_reduce_scatter(output, rs_buffer["tensor"], rs_buffer["handle"])
            return output.to(torch.bfloat16)
        else:
            # Fallback to NCCL
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
            return hidden_states.to(torch.bfloat16)

    def _token_combine_via_all_reduce_global(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Variant-B combine: all-reduce, return the full **global** tensor.

        The standard combine path does reduce-scatter so each rank ends up
        with [local_tokens, hidden]. Variant B keeps the global view across
        the entire segment so the next MoE's all-gather can be skipped:
        we all-reduce in place and return [global_tokens, hidden] on every
        rank.

        Implementations, in priority order:
        1. ``torch.ops.symm_mem.multimem_all_reduce_`` — PyTorch's NVLS
           multimem all-reduce. ~2x faster than reduce-scatter at small
           sizes on GB200 (microbench), and crucially is one collective
           instead of the RS+AG pair used by the standard path.
        2. NCCL ring all-reduce — fallback when symmetric memory or NVLS
           is unavailable. Same total bytes as RS+AG via ring algorithm.
        """
        nvls_eligible = (
            self.triton_nvls_kernels_allowed
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and are_tensors_nvls_eligible(hidden_states)
        )

        # Variant-B Opt-1: if a shared-experts local slice has been stashed
        # by the upstream MoE layer, fold it into the AR input at this
        # rank's slice range. The AR then sums each rank's contribution
        # into the matching global row, so the combined output already
        # includes shared without any extra collective. ``shared_local``
        # is cleared after consumption.
        shared_local = self._shared_local_slice
        shared_start = self._shared_local_start
        shared_end = self._shared_local_end
        self._shared_local_slice = None

        result = None
        if nvls_eligible:
            ar_buffer = self._maybe_allocate_rs_buffer(hidden_states)
            if ar_buffer["handle"] is not None:
                ar_buffer["tensor"].copy_(hidden_states)
                if shared_local is not None:
                    ar_buffer["tensor"][shared_start:shared_end].add_(shared_local)
                native_op = getattr(
                    getattr(torch.ops, "symm_mem", None), "multimem_all_reduce_", None
                )
                if native_op is not None:
                    sm = ar_buffer["tensor"].view(hidden_states.shape)
                    native_op(sm, "sum", self.tp_ep_group.group_name)
                    result = sm.to(torch.bfloat16)
                else:
                    output = torch.empty_like(hidden_states)
                    multimem_all_reduce(output, ar_buffer["tensor"], ar_buffer["handle"])
                    result = output.to(torch.bfloat16)

        if result is None:
            reduced = hidden_states.contiguous().clone()
            if shared_local is not None:
                reduced[shared_start:shared_end].add_(shared_local)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.tp_ep_group)
            result = reduced.to(torch.bfloat16)

        # Variant-B: rewrite ``self.hidden_shape`` so that the upcoming
        # ``combine_postprocess`` reshapes to the global view rather than
        # the local pre-AG view captured by ``dispatch_preprocess``. The
        # original first dim is local_tokens; replace it with global.
        old_shape = list(self.hidden_shape)
        if old_shape and old_shape[0] * self.ep_size == result.shape[0]:
            old_shape[0] = result.shape[0]
            self.hidden_shape = torch.Size(old_shape)
        return result

    def _token_combine_via_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Equivalent of token_combine but using AllReduce + local slice.

        Same arithmetic as the reduce-scatter path (sum across EP ranks,
        return this rank's slice) but reaches it via an all-reduce so the
        global view is materialized intermediately. Used by the
        attention-bounded-segments combine policy — the materialized
        global view is what lets a follow-up MoE layer skip its
        all-gather on this rank's slice.

        Implementations, in priority order:
        1. ``torch.ops.symm_mem.multimem_all_reduce_`` — PyTorch's native
           NVLS multimem all-reduce. Empirically ~2× faster than the
           multimem reduce-scatter baseline on GB200 at small sizes
           (e.g. ~66 µs vs ~124 µs for 128×4096 bf16). Fast path.
        2. NVLS multimem all-reduce via the Triton kernel in
           ``torch_symm_triton.collectives`` — works alongside the rest
           of the inference dispatcher's symmetric memory pool. Roughly
           parity with NCCL ring AR; kept as a fallback if the native
           op is unavailable.
        3. NCCL ring all-reduce + local slice — final fallback when no
           symmetric memory is available.

        Returns the local slice (same shape and contract as the standard
        reduce-scatter path) so callers don't need to know which
        collective ran.
        """
        rank = get_pg_rank(self.tp_ep_group)
        local_tokens = hidden_states.size(0) // self.ep_size
        start = rank * local_tokens
        end = start + local_tokens

        nvls_eligible = (
            self.triton_nvls_kernels_allowed
            and hidden_states.dtype in (torch.bfloat16, torch.float32)
            and are_tensors_nvls_eligible(hidden_states)
        )

        if nvls_eligible:
            ar_buffer = self._maybe_allocate_rs_buffer(hidden_states)
            if ar_buffer["handle"] is not None:
                # Stage in symmetric memory so the multicast load sees every
                # rank's contribution.
                ar_buffer["tensor"].copy_(hidden_states)
                native_op = getattr(
                    getattr(torch.ops, "symm_mem", None), "multimem_all_reduce_", None
                )
                if native_op is not None:
                    # Fast path: PyTorch's native NVLS all-reduce. Operates
                    # in-place on the symmetric memory buffer.
                    sm = ar_buffer["tensor"].view(hidden_states.shape)
                    native_op(sm, "sum", self.tp_ep_group.group_name)
                    return sm[start:end].to(torch.bfloat16).clone()
                # Fallback: our Triton multimem AR kernel (Stage 1 baseline).
                output = torch.empty_like(hidden_states)
                multimem_all_reduce(output, ar_buffer["tensor"], ar_buffer["handle"])
                return output[start:end].to(torch.bfloat16)

        # Final fallback: NCCL ring all-reduce.
        reduced = hidden_states.contiguous().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.tp_ep_group)
        return reduced[start:end].to(torch.bfloat16)
