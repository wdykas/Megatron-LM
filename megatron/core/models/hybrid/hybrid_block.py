# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.attention_bounded_segments import SegmentRuntime, summarize_segments
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor


@dataclass
class HybridStackSubmodules:
    """
    A class for the module specs for the HybridStack.
    """

    mamba_layer: Union[ModuleSpec, type] = IdentityOp
    gdn_layer: Union[ModuleSpec, type] = IdentityOp
    attention_layer: Union[ModuleSpec, type] = IdentityOp
    dsa_layer: Union[ModuleSpec, type] = IdentityOp
    mlp_layer: Union[ModuleSpec, type] = IdentityOp
    moe_layer: Union[ModuleSpec, type] = IdentityOp
    mtp_block_spec: Optional[ModuleSpec] = None


class HybridStack(MegatronModule):
    """
    Constructor for the HybridStack class.

    Args:
        config (TransformerConfig): the model configuration
        submodules (HybridStackSubmodules): the submodules for the stack
        pre_process (bool, optional): whether to include an embedding layer.
            Defaults to True.
        layer_type_list (list, optional): pre-computed list of layer type symbols for
            this pipeline segment. When provided (by HybridModel), pipeline stage
            selection has already been done via '|' separators in the pattern.
        pp_layer_offset (int, optional): the global layer offset for this pipeline
            segment. Defaults to 0.
        post_layer_norm (bool, optional): whether to include a final layer norm.
            Defaults to True.
        post_process (bool, optional): whether to include an output layer.
            Defaults to True.
        device (optional): the device to use. Defaults to None.
        dtype (optional): the data type to use. Defaults to None.
        pg_collection (ProcessGroupCollection): the required model communication
            process groups to use.
        is_mtp_layer (bool, optional): whether this is an MTP layer. Defaults to False.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: HybridStackSubmodules,
        pre_process: bool = True,
        layer_type_list: Optional[list[str]] = None,
        pp_layer_offset: int = 0,
        post_layer_norm: bool = True,
        post_process: bool = True,
        device=None,
        dtype=None,
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.pre_process = pre_process
        self.post_layer_norm = post_layer_norm
        self.post_process = post_process
        self.is_mtp_layer = is_mtp_layer

        assert pg_collection is not None, "pg_collection must be provided for HybridStack"

        self.pp_group = pg_collection.pp
        self.tp_group = pg_collection.tp

        # Required for pipeline parallel schedules
        self.input_tensor = None
        self.pg_collection = pg_collection

        assert layer_type_list is not None, (
            "layer_type_list must be provided. It should be pre-computed from "
            "--hybrid-layer-pattern by HybridModel."
        )
        self.layer_type_list = layer_type_list

        # Build layers from the pre-selected segment
        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(self.layer_type_list):
            layer_number = i + 1 + pp_layer_offset
            if self.config.fp8:
                quant_init_context = get_fp8_context(self.config, i + pp_layer_offset, is_init=True)
            elif self.config.fp4:
                quant_init_context = get_fp4_context(self.config, i + pp_layer_offset, is_init=True)
            else:
                quant_init_context = nullcontext()
            with quant_init_context:
                if layer_type == LayerSymbols.MAMBA:
                    layer = build_module(
                        submodules.mamba_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pp_layer_offset=pp_layer_offset,
                        pg_collection=pg_collection,
                    )
                elif layer_type == LayerSymbols.ATTENTION:
                    layer = build_module(
                        submodules.attention_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.DS_ATTENTION:
                    layer = build_module(
                        submodules.dsa_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        is_mtp_layer=is_mtp_layer,
                        add_layer_offset=False,
                        pp_layer_offset=pp_layer_offset,
                    )
                elif layer_type == LayerSymbols.MLP:
                    layer = build_module(
                        submodules.mlp_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.MOE:
                    layer = build_module(
                        submodules.moe_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        add_layer_offset=False,
                    )
                elif layer_type == LayerSymbols.GDN:
                    layer = build_module(
                        submodules.gdn_layer,
                        config=self.config,
                        layer_number=layer_number,
                        pg_collection=pg_collection,
                        # Set to False as we do not want to change offset.
                        add_layer_offset=False,
                    )
                else:
                    raise ValueError("unexpected layer_type")
            self.layers.append(layer)

        # Required for activation recomputation
        self.num_layers_per_pipeline_rank = len(self.layers)

        # Attention-bounded segment runtime. Lives on every pipeline stage
        # whether or not the feature is enabled — when disabled, all hooks
        # are no-ops and combine destinations resolve to "original_owner",
        # so behavior matches baseline. See
        # ``megatron/core/inference/attention_bounded_segments.py``.
        self.segment_runtime = SegmentRuntime.from_layer_type_list(
            self.layer_type_list,
            enabled=getattr(self.config, "enable_attention_bounded_segments", False),
            moe_combine_destination_policy=getattr(
                self.config, "moe_combine_destination_policy", "original_owner"
            ),
        )
        if self.segment_runtime.enabled:
            logging.getLogger(__name__).info(
                "[attention-bounded-segments] pp_layer_offset=%d layers=%d -> %s",
                pp_layer_offset,
                len(self.layer_type_list),
                summarize_segments(self.segment_runtime.segments),
            )

        # Annotate each layer with the segment runtime + its local index so
        # MoE layers (which are the layers that actually need to know about
        # combine destinations) can resolve their own segment in O(1) without
        # walking back up to HybridStack at forward time. MoE layers are
        # wrapped in a TransformerLayer (``self.mlp`` is the MoELayer); also
        # annotate the inner MoELayer when present.
        for local_idx, layer in enumerate(self.layers):
            layer.segment_runtime = self.segment_runtime
            layer.segment_local_layer_idx = local_idx
            inner_mlp = getattr(layer, "mlp", None)
            if inner_mlp is not None:
                inner_mlp.segment_runtime = self.segment_runtime
                inner_mlp.segment_local_layer_idx = local_idx

        # v34 fused AR + residual_add + RMSNorm: pre-link each MoE
        # TransformerLayer to the next layer's input-norm weight so the
        # fused kernel can apply that norm and we can skip the separate
        # input_layernorm kernel launch on the next layer. The next
        # layer is either another TransformerLayer (with
        # ``input_layernorm.weight``) or a MambaLayer (with
        # ``norm.weight``). Set ``_next_input_norm_weights`` to the
        # tensor or None — None means "no fusion possible here, fall
        # back to standalone v33 path".
        for i in range(len(self.layers) - 1):
            cur = self.layers[i]
            nxt = self.layers[i + 1]
            is_moe = (
                isinstance(cur, TransformerLayer)
                and getattr(cur, "is_moe_layer", False)
            )
            if not is_moe:
                continue
            next_norm = None
            nxt_iln = getattr(nxt, "input_layernorm", None)
            if nxt_iln is not None and not isinstance(nxt_iln, IdentityOp):
                next_norm = getattr(nxt_iln, "weight", None)
            else:
                nxt_norm = getattr(nxt, "norm", None)
                if nxt_norm is not None and not isinstance(nxt_norm, IdentityOp):
                    next_norm = getattr(nxt_norm, "weight", None)
            # Wrap in a list to prevent nn.Module from registering the
            # Parameter as a sub-parameter of this layer (which would
            # break checkpoint loading by adding spurious keys to the
            # MoE layer's state_dict).
            cur._next_input_norm_weights_ref = [next_norm] if next_norm is not None else None

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_norm = TENorm(
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )

    def set_input_tensor(self, input_tensor: Tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def mamba_state_shapes_per_request(self) -> Optional[Tuple[Tuple[int], Tuple[int]]]:
        """
        Returns the Mamba conv and ssm states shapes per input sequence
        if this block contains Mamba layers (this may not be the case with PP > 1).
        """
        for layer_type, layer in zip(self.layer_type_list, self.layers):
            if layer_type == LayerSymbols.MAMBA:
                return layer.mamba_state_shapes_per_request()
        return None

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Tensor,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask=None,
    ):
        """
        Forward function of the HybridStack class.

        It either returns the Loss values if labels are given or the
            final hidden units

        Args:
            hidden_states (Union[Tensor, WrappedTensor]): the input tensor.
                Can be passed as a WrappedTensor during inference to avoid an obsolete
                reference in the calling function.
            attention_mask (Tensor): the attention mask.
            inference_context (BaseInferenceContext): the inference parameters.
            rotary_pos_emb (Tensor, optional): the rotary positional embeddings.
                Defaults to None.
        Returns:
            Tensor: the output tensor.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Delete the obsolete reference to the initial input tensor if necessary
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if inference_context and inference_context.is_static_batching():
            # NOTE(bnorick): match BaseInferenceContext attributes for
            # mamba_ssm.utils.generation.BaseInferenceContext,
            # this hack supports eval
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset

        if (
            (
                (
                    self.config.cuda_graph_impl == "local"
                    and CudaGraphScope.full_iteration not in self.config.cuda_graph_scope
                )
                or self.config.flash_decode
            )
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            current_batch_size = hidden_states.shape[1]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device='cuda',
            )
        else:
            sequence_len_offset = None

        # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
        # otherwise do nothing extra at the outer level
        # if we are using other fp8 recipes, then the context manager enter&exit are free
        # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
        # control which layer will be fp8 or bf16
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        use_fp4_context = self.config.fp4 is not None
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

        if use_inner_fp8_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp8_context(config, layer_number)

        elif use_fp4_context:

            def get_inner_quant_context(config, layer_number):
                return get_fp4_context(config, layer_number)

        else:

            def get_inner_quant_context(config, layer_number):
                return nullcontext()

        # Variant-B path is active when (a) the segment runtime is on
        # with ``current_segment_owner`` policy AND (b) the inference
        # context's per-step ``fast_path_active`` flag is set. The engine
        # owns the per-step gate: today it tracks
        # ``InferenceConfig.inference_replicate_requests`` (the always-
        # replicate path), and the decode-only mode will toggle it per
        # step based on whether the batch is pure decode AND every
        # active request's mamba state has been migrated to all ranks.
        # When active, the dispatcher skips the AG before MoE and
        # replaces RS at combine with an AR returning the full global
        # view, so successive MoE layers chain in [G, hidden] form.
        abs_active = (
            self.segment_runtime.enabled
            and getattr(self.config, "moe_combine_destination_policy", "original_owner")
            == "current_segment_owner"
            and inference_context is not None
            and getattr(inference_context, "fast_path_active", False)
        )

        def _set_segment_dispatch_flag(layer, flag_value):
            inner_mlp = getattr(layer, "mlp", None)
            if inner_mlp is None:
                return
            tok_disp = getattr(inner_mlp, "token_dispatcher", None)
            if tok_disp is None:
                return
            if hasattr(tok_disp, "_segment_input_is_global"):
                tok_disp._segment_input_is_global = bool(flag_value)

        # Partitioned-state Variant-B opt-in: if engine config has
        # ``inference_partitioned_state=True`` AND abs_active, place a
        # single AGV-V right before each MoE so MoE skip-AG fires on
        # the global view, and slice global → per-rank for non-MoE
        # layers (mamba / attention) which need per-rank shape. This
        # gives partitioned mode the same per-(M+E)-pair collective
        # count as default (1 AGV-V + 1 AR), matching default speed.
        # The next optimization (fused matmul-multicast in mamba's
        # out_proj) eliminates the AGV-V — see
        # ``torch_symm_triton/fused_matmul_multicast.py``.
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as _LS
        _MOE_SYM = _LS.MOE
        decode_req_count = (
            inference_context.padded_batch_dimensions.decode_req_count
            if inference_context is not None
            else 0
        )
        prefill_req_count = (
            inference_context.padded_batch_dimensions.prefill_req_count
            if inference_context is not None
            else 0
        )
        partitioned_active = (
            abs_active
            and inference_context is not None
            and getattr(inference_context, "_inference_partitioned_state", False)
            and not getattr(inference_context, "_inference_replicate_requests", False)
            and hidden_states.dim() == 3
            and hidden_states.shape[1] == 1
            and decode_req_count > 0
            and prefill_req_count == 0
        )
        in_global_view = False

        with outer_fp8_context:
            for local_idx, layer in enumerate(self.layers):
                # Layers have 1-indexed layer numbers attribute.
                inner_quant_context = get_inner_quant_context(self.config, layer.layer_number - 1)

                if partitioned_active:
                    layer_type = (
                        self.layer_type_list[local_idx]
                        if local_idx < len(self.layer_type_list)
                        else None
                    )
                    is_moe = layer_type == _MOE_SYM
                    if is_moe and not in_global_view:
                        hidden_states = self._partitioned_pre_moe_agv(
                            hidden_states, inference_context
                        )
                        in_global_view = True
                    elif (not is_moe) and in_global_view:
                        hidden_states = self._partitioned_slice_to_local(
                            hidden_states, inference_context
                        )
                        in_global_view = False

                # Tell this layer's MoE dispatcher whether its input is
                # already global (skip-AG fires) or per-rank (normal
                # AGV path needed).
                if abs_active:
                    flag = (in_global_view if partitioned_active else True)
                    _set_segment_dispatch_flag(layer, flag)
                with inner_quant_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                    else:  # MambaLayer, Expert, or MLP
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )

                # The attention layer (currently a simplified transformer layer)
                # outputs a tuple of (hidden_states, context). Context is intended
                # for cross-attention, and is not needed in our model.
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]

        # Partitioned exit: if the layer loop ended in global view
        # (last layer was MoE, no following attention to slice us back),
        # slice down to per-rank for the engine's per-rank output path.
        if partitioned_active and in_global_view:
            hidden_states = self._partitioned_slice_to_local(
                hidden_states, inference_context
            )

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)

        # Ensure that the tensor passed between pipeline parallel stages is
        # viewless. See related notes in TransformerBlock and TransformerLayer
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return hidden_states

    def _partitioned_pre_moe_agv(
        self,
        hidden_states: torch.Tensor,
        inference_context: BaseInferenceContext,
    ) -> torch.Tensor:
        """Variable-count NVLS multimem all-gather right before each MoE.

        Caller passes 3-D ``[local_tokens, 1, hidden]`` (the dynamic
        engine's decode-step layout). We squeeze the seq dim to 2-D
        for AGV, run the kernel, then unsqueeze back. AGV writes at
        compact prefix-sum offsets (``rank * local_tokens``) since EP
        sync makes every rank's ``local_tokens`` the same.

        Returns ``[per_rank_max * ep_size, 1, hidden]`` on every rank;
        valid contributions are at offsets ``[rank * local_tokens,
        (rank+1) * local_tokens)`` for each rank.
        """
        from megatron.core.inference.communication.torch_symm_triton import (
            are_tensors_nvls_eligible,
            multicast_publish_constexpr,
        )
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            NVLSAllGatherVDispatcher,
        )

        ep_group = getattr(self.pg_collection, "ep", None)
        if ep_group is None or ep_group.size() <= 1:
            return hidden_states

        was_3d = False
        if hidden_states.dim() == 3 and hidden_states.shape[1] == 1:
            hidden_states_2d = hidden_states.squeeze(1)
            was_3d = True
        elif hidden_states.dim() == 2:
            hidden_states_2d = hidden_states
        else:
            return hidden_states

        local_tokens, hidden_size = hidden_states_2d.shape
        if not (
            hidden_states_2d.dtype in (torch.bfloat16, torch.float32)
            and are_tensors_nvls_eligible(hidden_states_2d)
        ):
            return hidden_states

        agv_h = NVLSAllGatherVDispatcher._symm_agv_hidden
        if agv_h is None or agv_h.get("handle") is None:
            return hidden_states
        per_rank_max = NVLSAllGatherVDispatcher._per_rank_worst_case_token_count
        ep_size = ep_group.size()
        rank = ep_group.rank()
        global_max = per_rank_max * ep_size

        # Use the constexpr-specialized multicast-publish kernel:
        # launches exactly local_tokens CTAs (no per-rank-max padding,
        # no ep_max_tokens runtime load). Saves the fused_metadata_update
        # collective AND ~115 wasted CTAs from the variable AGV-V kernel.
        output_2d_full = agv_h["tensor"].view(global_max, hidden_size)
        multicast_publish_constexpr(
            local_tensor=hidden_states_2d.contiguous(),
            output_global_buffer=output_2d_full,
            output_symm_mem_handle=agv_h["handle"],
            rank=rank,
            rank_token_offset=rank * local_tokens,
        )
        # Compact prefix-sum layout: valid rows are at [0, ep_size *
        # local_tokens). Slice to that exact extent so the downstream
        # MoE dispatcher sees ``_local_tokens = ep_size * local_tokens``
        # (≈52 in our bench) instead of ``per_rank_max * ep_size``
        # (=8192) — otherwise the combine AR runs over the entire
        # registered buffer and costs 100× more than needed.
        g_total = ep_size * local_tokens
        output_2d = output_2d_full[:g_total]
        if was_3d:
            return output_2d.unsqueeze(1)
        return output_2d

    def _partitioned_slice_to_local(
        self,
        hidden_states: torch.Tensor,
        inference_context: BaseInferenceContext,
    ) -> torch.Tensor:
        """Slice the global view back to this rank's per-rank chunk.

        Multimem AGV-V layout (compact prefix-sum): with EP sync every
        rank has the same ``local_tokens``, so rank r's data is at
        offsets ``[r * local_tokens, (r+1) * local_tokens)``.
        """
        ep_group = getattr(self.pg_collection, "ep", None)
        if ep_group is None or ep_group.size() <= 1:
            return hidden_states
        if hidden_states.dim() not in (2, 3):
            return hidden_states
        rank = ep_group.rank()
        decode_req_count = inference_context.padded_batch_dimensions.decode_req_count
        seq_len = 1 + getattr(inference_context, "num_speculative_tokens", 0)
        local_tokens = decode_req_count * seq_len
        return hidden_states[
            rank * local_tokens : (rank + 1) * local_tokens
        ].contiguous()

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Optional[tuple] = None,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """
        Returns a sharded state dictionary for the current object.

        This function constructs a sharded state dictionary by iterating over the layers
        in the current object, computing the sharded state dictionary for each layer,
        and combining the results into a single dictionary.

        Parameters:
            prefix (str): The prefix to use for the state dictionary keys.
            sharded_offsets (tuple): The sharded offsets to use for the state dictionary.
            metadata (dict): Additional metadata to use when computing the sharded state dictionary.

        Returns:
            dict: The sharded state dictionary for the current object.
        """

        sharded_state_dict = {}
        layer_prefix = f'{prefix}layers.'

        for local_layer_idx, layer in enumerate(self.layers):

            global_layer_offset = layer.layer_number - 1  # self.layer_number starts at 1
            state_dict_prefix = (
                f'{layer_prefix}{local_layer_idx}.'  # module list index in HybridStack
            )

            sharded_prefix = f'{layer_prefix}{global_layer_offset}.'
            sharded_pp_offset = []

            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )

            replace_prefix_for_sharding(layer_sharded_state_dict, state_dict_prefix, sharded_prefix)

            sharded_state_dict.update(layer_sharded_state_dict)

        # Add modules other than self.layers
        for name, module in self.named_children():
            if not module is self.layers:
                sharded_state_dict.update(
                    sharded_state_dict_default(
                        module,
                        f'{prefix}{name}.',
                        sharded_offsets,
                        metadata,
                        tp_group=self.tp_group,
                    )
                )

        return sharded_state_dict


# Backward-compatible aliases
MambaStackSubmodules = HybridStackSubmodules
MambaStack = HybridStack
