# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Multi-GPU equivalence test for the attention-bounded-segments combine path.

The MVP swaps ``InferenceCUDAGraphTokenDispatcher.token_combine``'s
ReduceScatter with an equivalent AllReduce + local slice when
``enable_attention_bounded_segments`` is set with the
``current_segment_owner`` policy. Both paths sum across EP ranks and return
each rank's local slice; this test runs both on the same input and verifies
they agree numerically. It exists so that turning on the feature flag can be
trusted not to silently break model output.

Run with EP=4:

    torchrun --nproc_per_node=4 -m pytest \
        tests/unit_tests/inference/test_attention_bounded_combine.py
"""

import os

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

# Same scaled-down nanov3-style config used by tests/test_moe_inference.py
NANOV3_BASE = dict(
    num_layers=1,
    hidden_size=128,
    ffn_hidden_size=128,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=8,
    moe_ffn_hidden_size=128,
    moe_router_topk=6,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.5,
    moe_shared_expert_intermediate_size=256,
    moe_router_dtype='fp32',
    moe_shared_expert_overlap=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.01,
    activation_func=squared_relu,
    normalization="RMSNorm",
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
)


def _make_config(**overrides):
    return TransformerConfig(**{**NANOV3_BASE, **overrides})


@pytest.mark.internal
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="Needs at least 2 EP ranks to exercise the combine collective",
)
class TestSegmentCombineEquivalence:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def _make_dispatcher(self, **overrides):
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.moe.token_dispatcher_inference import (
            InferenceCUDAGraphTokenDispatcher,
        )

        overrides.setdefault("expert_model_parallel_size", Utils.world_size)
        config = _make_config(**overrides)
        num_local = config.num_moe_experts // Utils.world_size
        rank = torch.distributed.get_rank()
        local_indices = [rank * num_local + i for i in range(num_local)]
        return InferenceCUDAGraphTokenDispatcher(
            num_local_experts=num_local,
            local_expert_indices=local_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
        )

    @pytest.mark.parametrize("seed", [0, 7, 123])
    @pytest.mark.parametrize("num_local_tokens", [16, 64, 128])
    def test_all_reduce_path_matches_reduce_scatter(self, num_local_tokens, seed):
        """AllReduce+slice must agree with ReduceScatter modulo NCCL fp ordering.

        Both compute ``sum_over_ranks(global_hidden)[start:end]``. The result
        is bf16 so we allow a tiny numerical tolerance to absorb the
        different reduction tree topologies between the two NCCL kernels.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        rank = torch.distributed.get_rank()
        ep = Utils.world_size
        hidden = NANOV3_BASE["hidden_size"]
        global_tokens = num_local_tokens * ep

        # Construct a synthetic post-experts global view that's identical
        # across ranks (broadcast from rank 0). This mirrors how the
        # dispatcher uses the tensor: after AllGather every rank has the
        # same payload, then experts contribute their share. For this test
        # we skip experts and just verify the combine collective.
        global_hidden = torch.randn(global_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        torch.distributed.broadcast(global_hidden, src=0)

        # --- Standard reduce-scatter path ---
        baseline_dispatcher = self._make_dispatcher()
        rs_out = baseline_dispatcher.token_combine(global_hidden.clone())

        # --- AllReduce + slice path (segments enabled, current_segment_owner) ---
        ar_dispatcher = self._make_dispatcher(
            enable_attention_bounded_segments=True,
            moe_combine_destination_policy="current_segment_owner",
        )
        ar_out = ar_dispatcher.token_combine(global_hidden.clone())

        # Both are bf16, both should equal sum_over_ranks(global_hidden)[start:end].
        # Since global_hidden is identical on all ranks the sum is just ep * global_hidden.
        start = rank * num_local_tokens
        end = start + num_local_tokens
        expected = (global_hidden[start:end].float() * ep).bfloat16()

        # bf16 tolerance: NCCL reduction trees can produce different fp ordering.
        torch.testing.assert_close(rs_out, expected, atol=0, rtol=0)
        torch.testing.assert_close(ar_out, expected, atol=0, rtol=0)
        torch.testing.assert_close(ar_out, rs_out, atol=0, rtol=0)

    @pytest.mark.parametrize("num_local_tokens", [16, 64])
    def test_baseline_policy_does_not_change_collective(self, num_local_tokens):
        """With ABS enabled but baseline policies, behavior matches plain RS.

        This is the contract that lets us turn on ``--enable-attention-bounded-segments``
        without behavioral risk: only when a non-baseline combine policy is
        also set should the actual collective change.
        """
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        rank = torch.distributed.get_rank()
        ep = Utils.world_size
        hidden = NANOV3_BASE["hidden_size"]
        global_tokens = num_local_tokens * ep

        global_hidden = torch.randn(global_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        torch.distributed.broadcast(global_hidden, src=0)

        baseline = self._make_dispatcher().token_combine(global_hidden.clone())
        # Flag on, but combine policy left at "original_owner" -> should still
        # take the reduce-scatter branch.
        flagged = self._make_dispatcher(
            enable_attention_bounded_segments=True,
            moe_combine_destination_policy="original_owner",
        ).token_combine(global_hidden.clone())

        torch.testing.assert_close(flagged, baseline, atol=0, rtol=0)
