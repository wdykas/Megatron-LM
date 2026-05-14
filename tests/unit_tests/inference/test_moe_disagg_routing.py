# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE routing under layer-kind disagg.

The v1 disagg model routes whole blocks (not sub-block sublayers).
MoE blocks bundle router + dispatch + experts + combine as one unit;
when an MoE block lives on the E shard, it runs end-to-end locally on
that shard's TP/EP groups. The hidden state arrives from the attention
shard, MoE runs its router internally, experts compute, residual goes
out to the next hop. No router-output payload travels alongside the
activation in v1.

This test confirms the assumption by inspecting how an E-kind block
maps under the dispatcher's plan — the dispatcher hands the whole MoE
block to ``run_local`` and never peeks inside the router.
"""

import pytest
import torch

from megatron.core.inference.route_dispatcher import (
    LayerAction,
    _LayerPlan,
)
from megatron.rl.inference.route_planner import Route, RouteHop


class _MockMoELayer(torch.nn.Module):
    """Stand-in for ``MoELayer`` that records whether forward was
    called as a whole-block unit (the v1 contract)."""

    def __init__(self, out_value: float = 7.0):
        super().__init__()
        self.calls = 0
        self.out_value = out_value

    def forward(self, hidden_states):
        # Real MoELayer does router → dispatch → experts → combine here.
        # The dispatcher treats it as one opaque unit.
        self.calls += 1
        return torch.full_like(hidden_states, self.out_value), None


class _MockDispatcher:
    """Replays the dispatcher contract that ``run_local`` is the
    layer's forward — the dispatcher never separately invokes the
    router."""

    def __init__(self, my_shard_idx, plan):
        from megatron.core.inference.route_dispatcher import RouteDispatcher

        self._route = Route(hops=(RouteHop(shard_idx=my_shard_idx, layer_indices=(0,)),))
        self._my_shard = my_shard_idx
        self._plan = plan
        self.sent = []
        self.received = []

    def dispatch_layer(self, layer_idx, hidden, run_local):
        from megatron.core.inference.route_dispatcher import LayerAction

        plan = self._plan.get(layer_idx)
        if plan is None:
            return hidden, LayerAction.NOT_MY_REQUEST
        if plan.receive_from_pe is not None:
            self.received.append(plan.receive_from_pe)
            hidden = torch.full((4,), 100.0)
        hidden = run_local(hidden)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if plan.send_to_pes:
            self.sent.append((plan.send_to_pes, hidden.clone()))
            return None, LayerAction.SEND
        return hidden, LayerAction.LOCAL


def test_moe_block_runs_end_to_end_on_e_shard():
    """When the dispatcher routes an MoE block to this shard, the
    block's forward is called once with the incoming hidden. The
    router runs inside the block's forward — not as a separate hop.
    """
    moe = _MockMoELayer(out_value=42.0)
    plan = {
        0: _LayerPlan(receive_from_pe=3, send_to_pes=(5,))  # entry + exit hop
    }
    disp = _MockDispatcher(my_shard_idx=2, plan=plan)
    hidden, action = disp.dispatch_layer(
        0,
        torch.zeros(4),
        lambda h: moe(h),
    )
    # MoE called exactly once as a unit (not separately for router /
    # experts).
    assert moe.calls == 1
    # Receive happened before run_local; send happened after.
    assert disp.received == [3]
    assert len(disp.sent) == 1
    assert disp.sent[0][0] == (5,)
    # The returned hidden is from the MoE block's output (42.0 fill).
    assert torch.allclose(disp.sent[0][1], torch.full((4,), 42.0))


def test_moe_block_layer_kind_is_E():
    """The ``E`` kind in ``layer_type_list`` maps to a MoE block. The
    factory loop in HybridStack instantiates the MoE submodule at
    that position (or an IdentityLayer stub if this shard doesn't
    own it). The dispatcher never peeks at the kind — it just routes
    by global layer index.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    assert "E" in Symbols.VALID_LAYERS
    # MoE block instantiation is keyed off the layer-type symbol.
    # Disagg shards with kinds=("E",) own exactly the E positions.


def test_moe_router_runs_on_E_shard_not_attention_shard():
    """A 3-shard layout with M / * / E kinds: the MoE block (including
    its router) is constructed on the E shard alone. Attention shard
    never instantiates the router parameters.

    This is the v1 model: whole-block ownership. Router-on-attention
    -experts-on-E would be a sub-block split (out of scope for v1).
    """
    from megatron.rl.inference.shards_spec import (
        compute_layer_indices_for_kinds,
    )

    pattern = ("M", "*", "E", "M", "*", "E")
    # E shard owns positions 2 and 5.
    e_owned = compute_layer_indices_for_kinds(("E",), pattern)
    assert e_owned == (2, 5)
    # Attention shard owns 1 and 4 (no E).
    attn_owned = compute_layer_indices_for_kinds(("*",), pattern)
    assert attn_owned == (1, 4)
    # No overlap — MoE block (router included) is exclusively the E
    # shard's responsibility.
    assert set(e_owned).isdisjoint(set(attn_owned))
