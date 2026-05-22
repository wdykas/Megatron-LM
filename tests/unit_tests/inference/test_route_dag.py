# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DAG routes: fan-out + reduce.

A linear ``Route`` is the historical case — each hop chains
``hop_pos → hop_pos + 1``. The DAG extension allows a hop to:

- **Fan out** to several parallel successors (``parallel_succs``)
- **Reduce** several inbound activations (``reduce_from`` +
  ``reduce_op``)

The canonical use case is MoE: a backbone shard fans the MoE layer
to N expert shards in parallel and reduces the experts' outputs
back on the backbone. The :func:`make_moe_dag_route` helper builds
exactly this shape.

These tests drive the dispatcher with the mock activation transport
backend (no real NVSHMEM, no CUDA) and verify the predecessor /
successor PE lists, the reduce kernel, and the wire-format round-trip.
"""

from typing import List
from unittest.mock import patch

import pytest
import torch

from megatron.core.inference.route_dispatcher import RouteDispatcher, _LayerPlan
from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    set_activation_transport_backend,
)
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    deserialize_route,
    make_moe_dag_route,
    serialize_route,
)


class _RecordingBackend(ActivationTransportBackend):
    """Records every send/recv and returns scripted payloads for recv.

    ``recv_payloads`` maps ``src_pe → list of tensors`` consumed in FIFO
    order, letting tests drive multi-branch reduces with distinct
    inbound activations."""

    def __init__(self, recv_payloads: dict) -> None:
        self.sends: list = []
        self.recv_calls: list = []
        self._recv_payloads = {k: list(v) for k, v in recv_payloads.items()}

    def is_initialized(self) -> bool:
        return True

    def init(self, **kwargs) -> None:
        pass

    def stream(self):
        return None

    def send_hidden(self, my_pe, dst_pe, hidden, payload_nbytes, *, stream=None):
        self.sends.append((my_pe, dst_pe, hidden.clone(), payload_nbytes))

    def receive_hidden(self, my_pe, src_pe, hidden_shape, hidden_dtype,
                       payload_nbytes, *, stream=None):
        self.recv_calls.append((my_pe, src_pe))
        # Pop the next scripted payload for this src; default to zeros
        # when the test didn't script one (verifies plumbing without
        # caring about the exact values).
        bucket = self._recv_payloads.get(src_pe)
        if bucket:
            return bucket.pop(0)
        return torch.zeros(hidden_shape, dtype=hidden_dtype)


# -------- Route data model -------------------------------------------------


def test_predecessors_and_successors_linear_route():
    """Linear route: predecessors/successors collapse to ``hop_pos ± 1``."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
            RouteHop(shard_idx=2, layer_indices=(2,)),
        )
    )
    assert route.predecessors(0) == ()
    assert route.predecessors(1) == (0,)
    assert route.predecessors(2) == (1,)
    assert route.successors(0) == (1,)
    assert route.successors(1) == (2,)
    assert route.successors(2) == ()
    assert not route.is_dag()


def test_predecessors_and_successors_dag_route():
    """DAG: explicit ``succs`` / ``preds`` fully replace the implicit
    linear chain at branch points so parallel siblings don't accidentally
    chain to each other."""
    route = Route(
        hops=(
            # Pre-hop fans out to BOTH experts (positions 1 and 2).
            RouteHop(shard_idx=0, layer_indices=(0,), succs=(1, 2)),
            # Expert E0 sends directly to the post-hop (position 3).
            RouteHop(shard_idx=1, layer_indices=(1,), succs=(3,)),
            # Expert E1 sends directly to the post-hop.
            RouteHop(shard_idx=2, layer_indices=(1,), succs=(3,)),
            # Post-hop ingests both experts and reduces.
            RouteHop(
                shard_idx=0,
                layer_indices=(2,),
                preds=(1, 2),
                reduce_op="sum",
            ),
        )
    )
    assert route.successors(0) == (1, 2)
    # Each expert hop's successor is hop 3 — they do NOT chain through
    # each other.
    assert route.successors(1) == (3,)
    assert route.successors(2) == (3,)
    # Post-hop ingests both experts.
    assert route.predecessors(3) == (1, 2)
    assert route.is_dag()


def test_make_moe_dag_route_basic():
    """The helper produces a backbone(pre) → [E0, E1] → backbone(post)
    DAG with the expected hop shape and reduce wiring."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=2,
        backbone_pre_layers=(0, 1),
        backbone_post_layers=(3, 4),
    )
    assert route.is_dag()
    assert len(route.hops) == 4
    # Pre hop on backbone with explicit fan-out to BOTH experts (hops 1
    # and 2). Without the explicit succs override the implicit linear
    # chain would only send to hop 1.
    pre = route.hops[0]
    assert pre.shard_idx == 0
    assert pre.layer_indices == (0, 1)
    assert pre.succs == (1, 2)
    # Expert hops have explicit succs straight to the post hop so they
    # don't accidentally chain through each other.
    e0 = route.hops[1]
    e1 = route.hops[2]
    assert (e0.shard_idx, e0.layer_indices, e0.succs) == (1, (2,), (3,))
    assert (e1.shard_idx, e1.layer_indices, e1.succs) == (2, (2,), (3,))
    # Post hop reduces both experts.
    post = route.hops[3]
    assert post.shard_idx == 0
    assert post.layer_indices == (3, 4)
    assert post.preds == (1, 2)
    assert post.reduce_op == "sum"
    # Entry/exit are both backbone.
    assert route.entry_shard == 0
    assert route.exit_shard == 0


def test_make_moe_dag_route_single_expert_degenerates_to_linear():
    """A single expert is still valid; it produces a linear chain
    (no parallel_succs / reduce_from), so dispatchers fall back to the
    fast path."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1,),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    assert not route.is_dag()
    assert len(route.hops) == 3


def test_make_moe_dag_route_requires_post_layers_when_fanning_out():
    """Without a post hop the experts have nowhere to converge — the
    reduce point would be missing. Reject at planning time."""
    with pytest.raises(AssertionError):
        make_moe_dag_route(
            backbone_shard=0,
            expert_shards=(1, 2),
            moe_layer=0,
            backbone_pre_layers=(),
            backbone_post_layers=(),
        )


# -------- Wire format ------------------------------------------------------


def test_serialize_linear_route_keeps_2_element_form():
    """Back-compat: linear routes still emit the original 2-element hop
    payloads. External consumers don't see a wire change unless they
    opt into DAG routes."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 1)),
            RouteHop(shard_idx=1, layer_indices=(2,)),
        )
    )
    wire = serialize_route(route)
    assert wire == [[0, [0, 1]], [1, [2]]]
    assert deserialize_route(wire) == route


def test_serialize_dag_route_emits_5_element_form():
    """DAG routes serialize with the extended hop payload so the
    deserializer can reconstruct parallel/reduce edges."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=2,
        backbone_pre_layers=(0, 1),
        backbone_post_layers=(3,),
    )
    wire = serialize_route(route)
    # Every hop is 5-element in DAG mode.
    assert all(len(h) == 5 for h in wire)
    # Round-trip preserves DAG edges.
    assert deserialize_route(wire) == route


def test_deserialize_rejects_multi_pred_without_op():
    """A hop with multiple ``preds`` but no ``reduce_op`` is malformed;
    reject at deserialize time so a bad wire payload doesn't reach the
    dispatcher."""
    bad_wire = [
        [0, [0], None, None, None],
        [1, [1], None, [0, 0], None],  # preds has 2 entries, reduce_op missing
    ]
    with pytest.raises(AssertionError):
        deserialize_route(bad_wire)


# -------- Dispatcher integration ------------------------------------------


def test_dispatcher_fan_out_at_pre_moe():
    """Backbone pre-hop's exit layer sends the same hidden state to
    each expert shard via the backend. Verify the send list."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    backend = _RecordingBackend(recv_payloads={})
    set_activation_transport_backend(backend)
    try:
        disp = RouteDispatcher(
            route=route,
            my_shard_idx=0,
            my_pe=0,
            my_tp_offset=0,
            shard_tp=[1, 1, 1],
            shard_rank_offset=[0, 1, 2],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        hidden = torch.ones(2, 4)
        # Layer 0 is the only pre-MoE backbone layer → SEND at its exit.
        out, _ = disp.dispatch_layer(0, hidden, lambda h: h)
        assert out is None
        # Two sends: one per expert shard.
        dst_pes = sorted(s[1] for s in backend.sends)
        assert dst_pes == [1, 2]
    finally:
        set_activation_transport_backend(None)


def test_dispatcher_reduces_at_post_moe():
    """Backbone post-hop's entry layer receives one activation per
    expert and reduces (sum). The reduced tensor feeds into ``run_local``."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    # Backend scripts payloads such that expert-1 sends 3s and expert-2
    # sends 5s; the reduce should land 8s into run_local.
    backend = _RecordingBackend(
        recv_payloads={
            1: [torch.full((2, 4), 3.0)],
            2: [torch.full((2, 4), 5.0)],
        }
    )
    set_activation_transport_backend(backend)
    try:
        disp = RouteDispatcher(
            route=route,
            my_shard_idx=0,
            my_pe=0,
            my_tp_offset=0,
            shard_tp=[1, 1, 1],
            shard_rank_offset=[0, 1, 2],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        captured: List[torch.Tensor] = []

        def run_local(h):
            captured.append(h.clone())
            return h

        # Layers 0 (pre), 1 (expert — not ours), 2 (post-reduce, ours).
        disp.dispatch_layer(0, torch.ones(2, 4), run_local)
        disp.dispatch_layer(1, None, run_local)  # NOT_MY_REQUEST
        disp.dispatch_layer(2, None, run_local)  # RECEIVE+REDUCE+LOCAL
        # The reduced tensor handed to run_local at layer 2 should be 8s.
        assert torch.allclose(captured[-1], torch.full((2, 4), 8.0))
        # Two recvs at layer 2, one from each expert PE.
        assert sorted(c[1] for c in backend.recv_calls) == [1, 2]
    finally:
        set_activation_transport_backend(None)


def test_dispatcher_reduces_with_mean_op():
    """``reduce_op="mean"`` divides by branch count."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,), succs=(1, 2)),
            RouteHop(shard_idx=1, layer_indices=(1,), succs=(3,)),
            RouteHop(shard_idx=2, layer_indices=(1,), succs=(3,)),
            RouteHop(
                shard_idx=0,
                layer_indices=(2,),
                preds=(1, 2),
                reduce_op="mean",
            ),
        )
    )
    backend = _RecordingBackend(
        recv_payloads={
            1: [torch.full((2, 4), 6.0)],
            2: [torch.full((2, 4), 2.0)],
        }
    )
    set_activation_transport_backend(backend)
    try:
        disp = RouteDispatcher(
            route=route,
            my_shard_idx=0,
            my_pe=0,
            my_tp_offset=0,
            shard_tp=[1, 1, 1],
            shard_rank_offset=[0, 1, 2],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        captured: List[torch.Tensor] = []

        def run_local(h):
            captured.append(h.clone())
            return h

        disp.dispatch_layer(0, torch.ones(2, 4), run_local)
        disp.dispatch_layer(1, None, run_local)
        disp.dispatch_layer(2, None, run_local)
        assert torch.allclose(captured[-1], torch.full((2, 4), 4.0))
    finally:
        set_activation_transport_backend(None)


def test_expert_shard_only_runs_moe_layer_and_sends():
    """From an expert shard's POV: NOT_MY_REQUEST on backbone layers,
    one local + send on the MoE layer."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    # Expert shard 2: receives from backbone (PE 0) at layer 1, sends to
    # backbone (PE 0) at the same layer (exit).
    backend = _RecordingBackend(
        recv_payloads={0: [torch.full((2, 4), 7.0)]},
    )
    set_activation_transport_backend(backend)
    try:
        disp = RouteDispatcher(
            route=route,
            my_shard_idx=2,
            my_pe=2,
            my_tp_offset=0,
            shard_tp=[1, 1, 1],
            shard_rank_offset=[0, 1, 2],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        out0, a0 = disp.dispatch_layer(0, None, lambda h: h)
        assert out0 is None  # NOT_MY_REQUEST
        # Layer 1 = MoE: recv from backbone PE 0, run local, send back to PE 0.
        out1, _ = disp.dispatch_layer(1, None, lambda h: h)
        assert out1 is None  # SEND
        assert backend.recv_calls == [(2, 0)]
        # Send target is backbone PE 0 (the reduce hop's entry).
        assert [s[1] for s in backend.sends] == [0]
        out2, _ = disp.dispatch_layer(2, None, lambda h: h)
        assert out2 is None  # NOT_MY_REQUEST (backbone runs layer 2)
    finally:
        set_activation_transport_backend(None)


def test_layer_plan_carries_reduce_op():
    """The dispatcher's precomputed plan stores ``reduce_op`` on the
    entry layer of a multi-input hop and ``None`` everywhere else."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2, 3),
    )
    set_activation_transport_backend(_RecordingBackend(recv_payloads={}))
    try:
        disp = RouteDispatcher(
            route=route,
            my_shard_idx=0,
            my_pe=0,
            my_tp_offset=0,
            shard_tp=[1, 1, 1],
            shard_rank_offset=[0, 1, 2],
            hidden_shape=(2, 4),
            hidden_dtype=torch.float32,
        )
        # Layer 2 is the post hop's entry — reduce_op="sum" carried here.
        assert isinstance(disp._plan[2], _LayerPlan)
        assert disp._plan[2].reduce_op == "sum"
        assert len(disp._plan[2].receive_from_pes) == 2
        # Layer 3 is mid-hop — no reduce, no recv, no send (until exit).
        assert disp._plan[3].reduce_op is None
        assert disp._plan[3].receive_from_pes == ()
    finally:
        set_activation_transport_backend(None)
