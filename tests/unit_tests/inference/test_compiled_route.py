# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static-action specialization for ``torch.compile``-friendly routes.

:class:`CompiledRoute` pre-resolves the per-layer dispatch decisions
at construction time, so its ``run`` method is a flat sequence of
``RECV / LOCAL / SEND / CACHE_HIT / CACHE_BOUNDARY`` actions with no
runtime control flow keyed on layer_idx. This makes the resulting
function torch.compile-friendly (the for-loop unrolls; only transport
calls graph-break, and a follow-up will register those as torch ops).

These tests verify the action list shape against the standard
:class:`RouteDispatcher`'s behavior on the same route + that ``run``
produces the same output. The mock activation transport backend is
used so no real NVSHMEM is required.
"""

from typing import List

import pytest
import torch

from megatron.core.inference.activation_cache import PrefixActivationCache
from megatron.core.inference.compiled_route import (
    CompiledRoute,
    _Action,
    _ActionKind,
)
from megatron.core.inference.route_dispatcher import LayerAction, RouteDispatcher
from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    set_activation_transport_backend,
)
from megatron.rl.inference.route_planner import (
    Route,
    RouteHop,
    make_moe_dag_route,
)


class _RecordingBackend(ActivationTransportBackend):
    """Records sends + serves scripted recv payloads.

    ``recv_payloads`` maps ``src_pe → list of tensors`` consumed FIFO,
    so multi-branch DAG tests can drive distinct activations through
    distinct branches."""

    def __init__(self, recv_payloads=None) -> None:
        self.sends: list = []
        self.recv_calls: list = []
        self._recv_payloads = {k: list(v) for k, v in (recv_payloads or {}).items()}

    def is_initialized(self) -> bool:
        return True

    def init(self, **kwargs) -> None:
        pass

    def stream(self):
        return None

    def send_hidden(self, my_pe, dst_pe, hidden, payload_nbytes, *, stream=None):
        self.sends.append((dst_pe, hidden.clone()))

    def receive_hidden(self, my_pe, src_pe, hidden_shape, hidden_dtype,
                       payload_nbytes, *, stream=None):
        self.recv_calls.append(src_pe)
        bucket = self._recv_payloads.get(src_pe)
        if bucket:
            return bucket.pop(0)
        return torch.zeros(hidden_shape, dtype=hidden_dtype)


@pytest.fixture
def backend():
    b = _RecordingBackend()
    set_activation_transport_backend(b)
    yield b
    set_activation_transport_backend(None)


def _make_compiled(route, my_shard_idx, cache=None, prompt_tokens=None):
    max_shard = max(h.shard_idx for h in route.hops)
    return CompiledRoute(
        route=route,
        my_shard_idx=my_shard_idx,
        my_pe=my_shard_idx,
        my_tp_offset=0,
        shard_tp=[1] * (max_shard + 1),
        shard_rank_offset=list(range(max_shard + 1)),
        hidden_shape=(1, 4),
        hidden_dtype=torch.float32,
        cache=cache,
        prompt_tokens=prompt_tokens,
    )


# ---- Action list shape ---------------------------------------------------


def test_entry_shard_two_layer_hop_emits_two_locals_then_send(backend):
    """Standard prefill pattern: entry shard owns 2 layers, then
    sends to the next shard. Expect [LOCAL, LOCAL, SEND]."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 1)),
            RouteHop(shard_idx=1, layer_indices=(2,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=0)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [
        _ActionKind.LOCAL,
        _ActionKind.LOCAL,
        _ActionKind.SEND,
    ]
    # The SEND action targets PE 1 (the next shard's base rank).
    assert cr.actions[-1].pes == (1,)


def test_middle_shard_emits_recv_then_locals_then_send(backend):
    """Middle shard pattern: RECV at hop entry, LOCALs across the
    hop's layers, SEND at hop exit."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1, 2)),
            RouteHop(shard_idx=2, layer_indices=(3,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=1)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [
        _ActionKind.RECV,
        _ActionKind.LOCAL,
        _ActionKind.LOCAL,
        _ActionKind.SEND,
    ]


def test_exit_shard_emits_recv_then_locals_no_send(backend):
    """Exit shard receives, runs locally, returns hidden (no SEND)."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=1)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [_ActionKind.RECV, _ActionKind.LOCAL]


def test_not_my_request_layers_dont_appear_in_action_list(backend):
    """Layers owned by other shards have no entry in the action list
    — only owned layers do. Lets the compiled run() iterate without
    "is this my layer?" checks."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1, 2)),
            RouteHop(shard_idx=2, layer_indices=(3,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=0)
    assert cr.owned_layers == (0,)
    # Only layer 0 appears.
    assert all(a.layer_idx == 0 for a in cr.actions)


# ---- DAG fan-out / reduce in compiled form -------------------------------


def test_dag_pre_hop_emits_send_to_every_expert(backend):
    """MoE pre-hop on the backbone: one LOCAL, then a SEND whose
    ``pes`` lists every expert shard."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    cr = _make_compiled(route, my_shard_idx=0)
    kinds = [a.kind for a in cr.actions]
    # Pre-hop: LOCAL + SEND; post-hop on the same shard: RECV + LOCAL.
    assert kinds == [
        _ActionKind.LOCAL, _ActionKind.SEND,    # pre
        _ActionKind.RECV, _ActionKind.LOCAL,    # post
    ]
    pre_send = cr.actions[1]
    post_recv = cr.actions[2]
    assert sorted(pre_send.pes) == [1, 2]    # fans to both experts
    assert sorted(post_recv.pes) == [1, 2]   # reduces from both experts
    assert post_recv.reduce_op == "sum"


def test_dag_expert_shard_emits_recv_then_local_then_send(backend):
    """Expert shard's POV: receives from backbone, runs the MoE
    layer, sends to backbone's post hop."""
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    cr = _make_compiled(route, my_shard_idx=2)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [_ActionKind.RECV, _ActionKind.LOCAL, _ActionKind.SEND]
    # Recv from backbone PE 0; send back to backbone PE 0.
    assert cr.actions[0].pes == (0,)
    assert cr.actions[-1].pes == (0,)


# ---- Cache integration in compiled form ----------------------------------


def test_compiled_route_emits_cache_hits_for_skipped_layers(backend):
    """When the cache covers layers 0..K-1, expect K CACHE_HIT actions
    then a CACHE_BOUNDARY at layer K (which substitutes the cached
    hidden as input)."""
    cache = PrefixActivationCache()
    prompt = [1, 2, 3]
    # Pre-populate so a hit covers layers 0 + 1.
    cache.store(prompt, 3, 0, torch.full((1, 4), 10.0))
    cache.store(prompt, 3, 1, torch.full((1, 4), 20.0))

    # Single-hop route covering layers 0..2.
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1, 2)),))
    cr = _make_compiled(route, my_shard_idx=0, cache=cache, prompt_tokens=prompt)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [
        _ActionKind.CACHE_HIT,      # layer 0
        _ActionKind.CACHE_HIT,      # layer 1
        _ActionKind.CACHE_BOUNDARY, # layer 2 (first uncached)
    ]
    assert cr.cache_skip_depth == 2


def test_compiled_route_no_cache_hit_emits_pure_locals(backend):
    """No cache passed → no CACHE_HIT / CACHE_BOUNDARY actions,
    every owned layer is LOCAL."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1, 2)),))
    cr = _make_compiled(route, my_shard_idx=0)
    kinds = [a.kind for a in cr.actions]
    assert kinds == [_ActionKind.LOCAL] * 3


# ---- Execution: run() produces correct output ----------------------------


def test_run_entry_shard_threads_hidden_through_layers(backend):
    """Entry shard with two local layers: ``run`` calls each
    layer_callable in order with the previous output as input."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1)),))
    cr = _make_compiled(route, my_shard_idx=0)
    # layer 0: add 10; layer 1: multiply by 2.
    callables = [lambda h: h + 10, lambda h: h * 2]
    out = cr.run(torch.zeros(1, 4), layer_callables=callables)
    # (0 + 10) * 2 = 20.
    assert torch.allclose(out, torch.full((1, 4), 20.0))


def test_run_sends_at_hop_exit(backend):
    """Entry shard with a hop exit: ``run`` returns ``None`` after
    sending, and the backend recorded the send."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=0)
    out = cr.run(torch.ones(1, 4), layer_callables=[lambda h: h + 1])
    assert out is None
    assert len(backend.sends) == 1
    assert backend.sends[0][0] == 1  # sent to PE 1


def test_run_receives_at_hop_entry_and_threads_through(backend):
    """Middle shard: ``run`` receives, runs locally, sends. The
    received tensor (zeros from the mock backend) is what
    layer_callable[1] sees as input."""
    backend._recv_payloads = {0: [torch.full((1, 4), 7.0)]}
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
            RouteHop(shard_idx=2, layer_indices=(2,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=1)
    captured = []

    def layer_1(h):
        captured.append(h.clone())
        return h * 3

    out = cr.run(None, layer_callables=[None, layer_1])
    # Sent at hop exit → None returned.
    assert out is None
    # Layer 1 saw the recv'd payload (7s).
    assert torch.allclose(captured[0], torch.full((1, 4), 7.0))
    # Send fired with 7 * 3 = 21s.
    sent_hidden = backend.sends[0][1]
    assert torch.allclose(sent_hidden, torch.full((1, 4), 21.0))


def test_run_with_cache_hit_skips_local_compute(backend):
    """Cache hit: skipped layers never call into layer_callables.
    Boundary layer substitutes the cached hidden as input."""
    cache = PrefixActivationCache()
    prompt = [1, 2, 3]
    cache.store(prompt, 3, 0, torch.full((1, 4), 100.0))

    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1)),))
    cr = _make_compiled(
        route, my_shard_idx=0, cache=cache, prompt_tokens=prompt
    )

    calls = []

    def layer_0(h):
        calls.append(("l0", h.clone()))
        return h

    def layer_1(h):
        calls.append(("l1", h.clone()))
        return h * 2

    out = cr.run(torch.zeros(1, 4), layer_callables=[layer_0, layer_1])
    # Layer 0 was cached → not called. Layer 1 saw cached hidden (100s)
    # → output is 200s.
    assert [c[0] for c in calls] == ["l1"]
    assert torch.allclose(calls[0][1], torch.full((1, 4), 100.0))
    assert torch.allclose(out, torch.full((1, 4), 200.0))


def test_run_reduces_multi_input_recv(backend):
    """DAG: post-hop receives from two experts and sums them."""
    backend._recv_payloads = {
        1: [torch.full((1, 4), 3.0)],
        2: [torch.full((1, 4), 5.0)],
    }
    route = make_moe_dag_route(
        backbone_shard=0,
        expert_shards=(1, 2),
        moe_layer=1,
        backbone_pre_layers=(0,),
        backbone_post_layers=(2,),
    )
    cr = _make_compiled(route, my_shard_idx=0)
    captured = []

    def layer_2(h):
        captured.append(h.clone())
        return h

    out = cr.run(
        torch.ones(1, 4),
        layer_callables=[lambda h: h, None, layer_2],
    )
    # Post-hop layer 2 sees the sum of expert outputs = 3 + 5 = 8.
    assert torch.allclose(captured[0], torch.full((1, 4), 8.0))
    assert torch.allclose(out, torch.full((1, 4), 8.0))


# ---- Cross-check against standard dispatcher -----------------------------


def test_compiled_route_matches_standard_dispatcher_output(backend):
    """For a non-DAG, non-cached route, ``CompiledRoute.run`` and the
    standard dispatcher loop must produce identical output. (Same
    operations, just specialized control flow.)"""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0, 1)),
            RouteHop(shard_idx=1, layer_indices=(2,)),
        )
    )
    # Both run on entry shard 0.
    layer_callables = [lambda h: h + 1, lambda h: h * 2, None]

    # Compiled path.
    cr = _make_compiled(route, my_shard_idx=0)
    backend.sends.clear()
    out_compiled = cr.run(torch.zeros(1, 4), layer_callables=layer_callables)
    compiled_send_payload = backend.sends[0][1].clone()

    # Standard dispatcher path (run the same forward).
    backend.sends.clear()
    disp = RouteDispatcher(
        route=route,
        my_shard_idx=0,
        my_pe=0,
        my_tp_offset=0,
        shard_tp=[1, 1],
        shard_rank_offset=[0, 1],
        hidden_shape=(1, 4),
        hidden_dtype=torch.float32,
    )
    h = torch.zeros(1, 4)
    for li in range(3):
        # ``run_local`` mirrors layer_callables[li], skipping None entries.
        fn = layer_callables[li] or (lambda h: h)
        h, _ = disp.dispatch_layer(li, h, fn)
    dispatch_send_payload = backend.sends[0][1].clone()

    # Both paths return None (SEND at hop exit); but the SEND payload
    # bytes must match — that's the actual outbound tensor.
    assert out_compiled is None
    assert torch.allclose(compiled_send_payload, dispatch_send_payload)


# ---- torch.compile compatibility (best-effort smoke test) ----------------


def test_compiled_route_run_is_dynamo_traceable(backend):
    """torch.compile produces a graph over ``run`` without raising.
    Transport calls graph-break for now (they're not registered torch
    ops yet); but the local-compute portions DO compile cleanly,
    which is the whole point of the specialization.

    This test verifies the smoke path — that wrapping ``run`` with
    ``torch.compile`` doesn't blow up on the dispatch logic itself.
    """
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1)),))
    cr = _make_compiled(route, my_shard_idx=0)
    layer_callables = [lambda h: h + 1, lambda h: h * 2]

    @torch.compile(fullgraph=False)
    def compiled_run(h):
        return cr.run(h, layer_callables=layer_callables)

    out = compiled_run(torch.zeros(1, 4))
    assert torch.allclose(out, torch.full((1, 4), 2.0))
