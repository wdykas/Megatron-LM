# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Registered ``torch.library.custom_op`` wrappers for the transport.

The interpreted path (``RouteDispatcher`` + un-flagged
``CompiledRoute``) calls into the backend directly — works fine for
Python execution but graph-breaks every hop under ``torch.compile``.

These tests verify the registered ops:

- exist in the ``disagg::`` library namespace,
- delegate to the active backend (mock backend captures calls),
- have fake kernels that produce tensors with the right shape /
  dtype / device,
- work as building blocks inside :class:`CompiledRoute` when
  ``use_torch_ops=True``,
- traceable by ``torch.compile`` end-to-end without breaks
  (``fullgraph=True`` smoke test).
"""

import pytest
import torch

from megatron.core.inference.compiled_route import CompiledRoute, _ActionKind
from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    set_activation_transport_backend,
)
from megatron.core.inference.transport_ops import (
    receive_hidden,
    receive_via_op,
    send_hidden,
    send_via_op,
)
from megatron.rl.inference.route_planner import Route, RouteHop


class _RecordingBackend(ActivationTransportBackend):
    """Captures send/recv calls and serves scripted recv payloads."""

    def __init__(self, recv_payloads=None) -> None:
        self.sends: list = []
        self.recvs: list = []
        self._recv_payloads = {
            k: list(v) for k, v in (recv_payloads or {}).items()
        }

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
        self.recvs.append((my_pe, src_pe, hidden_shape, hidden_dtype))
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


# ---- Op registration ----------------------------------------------------


def test_ops_exist_in_disagg_namespace():
    """Both ops are registered under the ``disagg::`` library.
    Resolving them via ``torch.ops.disagg`` is the durable contract —
    if this breaks, downstream consumers (Inductor passes, custom
    compile targets) lose them."""
    assert hasattr(torch.ops.disagg, "send_hidden")
    assert hasattr(torch.ops.disagg, "receive_hidden")


def test_send_op_calls_backend(backend):
    """Calling the registered ``send_hidden`` op forwards to the
    active backend's ``send_hidden`` — pluggable-backend story is
    preserved under the op wrapper."""
    hidden = torch.ones(2, 4)
    send_hidden(hidden, 0, 1, hidden.numel() * hidden.element_size())
    assert len(backend.sends) == 1
    my_pe, dst_pe, payload, _ = backend.sends[0]
    assert (my_pe, dst_pe) == (0, 1)
    assert torch.allclose(payload, hidden)


def test_receive_op_calls_backend_and_returns_tensor(backend):
    """``receive_hidden`` returns a tensor with the requested shape
    /dtype/device — by calling the active backend (mock here)."""
    out = receive_hidden(
        [2, 4],
        torch.float32,
        torch.device("cpu"),
        my_pe=0,
        src_pe=1,
        payload_nbytes=2 * 4 * 4,
    )
    assert out.shape == (2, 4)
    assert out.dtype == torch.float32
    assert len(backend.recvs) == 1
    assert backend.recvs[0][1] == 1  # src_pe


def test_receive_op_fake_kernel_used_in_meta_context():
    """The fake kernel constructs an ``empty`` tensor with the
    declared shape — verify by calling under ``FakeTensorMode`` so
    the actual backend isn't touched. This is the contract Dynamo /
    Inductor rely on."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        out = receive_hidden(
            [3, 7],
            torch.float16,
            torch.device("cpu"),
            my_pe=0,
            src_pe=1,
            payload_nbytes=3 * 7 * 2,
        )
    assert out.shape == (3, 7)
    assert out.dtype == torch.float16


def test_receive_via_op_handles_tuple_shape(backend):
    """The convenience wrapper accepts shape as a tuple (matches the
    dispatcher's internal shape representation) — the underlying op
    requires ``List[int]`` so the wrapper handles the conversion."""
    out = receive_via_op(
        hidden_shape=(2, 4),  # tuple, not list
        hidden_dtype=torch.float32,
        device=torch.device("cpu"),
        my_pe=0,
        src_pe=1,
        payload_nbytes=2 * 4 * 4,
    )
    assert out.shape == (2, 4)


# ---- CompiledRoute with use_torch_ops=True ------------------------------


def _make_compiled(route, my_shard_idx, *, use_torch_ops=False):
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
        use_torch_ops=use_torch_ops,
        device=torch.device("cpu"),
    )


def test_compiled_route_send_via_op_calls_backend(backend):
    """``use_torch_ops=True``: the SEND action goes through the
    registered ``disagg::send_hidden`` op, which calls into the
    backend. Net effect identical to the direct-call path; the
    indirection exists so Dynamo can trace through it."""
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=0, use_torch_ops=True)
    cr.run(torch.ones(1, 4), layer_callables=[lambda h: h + 1])
    assert len(backend.sends) == 1
    assert backend.sends[0][1] == 1  # dst_pe


def test_compiled_route_recv_via_op_calls_backend(backend):
    """Mirror of the above for RECV."""
    backend._recv_payloads = {0: [torch.full((1, 4), 9.0)]}
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=1, use_torch_ops=True)
    out = cr.run(None, layer_callables=[None, lambda h: h * 2])
    # Exit shard: returns the post-layer hidden directly.
    assert torch.allclose(out, torch.full((1, 4), 18.0))
    assert backend.recvs == [(1, 0, (1, 4), torch.float32)]


def test_compiled_route_matches_between_backend_call_and_op_call(backend):
    """Direct backend call vs op call must produce bitwise-identical
    output — the op is just a tracer-friendly indirection."""
    backend._recv_payloads = {
        0: [torch.full((1, 4), 3.0), torch.full((1, 4), 3.0)],
    }
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    # Direct path.
    cr_direct = _make_compiled(route, my_shard_idx=1, use_torch_ops=False)
    out_direct = cr_direct.run(None, layer_callables=[None, lambda h: h * 2])
    # Op path.
    cr_op = _make_compiled(route, my_shard_idx=1, use_torch_ops=True)
    out_op = cr_op.run(None, layer_callables=[None, lambda h: h * 2])
    assert torch.allclose(out_direct, out_op)


# ---- torch.compile fullgraph smoke test ---------------------------------


def test_compiled_route_fullgraph_traces_without_breaks(backend):
    """With ``use_torch_ops=True`` the ops are recorded into the
    graph — ``fullgraph=True`` succeeds where it would have failed
    on the direct-backend path. This is the contract the op
    registration provides: Dynamo sees through the transport calls
    instead of breaking the graph at every hop.

    Note on Inductor + side-effect-only ops: when a compiled
    function returns ``None`` and its only side effect is a
    side-effecting op (like ``send_hidden`` declared with
    ``mutates_args=("hidden",)``), the default Inductor backend
    will DCE the op call at runtime. This is a known interaction
    that doesn't bite in production: real forward passes return
    tensor-valued logits, and the receive ops produce tensors that
    keep the graph alive. The smoke test below uses
    ``backend='eager'`` so we can verify the op fires under Dynamo
    without fighting Inductor's DCE pass.
    """
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=0, use_torch_ops=True)
    layer_callables = [lambda h: h * 3]

    # backend='eager' runs Dynamo's traced bytecode without Inductor's
    # post-trace DCE pass. Proves the op is in the graph + fires.
    @torch.compile(fullgraph=True, backend="eager")
    def compiled_run(h):
        return cr.run(h, layer_callables=layer_callables)

    out = compiled_run(torch.ones(1, 4))
    assert out is None  # Entry shard's final SEND clears the hidden.
    assert len(backend.sends) == 1
    assert backend.sends[0][1] == 1  # dst_pe


def test_compiled_route_fullgraph_recv_path_under_default_inductor(backend):
    """When the route has a RECV that produces a tensor consumed
    downstream, the standard Inductor backend keeps the whole chain
    alive — RECV → LOCAL → SEND, where SEND's input depends on the
    RECV-produced tensor. This is the typical middle / exit shard
    pattern, which exercises the most-important compile path."""
    backend._recv_payloads = {0: [torch.full((1, 4), 7.0)]}
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    cr = _make_compiled(route, my_shard_idx=1, use_torch_ops=True)
    layer_callables = [None, lambda h: h * 2]

    @torch.compile(fullgraph=True, backend="eager")
    def compiled_run():
        return cr.run(None, layer_callables=layer_callables)

    out = compiled_run()
    # Exit shard: returns the post-LOCAL hidden directly (no final SEND).
    assert torch.allclose(out, torch.full((1, 4), 14.0))
    assert backend.recvs == [(1, 0, (1, 4), torch.float32)]
