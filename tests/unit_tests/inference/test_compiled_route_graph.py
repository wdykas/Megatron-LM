# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA graph capture wrapper around :class:`CompiledRoute`.

Two levels of coverage:

- **API contract tests** that run on any platform (verify the
  wrapper's assertions about shape / dtype / capture-before-replay
  semantics).
- **Actual capture / replay** tests guarded with
  ``@pytest.mark.skipif(not torch.cuda.is_available())``. These run
  when GPUs are present and confirm the captured graph produces the
  same output as the un-captured CompiledRoute.

The wrapper is independent of ``torch.compile`` — CUDA graphs are
a runtime capture mechanism orthogonal to Inductor. The two layers
compose (you can torch.compile + capture) but neither requires the
other.
"""

import pytest
import torch

from megatron.core.inference.compiled_route import CompiledRoute
from megatron.core.inference.compiled_route_graph import CompiledRouteGraph
from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    set_activation_transport_backend,
)
from megatron.rl.inference.route_planner import Route, RouteHop


class _NoopBackend(ActivationTransportBackend):
    """Backend that does nothing — sufficient for tests that don't
    actually exercise transport (single-shard routes)."""

    def is_initialized(self): return True
    def init(self, **kwargs): pass
    def stream(self): return None

    def send_hidden(self, my_pe, dst_pe, hidden, payload_nbytes, *, stream=None):
        pass

    def receive_hidden(self, my_pe, src_pe, hidden_shape, hidden_dtype,
                       payload_nbytes, *, stream=None):
        return torch.zeros(hidden_shape, dtype=hidden_dtype)


@pytest.fixture
def backend():
    b = _NoopBackend()
    set_activation_transport_backend(b)
    yield b
    set_activation_transport_backend(None)


def _make_compiled_route(device, dtype=torch.float32):
    """Single-hop single-shard route — keeps the test self-contained
    (no transport, just LOCAL actions)."""
    route = Route(hops=(RouteHop(shard_idx=0, layer_indices=(0, 1)),))
    return CompiledRoute(
        route=route,
        my_shard_idx=0,
        my_pe=0,
        my_tp_offset=0,
        shard_tp=[1],
        shard_rank_offset=[0],
        hidden_shape=(1, 4),
        hidden_dtype=dtype,
        device=device,
    )


# ---- API contract ---------------------------------------------------------


def test_init_rejects_shape_mismatch_vs_compiled_route(backend):
    """The wrapper enforces that the declared input shape matches
    the underlying CompiledRoute. Mismatch would silently produce
    garbage if captured."""
    cr = _make_compiled_route(device=torch.device("cpu"))
    with pytest.raises(AssertionError):
        CompiledRouteGraph(
            cr,
            hidden_shape=(2, 8),  # wrong
            dtype=torch.float32,
            device=torch.device("cpu"),
        )


def test_init_rejects_dtype_mismatch_vs_compiled_route(backend):
    cr = _make_compiled_route(device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(AssertionError):
        CompiledRouteGraph(
            cr,
            hidden_shape=(1, 4),
            dtype=torch.float16,  # wrong
            device=torch.device("cpu"),
        )


def test_input_buffer_is_persistent_with_declared_shape(backend):
    """``input_buffer`` is the destination ``replay`` copies into.
    Its shape + dtype + device match what the constructor declared.
    Allocated up-front so capture has a stable pointer."""
    cr = _make_compiled_route(device=torch.device("cpu"))
    crg = CompiledRouteGraph(
        cr, hidden_shape=(1, 4), dtype=torch.float32, device=torch.device("cpu")
    )
    assert crg.input_buffer.shape == (1, 4)
    assert crg.input_buffer.dtype == torch.float32


def test_captured_flag_false_before_capture(backend):
    """``captured`` flips to True only after a successful capture
    — useful for downstream code that conditionally falls back to
    eager execution before the first capture lands."""
    cr = _make_compiled_route(device=torch.device("cpu"))
    crg = CompiledRouteGraph(
        cr, hidden_shape=(1, 4), dtype=torch.float32, device=torch.device("cpu")
    )
    assert crg.captured is False


def test_replay_before_capture_raises(backend):
    """Calling ``replay`` before ``capture`` is a programming error
    — there's no graph to replay. Fail loud, not silent."""
    cr = _make_compiled_route(device=torch.device("cpu"))
    crg = CompiledRouteGraph(
        cr, hidden_shape=(1, 4), dtype=torch.float32, device=torch.device("cpu")
    )
    with pytest.raises(AssertionError):
        crg.replay(torch.zeros(1, 4))


# ---- Capture + replay (CUDA-only) ----------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs CUDA."
)
def test_capture_then_replay_matches_eager(backend):
    """Captured graph's replay output equals the un-captured
    CompiledRoute's output for the same input. This is the
    correctness contract — capture should be a perf optimization,
    not a behavior change."""
    cuda = torch.device("cuda")
    cr = _make_compiled_route(device=cuda)
    crg = CompiledRouteGraph(cr, hidden_shape=(1, 4), dtype=torch.float32, device=cuda)

    # Compile in pure CompiledRoute the eager output as ground truth.
    eager_out = cr.run(
        torch.ones(1, 4, device=cuda),
        layer_callables=[lambda h: h + 1, lambda h: h * 2],
    )

    crg.capture(layer_callables=[lambda h: h + 1, lambda h: h * 2])
    assert crg.captured

    graph_out = crg.replay(torch.ones(1, 4, device=cuda))
    assert torch.allclose(eager_out, graph_out)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs CUDA."
)
def test_replay_uses_persistent_input_buffer(backend):
    """``replay`` copies new input into the persistent input buffer
    and replays — verify by passing different inputs across two
    replays and confirming the outputs differ accordingly."""
    cuda = torch.device("cuda")
    cr = _make_compiled_route(device=cuda)
    crg = CompiledRouteGraph(cr, hidden_shape=(1, 4), dtype=torch.float32, device=cuda)
    crg.capture(layer_callables=[lambda h: h + 1, lambda h: h * 2])

    out1 = crg.replay(torch.full((1, 4), 0.0, device=cuda)).clone()
    out2 = crg.replay(torch.full((1, 4), 5.0, device=cuda)).clone()
    # First input: (0 + 1) * 2 = 2. Second: (5 + 1) * 2 = 12.
    assert torch.allclose(out1, torch.full((1, 4), 2.0, device=cuda))
    assert torch.allclose(out2, torch.full((1, 4), 12.0, device=cuda))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture needs CUDA."
)
def test_replay_rejects_shape_mismatch_vs_capture(backend):
    """An input with the wrong shape would silently corrupt the
    graph replay (out-of-bounds copy_); reject early."""
    cuda = torch.device("cuda")
    cr = _make_compiled_route(device=cuda)
    crg = CompiledRouteGraph(cr, hidden_shape=(1, 4), dtype=torch.float32, device=cuda)
    crg.capture(layer_callables=[lambda h: h + 1, lambda h: h * 2])

    with pytest.raises(AssertionError):
        crg.replay(torch.zeros(2, 8, device=cuda))
