# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prefix activation cache integration with the route dispatcher.

The dispatcher accepts an optional :class:`PrefixActivationCache` +
``prompt_tokens``. On the entry shard's first forward pass:

- Cache lookup picks the deepest cached layer for the longest
  matching prefix.
- Layers ``0 .. skip_depth - 1`` return ``CACHED`` from
  ``dispatch_layer`` — ``run_local`` is skipped entirely.
- The boundary layer (``skip_depth``) uses the cached hidden as
  its inbound activation instead of ``plan.receive_from_pes``.
- Subsequent layers (past the cache) run normally and **populate
  the cache** with their computed outputs so future requests with
  matching prefixes get a deeper hit.

The cache only fires on the FIRST forward pass after dispatcher
construction (the prefill). Decode-step passes through the same
dispatcher must run every layer normally — verified by the
"second-pass-runs-every-layer" test below.

Non-entry shards never apply the cache (their input comes from
NVSHMEM, not the local cache) — verified by the "non-entry-shard
ignores cache" test.
"""

from typing import List

import pytest
import torch

from megatron.core.inference.activation_cache import PrefixActivationCache
from megatron.core.inference.route_dispatcher import (
    LayerAction,
    RouteDispatcher,
)
from megatron.core.inference.transport_backend import (
    ActivationTransportBackend,
    set_activation_transport_backend,
)
from megatron.rl.inference.route_planner import Route, RouteHop


class _NoopBackend(ActivationTransportBackend):
    """Records sends + scripted recvs (same shape as the DAG tests'
    backend, deliberately minimal — these tests focus on the cache
    path, not transport)."""

    def __init__(self) -> None:
        self.sends: list = []
        self.recv_calls: list = []

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
        return torch.zeros(hidden_shape, dtype=hidden_dtype)


@pytest.fixture(autouse=True)
def _mock_backend():
    """Every test in this file uses the no-op backend (no real
    NVSHMEM). Reset between tests so call counts start at zero."""
    backend = _NoopBackend()
    set_activation_transport_backend(backend)
    yield backend
    set_activation_transport_backend(None)


def _entry_shard_route(num_layers: int) -> Route:
    """Single-hop route on shard 0 covering ``num_layers`` layers —
    the simplest case for testing cache integration."""
    return Route(
        hops=(RouteHop(shard_idx=0, layer_indices=tuple(range(num_layers))),)
    )


def _make_dispatcher(route, my_shard_idx=0, cache=None, prompt_tokens=None):
    return RouteDispatcher(
        route=route,
        my_shard_idx=my_shard_idx,
        my_pe=my_shard_idx,
        my_tp_offset=0,
        shard_tp=[1] * (max(h.shard_idx for h in route.hops) + 1),
        shard_rank_offset=list(range(max(h.shard_idx for h in route.hops) + 1)),
        hidden_shape=(1, 4),
        hidden_dtype=torch.float32,
        cache=cache,
        prompt_tokens=prompt_tokens,
    )


# ---- Construction-time lookup --------------------------------------------


def test_no_cache_arg_disables_cache_path():
    """Default (no cache passed) behaves identically to v1 — every
    layer runs locally, no CACHED actions."""
    route = _entry_shard_route(num_layers=3)
    disp = _make_dispatcher(route)
    hidden = torch.ones(1, 4)
    actions = []
    for li in range(3):
        hidden, action = disp.dispatch_layer(li, hidden, lambda h: h + 1)
        actions.append(action)
    assert actions == [LayerAction.LOCAL, LayerAction.LOCAL, LayerAction.LOCAL]


def test_cache_miss_runs_every_layer_and_populates():
    """First request through a fresh cache: full miss, every layer
    runs, every output gets stored."""
    cache = PrefixActivationCache()
    route = _entry_shard_route(num_layers=3)
    disp = _make_dispatcher(route, cache=cache, prompt_tokens=[1, 2, 3])

    hidden = torch.zeros(1, 4)
    actions = []
    for li in range(3):
        hidden, action = disp.dispatch_layer(li, hidden, lambda h: h + 1)
        actions.append(action)

    # No CACHED actions — full miss.
    assert all(a == LayerAction.LOCAL for a in actions)
    # Three entries populated (one per layer of the entry shard).
    assert len(cache) == 3


def test_cache_hit_skips_cached_layers():
    """Second request with the same prefix: full prefix hit, every
    cached layer returns CACHED without invoking ``run_local``."""
    cache = PrefixActivationCache()
    route = _entry_shard_route(num_layers=3)

    # First request populates the cache with layers 0..2.
    disp1 = _make_dispatcher(route, cache=cache, prompt_tokens=[7, 8, 9])
    hidden = torch.zeros(1, 4)
    for li in range(3):
        hidden, _ = disp1.dispatch_layer(li, hidden, lambda h: h + 1)
    assert len(cache) == 3

    # Second request, same prompt → should hit on all 3 cached layers.
    disp2 = _make_dispatcher(route, cache=cache, prompt_tokens=[7, 8, 9])
    run_local_calls = 0

    def counting_run_local(h):
        nonlocal run_local_calls
        run_local_calls += 1
        return h + 1

    actions = []
    hidden = torch.zeros(1, 4)
    for li in range(3):
        hidden, action = disp2.dispatch_layer(li, hidden, counting_run_local)
        actions.append(action)

    assert actions == [LayerAction.CACHED, LayerAction.CACHED, LayerAction.CACHED]
    assert run_local_calls == 0


def test_partial_cache_hit_resumes_at_boundary():
    """Shared 5-token prefix, the cached request only ran the first
    2 layers (cache has 0 + 1). A new 3-layer request hits on 0 + 1
    and resumes at layer 2 using the cached hidden."""
    cache = PrefixActivationCache()
    # Manually populate the cache as if a partial prefill happened.
    cache.store(
        [1, 2, 3, 4, 5], prefix_len=5, layer_idx=0,
        hidden=torch.full((1, 4), 10.0),
    )
    cache.store(
        [1, 2, 3, 4, 5], prefix_len=5, layer_idx=1,
        hidden=torch.full((1, 4), 20.0),
    )

    route = _entry_shard_route(num_layers=3)
    disp = _make_dispatcher(
        route, cache=cache, prompt_tokens=[1, 2, 3, 4, 5]
    )

    captured = []

    def run_local(h):
        captured.append(h.clone())
        return h + 1

    hidden = torch.zeros(1, 4)
    actions = []
    for li in range(3):
        hidden, action = disp.dispatch_layer(li, hidden, run_local)
        actions.append(action)

    # Layers 0 + 1 are cached; layer 2 runs with cached hidden as input.
    assert actions[0] == LayerAction.CACHED
    assert actions[1] == LayerAction.CACHED
    assert actions[2] == LayerAction.LOCAL
    # Layer 2 saw the cached hidden (= output of layer 1 = 20s) as input.
    assert torch.allclose(captured[0], torch.full((1, 4), 20.0))


# ---- Pass-counting (cache only fires on prefill) -------------------------


def test_second_pass_runs_every_layer_normally():
    """The dispatcher persists across decode steps. On the SECOND
    call to ``dispatch_layer(0, ...)`` the cache must NOT fire —
    decode requires running every layer with the new token's
    state. This is the key correctness property."""
    cache = PrefixActivationCache()
    route = _entry_shard_route(num_layers=3)

    # Pre-populate the cache so a hit would occur on the first pass.
    for L in range(3):
        cache.store([1, 2, 3], 3, L, torch.full((1, 4), float(L)))

    disp = _make_dispatcher(route, cache=cache, prompt_tokens=[1, 2, 3])

    run_local_calls = 0

    def counting_run_local(h):
        nonlocal run_local_calls
        run_local_calls += 1
        return h

    # First pass — full cache hit, run_local never called.
    hidden = torch.zeros(1, 4)
    for li in range(3):
        hidden, _ = disp.dispatch_layer(li, hidden, counting_run_local)
    assert run_local_calls == 0

    # Second pass (the "decode step") — every layer must run.
    hidden = torch.zeros(1, 4)
    actions = []
    for li in range(3):
        hidden, action = disp.dispatch_layer(li, hidden, counting_run_local)
        actions.append(action)

    assert all(a == LayerAction.LOCAL for a in actions)
    assert run_local_calls == 3


def test_decode_pass_does_not_repopulate_cache():
    """After the cache is consumed (first pass done), subsequent
    layer dispatches must NOT write to the cache — decode's
    activations depend on more than the prompt prefix."""
    cache = PrefixActivationCache()
    route = _entry_shard_route(num_layers=2)
    disp = _make_dispatcher(route, cache=cache, prompt_tokens=[5, 5])

    # First pass (miss → populates).
    hidden = torch.zeros(1, 4)
    for li in range(2):
        hidden, _ = disp.dispatch_layer(li, hidden, lambda h: h + 1)
    assert len(cache) == 2

    # Second pass (no new entries).
    cache_size_before_second_pass = len(cache)
    hidden = torch.zeros(1, 4)
    for li in range(2):
        hidden, _ = disp.dispatch_layer(li, hidden, lambda h: h + 99)
    assert len(cache) == cache_size_before_second_pass


# ---- Non-entry shard ignores cache ---------------------------------------


def test_non_entry_shard_never_applies_cache():
    """Shards other than ``route.entry_shard`` get their inbound via
    NVSHMEM. Even if a cache is passed, lookup must not happen on
    them (a hit there would corrupt the activation arriving from a
    peer shard). v1 scope: cache is entry-shard-only."""
    cache = PrefixActivationCache()
    # Populate so any naive lookup would hit.
    for L in range(2):
        cache.store([1, 2], 2, L, torch.full((1, 4), float(L)))

    # Two-hop route: shard 0 (entry) → shard 1 (this dispatcher).
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
        )
    )
    disp = RouteDispatcher(
        route=route,
        my_shard_idx=1,  # NOT entry
        my_pe=1,
        my_tp_offset=0,
        shard_tp=[1, 1],
        shard_rank_offset=[0, 1],
        hidden_shape=(1, 4),
        hidden_dtype=torch.float32,
        cache=cache,
        prompt_tokens=[1, 2],
    )
    # Skip depth should remain 0 on non-entry shard regardless of cache content.
    assert disp._cache_skip_depth == 0


# ---- Cache miss after weight reload --------------------------------------


def test_cache_clear_invalidates_hits():
    """``cache.clear()`` is the weight-reload hook. After clearing,
    a previously-hit prefix misses cleanly."""
    cache = PrefixActivationCache()
    route = _entry_shard_route(num_layers=2)

    # Populate.
    disp1 = _make_dispatcher(route, cache=cache, prompt_tokens=[1, 2])
    hidden = torch.zeros(1, 4)
    for li in range(2):
        hidden, _ = disp1.dispatch_layer(li, hidden, lambda h: h + 1)
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0

    # New dispatcher with same prompt → full miss now.
    disp2 = _make_dispatcher(route, cache=cache, prompt_tokens=[1, 2])
    run_local_calls = 0

    def counting_run_local(h):
        nonlocal run_local_calls
        run_local_calls += 1
        return h + 1

    hidden = torch.zeros(1, 4)
    for li in range(2):
        hidden, _ = disp2.dispatch_layer(li, hidden, counting_run_local)
    assert run_local_calls == 2
