# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MegatronLocalMulti's route handler wiring.

Verifies that when a ROUTE_REQUEST signal arrives the handler:
- Builds a ``RouteDispatcher`` for the request,
- Registers it on the engine,
- Attaches the engine to the model's HybridStack so the forward
  loop can find dispatchers.

Non-distributed; mocks the engine + model surfaces it touches.
"""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.core.inference.route_dispatcher import RouteDispatcher
from megatron.core.inference.shards import InferenceShard
from megatron.rl.inference.route_planner import Route, RouteHop


def _route_visiting(*shards) -> Route:
    """Build a simple linear route visiting the given shard indices."""
    hops = tuple(
        RouteHop(shard_idx=s, layer_indices=(i,))
        for i, s in enumerate(shards)
    )
    return Route(hops=hops)


@pytest.fixture
def fake_inference_server():
    """Build a stripped-down MegatronLocalMulti instance with the
    bookkeeping fields populated. Skips pydantic validation by
    going through __new__ + manual attribute assignment."""
    from megatron.rl.inference.multi_shard import MegatronLocalMulti

    # Build 3 shards, each occupying 2 contiguous ranks.
    shards = []
    for i in range(3):
        shards.append(
            InferenceShard(
                index=i,
                spec={"tp": 2, "pp": 1, "ep": 1, "expt_tp": 2, "dp": 1, "kinds": ("M",) if i == 0 else ("*",) if i == 1 else ("E",)},
                rank_offset=i * 2,
                world_size=2,
                pg_collection=None,
                kinds=("M",) if i == 0 else ("*",) if i == 1 else ("E",),
                layer_indices=(i,),
            )
        )

    instance = MegatronLocalMulti.__new__(MegatronLocalMulti)
    instance._shards = shards
    instance._my_shard_index = 1  # we are the attention shard
    # Stub engine surface used by the handler.
    engine = MagicMock()
    engine.max_requests = 16
    engine._route_handler = None

    # Build a mocked model with config exposing hidden_size + dtype.
    mock_cfg = MagicMock()
    mock_cfg.hidden_size = 64
    mock_cfg.params_dtype = torch.bfloat16
    mock_cfg.pipeline_dtype = torch.bfloat16
    engine.controller.inference_wrapped_model.model.config = mock_cfg
    decoder_mock = MagicMock()
    engine.controller.inference_wrapped_model.model.decoder = decoder_mock
    instance._my_engine = engine
    return instance, engine, decoder_mock


def test_route_handler_skips_when_shard_not_on_route(fake_inference_server, monkeypatch):
    """If this shard isn't in the route's hops, handler should be a
    no-op (no dispatcher registered)."""
    instance, engine, decoder = fake_inference_server
    route = _route_visiting(0, 2)
    monkeypatch.setattr(
        "torch.distributed.is_initialized", lambda: False
    )
    instance._on_route_request_signal(request_id=42, route=route)
    engine.register_route_dispatcher.assert_not_called()


def test_route_handler_registers_dispatcher_when_on_route(
    fake_inference_server, monkeypatch
):
    """When this shard IS on the route, handler builds a dispatcher
    and registers it on the engine."""
    instance, engine, decoder = fake_inference_server
    route = _route_visiting(0, 1, 2)  # we (shard 1) are on this route
    monkeypatch.setattr(
        "torch.distributed.is_initialized", lambda: False
    )
    instance._on_route_request_signal(request_id=42, route=route)
    engine.register_route_dispatcher.assert_called_once()
    rid, dispatcher = engine.register_route_dispatcher.call_args[0]
    assert rid == 42
    assert isinstance(dispatcher, RouteDispatcher)


def test_route_handler_registers_each_request_separately(
    fake_inference_server, monkeypatch
):
    """Each ROUTE_REQUEST registers its own dispatcher; the
    engine→decoder push happens later via activate_disagg_request
    (engine API), not at registration time."""
    instance, engine, decoder = fake_inference_server
    route = _route_visiting(0, 1, 2)
    monkeypatch.setattr(
        "torch.distributed.is_initialized", lambda: False
    )
    instance._on_route_request_signal(request_id=1, route=route)
    instance._on_route_request_signal(request_id=2, route=route)
    assert engine.register_route_dispatcher.call_count == 2


def test_route_handler_skips_when_no_engine():
    """If the inference server has no engine yet (rank not in any
    shard, or pre-launch), the handler is a defensive no-op."""
    from megatron.rl.inference.multi_shard import MegatronLocalMulti

    instance = MegatronLocalMulti.__new__(MegatronLocalMulti)
    instance._shards = []
    instance._my_shard_index = None
    instance._my_engine = None

    # Should not raise even though we have no engine / shards.
    instance._on_route_request_signal(request_id=1, route=_route_visiting(0))


def test_route_handler_rejects_hetero_tp_between_disagg_shards(monkeypatch):
    """Layer-kind disagg requires every participating shard to share TP.
    The shard_to_pe mapping pairs my tp_offset with the same tp_offset
    on peers, which only makes sense with matched TP. Mismatched TP
    fails loudly with a pointer to the constraint."""
    from megatron.rl.inference.multi_shard import MegatronLocalMulti

    # Shard 0: tp=2; shard 1: tp=4 (hetero).
    shards = [
        InferenceShard(
            index=0,
            spec={"tp": 2, "pp": 1, "ep": 1, "expt_tp": 2, "dp": 1, "kinds": ("M",)},
            rank_offset=0,
            world_size=2,
            pg_collection=None,
            kinds=("M",),
            layer_indices=(0,),
        ),
        InferenceShard(
            index=1,
            spec={"tp": 4, "pp": 1, "ep": 1, "expt_tp": 4, "dp": 1, "kinds": ("*",)},
            rank_offset=2,
            world_size=4,
            pg_collection=None,
            kinds=("*",),
            layer_indices=(1,),
        ),
    ]
    instance = MegatronLocalMulti.__new__(MegatronLocalMulti)
    instance._shards = shards
    instance._my_shard_index = 0
    engine = MagicMock()
    engine.max_requests = 16
    mock_cfg = MagicMock()
    mock_cfg.hidden_size = 64
    mock_cfg.params_dtype = torch.bfloat16
    engine.controller.inference_wrapped_model.model.config = mock_cfg
    instance._my_engine = engine
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

    with pytest.raises(AssertionError, match="matched TP"):
        instance._on_route_request_signal(request_id=42, route=_route_visiting(0, 1))
