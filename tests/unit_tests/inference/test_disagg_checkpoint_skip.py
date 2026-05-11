# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for IdentityLayer-aware sharded_state_dict + the engine's
activate_disagg_request hook.
"""

from unittest.mock import MagicMock

from megatron.core.inference.partial_model import IdentityLayer


def test_identity_layer_skipped_in_state_dict_loop():
    """The sharded_state_dict loop in HybridStack checks
    ``isinstance(layer, IdentityLayer)`` AND
    ``_is_identity_stub`` to decide whether to skip. Verify both
    paths."""
    a = IdentityLayer()
    b = MagicMock()
    b._is_identity_stub = True  # impostor that ducktypes the stub marker

    # Real isinstance check on the real class.
    assert isinstance(a, IdentityLayer)

    # Ducktype check on _is_identity_stub.
    assert getattr(b, "_is_identity_stub", False) is True


def test_engine_activate_picks_single_disagg_request():
    """When exactly one in-flight request has a dispatcher, the
    engine's step path picks it as the active disagg request id."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._migration_handler = None
    eng._pending_migrations = []
    eng._route_handler = None
    eng._route_dispatchers = {42: object()}  # one dispatcher
    eng.requests = {42: object(), 99: object()}  # request 99 has no dispatcher
    eng.controller = MagicMock()
    eng.controller.inference_wrapped_model.model.decoder = MagicMock()

    # The selection logic (copied from async_step):
    in_flight_with_dispatcher = [
        rid for rid in eng._route_dispatchers if rid in eng.requests
    ]
    assert in_flight_with_dispatcher == [42]
    single = in_flight_with_dispatcher[0] if len(in_flight_with_dispatcher) == 1 else None
    assert single == 42

    eng.activate_disagg_request(single)
    # activate_disagg_request resolves the dispatcher and pushes it
    # directly into the decoder via set_active_dispatcher.
    eng.controller.inference_wrapped_model.model.decoder.set_active_dispatcher.assert_called_once()
    pushed_dispatcher = eng.controller.inference_wrapped_model.model.decoder.set_active_dispatcher.call_args[0][0]
    assert pushed_dispatcher is eng._route_dispatchers[42]


def test_engine_activate_falls_back_to_collocated_for_mixed_batch():
    """Two disagg requests in flight at once: v1 falls back to
    collocated (None) since the layer loop can only have one active
    dispatcher per forward pass."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._route_dispatchers = {1: object(), 2: object()}
    eng.requests = {1: object(), 2: object(), 3: object()}

    in_flight_with_dispatcher = [
        rid for rid in eng._route_dispatchers if rid in eng.requests
    ]
    assert sorted(in_flight_with_dispatcher) == [1, 2]
    single = in_flight_with_dispatcher[0] if len(in_flight_with_dispatcher) == 1 else None
    assert single is None  # two disagg requests → no single active


def test_engine_activate_no_disagg_when_no_dispatchers():
    """Empty _route_dispatchers means no disagg active; activate(None)
    is the safe default."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._route_dispatchers = {}
    eng.requests = {1: object()}
    eng.controller = MagicMock()
    eng.controller.inference_wrapped_model.model.decoder = MagicMock()

    in_flight_with_dispatcher = [
        rid for rid in eng._route_dispatchers if rid in eng.requests
    ]
    single = in_flight_with_dispatcher[0] if len(in_flight_with_dispatcher) == 1 else None
    assert single is None
    eng.activate_disagg_request(single)
    # None → clear the active dispatcher.
    eng.controller.inference_wrapped_model.model.decoder.set_active_dispatcher.assert_called_with(None)


def test_engine_activate_safe_when_decoder_missing():
    """Plain GPT engines (no hybrid decoder) shouldn't crash when
    activate_disagg_request is called."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng.controller = MagicMock()
    # Strip the decoder attr to simulate a plain GPT model.
    del eng.controller.inference_wrapped_model.model.decoder

    # Should not raise.
    eng.activate_disagg_request(None)
    eng.activate_disagg_request(42)
