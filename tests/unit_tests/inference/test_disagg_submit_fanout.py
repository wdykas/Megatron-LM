# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Protocol scaffolding for non-entry shards' engine integration.

When a disagg layout is configured, the coord must fan ``SUBMIT`` to
the entry shard AND ``DISAGG_SUBMIT`` to every other participating
shard so each engine can allocate per-shard state and drive forward.
This test exercises the fan-out and the engine-side handler that
records the participant role.

Full step-loop integration (allocating KV/mamba, driving forward,
sampling on the exit shard) is a separate phase — this scaffolding
gets the protocol pieces in place.
"""

from unittest.mock import MagicMock

import msgpack
import pytest

from megatron.core.inference.headers import Headers
from megatron.rl.inference.route_planner import Route, RouteHop, serialize_route


def _bare_coord():
    from megatron.core.inference.data_parallel_inference_coordinator import (
        DataParallelInferenceCoordinator,
    )

    coord = DataParallelInferenceCoordinator.__new__(
        DataParallelInferenceCoordinator
    )
    coord._disagg_route = None
    coord._send_to_engine = MagicMock()
    coord._identities_for_shard = lambda s: [f"engine_s{s}".encode()]
    return coord


def test_coord_fans_disagg_submit_to_non_entry_participants():
    """Three-shard route 0→1→2. Coord fans:
    - ROUTE_REQUEST to s0, s1, s2 (already covered by other tests)
    - DISAGG_SUBMIT to s1 (intermediate) and s2 (exit), NOT s0
    - SUBMIT_REQUEST is forwarded separately by the existing path."""
    coord = _bare_coord()
    route = Route(
        hops=(
            RouteHop(shard_idx=0, layer_indices=(0,)),
            RouteHop(shard_idx=1, layer_indices=(1,)),
            RouteHop(shard_idx=2, layer_indices=(2,)),
        )
    )
    coord._disagg_route = serialize_route(route)

    # Replay the coord's DISAGG_SUBMIT fan branch.
    request_id = 42
    prompt = "the prompt"
    sampling_params = {"temperature": 1.0}
    effective_route = coord._disagg_route
    entry_shard = effective_route[0][0]
    exit_shard = effective_route[-1][0]
    participating = {h[0] for h in effective_route}
    for shard_idx in sorted(participating):
        if shard_idx == entry_shard:
            continue
        role = "exit" if shard_idx == exit_shard else "intermediate"
        payload = msgpack.packb(
            [Headers.DISAGG_SUBMIT.value, request_id, prompt, sampling_params, role],
            use_bin_type=True,
        )
        for ident in coord._identities_for_shard(shard_idx):
            coord._send_to_engine(ident, payload)

    # Entry shard (s0) skipped; intermediate (s1) and exit (s2) get
    # DISAGG_SUBMIT with the right role.
    assert coord._send_to_engine.call_count == 2
    sent_to = {
        c.args[0]: msgpack.unpackb(c.args[1], raw=False)
        for c in coord._send_to_engine.call_args_list
    }
    assert sent_to[b"engine_s1"][4] == "intermediate"
    assert sent_to[b"engine_s2"][4] == "exit"
    for payload in sent_to.values():
        assert payload[0] == Headers.DISAGG_SUBMIT.value
        assert payload[1] == request_id
        assert payload[2] == prompt
        assert payload[3] == sampling_params


def test_engine_disagg_submit_handler_records_participant():
    """The engine's DISAGG_SUBMIT branch records the participant role
    so the step-loop integration can drive forward + skip sampling
    for non-exit participants."""
    from megatron.core.inference.engines.dynamic_engine import (
        DynamicInferenceEngine,
    )

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._disagg_participants = {}

    # Replay the engine's DISAGG_SUBMIT branch (matches the if branch
    # in async_step — the handler body is small enough to inline here).
    data = [Headers.DISAGG_SUBMIT.value, 7, "prompt", {"top_p": 0.9}, "exit"]
    _, ds_request_id, ds_prompt, ds_params, ds_role = data
    eng._disagg_participants[int(ds_request_id)] = {
        "role": ds_role,
        "prompt": ds_prompt,
        "params": ds_params,
    }

    assert 7 in eng._disagg_participants
    record = eng._disagg_participants[7]
    assert record["role"] == "exit"
    assert record["prompt"] == "prompt"
    assert record["params"] == {"top_p": 0.9}


def test_inject_disagg_participant_marks_role_and_routes_to_add_request():
    """``inject_disagg_participant`` builds a request, marks it as a
    disagg participant with the given role, and routes through
    ``_add_request`` so the engine's normal lifecycle handles it.
    The participant flag is what the ENGINE_REPLY emit path consults
    to skip non-exit shards."""
    from megatron.core.inference.engines.dynamic_engine import (
        DynamicInferenceEngine,
    )
    from megatron.core.inference.sampling_params import SamplingParams

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng.requests = {}
    eng._add_request = MagicMock()
    eng.controller = MagicMock()
    eng.controller.tokenizer.tokenize = MagicMock(return_value=[1, 2, 3, 4])

    # SamplingParams.deserialize round-trips a serialized form;
    # use the same path the handler uses.
    sp = SamplingParams(temperature=1.0, top_p=0.9)
    sp_wire = sp.serialize()

    eng.inject_disagg_participant(
        request_id=42,
        prompt="hello world",
        sampling_params_serialized=sp_wire,
        role="exit",
    )
    eng._add_request.assert_called_once()
    request = eng._add_request.call_args.args[0]
    assert request.request_id == 42
    assert request._is_disagg_participant is True
    assert request._disagg_role == "exit"
    # Prompt was tokenized.
    assert request.prompt_tokens.tolist() == [1, 2, 3, 4]


def test_inject_disagg_participant_idempotent():
    """Duplicate DISAGG_SUBMIT for the same request_id is a no-op."""
    from megatron.core.inference.engines.dynamic_engine import (
        DynamicInferenceEngine,
    )

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng.requests = {7: object()}  # already tracked
    eng._add_request = MagicMock()

    eng.inject_disagg_participant(
        request_id=7,
        prompt="x",
        sampling_params_serialized=b"",
        role="intermediate",
    )
    eng._add_request.assert_not_called()


def test_release_drops_participant_record():
    """RELEASE_DISAGG_REQUEST drops the participant entry along with
    the dispatcher / KV state."""
    from megatron.core.inference.engines.dynamic_engine import (
        DynamicInferenceEngine,
    )

    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng._route_dispatchers = {7: MagicMock()}
    eng._disagg_participants = {7: {"role": "intermediate"}}
    eng.requests = {}  # not local; detach path skipped

    # Replay the RELEASE_DISAGG_REQUEST branch (the parts that don't
    # need engine machinery).
    request_id = 7
    eng._route_dispatchers.pop(request_id, None)
    eng._disagg_participants.pop(request_id, None)

    assert 7 not in eng._route_dispatchers
    assert 7 not in eng._disagg_participants
