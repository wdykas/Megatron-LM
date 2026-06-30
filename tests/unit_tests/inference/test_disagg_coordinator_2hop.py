# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coordinator-side 2-hop handlers (CPU; fake socket, no ZMQ/engines).

Drives ``_route_submit_disagg`` / ``_handle_prefill_done`` on a coordinator
instance built without ``__init__`` (which needs ZMQ) and asserts the right
control messages go to the right engines: SUBMIT->prefill, then on
PREFILL_DONE a SEND_KV to prefill (decode layouts) + RECV_KV to decode
(prefill layouts + prompt + sampling).
"""

import pytest

msgpack = pytest.importorskip("msgpack")

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.disaggregation.coordinator_routing import DisaggRouting
from megatron.core.inference.headers import Headers


def _coord():
    """Coordinator with just the disagg state populated (bypass ZMQ __init__)."""
    c = DataParallelInferenceCoordinator.__new__(DataParallelInferenceCoordinator)
    c.disaggregated = True
    c._disagg = DisaggRouting()
    c._engine_layouts = {b"p0": ["PREFILL_LAYOUTS"], b"d0": ["DECODE_LAYOUTS"]}
    c._req_meta = {}
    # Credit flow-control state (push by default -> unthrottled, like a real
    # push-backend registration with no is_pull flag).
    c._engine_is_pull = {}
    c._disagg_prefill_of = {}
    c._disagg_outstanding = {}
    c._disagg_submit_queue = {}
    c._disagg_credit_window = 32
    c._disagg.register(b"p0", "prefill")
    c._disagg.register(b"d0", "decode")
    c.sent = []  # (identity, [header, *parts])
    c._send_to_engine = lambda ident, payload: (
        c.sent.append((ident, msgpack.unpackb(payload, raw=False))) or True
    )
    return c


def test_submit_routes_to_prefill_and_stashes_meta():
    c = _coord()
    c._route_submit_disagg(5, [1, 2, 3], {"temperature": 0.0})
    assert c._req_meta[5] == ([1, 2, 3], {"temperature": 0.0})
    assert len(c.sent) == 1
    ident, msg = c.sent[0]
    assert ident == b"p0"
    assert Headers(msg[0]) == Headers.SUBMIT_REQUEST and msg[1] == 5


def test_prefill_done_emits_send_kv_and_recv_kv():
    c = _coord()
    c._route_submit_disagg(5, [1, 2, 3], {"temperature": 0.0})
    c.sent.clear()
    c._handle_prefill_done(5)

    by_ident = {ident: msg for ident, msg in c.sent}
    # SEND_KV -> prefill engine, carrying the decode instance's layouts
    send_kv = by_ident[b"p0"]
    assert Headers(send_kv[0]) == Headers.SEND_KV
    assert send_kv[1] == 5 and send_kv[2] == ["DECODE_LAYOUTS"]
    # RECV_KV -> decode engine, carrying prefill layouts + prompt + sampling
    recv_kv = by_ident[b"d0"]
    assert Headers(recv_kv[0]) == Headers.RECV_KV
    assert recv_kv[1] == 5 and recv_kv[2] == ["PREFILL_LAYOUTS"]
    assert recv_kv[3] == [1, 2, 3] and recv_kv[4] == {"temperature": 0.0}
