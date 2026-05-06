# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the unified DataParallelInferenceCoordinator.

A unified coord serves heterogeneous shards through a single ZMQ ROUTER
socket: engines register with a shard index, submits carry an optional
target_shard_index, the coord acks server-side ids, and cross-shard
migration is a metadata update (UPDATE_REQUEST_RANK). These tests spin
up a real coord subprocess and talk to it with raw ZMQ sockets —
impersonating engines and a client — so the behavior is exercised end
to end without the rest of the inference stack.
"""
import contextlib
import multiprocessing
import time

import msgpack
import pytest

zmq = pytest.importorskip("zmq")

from megatron.core.inference.data_parallel_inference_coordinator import (
    DataParallelInferenceCoordinator,
)
from megatron.core.inference.headers import Headers


class _NullTokenizer:
    """Minimal tokenizer stub the coord only uses on the reply path.

    The reply path runs detokenize on the engine's finished request; our
    tests send a pre-rendered finished_request dict so detokenize is
    exercised but produces only an empty generated_text.
    """

    def detokenize(self, token_ids):  # pragma: no cover - unused in these tests
        return ""


def _spawn_coord(data_parallel_size: int = 2, max_requests: int = 32):
    ctx = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = ctx.Pipe()
    ready = ctx.Event()
    proc = ctx.Process(
        target=DataParallelInferenceCoordinator.entrypoint,
        kwargs={
            "pipe_connection": child_pipe,
            "ready_event": ready,
            "data_parallel_size": data_parallel_size,
            "tokenizer": _NullTokenizer(),
            "max_requests": max_requests,
            "inference_coordinator_port": None,
            "deterministic_mode": True,
            "block_size_tokens": None,
            "enable_prefix_caching": False,
            "hostname": "127.0.0.1",
        },
    )
    proc.start()
    addr = parent_pipe.recv()
    parent_pipe.close()
    return proc, ready, addr


def _recv_with_timeout(sock, timeout_s: float = 5.0):
    if sock.poll(int(timeout_s * 1000)) == 0:
        raise TimeoutError("no message within timeout")
    return sock.recv()


@pytest.mark.internal
class TestUnifiedCoord:
    """Spawn a real coord process and drive it with raw ZMQ sockets."""

    def _make_engine(self, zctx, addr, identity: bytes, shard_index: int):
        sock = zctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, identity)
        sock.connect(addr)
        sock.send(
            msgpack.packb(
                [Headers.REGISTER_DP_RANK.value, shard_index], use_bin_type=True
            )
        )
        return sock

    def _make_client(self, zctx, addr, identity: bytes = b"client-0"):
        sock = zctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, identity)
        sock.connect(addr)
        # Handshake.
        sock.send(msgpack.packb([Headers.CONNECT.value], use_bin_type=True))
        data = msgpack.unpackb(_recv_with_timeout(sock), raw=False)
        assert Headers(data[0]) == Headers.CONNECT_ACK
        return sock

    @contextlib.contextmanager
    def _coord_up(self, num_shards: int = 2):
        """Spin up the coord with ``num_shards`` engines + one client,
        yield ``(engines, client)``, and clean up on exit.

        Replaces ~15 lines of per-test setup/teardown boilerplate with
        a single ``with`` statement.
        """
        proc, ready, addr = _spawn_coord(data_parallel_size=num_shards)
        zctx = zmq.Context()
        sockets: list = []
        try:
            engines = [
                self._make_engine(
                    zctx, addr, f"eng-s{i}".encode(), shard_index=i
                )
                for i in range(num_shards)
            ]
            sockets += engines
            # ready fires once the coord has processed all N REGISTER_DP_RANK
            # messages from the initial barrier.
            assert ready.wait(timeout=10), "coord did not ready within 10s"
            client = self._make_client(zctx, addr)
            sockets.append(client)
            yield engines, client
        finally:
            for s in sockets:
                try:
                    s.close(linger=0)
                except Exception:
                    pass
            try:
                zctx.term()
            except Exception:
                pass
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)

    def test_shard_aware_routing_and_submit_ack(self):
        """SUBMIT with target_shard routes to that shard only; the coord
        also echoes a SUBMIT_REQUEST_ACK carrying the server_request_id.
        """
        with self._coord_up(num_shards=2) as (engines, client):
            eng0, eng1 = engines
            # Submit targeted at shard 0. The coord should dispatch the
            # engine-facing payload to eng0 (not eng1) and concurrently
            # ack the client with the server_request_id.
            client.send(
                msgpack.packb(
                    [
                        Headers.SUBMIT_REQUEST.value,
                        42,  # client_request_id
                        [1, 2, 3],  # prompt token ids
                        {},  # sampling_params serialized stub
                        0,  # target_shard_index
                    ],
                    use_bin_type=True,
                )
            )

            # eng0 should receive the SUBMIT_REQUEST payload.
            data = msgpack.unpackb(_recv_with_timeout(eng0), raw=False)
            assert Headers(data[0]) == Headers.SUBMIT_REQUEST
            server_request_id_sent_to_engine = data[1]

            # eng1 should NOT receive anything (shard-scoped routing).
            assert eng1.poll(500) == 0, "shard 1 engine wrongly received a shard-0 submit"

            # Client receives the ack with matching server_request_id.
            data = msgpack.unpackb(_recv_with_timeout(client), raw=False)
            assert Headers(data[0]) == Headers.SUBMIT_REQUEST_ACK
            client_request_id, server_request_id_acked = data[1], data[2]
            assert client_request_id == 42
            assert server_request_id_acked == server_request_id_sent_to_engine

            # Now target shard 1; eng1 should get it, eng0 should not.
            client.send(
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 43, [4, 5], {}, 1],
                    use_bin_type=True,
                )
            )
            data = msgpack.unpackb(_recv_with_timeout(eng1), raw=False)
            assert Headers(data[0]) == Headers.SUBMIT_REQUEST
            assert eng0.poll(500) == 0
            # Drain eng1's ack echo to the client.
            _ = msgpack.unpackb(_recv_with_timeout(client), raw=False)

    def test_pause_broadcast_is_shard_scoped(self):
        """PAUSE with a shard_indices scope only reaches the listed shards.

        Without scoping, the coord broadcasts control signals to every
        registered engine; with scoping, other shards keep running. This
        is what lets a cross-shard migration quiesce two shards without
        stopping the world.
        """
        with self._coord_up(num_shards=3) as (engines, client):
            eng0, eng1, eng2 = engines
            # Scoped PAUSE: only shards 0 and 2 should see it.
            client.send(
                msgpack.packb([Headers.PAUSE.value, [0, 2]], use_bin_type=True)
            )

            data = msgpack.unpackb(_recv_with_timeout(eng0), raw=False)
            assert Headers(data[0]) == Headers.PAUSE
            data2 = msgpack.unpackb(_recv_with_timeout(eng2), raw=False)
            assert Headers(data2[0]) == Headers.PAUSE
            # Shard 1 must NOT have received anything; give it a beat to
            # prove no message is en route.
            assert eng1.poll(500) == 0, (
                "shard 1 wrongly received a PAUSE that was scoped to shards 0 and 2"
            )
            # Engines must not see the scope list — it's coord-internal.
            assert len(data) == 1, (
                f"engine payload must be stripped of shard scope; got {data!r}"
            )

    def test_submit_with_disagg_tag_forwards_to_engine(self):
        """SUBMIT with disagg_dst_shard_index threads through to the engine.

        A 5-field SUBMIT payload carries (client_req_id, prompt, params,
        target_shard, disagg_dst). The coord routes via target_shard and
        includes disagg_dst in the engine-facing SUBMIT so the engine
        can stash it on the request for the auto-disagg scheduler.
        """
        with self._coord_up(num_shards=2) as (engines, client):
            eng0, eng1 = engines
            client.send(
                msgpack.packb(
                    [
                        Headers.SUBMIT_REQUEST.value,
                        101,
                        [1, 2, 3],
                        {},
                        0,  # target_shard_index = prefill
                        1,  # disagg_dst_shard_index = decode
                    ],
                    use_bin_type=True,
                )
            )

            data = msgpack.unpackb(_recv_with_timeout(eng0), raw=False)
            assert Headers(data[0]) == Headers.SUBMIT_REQUEST
            assert len(data) == 5, (
                f"engine-facing SUBMIT must include disagg_dst; got {data!r}"
            )
            assert data[4] == 1
            assert eng1.poll(500) == 0, (
                "shard 1 wrongly received a disagg-routed submit"
            )

            # Untagged submit → 4-field engine payload (no trailing tag).
            _ = _recv_with_timeout(client)  # drain the ack
            client.send(
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 102, [4, 5], {}, 1],
                    use_bin_type=True,
                )
            )
            data2 = msgpack.unpackb(_recv_with_timeout(eng1), raw=False)
            assert Headers(data2[0]) == Headers.SUBMIT_REQUEST
            assert len(data2) == 4, (
                f"untagged SUBMIT must stay 4 fields; got {data2!r}"
            )

    def test_update_request_rank_redispatches_reply(self):
        """After UPDATE_REQUEST_RANK moves a live request from shard 0 to
        shard 1, an ENGINE_REPLY sent from shard 1's engine reaches the
        original client — the coord's client table still points at that
        client even though the reply came from the 'wrong' engine post
        migration.
        """
        with self._coord_up(num_shards=2) as (engines, client):
            eng0, eng1 = engines
            client.send(
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 7, [10, 11], {}, 0],
                    use_bin_type=True,
                )
            )
            data = msgpack.unpackb(_recv_with_timeout(eng0), raw=False)
            assert Headers(data[0]) == Headers.SUBMIT_REQUEST
            server_request_id = data[1]
            ack = msgpack.unpackb(_recv_with_timeout(client), raw=False)
            assert Headers(ack[0]) == Headers.SUBMIT_REQUEST_ACK
            assert ack[2] == server_request_id

            # Tell the coord the request moved to shard 1, DP-rank 0.
            client.send(
                msgpack.packb(
                    [
                        Headers.UPDATE_REQUEST_RANK.value,
                        server_request_id,
                        1,
                        0,
                    ],
                    use_bin_type=True,
                )
            )
            # Give the coord a beat to ingest the control message.
            time.sleep(0.2)

            finished = {
                "request_id": server_request_id,
                "prompt": "",
                "prompt_tokens": [[0], [10, 11]],
                "generated_tokens": [99, 100],
                "sampling_params": {},
            }
            eng1.send(
                msgpack.packb(
                    [Headers.ENGINE_REPLY.value, [finished]], use_bin_type=True
                )
            )

            reply = msgpack.unpackb(_recv_with_timeout(client, timeout_s=3.0), raw=False)
            assert Headers(reply[0]) == Headers.ENGINE_REPLY
            assert reply[1] == 7, (
                f"client should see its client_request_id (7), got {reply[1]}"
            )
            assert reply[2]["request_id"] == server_request_id
            assert reply[2]["generated_tokens"] == [99, 100]
