# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
import time
from typing import List, Optional, Union

from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_asyncio_loop, trace_async_exceptions

from .headers import Headers

try:
    import zmq

    HAVE_ZMQ = True
except:
    HAVE_ZMQ = False

try:
    import msgpack

    HAVE_MSGPACK = True
except:
    HAVE_MSGPACK = False


class InferenceClient:
    """
    An asynchronous client for communicating with an inference coordinator service.

    This client uses ZeroMQ (ZMQ) for messaging and MessagePack (msgpack) for
    serialization. It is designed to work within an asyncio event loop. It can
    submit inference requests, listen for completed results, and send control
    signals (e.g., pause, stop) to the inference engines.

    The client operates by connecting a ZMQ DEALER socket to the inference
    coordinator's ROUTER socket. Requests are sent with a unique ID, and an
    `asyncio.Future` is created for each request. A background task listens for
    replies from the coordinator, and when a reply is received, it resolves the
    corresponding future with the result.

    Attributes:
        context (zmq.Context): The ZeroMQ context.
        socket (zmq.Socket): The ZMQ DEALER socket used for communication.
        completion_futures (dict[int, asyncio.Future]): A dictionary mapping
            request IDs to the asyncio Future objects that will hold the results.
        next_request_id (int): A counter for generating unique request IDs.
        listener_task (asyncio.Task): The background task that listens for
            completed requests.
    """

    def __init__(self, inference_coordinator_address: str, deserialize: bool = False):
        """
        Initializes the InferenceClient.

        Args:
            inference_coordinator_address (str): The address on which the
                inference coordinator is listening.
            deserialize (bool): If True, deserialize completed requests
                into DynamicInferenceRequest objects. If False (default), return
                the raw serialized dict for lower overhead.
        """
        assert (
            HAVE_ZMQ
        ), "please install the pyzmq library to use InferenceClient - pip install pyzmq"
        assert (
            HAVE_MSGPACK
        ), "please install the messagepack library to use InferenceClient - pip install msgpack"
        self.context = zmq.Context()
        socket = self.context.socket(zmq.DEALER)

        # Prevent socket.send() from thread-blocking at >1000 concurrent requests
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVHWM, 0)

        socket.connect(inference_coordinator_address)

        self._loop = None
        self.socket = socket
        self.deserialize = deserialize
        self.completion_futures = {}
        # client_request_id → asyncio.Future[int] resolved when the coord
        # acks the server-side id via Headers.SUBMIT_REQUEST_ACK. Callers
        # that need the server id (to drive cross-shard migration of the
        # in-flight request) await `wait_for_server_id(client_request_id)`.
        self._server_id_futures = {}
        self.request_submission_times = {}
        self.next_request_id = 0

    def add_request(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        *,
        target_shard_index: Optional[int] = None,
        disagg_dst_shard_index: Optional[int] = None,
        late_dst_shard_index: Optional[int] = None,
        late_dst_min_tokens: Optional[int] = None,
    ) -> asyncio.Future:
        """
        Submits a new inference request to the coordinator.

        This method sends the prompt and sampling parameters to the inference
        coordinator. It immediately returns an asyncio.Future, which can be
        awaited to get the result of the inference request when it is complete.

        Args:
            prompt (str): The input prompt to send to the language model.
            sampling_params: An object containing the sampling parameters for
                text generation (e.g., temperature, top_p). It must have a
                `serialize()` method.
            target_shard_index: If given, instruct the coordinator to route
                this submit to an engine registered for that shard
                (heterogeneous inference, disaggregated prefill/decode).
                ``None`` preserves the legacy unscoped scheduling. Ignored
                by coords that have no engines tagged with shards.
            disagg_dst_shard_index: If given, the auto-disagg scheduler on
                the serving side migrates this request to the named shard
                as soon as it has produced its first decode token. This
                is the per-request opt-in for HTTP-transparent disagg;
                ``None`` means the request is not eligible for migration.

        Returns:
            asyncio.Future: A future that will be resolved with a
            `DynamicInferenceRequest` object (if deserialize=True) or a raw
            serialized dict (if deserialize=False) containing the completed result.
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        # Payload is variable-length so legacy 3-field clients keep
        # working. The coord decodes positionally:
        #   [..., target_shard, disagg_dst, late_dst, late_min]
        # Each later field requires all earlier ones to be present
        # (None is a valid placeholder).
        payload: list = [
            Headers.SUBMIT_REQUEST.value,
            request_id,
            prompt,
            sampling_params.serialize(),
        ]
        late_set = (
            late_dst_shard_index is not None and late_dst_min_tokens is not None
        )
        any_extra = (
            target_shard_index is not None
            or disagg_dst_shard_index is not None
            or late_set
        )
        if any_extra:
            payload.append(target_shard_index)
        if disagg_dst_shard_index is not None or late_set:
            payload.append(disagg_dst_shard_index)
        if late_set:
            payload.append(late_dst_shard_index)
            payload.append(late_dst_min_tokens)
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)
        assert request_id not in self.completion_futures
        loop = asyncio.get_running_loop()
        self.completion_futures[request_id] = loop.create_future()
        # Register an ack-future so a caller can recover the server-side id
        # the coord assigns (see Headers.SUBMIT_REQUEST_ACK). Always create
        # this; it's cheap and callers that don't need the id simply never
        # await it.
        self._server_id_futures[request_id] = loop.create_future()
        self.request_submission_times[request_id] = time.perf_counter()
        return self.completion_futures[request_id]

    async def wait_for_server_id(self, client_request_id: int) -> int:
        """Return the server-side id the coord assigned for this submit.

        Resolves when the coord's SUBMIT_REQUEST_ACK arrives. Callers that
        need the server id (cross-shard migration, direct engine inspection)
        should await this before triggering the downstream op; the engine
        only knows the request by its server id.
        """
        fut = self._server_id_futures.get(client_request_id)
        if fut is None:
            raise KeyError(
                f"no server_id future for client_request_id={client_request_id}; "
                "add_request was not called through this client"
            )
        return await fut

    def update_request_ranks_batch(
        self,
        request_ids: List[int],
        new_shard_index: int,
        new_dp_rank_within_shard: int = 0,
    ) -> None:
        """Inform the coordinator that the listed requests have been
        migrated to ``(new_shard_index, new_dp_rank_within_shard)``.

        Sent by the migration handler after a successful cross-shard
        transfer. The coord rewrites ``request_id_to_rank`` for each id
        and shifts pending-count accounting to the new owner so each
        request's eventual ENGINE_REPLY reaches the original HTTP
        client. All ids must share the same destination.
        """
        self._send_signal_to_engines(
            Headers.UPDATE_REQUEST_RANKS_BATCH,
            list(request_ids),
            int(new_shard_index),
            int(new_dp_rank_within_shard),
        )

    @trace_async_exceptions
    async def _recv_task(self):
        """
        Listens for completed inference requests from the coordinator.

        This coroutine runs in an infinite loop, continuously polling the socket
        for data.
        When a request reply is received, it unpacks the message, finds the
        corresponding Future using the request ID, and sets the result.
        Other control packets are handled appropriately.

        This method is started as a background task by the `start()` method.
        """
        while True:
            try:
                data = msgpack.unpackb(self.socket.recv(flags=zmq.NOBLOCK), raw=False)
                header = Headers(data[0])
                if header == Headers.ENGINE_REPLY:
                    request_id, reply = data[1:]
                    reply['latency'] = time.perf_counter() - self.request_submission_times.pop(
                        request_id
                    )
                    completion_future = self.completion_futures.pop(request_id)
                    # Clean up the server-id future if it was never
                    # awaited so we don't leak entries.
                    self._server_id_futures.pop(request_id, None)
                    if completion_future.done():
                        logging.warning(f"Client: The future for {request_id} has been cancelled!")
                        continue
                    completed_request = (
                        DynamicInferenceRequest.deserialize(reply) if self.deserialize else reply
                    )
                    completion_future.set_result(completed_request)
                elif header == Headers.SUBMIT_REQUEST_ACK:
                    client_request_id, server_request_id = data[1:]
                    fut = self._server_id_futures.get(client_request_id)
                    if fut is not None and not fut.done():
                        fut.set_result(server_request_id)
            except zmq.Again:
                await asyncio.sleep(0.005)
                continue
            except KeyboardInterrupt:
                break

    def _connect_with_inference_coordinator(self):
        """
        Performs the initial handshake with the inference coordinator.

        Sends a CONNECT signal and waits for a CONNECT_ACK reply to ensure the
        connection is established and acknowledged by the coordinator.
        """
        payload = [Headers.CONNECT.value]
        self.socket.send(msgpack.packb(payload, use_bin_type=True))
        reply = msgpack.unpackb(self.socket.recv(), raw=False)[0]
        assert Headers(reply) == Headers.CONNECT_ACK

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Connects to the coordinator and starts the background listener task.

        This must be called before submitting any requests. It handles
        the initial handshake and spawns the `listen_for_completed_requests`
        coroutine.
        """
        logging.info("Client: Connecting to InferenceCoordinator...")
        self._loop = get_asyncio_loop(loop)
        self._connect_with_inference_coordinator()
        self.listener_task = self._loop.create_task(self._recv_task())

    def _send_signal_to_engines(self, signal, *args):
        """
        Sends a generic control signal to the inference coordinator.

        Args:
            signal: The signal to send, typically a value from the `Headers` enum.
            *args: Optional extra values to include in the payload.
        """
        payload = [signal.value, *args]
        payload_serialized = msgpack.packb(payload, use_bin_type=True)
        self.socket.send(payload_serialized)

    def pause_engines(self, *, shard_indices: Optional[List[int]] = None):
        """Sends PAUSE to engines via coordinator.

        The coordinator broadcasts PAUSE. Each engine reaches EP consensus,
        then synchronizes via a world-wide barrier before transitioning to
        PAUSED. Callers should await engine.paused for confirmation.

        Args:
            shard_indices: If provided, only engines registered with
                those shard indices receive the PAUSE — other shards
                keep serving. ``None`` preserves whole-world behavior.

                The cross-shard request-migration handler in this repo
                does **not** use scoped pause — its NVSHMEM transport
                runs without quiescing either shard. The scope hook is
                kept as a downstream-friendly extension point for
                external frameworks (NeMo-RL, verl, etc.) that may want
                to lifecycle individual shards independently.
        """
        self._send_signal_to_engines(Headers.PAUSE, shard_indices)

    def unpause_engines(self, *, shard_indices: Optional[List[int]] = None) -> None:
        """Sends UNPAUSE to all engines. No synchronization needed."""
        self._send_signal_to_engines(Headers.UNPAUSE, shard_indices)

    def set_generation_epoch(self, generation_epoch: int):
        """Sends a signal to stamp all in-flight requests with the given generation epoch.

        Args:
            generation_epoch: The current generation epoch number.
        """
        self._send_signal_to_engines(Headers.SET_GENERATION_EPOCH, generation_epoch)

    def suspend_engines(self, *, shard_indices: Optional[List[int]] = None):
        """Sends SUSPEND to all engines via coordinator. Requires PAUSED.

        Callers should await engine.suspended for confirmation. See
        :meth:`pause_engines` for the ``shard_indices`` contract.
        """
        self._send_signal_to_engines(Headers.SUSPEND, shard_indices)

    def resume_engines(self, *, shard_indices: Optional[List[int]] = None):
        """Sends RESUME to all engines via coordinator. Requires SUSPENDED.

        Callers should await engine.paused (or engine.running after UNPAUSE) for confirmation.
        """
        self._send_signal_to_engines(Headers.RESUME, shard_indices)

    def stop_engines(self):
        """Sends STOP to all engines via coordinator. Requires PAUSED or SUSPENDED.

        Callers should await engine.stopped for confirmation.
        Does not affect the coordinator.
        """
        self._send_signal_to_engines(Headers.STOP)

    def migrate_request_batch(
        self,
        request_ids: List[int],
        src_shard_index: int,
        dst_shard_index: int,
        bundles: Optional[List[dict]] = None,
        dst_dp_rank_within_shard: int = 0,
    ) -> None:
        """Tell engines in src + dst shards to migrate ``request_ids`` from
        src to dst. The coordinator forwards a ``MIGRATE_BATCH`` signal to
        the named shards' engines; each engine pops the signal and runs
        its registered migration handler.

        ``bundles`` is the per-request serialized
        :class:`RequestMigrationBundle` (msgpack-compatible dict, one per
        ``request_ids`` entry). Carrying bundles inline lets engines run
        migration without a cross-shard broadcast — the dst engine has
        the metadata it needs to allocate KV blocks the moment it sees
        the signal. Pass ``None`` only for backward-compat paths that
        rebuild bundles via the cross-shard collective.

        Fire-and-forget: returns once the message is enqueued.
        """
        if bundles is None:
            bundles = []
        self._send_signal_to_engines(
            Headers.MIGRATE_BATCH,
            list(request_ids),
            int(src_shard_index),
            int(dst_shard_index),
            list(bundles),
            int(dst_dp_rank_within_shard),
        )

    def shutdown_coordinator(self):
        """Tells the coordinator process to exit its main loop.

        Does not affect the engines.
        """
        self._send_signal_to_engines(Headers.SHUTDOWN)

    def stop(self):
        """
        Stops the client and cleans up all resources.

        This method cancels the background listener task, closes the ZMQ socket,
        and terminates the ZMQ context. It should be called when the client is
        no longer needed to ensure a graceful shutdown.
        """
        if hasattr(self, 'listener_task') and not self.listener_task.done():
            self.listener_task.cancel()
        # Wake up any listeners.
        for future in self.completion_futures.values():
            if not future.done():
                future.cancel()
        self.completion_futures.clear()
        for future in self._server_id_futures.values():
            if not future.done():
                future.cancel()
        self._server_id_futures.clear()
        self.socket.close(linger=0)
        self.context.term()
