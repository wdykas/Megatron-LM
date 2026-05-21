# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import errno
import faulthandler
import json
import logging
import signal
import socket
from collections import deque
from enum import Enum, auto
from multiprocessing import Event
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np
import torch

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import compute_block_hashes_batched
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

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

# Register faulthandler to emit stack traces upon process kill.
faulthandler.enable()
faulthandler.register(signal.SIGTERM, all_threads=False, chain=True)
faulthandler.register(signal.SIGINT, all_threads=False, chain=True)


class DataParallelInferenceCoordinator:
    """
    Coordinates inference requests between clients and distributed model engines.

    This class acts as a central server. It uses a ZMQ ROUTER socket to manage
    communication flows between multiple clients and multiple data parallel ranks.

    The coordinator's main responsibilities are:
    1.  **Worker Registration**: It waits for a specified number of data parallel ranks
        (representing distributed model instances) to connect and register themselves.
    2.  **Client Connection**: It accepts connections from external clients, like
        `InferenceClient`, and performs a simple handshake.
    3.  **Request Forwarding**: It receives inference requests from clients, assigns a
        unique server-side request ID, tokenizes the prompt, and forwards the request
        to one of the available data parallel rank using a round-robin scheduling
        strategy.
    4.  **Response Routing**: It receives completed results from
        the data parallel ranks and routes them back to the original client that made the
        request.
    5.  **Control Signal Broadcasting**: It relays control signals (e.g., PAUSE, STOP)
        from a client to all connected data parallel ranks.

    Attributes:
        router_socket (zmq.Socket): The central ZMQ ROUTER socket for all communication.
        data_parallel_size (int): The number of data parallel workers to expect.
        identities_of_data_parallel_ranks (deque): A deque holding the ZMQ
            identities of connected TP-coordinators, used for round-robin scheduling.
        request_id_to_client_id (dict): Maps server-side request IDs to the ZMQ
            identity of the client that initiated the request.
        request_id_to_client_request_id (dict): Maps server-side request IDs to the
            original request ID provided by the client.
        next_request_id (int): A counter for generating unique server-side request IDs.
    """

    class CoordinatorState(Enum):
        """State machine for the coordinator."""

        RUNNING = auto()
        PAUSED = auto()
        SUSPENDED = auto()
        STOPPING = auto()

    def __init__(
        self,
        pipe_connection: Connection,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Initializes the inference coordinator.

        This sets up the ZMQ context and a ROUTER socket, binding it to the given
        port. It then enters a blocking loop to wait for all expected data parallel
        ranks to connect before proceeding.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            data_parallel_size (int): The number of TP-coordinator workers that are
                expected to connect.
            tokenizer: The tokenizer to use for prompt tokenization and detokenization.
            inference_coordinator_port (Optional[int]): The TCP port number to bind the server to.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score:
                score = alpha * match + (1 - alpha) * normalized_load.
            max_requests (int): Max concurrent requests per rank, used to
                compute normalized_load for prefix-aware scoring.
        """
        assert HAVE_ZMQ, (
            "please install the pyzmq library to use DataParallelInferenceCoordinator\n"
            "pip install pyzmq"
        )
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use DataParallelInferenceCoordinator\n"
            "pip install msgpack"
        )
        self.pipe_connection = pipe_connection
        self.data_parallel_size = data_parallel_size
        self.context = zmq.Context()

        # This is the central router socket
        # 1. data parallel ranks connect to this socket to register themselves
        # 2. Users connect to this socket and submit their requests. We transmit them to
        #    data parallel ranks in a round robin fashion
        # 3. data parallel ranks return completed requests to this socket. We route them back to
        #    the user that had submitted the request originally.

        # Get local IP.
        local_ip = hostname or socket.gethostname()

        self.router_socket = self.context.socket(zmq.ROUTER)
        # Raise error if the other side of the connection has dropped.
        self.router_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        is_bound = False
        if inference_coordinator_port is not None:
            try:
                self.router_socket.bind(f"tcp://{local_ip}:{inference_coordinator_port}")
                is_bound = True
            except zmq.error.ZMQError as e:
                if e.errno == errno.EADDRINUSE:
                    logging.warning(
                        f"Port {inference_coordinator_port} is already in use. "
                        "Binding to a random available port instead."
                    )
            except Exception:
                logging.warning(
                    f"Unknown error when binding to port {inference_coordinator_port}. "
                    "Attempting to bind to a random available port instead."
                )
        if not is_bound:
            self.router_socket.bind_to_random_port(f"tcp://{local_ip}")
        self.addr = self.router_socket.getsockopt_string(zmq.LAST_ENDPOINT)

        # Send the address to the parent process.
        self.pipe_connection.send(self.addr)
        self.pipe_connection.close()

        logging.info("Inference Coordinator: waiting for connections from data parallel ranks...")
        # First wait for all data parallel ranks to establish connections.
        self.identities_of_data_parallel_ranks = deque([])
        # identity → shard_index collected during the initial barrier.
        # REGISTER_DP_RANK carries the tag; legacy empty-bytes leaves it
        # None (unscoped, matches any target_shard_index). The third
        # field (max_requests) lets the coord capacity-check
        # ``MIGRATE_BATCH`` payloads against the dst shard's slot-table
        # before forwarding them.
        shard_tags: dict[bytes, int | None] = {}
        engine_max_requests: dict[bytes, int | None] = {}
        for _ in range(data_parallel_size):
            identity, serialized_payload = self.router_socket.recv_multipart()
            assert identity not in self.identities_of_data_parallel_ranks
            self.identities_of_data_parallel_ranks.append(identity)
            if serialized_payload == b"":
                shard_tags[identity] = None
                engine_max_requests[identity] = None
                continue
            try:
                deserialized = msgpack.unpackb(serialized_payload, raw=False)
                assert Headers(deserialized[0]) == Headers.REGISTER_DP_RANK
                shard_tags[identity] = (
                    deserialized[1] if len(deserialized) > 1 else None
                )
                engine_max_requests[identity] = (
                    int(deserialized[2]) if len(deserialized) > 2 else None
                )
            except Exception as e:
                logging.warning(
                    "Coordinator: could not parse registration from %s: %s", identity, e
                )
                shard_tags[identity] = None
                engine_max_requests[identity] = None
        logging.info("Inference Coordinator: Connected with data parallel ranks...")

        # In deterministic mode, sort identities for consistent scheduling order.
        if deterministic_mode:
            self.identities_of_data_parallel_ranks = deque(
                sorted(self.identities_of_data_parallel_ranks)
            )
        self._round_robin_idx = 0

        self.request_id_to_client_id = {}
        self.request_id_to_client_request_id = {}
        self.request_id_to_rank = {}  # Maps request_id → rank identity for pending count tracking

        self.next_request_id = 0
        self.tokenizer = tokenizer
        self.state = self.CoordinatorState.RUNNING

        # Layout-wide layer-kind disagg route. ``None`` means no disagg
        # is configured for this coord; every SUBMIT_REQUEST is processed
        # collocated. When set (via ``Headers.SET_DISAGG_ROUTE`` from a
        # client at launch time), every SUBMIT_REQUEST triggers an
        # auto-fanned ``ROUTE_REQUEST(server_request_id, stored_route)``
        # to participating shards before forwarding the SUBMIT. Stored as
        # the wire form (list-of-hops) — coord doesn't reconstruct the
        # ``Route`` object.
        self._disagg_route: Optional[list] = None

        # Prefix caching state for routing.
        self.block_size_tokens = block_size_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_caching_coordinator_policy = prefix_caching_coordinator_policy
        self.prefix_caching_routing_alpha = prefix_caching_routing_alpha
        self.max_requests = max_requests
        assert self.max_requests is not None and self.max_requests > 0

        # Schedule recording.
        self.schedule_output_path = schedule_output_path
        self.schedule_records = [] if schedule_output_path else None

        # Deterministic rank index mapping (sorted identity -> 0-based index).
        sorted_identities = sorted(self.identities_of_data_parallel_ranks)
        self.identity_to_rank_index = {
            identity: idx for idx, identity in enumerate(sorted_identities)
        }

        # Numpy arrays for vectorized scoring (indexed by rank index).
        n_ranks = len(sorted_identities)
        self._identities_list = list(sorted_identities)  # rank_index → identity
        self._pending_counts = np.zeros(n_ranks, dtype=np.int32)

        # Per-engine slot-table cap, used to gate ``MIGRATE_BATCH``
        # against the dst shard's headroom. Engines that registered
        # without a cap (legacy ``b""`` handshake or older
        # ``REGISTER_DP_RANK`` payloads without the field) effectively
        # opt out — ``-1`` is treated as "no cap" and migrations are
        # always forwarded for those engines (legacy behaviour).
        self._engine_max_requests = np.array(
            [
                engine_max_requests.get(ident) if engine_max_requests.get(ident) is not None else -1
                for ident in sorted_identities
            ],
            dtype=np.int64,
        )

        # Hash → {rank_idx: timestamp} dict for prefix cache affinity routing.
        # Each key is a block hash; each value maps rank indices to assignment
        # timestamps (positive int).  Missing entries are implicitly zero.
        self._hash_table: dict[int, dict[int, int]] = {}
        self._hash_assignment_counter = 0

        # Shard-aware routing: heterogeneous serving tags each engine
        # with a shard_index so submits can carry target_shard_index and
        # migration is a rank-mapping update (Headers.UPDATE_REQUEST_RANK).
        # _shard_to_identities is a lazy cache keyed by shard_index.
        self.identity_to_shard_index: dict[bytes, int | None] = {
            ident: shard_tags.get(ident) for ident in self._identities_list
        }
        self._shard_to_identities: dict[int, list[bytes]] = {}

    def get_next_data_parallel_rank(self, shard_index: int | None = None):
        """
        Selects the next data parallel rank using round-robin scheduling.

        Args:
            shard_index: If given, round-robin only across engines that
                registered with that shard_index. Engines that registered
                without shard metadata never match a shard-scoped pick.

        Returns:
            bytes: The ZMQ identity of the next data parallel rank to receive a request.
        """
        identities = self._identities_for_shard(shard_index)
        if not identities:
            raise RuntimeError(
                f"No engines connected"
                + (f" for shard {shard_index}" if shard_index is not None else "")
            )
        idx = self._round_robin_idx % len(identities)
        self._round_robin_idx = idx + 1
        return identities[idx]

    def _identities_for_shard(self, shard_index: int | None) -> list[bytes]:
        """Ordered list of engine identities eligible for a routing decision.

        Unscoped (shard_index=None) returns every connected engine. A scoped
        pick returns only those engines whose REGISTER_DP_RANK carried that
        shard_index; registration metadata is stable across a session so we
        rebuild the list whenever membership changes.
        """
        if shard_index is None:
            return list(self.identities_of_data_parallel_ranks)
        cached = self._shard_to_identities.get(shard_index)
        if cached is None:
            cached = [
                ident
                for ident in self.identities_of_data_parallel_ranks
                if self.identity_to_shard_index.get(ident) == shard_index
            ]
            self._shard_to_identities[shard_index] = cached
        return cached

    def _invalidate_shard_cache(self):
        self._shard_to_identities.clear()

    def _identity_for_dp_rank(
        self, shard_index: int, dp_rank: int
    ) -> bytes | None:
        """Return the engine identity ``mp-coord-s{shard_index}-{dp_rank}``,
        or ``None`` if no engine in the shard advertises that suffix.

        Match by suffix (not by registration order) so MIGRATE_BATCH and
        UPDATE_REQUEST_RANK always pick the same physical rank.
        """
        suffix = f"-{dp_rank}".encode()
        for ident in self._identities_for_shard(shard_index):
            if ident.endswith(suffix):
                return ident
        return None

    def _register_rank_identity(
        self,
        identity,
        shard_index: int | None = None,
        max_requests: int | None = None,
    ):
        """Register a new rank identity in the scoring data structures.

        Called when a rank dynamically connects to a running coordinator
        (e.g. in tests that spawn the coordinator with data_parallel_size=0
        and let engines register after the fact).

        Args:
            shard_index: Optional shard tag for this engine. When given, the
                coord will only dispatch shard-scoped submits to engines that
                share this index. ``None`` preserves legacy unscoped behavior.
            max_requests: Slot-table cap for this engine, used to gate
                MIGRATE_BATCH against the dst shard's headroom. ``None``
                or unset means no cap (legacy behaviour — migrations
                always forwarded).
        """
        if identity in self.identity_to_rank_index:
            # Identity already known; allow a late shard_index to attach.
            if shard_index is not None and self.identity_to_shard_index.get(identity) is None:
                self.identity_to_shard_index[identity] = shard_index
                self._invalidate_shard_cache()
            return
        new_idx = len(self._identities_list)
        self.identity_to_rank_index[identity] = new_idx
        self._identities_list.append(identity)
        self._pending_counts = np.append(self._pending_counts, np.int32(0))
        # Keep ``_engine_max_requests`` in lockstep with the rank
        # index space, otherwise the MIGRATE_BATCH capacity check
        # indexes past the array end when a late-registered engine
        # is the dst.
        cap = -1 if max_requests is None else int(max_requests)
        self._engine_max_requests = np.append(
            self._engine_max_requests, np.int64(cap)
        )
        self.identity_to_shard_index[identity] = shard_index
        self._invalidate_shard_cache()
        logging.info(
            "Coordinator: registered engine %s as rank index %d"
            " (shard=%s, now %d engines)",
            identity,
            new_idx,
            shard_index,
            len(self._identities_list),
        )

    def _remove_engine(self, identity):
        """Remove a disconnected engine from the routing pool."""
        self.identities_of_data_parallel_ranks.remove(identity)
        self.identity_to_shard_index.pop(identity, None)
        self._invalidate_shard_cache()
        logging.warning(
            "Coordinator: removed engine %s (now %d engines)",
            identity,
            len(self.identities_of_data_parallel_ranks),
        )

    def _send_to_engine(self, identity, payload):
        """Send payload to an engine, removing it from the pool if unreachable.

        Returns:
            True if the send succeeded, False if the engine was unreachable and removed.
        """
        try:
            self.router_socket.send_multipart([identity, payload])
            return True
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EHOSTUNREACH:
                self._remove_engine(identity)
                return False
            raise

    def compute_request_hashes(self, prompt):
        """Compute block hashes for a prompt on CPU.

        Args:
            prompt: Either a string (to be tokenized) or a list of token IDs.

        Returns:
            List of integer block hashes, or empty list if prefix caching is disabled.
        """
        if not self.enable_prefix_caching or self.block_size_tokens is None:
            return []
        if isinstance(prompt, str):
            tokens = self.tokenizer.tokenize(prompt)
        else:
            tokens = list(prompt)
        token_tensor = torch.tensor(tokens, dtype=torch.int64)
        return compute_block_hashes_batched(token_tensor, self.block_size_tokens)

    def get_best_data_parallel_rank(self, request_hashes, shard_index: int | None = None):
        """Select the best DP rank based on prefix cache affinity and load.

        Uses a scoring function: score = alpha * match + (1 - alpha) * normalized_load
        where *match* is a policy-dependent affinity score in [0, 1] (binary for
        ``first_prefix_block``, normalized prefix depth for ``longest_prefix``)
        and normalized_load = free_slots / max_requests (higher means more free
        capacity).

        Args:
            request_hashes: List of block hashes for the request.
            shard_index: If given, restrict the candidate pool to engines
                that registered with this shard_index. Used by unified coord
                to implement heterogeneous shard routing; ``None`` picks
                from every connected engine.

        Returns:
            bytes: The ZMQ identity of the selected data parallel rank.
        """
        if self.prefix_caching_coordinator_policy == PrefixCachingCoordinatorPolicy.ROUND_ROBIN:
            return self.get_next_data_parallel_rank(shard_index=shard_index)

        if not self.enable_prefix_caching or not request_hashes:
            return self.get_next_data_parallel_rank(shard_index=shard_index)

        match, recency = self._match_vector(request_hashes)

        alpha = self.prefix_caching_routing_alpha

        # Vectorized score: alpha * match + (1-alpha) * free_capacity_fraction.
        free_slots = np.maximum(0, self.max_requests - self._pending_counts).astype(np.float64)
        scores = alpha * match + (1.0 - alpha) * (free_slots / self.max_requests)

        n_ranks = len(self._identities_list)
        if shard_index is not None:
            # Mask out identities outside the target shard.
            eligible = np.array(
                [
                    self.identity_to_shard_index.get(self._identities_list[i]) == shard_index
                    for i in range(n_ranks)
                ],
                dtype=bool,
            )
            if not eligible.any():
                raise RuntimeError(
                    f"No engines connected for shard {shard_index}"
                )
            # Push non-eligible scores to -inf so lexsort never picks them.
            scores = np.where(eligible, scores, -np.inf)
            recency = np.where(eligible, recency, -np.inf)

        # Tiebreak: highest score, then highest recency, then lowest rank index.
        order = np.lexsort((np.arange(n_ranks), -recency, -scores))
        best_idx = int(order[0])
        return self._identities_list[best_idx]

    def _update_rank_hashes(self, rank_identity, request_hashes):
        """Record that a rank owns the given hashes.

        Args:
            rank_identity: ZMQ identity of the target rank.
            request_hashes: List of block hashes assigned to this rank.
        """
        rank_idx = self.identity_to_rank_index[rank_identity]
        self._hash_assignment_counter += 1
        ts = self._hash_assignment_counter
        for h in request_hashes:
            self._hash_table.setdefault(h, {})[rank_idx] = ts

    def _match_vector(self, hashes):
        """Return ``(match, recency)`` vectors of shape ``(n_ranks,)``.

        *match* is binary depth: ``(depth + 1) / len(hashes)`` for ranks that
        have the deepest cached block, 0 otherwise.  *recency* is the raw
        assignment timestamp for each matching rank (0 for non-matching ranks).

        For ``FIRST_PREFIX_BLOCK`` the caller already truncates *hashes* to a
        single element, so the same logic yields a binary 0/1 match score.
        """
        n_ranks = len(self._identities_list)
        n = len(hashes)
        zeros = np.zeros(n_ranks, dtype=np.float64)
        if n == 0:
            return zeros, zeros.copy()
        for i in range(n - 1, -1, -1):
            row = self._hash_table.get(hashes[i])
            if row is None:
                continue
            rank_idxs = np.fromiter(row.keys(), dtype=np.intp)
            present = np.zeros(n_ranks, dtype=bool)
            present[rank_idxs] = True
            recency = np.zeros(n_ranks, dtype=np.float64)
            recency[rank_idxs] = np.fromiter(row.values(), dtype=np.float64)
            if present.any():
                return present.astype(np.float64) * ((i + 1.0) / n), recency
        return zeros, zeros.copy()

    def _advance_lifecycle_state(self, header) -> bool:
        """Validate + apply a whole-world coord state transition for
        PAUSE / UNPAUSE / SUSPEND / RESUME / STOP.

        Returns ``True`` if the caller should proceed to broadcast the
        signal; ``False`` if the header was ignored (idempotent or
        out-of-order) and the caller should continue.
        """
        State = self.CoordinatorState
        if header == Headers.PAUSE:
            if self.state == State.RUNNING:
                self.state = State.PAUSED
                return True
            if self.state in (State.PAUSED, State.SUSPENDED):
                return False  # idempotent
        elif header == Headers.UNPAUSE:
            if self.state == State.PAUSED:
                self.state = State.RUNNING
                return True
        elif header == Headers.SUSPEND:
            if self.state == State.PAUSED:
                self.state = State.SUSPENDED
                return True
        elif header == Headers.RESUME:
            if self.state == State.SUSPENDED:
                self.state = State.PAUSED
                return True
        elif header == Headers.STOP:
            if self.state in (State.PAUSED, State.SUSPENDED):
                self.state = State.STOPPING
                return True
        else:
            # Other headers don't drive the lifecycle state machine.
            return True
        logging.warning("Coordinator: ignoring %s in state %s", header.name, self.state)
        return False

    def start(self):
        """
        Starts the main event loop for the coordinator.

        This method runs an infinite loop, continuously listening for incoming
        messages on the ZMQ ROUTER socket. It parses the message header to
        determine the message type and takes appropriate action, such as
        handling new client connections, forwarding requests, broadcasting
        control signals, or processing replies from the engines.
        """
        # Todo [Siddharth]: Make this more robust to handle invalid messages.
        known_clients = set()
        while True:
            sender_identity, serialized_payload = self.router_socket.recv_multipart()

            # Allow for re-registration if connecting to a running coordinator.
            if serialized_payload == b"":
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    self.identities_of_data_parallel_ranks.append(sender_identity)
                    self._register_rank_identity(sender_identity)
                continue

            deserialized_payload = msgpack.unpackb(serialized_payload, raw=False)
            header = Headers(deserialized_payload[0])

            if header == Headers.REGISTER_DP_RANK:
                # Engine-side shard-aware registration. Payload:
                # [REGISTER_DP_RANK, shard_index_or_None,
                #  max_requests_or_absent]. The third field is the
                # engine's slot-table cap, used to gate MIGRATE_BATCH
                # against dst headroom.
                shard_index = deserialized_payload[1] if len(deserialized_payload) > 1 else None
                max_requests = (
                    int(deserialized_payload[2])
                    if len(deserialized_payload) > 2
                    else None
                )
                if sender_identity not in self.identities_of_data_parallel_ranks:
                    self.identities_of_data_parallel_ranks.append(sender_identity)
                self._register_rank_identity(
                    sender_identity,
                    shard_index=shard_index,
                    max_requests=max_requests,
                )
                continue

            if header == Headers.CONNECT:
                if sender_identity in known_clients:
                    logging.info(
                        f"Client {sender_identity} sent a duplicate connect request. Ignoring .."
                    )
                    continue

                # print(f"New client connected: {sender_identity}")
                known_clients.add(sender_identity)
                self.router_socket.send_multipart(
                    [sender_identity, msgpack.packb([Headers.CONNECT_ACK.value], use_bin_type=True)]
                )

            elif header == Headers.SUBMIT_REQUEST:
                # ToDo [Siddharth]: We might want to tokenize the prompt on the
                # assigned data parallel rank for this process instead
                # of the coordinator.

                # Message from a known client
                if sender_identity not in known_clients:
                    logging.info(
                        f"Received message from unknown client {sender_identity}. Ignoring."
                    )
                    continue
                # this is a message from a client.
                # route it to a data parallel rank
                # Variable-length SUBMIT:
                #   3: (client_req_id, prompt, params)
                #   4: + target_shard_index (heterogeneous routing)
                #   5: + disagg_dst_shard_index (auto-disagg opt-in)
                #   7: + late_dst_shard_index, late_dst_min_tokens
                #      (two-stage tail-cut migration)
                #   8: + per_request_disagg_route (overrides stored layout
                #        route for this submit; layer-kind disagg)
                fields = deserialized_payload[1:]
                if len(fields) not in (3, 4, 5, 7, 8):
                    raise ValueError(
                        f"SUBMIT_REQUEST expected 3, 4, 5, 7, or 8 fields, got {len(fields)}"
                    )
                client_request_id, prompt, sampling_params = fields[:3]
                target_shard_index = fields[3] if len(fields) >= 4 else None
                disagg_dst_shard_index = fields[4] if len(fields) >= 5 else None
                late_dst_shard_index = fields[5] if len(fields) >= 7 else None
                late_dst_min_tokens = fields[6] if len(fields) >= 7 else None
                per_request_route = fields[7] if len(fields) >= 8 else None
                # map client request_id to server request_id
                # necessary because multiple clients might have the same request_id.
                request_id = self.next_request_id
                self.next_request_id += 1
                self.request_id_to_client_id[request_id] = sender_identity
                self.request_id_to_client_request_id[request_id] = client_request_id

                # Echo the server-side id back to the client. Clients that
                # need to drive mid-flight operations on the request
                # (e.g. cross-shard migration) await this ack to learn the
                # server id that both the engine and UPDATE_REQUEST_RANK
                # expect.
                self.router_socket.send_multipart(
                    [
                        sender_identity,
                        msgpack.packb(
                            [
                                Headers.SUBMIT_REQUEST_ACK.value,
                                client_request_id,
                                request_id,
                            ],
                            use_bin_type=True,
                        ),
                    ]
                )

                # Layer-kind disagg: a per-request route (if supplied on
                # the SUBMIT) takes precedence over the layout-wide
                # stored route. Either source kicks off the same
                # ROUTE_REQUEST fan-out before forwarding the SUBMIT to
                # the entry shard. ZMQ DEALER/ROUTER preserves per-peer
                # order, so each engine sees ROUTE_REQUEST before SUBMIT
                # for this request_id and has the dispatcher registered
                # when forward fires.
                #
                # Non-entry participating shards additionally receive a
                # DISAGG_SUBMIT carrying the same prompt + params plus a
                # role tag, so each engine allocates per-shard state and
                # drives forward in lockstep with the entry shard. Only
                # role="exit" samples + emits ENGINE_REPLY.
                effective_route = (
                    per_request_route
                    if per_request_route is not None
                    else self._disagg_route
                )
                if effective_route is not None:
                    route_payload = msgpack.packb(
                        [
                            Headers.ROUTE_REQUEST.value,
                            request_id,
                            effective_route,
                        ],
                        use_bin_type=True,
                    )
                    entry_shard = effective_route[0][0]
                    exit_shard = effective_route[-1][0]
                    participating_shards = {h[0] for h in effective_route}
                    for shard_idx in sorted(participating_shards):
                        for ident in self._identities_for_shard(shard_idx):
                            self._send_to_engine(ident, route_payload)
                    # Fan DISAGG_SUBMIT to non-entry participating
                    # shards. Entry shard receives the normal SUBMIT
                    # forwarded below.
                    for shard_idx in sorted(participating_shards):
                        if shard_idx == entry_shard:
                            continue
                        role = "exit" if shard_idx == exit_shard else "intermediate"
                        disagg_submit_payload = msgpack.packb(
                            [
                                Headers.DISAGG_SUBMIT.value,
                                request_id,
                                prompt,
                                sampling_params,
                                role,
                            ],
                            use_bin_type=True,
                        )
                        for ident in self._identities_for_shard(shard_idx):
                            self._send_to_engine(ident, disagg_submit_payload)

                # Serialize prompt.
                if isinstance(prompt, (str, list)):
                    pass
                elif isinstance(prompt, torch.Tensor):
                    prompt = prompt.tolist()
                else:
                    raise Exception("specialize for <%s> prompt." % type(prompt).__name__)

                # Forward to engine. Include disagg / late tags when
                # set so the engine stashes them on the request for the
                # auto-disagg scheduler to read. Engines accept 3-, 4-,
                # or 6-field SUBMIT payloads.
                engine_fields: list = [
                    Headers.SUBMIT_REQUEST.value,
                    request_id,
                    prompt,
                    sampling_params,
                ]
                late_set = (
                    late_dst_shard_index is not None
                    and late_dst_min_tokens is not None
                )
                if disagg_dst_shard_index is not None or late_set:
                    engine_fields.append(disagg_dst_shard_index)
                if late_set:
                    engine_fields.append(late_dst_shard_index)
                    engine_fields.append(late_dst_min_tokens)
                payload = msgpack.packb(engine_fields, use_bin_type=True)

                request_hashes = self.compute_request_hashes(prompt)
                if (
                    self.prefix_caching_coordinator_policy
                    == PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
                ):
                    request_hashes = request_hashes[:1]

                # Account for the fact that some engines may have died.
                # Retries are bounded by either the total number of engines
                # (unscoped submit) or the engines in the target shard.
                candidates = self._identities_for_shard(target_shard_index)
                for _ in range(len(candidates)):
                    next_identity = self.get_best_data_parallel_rank(
                        request_hashes, shard_index=target_shard_index
                    )
                    if self._send_to_engine(next_identity, payload):
                        break
                else:
                    # If all engines have died, we are in an abnormal state, and must exit cleanly.
                    logging.error(
                        "Coordinator: no reachable engines for request %d (shard=%s)",
                        request_id,
                        target_shard_index,
                    )
                    del self.request_id_to_client_id[request_id]
                    del self.request_id_to_client_request_id[request_id]
                    return

                self.request_id_to_rank[request_id] = next_identity
                self._pending_counts[self.identity_to_rank_index[next_identity]] += 1
                if request_hashes:
                    self._update_rank_hashes(next_identity, request_hashes)
                if self.schedule_records is not None:
                    self.schedule_records.append(
                        {
                            "request_id": request_id,
                            "rank_index": self.identity_to_rank_index[next_identity],
                            "num_hashes": len(request_hashes),
                        }
                    )

            elif header in (
                Headers.PAUSE,
                Headers.UNPAUSE,
                Headers.SUSPEND,
                Headers.RESUME,
                Headers.SET_GENERATION_EPOCH,
                Headers.STOP,
            ):
                # Start by checking the current state against the control signal.
                if sender_identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue

                # Optional shard-scope: PAUSE/UNPAUSE/SUSPEND/RESUME may
                # carry a trailing ``shard_indices`` list (or ``None``).
                # When set, the broadcast is filtered to engines in those
                # shards only, and the coord's own state machine — which
                # tracks whole-world state — is intentionally bypassed
                # (per-engine state machines still validate). This is a
                # downstream-friendly extension point: in-tree the
                # cross-shard migration handler runs without scoped pause
                # (NVSHMEM transport, no quiesce required), but external
                # frameworks may use it to lifecycle individual shards.
                shard_indices: list | None = None
                if header in (
                    Headers.PAUSE,
                    Headers.UNPAUSE,
                    Headers.SUSPEND,
                    Headers.RESUME,
                ) and len(deserialized_payload) > 1:
                    trailing = deserialized_payload[1]
                    if trailing is None or isinstance(trailing, (list, tuple)):
                        shard_indices = list(trailing) if trailing else None

                if shard_indices is None:
                    # Whole-world path — validate + apply the coord
                    # state-machine transition. Scoped path bypasses the
                    # whole-world state (per-engine state machines still
                    # validate).
                    if not self._advance_lifecycle_state(header):
                        continue

                # Determine broadcast targets. When a scope is set, only
                # those shards' engines receive the signal; other shards
                # keep running.
                if shard_indices is None:
                    targets = list(self.identities_of_data_parallel_ranks)
                else:
                    targets = []
                    for sidx in shard_indices:
                        targets.extend(self._identities_for_shard(sidx))

                # Always broadcast a clean payload (without the shard
                # scope) to engines so engine-side handlers don't need
                # to care about the scoping knob.
                if header == Headers.SET_GENERATION_EPOCH:
                    # SET_GENERATION_EPOCH carries its data arg at pos 1.
                    forward_payload = msgpack.packb(
                        [header.value, deserialized_payload[1]], use_bin_type=True
                    )
                else:
                    forward_payload = msgpack.packb([header.value], use_bin_type=True)
                for data_parallel_rank_id in targets:
                    self._send_to_engine(data_parallel_rank_id, forward_payload)

                # STOP affects engines; reset coordinator to RUNNING to allow future engines.
                if header == Headers.STOP and shard_indices is None:
                    self.state = self.CoordinatorState.RUNNING

            elif header == Headers.ENGINE_REPLY:
                # This is the output of a single engine step on some data parallel rank.
                assert sender_identity in self.identities_of_data_parallel_ranks
                finished_requests = deserialized_payload[1]

                for finished_request in finished_requests:
                    self.detokenize(finished_request)
                    fid = finished_request["request_id"]
                    client_identity = self.request_id_to_client_id[fid]
                    client_request_identity = self.request_id_to_client_request_id[fid]
                    del self.request_id_to_client_id[fid]
                    del self.request_id_to_client_request_id[fid]
                    assigned_rank = self.request_id_to_rank.pop(fid, None)
                    if assigned_rank is not None:
                        idx = self.identity_to_rank_index.get(assigned_rank)
                        if idx is not None:
                            assert self._pending_counts[idx] >= 1
                            self._pending_counts[idx] -= 1

                    self.router_socket.send_multipart(
                        [
                            client_identity,
                            msgpack.packb(
                                [header.value, client_request_identity, finished_request],
                                use_bin_type=True,
                            ),
                        ]
                    )

            elif header == Headers.MIGRATE_BATCH:
                # Payload: [MIGRATE_BATCH, batch_id, request_ids,
                # src_shard_index, dst_shard_index, bundles, dst_dp_rank].
                # Bundles are the per-request serialized envelopes; carried
                # inline so engines can run async migration without a
                # cross-shard broadcast. Coord forwards the original bytes
                # to every rank in src + the single chosen dst dp_rank
                # **only if the dst engine has headroom for the batch**.
                # Two-phase commit prevents data-loss when dst is at
                # capacity: the decider's ``InferenceClient`` awaits the
                # MIGRATE_BATCH_ACK reply before posting
                # UPDATE_REQUEST_RANKS_BATCH, so a rejected batch leaves
                # both src and the coord's ownership map untouched and
                # the decider retries on a later tick.
                if sender_identity not in known_clients:
                    logging.warning(
                        "Coordinator: ignoring MIGRATE_BATCH from unknown client."
                    )
                    continue
                (
                    _,
                    batch_id,
                    request_ids_in_batch,
                    src_shard_index,
                    dst_shard_index,
                    _,
                    dst_dp_rank,
                ) = deserialized_payload
                src_targets = self._identities_for_shard(src_shard_index)
                dst_target = self._identity_for_dp_rank(dst_shard_index, dst_dp_rank)
                if dst_target is None:
                    dst_idents = self._identities_for_shard(dst_shard_index)
                    logging.warning(
                        "Coordinator: dst_dp_rank=%d not found among %s for "
                        "shard %d; falling back to first registered engine.",
                        dst_dp_rank,
                        [i.decode(errors='replace') for i in dst_idents],
                        dst_shard_index,
                    )
                    if not dst_idents:
                        # No dst engines at all — reply rejected so the
                        # decider doesn't hang on the ack.
                        self.router_socket.send_multipart(
                            [
                                sender_identity,
                                msgpack.packb(
                                    [Headers.MIGRATE_BATCH_ACK.value, batch_id, False],
                                    use_bin_type=True,
                                ),
                            ]
                        )
                        continue
                    dst_target = dst_idents[0]

                # Capacity check against the dst engine's slot-table cap.
                # ``_engine_max_requests`` is populated from
                # REGISTER_DP_RANK; engines that registered without a cap
                # (-1) are treated as unconstrained for backward compat.
                dst_idx = self.identity_to_rank_index.get(dst_target)
                accepted = True
                if dst_idx is not None:
                    cap = int(self._engine_max_requests[dst_idx])
                    if cap >= 0:
                        in_flight = int(self._pending_counts[dst_idx])
                        if in_flight + len(request_ids_in_batch) > cap:
                            accepted = False
                            logging.info(
                                "Coordinator: rejecting MIGRATE_BATCH (batch_id=%d, "
                                "n=%d) → shard %d dp_rank=%d: would exceed cap "
                                "(%d in flight + %d > %d); decider will retry.",
                                batch_id,
                                len(request_ids_in_batch),
                                dst_shard_index,
                                dst_dp_rank,
                                in_flight,
                                len(request_ids_in_batch),
                                cap,
                            )

                # Reply to the decider's client with the accept/reject
                # decision before forwarding (or instead of forwarding
                # on reject).
                self.router_socket.send_multipart(
                    [
                        sender_identity,
                        msgpack.packb(
                            [Headers.MIGRATE_BATCH_ACK.value, batch_id, accepted],
                            use_bin_type=True,
                        ),
                    ]
                )

                if accepted:
                    for data_parallel_rank_id in [*src_targets, dst_target]:
                        self._send_to_engine(data_parallel_rank_id, serialized_payload)

            elif header == Headers.ROUTE_REQUEST:
                # Layer-kind disagg: ship a per-request route plan to
                # every shard the request will visit. Wire payload:
                # [ROUTE_REQUEST, request_id, route_hops]. No
                # two-phase commit — disagg shards are forced
                # participants.
                if sender_identity not in known_clients:
                    logging.warning(
                        "Coordinator: ignoring ROUTE_REQUEST from unknown client."
                    )
                    continue
                _, route_request_id, route_hops = deserialized_payload
                participating_shards = {h[0] for h in route_hops}
                for shard_idx in sorted(participating_shards):
                    for ident in self._identities_for_shard(shard_idx):
                        self._send_to_engine(ident, serialized_payload)
                logging.debug(
                    "Coordinator: ROUTE_REQUEST request_id=%d entry=%d exit=%d "
                    "shards=%s",
                    route_request_id,
                    route_hops[0][0],
                    route_hops[-1][0],
                    sorted(participating_shards),
                )

            elif header == Headers.DISAGG_TOKEN:
                # Exit shard sampled a token; fan to every other
                # participating shard so they sync ``generated_tokens``
                # for the next step's embedding / KV bookkeeping.
                # Wire payload:
                # [DISAGG_TOKEN, request_id, token_id, participating_shards].
                _, dt_request_id, dt_token, dt_participating = deserialized_payload
                fanout_payload = msgpack.packb(
                    [Headers.DISAGG_TOKEN.value, int(dt_request_id), int(dt_token)],
                    use_bin_type=True,
                )
                for shard_idx in sorted(dt_participating):
                    for ident in self._identities_for_shard(shard_idx):
                        if ident == sender_identity:
                            continue  # don't echo back to the sender
                        self._send_to_engine(ident, fanout_payload)

            elif header == Headers.RELEASE_DISAGG_REQUEST:
                # Entry shard's engine finished a disagg request and is
                # telling the coord to fan release to every other
                # participating shard so they free dispatchers + KV /
                # mamba state. Wire payload:
                # [RELEASE_DISAGG_REQUEST, request_id, participating_shards].
                _, release_request_id, participating_shards = deserialized_payload
                fanout_payload = msgpack.packb(
                    [
                        Headers.RELEASE_DISAGG_REQUEST.value,
                        int(release_request_id),
                    ],
                    use_bin_type=True,
                )
                for shard_idx in sorted(participating_shards):
                    for ident in self._identities_for_shard(shard_idx):
                        if ident == sender_identity:
                            continue  # don't echo back to the sender
                        self._send_to_engine(ident, fanout_payload)

            elif header == Headers.SET_DISAGG_ROUTE:
                # Layer-kind disagg producer wires the layout-wide route
                # once at launch. The coord stores it and auto-fans
                # ROUTE_REQUEST on every subsequent SUBMIT_REQUEST. Pass
                # [SET_DISAGG_ROUTE, None] to clear (e.g. during shutdown
                # or layout swap). Payload: [SET_DISAGG_ROUTE, route_hops].
                if sender_identity not in known_clients:
                    logging.warning(
                        "Coordinator: ignoring SET_DISAGG_ROUTE from unknown client."
                    )
                    continue
                _, route_hops = deserialized_payload
                if route_hops is None:
                    self._disagg_route = None
                    logging.info("Coordinator: cleared disagg route.")
                else:
                    self._disagg_route = route_hops
                    participating = sorted({h[0] for h in route_hops})
                    logging.info(
                        "Coordinator: stored disagg route with %d hops, "
                        "participating shards=%s",
                        len(route_hops),
                        participating,
                    )
                # Ack the client so it can unblock launch before the
                # first SUBMIT lands.
                self.router_socket.send_multipart(
                    [
                        sender_identity,
                        msgpack.packb(
                            [Headers.SET_DISAGG_ROUTE_ACK.value],
                            use_bin_type=True,
                        ),
                    ]
                )

            elif header in (
                Headers.UPDATE_REQUEST_RANK,
                Headers.UPDATE_REQUEST_RANKS_BATCH,
            ):
                # Driver tells the coord that one or more live requests
                # have migrated to ``(new_shard_index, new_dp_rank)``.
                # Single payload: [hdr, request_id, shard, dp_rank].
                # Batch payload:  [hdr, [request_ids...], shard, dp_rank].
                # The coord rewrites ``request_id_to_rank`` and shifts
                # ``_pending_counts`` so subsequent ENGINE_REPLY routing
                # and load accounting target the new owner.
                if sender_identity not in known_clients:
                    logging.warning(
                        "Coordinator: ignoring %s from unknown client.",
                        header.name,
                    )
                    continue
                _, ids_field, new_shard_index, new_dp_rank_within_shard = (
                    deserialized_payload
                )
                request_ids_to_update = (
                    list(ids_field)
                    if header == Headers.UPDATE_REQUEST_RANKS_BATCH
                    else [ids_field]
                )
                new_identity = self._identity_for_dp_rank(
                    new_shard_index, new_dp_rank_within_shard
                )
                if new_identity is None:
                    logging.error(
                        "Coordinator: %s dp_rank=%d not found in shard %s "
                        "(engines=%s)",
                        header.name,
                        new_dp_rank_within_shard,
                        new_shard_index,
                        [
                            i.decode(errors='replace')
                            for i in self._identities_for_shard(new_shard_index)
                        ],
                    )
                    continue
                new_idx = self.identity_to_rank_index.get(new_identity)
                migrated = 0
                for request_id in request_ids_to_update:
                    if request_id not in self.request_id_to_client_id:
                        logging.warning(
                            "Coordinator: %s for unknown request %d; ignoring "
                            "(already replied or never submitted)",
                            header.name,
                            request_id,
                        )
                        continue
                    old_identity = self.request_id_to_rank.get(request_id)
                    if old_identity is not None:
                        old_idx = self.identity_to_rank_index.get(old_identity)
                        if old_idx is not None and self._pending_counts[old_idx] > 0:
                            self._pending_counts[old_idx] -= 1
                    self.request_id_to_rank[request_id] = new_identity
                    if new_idx is not None:
                        self._pending_counts[new_idx] += 1
                    migrated += 1
                if migrated:
                    logging.info(
                        "Coordinator: %d request(s) migrated → %s (shard %s)",
                        migrated,
                        new_identity,
                        new_shard_index,
                    )

            elif header == Headers.SHUTDOWN:
                if sender_identity not in known_clients:
                    logging.warning("Coordinator: ignoring signal from unknown client.")
                    continue
                break

            elif header == Headers.DISCONNECT:
                if sender_identity in self.identities_of_data_parallel_ranks:
                    self._remove_engine(sender_identity)

            else:
                raise UnknownHeaderError(header)

    def detokenize(self, finished_request):
        """
        Detokenizes the generated tokens in the finished request.

        This method uses the coordinator's tokenizer to convert the list of
        generated token IDs back into human-readable text.

        Args:
            finished_request (dict): The serialized merged request containing the
                generated tokens to be detokenized. It is modified in place.
        """
        if finished_request["prompt"] is None:
            finished_request["prompt"] = TextGenerationController.detokenize(
                self.tokenizer, finished_request["prompt_tokens"][1], remove_EOD=False
            )
        detokenize_stop_sequence = (finished_request.get("sampling_params", {}) or {}).get(
            "detokenize_stop_sequence", False
        )
        finished_request["generated_text"] = TextGenerationController.detokenize(
            self.tokenizer,
            finished_request["generated_tokens"],
            remove_EOD=not detokenize_stop_sequence,
        )

    @classmethod
    def entrypoint(
        cls,
        pipe_connection: Connection,
        ready_event: Event,
        data_parallel_size: int,
        tokenizer,
        max_requests,
        inference_coordinator_port: int | None = None,
        deterministic_mode: bool = False,
        block_size_tokens: int | None = None,
        enable_prefix_caching: bool = False,
        prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
            PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
        ),
        prefix_caching_routing_alpha: float = 0.5,
        schedule_output_path: str | None = None,
        hostname: str | None = None,
    ):
        """
        Class method to instantiate and run the coordinator, for use in a separate process.

        This method initializes the coordinator, signals a `ready_event` to indicate
        that it is fully initialized and listening, and then starts the main event loop.

        Args:
            pipe_connection (Connection): A connecting pipe to the parent process.
            ready_event (Event): A threading or multiprocessing event object that is set()
                once the coordinator is ready to accept connections.
            inference_coordinator_port (int): The port to bind to.
            data_parallel_size (int): The number of expected TP-coordinators.
            deterministic_mode (bool): Whether to enable deterministic scheduling.
            block_size_tokens (Optional[int]): Token block size for prefix caching hashing.
            enable_prefix_caching (bool): Whether prefix caching is enabled.
            prefix_caching_coordinator_policy (PrefixCachingCoordinatorPolicy): Routing policy.
            schedule_output_path (Optional[str]): Path to write scheduling decisions JSON.
            prefix_caching_routing_alpha (float): Weight for prefix-aware routing score.
            max_requests (int): Max concurrent requests per rank.
        """
        coordinator = cls(
            pipe_connection,
            data_parallel_size,
            tokenizer,
            max_requests,
            inference_coordinator_port,
            deterministic_mode=deterministic_mode,
            block_size_tokens=block_size_tokens,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=prefix_caching_coordinator_policy,
            prefix_caching_routing_alpha=prefix_caching_routing_alpha,
            schedule_output_path=schedule_output_path,
            hostname=hostname,
        )
        ready_event.set()
        try:
            coordinator.start()
        except KeyboardInterrupt:
            logging.info("Coordinator process interrupted. Exiting...")
        coordinator.stop()
        logging.info("Inference Coordinator: shut down successfully.")

    def stop(self):
        """
        Stops the inference coordinator, performing any necessary cleanup operations.
        """
        if self.schedule_output_path and self.schedule_records:
            schedule_data = {
                "policy": self.prefix_caching_coordinator_policy.value,
                "data_parallel_size": self.data_parallel_size,
                "num_requests": len(self.schedule_records),
                "records": self.schedule_records,
            }
            with open(self.schedule_output_path, "w") as f:
                json.dump(schedule_data, f, indent=2)
        self.router_socket.close()
        self.context.term()
