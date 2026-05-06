# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from enum import Enum, auto


class Headers(Enum):
    """
    Enum representing headers used for communication with the inference-coordinator.
    """

    CONNECT = auto()
    CONNECT_ACK = auto()
    SUBMIT_REQUEST = auto()
    ENGINE_REPLY = auto()
    PAUSE = auto()
    UNPAUSE = auto()
    SUSPEND = auto()
    RESUME = auto()
    SET_GENERATION_EPOCH = auto()
    STOP = auto()
    DISCONNECT = auto()
    SHUTDOWN = auto()
    # Coordinator → client: acknowledges a SUBMIT_REQUEST by reporting the
    # server-side request id the coord assigned. Lets clients that need the
    # server id (e.g. to drive cross-shard migration of the in-flight
    # request) resolve it without waiting for the final ENGINE_REPLY.
    # Payload: [SUBMIT_REQUEST_ACK, client_request_id, server_request_id].
    SUBMIT_REQUEST_ACK = auto()
    # Engine → coordinator: register a DP rank with optional shard metadata.
    # Payload: [REGISTER_DP_RANK, shard_index_or_None]. Replaces the legacy
    # empty-bytes registration when shard-aware routing is in play; the coord
    # still accepts b"" for backward compatibility.
    REGISTER_DP_RANK = auto()
    # Rank-0 driver → coordinator: after an engine-level cross-shard request
    # migration completes, tell the coord which shard + DP-rank-within-shard
    # now owns the request, so ENGINE_REPLY dispatch and load accounting stay
    # consistent. Payload: [UPDATE_REQUEST_RANK, request_id, new_shard_index,
    # new_dp_rank_within_shard].
    UPDATE_REQUEST_RANK = auto()
    # Rank-0 driver → coordinator → engines in src+dst shards: trigger a
    # batched cross-shard request migration. The coord forwards this header
    # to engines in the named shards; each engine's run loop pops it from
    # ``_pending_signals`` and invokes the migration callback registered
    # via ``engine.set_migration_handler(...)``. The callback runs sync
    # NCCL on ``cross_shard_group`` to move KV blocks for the listed
    # request ids from src → dst. Replaces the prior PG-33 broadcast
    # design (cross-PG NCCL ordering deadlock). Payload sent over the
    # wire: ``[MIGRATE_BATCH, request_ids, src_shard_index,
    # dst_shard_index]``. Forwarded to engines as the same 4-tuple.
    MIGRATE_BATCH = auto()


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
