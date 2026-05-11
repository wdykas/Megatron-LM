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
    # Batched form of UPDATE_REQUEST_RANK: rank-0 driver tells the coord
    # that a *list* of request_ids has all migrated to the same
    # ``(new_shard_index, new_dp_rank_within_shard)``. Saves 16 ZMQ
    # send+recv+unpack round trips per migration batch. Payload:
    # [UPDATE_REQUEST_RANKS_BATCH, request_ids, new_shard_index,
    # new_dp_rank_within_shard].
    UPDATE_REQUEST_RANKS_BATCH = auto()
    # Decider rank → coordinator → engines in src + chosen dst dp_rank:
    # trigger a batched cross-shard request migration. The coord forwards
    # this header to every rank in the src shard and to the single chosen
    # dst dp_rank's engines; each engine's run loop pops it from
    # ``_pending_signals`` and invokes the migration callback registered
    # via ``engine.set_migration_handler(...)``. The callback runs the
    # NVSHMEM one-sided ``put_signal`` / ``signal_wait`` transport on the
    # migration stream — no NCCL collective, no engine pause.
    # Wire payload: ``[MIGRATE_BATCH, batch_id, request_ids,
    # src_shard_index, dst_shard_index, bundles, dst_dp_rank]``.
    # ``batch_id`` is a monotonic counter the decider's
    # :class:`InferenceClient` uses to match the coord's
    # :data:`MIGRATE_BATCH_ACK` reply; ``bundles`` are the per-request
    # serialized envelopes (carrying them inline lets the dst engine
    # register the requests without a cross-shard broadcast).
    MIGRATE_BATCH = auto()
    # Coordinator → decider client: acknowledges a ``MIGRATE_BATCH``
    # after the coord has performed a capacity check against the dst
    # shard. Two-phase commit prevents data loss when dst is at
    # capacity — the decider only posts the follow-up
    # :data:`UPDATE_REQUEST_RANKS_BATCH` and marks the ids as migrated
    # if ``accepted`` is True. On rejection the coord drops the batch
    # and the decider retries on a later tick, when dst has freed
    # slots. Payload: ``[MIGRATE_BATCH_ACK, batch_id, accepted]``.
    MIGRATE_BATCH_ACK = auto()
    TP_BROADCAST = auto()


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
