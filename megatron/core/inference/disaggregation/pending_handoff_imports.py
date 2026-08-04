# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""State records for KV-cache and Mamba-state imports awaiting completion."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from megatron.core.inference.sampling_params import SamplingParams


@dataclass(kw_only=True)
class DeferredKvHandoff:
    """Decode handoff waiting for local cache capacity before transfer starts."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    kv_meta: dict
    src_block_ids: List[int]
    hashes: List[int]
    num_blocks: int
    future: asyncio.Future


@dataclass(kw_only=True)
class PendingPrefixReservation:
    """Prefix boundary reserved by one in-flight decode import."""

    request_id: int
    block_hash: int
    waiters: Deque[DeferredKvHandoff] = field(default_factory=deque)


class PendingPrefixReservations:
    """Keep in-flight hashes separate from the ready prefix cache."""

    def __init__(self) -> None:
        self._owner_by_hash: Dict[int, int] = {}
        self._by_request: Dict[int, PendingPrefixReservation] = {}
        self._waiter_count = 0

    @property
    def waiter_count(self) -> int:
        """Number of handoffs waiting for an overlapping import."""

        return self._waiter_count

    def find_owner(self, block_hash: Optional[int]) -> Optional[int]:
        """Return the owner of a reserved prefix boundary, if any."""

        return None if block_hash is None else self._owner_by_hash.get(int(block_hash))

    def reserve(self, request_id: int, block_hash: Optional[int]) -> None:
        """Reserve the first missing hash after destination blocks exist.

        Block hashes include their parent digest, so imports cannot collide on
        a later block unless they share this first missing hash.
        """

        if block_hash is None:
            return
        block_hash = int(block_hash)
        if request_id in self._by_request:
            raise RuntimeError(f"request {request_id} already owns a pending prefix reservation")
        owner = self.find_owner(block_hash)
        if owner is not None:
            raise RuntimeError(
                f"request {request_id} overlaps pending prefix reservation owned by {owner}"
            )
        reservation = PendingPrefixReservation(request_id=request_id, block_hash=block_hash)
        self._by_request[request_id] = reservation
        self._owner_by_hash[block_hash] = request_id

    def wait(self, owner_request_id: int, handoff: DeferredKvHandoff) -> None:
        """Attach a handoff to an existing reservation."""

        reservation = self._by_request.get(owner_request_id)
        if reservation is None:
            raise RuntimeError(f"pending prefix reservation {owner_request_id} disappeared")
        reservation.waiters.append(handoff)
        self._waiter_count += 1

    def release(self, request_id: int) -> List[DeferredKvHandoff]:
        """Release one reservation and return its waiters in arrival order."""

        reservation = self._by_request.pop(request_id, None)
        if reservation is None:
            return []
        if self._owner_by_hash.get(reservation.block_hash) == request_id:
            del self._owner_by_hash[reservation.block_hash]
        self._waiter_count -= len(reservation.waiters)
        return list(reservation.waiters)

    def drain_waiters(self) -> List[DeferredKvHandoff]:
        """Remove and return every waiter while preserving reservation order."""

        waiters = []
        for reservation in self._by_request.values():
            waiters.extend(reservation.waiters)
            reservation.waiters.clear()
        self._waiter_count = 0
        return waiters


@dataclass(kw_only=True)
class PendingMambaImport:
    """Mamba state transfers attached to a pending KV-cache import."""

    handles: dict[str, Any]
    local_slots: List[int]
    target_blocks: List[int]
    positions: List[int]


@dataclass(kw_only=True)
class PendingKvImport:
    """Decode request waiting for an asynchronous KV-cache import."""

    request_id: int
    prompt: list
    sampling_params: SamplingParams
    local_blocks: List[int]
    hashes: list
    hashes_to_register: int
    hash_registration_start: int
    handle: Any
    future: asyncio.Future
    mamba: Optional[PendingMambaImport] = None
