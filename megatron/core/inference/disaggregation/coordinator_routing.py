# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pure 2-hop routing state for coordinator-native prefill->decode disagg."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

PREFILL = "prefill"
DECODE = "decode"


class DisaggRouting:
    """Sequences a request prefill-engine -> (KV handoff) -> decode-engine.

    The :class:`DataParallelInferenceCoordinator` delegates routing decisions
    here so the policy is isolated and unit-testable without ZMQ. Engines are
    identified by their opaque transport identity (bytes for ZMQ; any hashable
    in tests). This object holds no sockets and does no I/O -- it only decides
    *which* engine each hop goes to and remembers the per-request pairing so
    the final reply can be routed home.

    Selection is round-robin within each role (deterministic given the
    registration order); a prefix/load-aware decode policy can replace
    :meth:`_pick_decode` later without changing callers.
    """

    def __init__(self) -> None:
        self.prefill_engines: List = []
        self.decode_engines: List = []
        self._prefill_rr = 0
        self._decode_rr = 0
        # request_id -> (prefill_identity, decode_identity|None)
        self._req_prefill: Dict[int, object] = {}
        self._req_decode: Dict[int, object] = {}

    # --- registration ------------------------------------------------------

    def register(self, identity, role: str) -> None:
        """Record an engine and its disagg role (idempotent)."""
        if role == PREFILL:
            pool = self.prefill_engines
        elif role == DECODE:
            pool = self.decode_engines
        else:
            raise ValueError(f"disagg engine role must be 'prefill'/'decode'; got {role!r}")
        if identity not in pool:
            pool.append(identity)

    def remove(self, identity) -> None:
        """Drop a disconnected engine from both pools."""
        for pool in (self.prefill_engines, self.decode_engines):
            if identity in pool:
                pool.remove(identity)

    @property
    def ready(self) -> bool:
        """Whether at least one prefill and one decode engine are registered."""
        return bool(self.prefill_engines) and bool(self.decode_engines)

    # --- per-request routing ----------------------------------------------

    def route_submit(self, request_id: int):
        """Hop 1: pick the prefill engine for a newly submitted request."""
        if not self.prefill_engines:
            raise RuntimeError("no prefill engines registered")
        ident = self.prefill_engines[self._prefill_rr % len(self.prefill_engines)]
        self._prefill_rr += 1
        self._req_prefill[request_id] = ident
        return ident

    def route_prefill_done(self, request_id: int) -> Tuple[object, object]:
        """Hop 2: a request finished prefill -- pick its decode engine.

        Returns ``(prefill_identity, decode_identity)``: the coordinator sends
        SEND_KV to the prefill engine and RECV_KV to the decode engine.
        """
        if not self.decode_engines:
            raise RuntimeError("no decode engines registered")
        dec = self._pick_decode(request_id)
        self._req_decode[request_id] = dec
        prefill = self._req_prefill.get(request_id)
        return prefill, dec

    def decode_of(self, request_id: int) -> Optional[object]:
        """The decode engine a request was routed to (for reply accounting)."""
        return self._req_decode.get(request_id)

    def forget(self, request_id: int) -> None:
        """Drop per-request state once the reply has been routed to the client."""
        self._req_prefill.pop(request_id, None)
        self._req_decode.pop(request_id, None)

    def _pick_decode(self, request_id: int):
        dec = self.decode_engines[self._decode_rr % len(self.decode_engines)]
        self._decode_rr += 1
        return dec
