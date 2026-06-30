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
    TP_BROADCAST = auto()
    # Disaggregated prefill->decode (engine roles + 2-hop KV handoff).
    REGISTER_ROLE = auto()  # engine -> coord: declare role + KV layout at registration
    PREFILL_DONE = auto()   # prefill engine -> coord: request finished prefill, KV staged
    SEND_KV = auto()        # coord -> prefill engine: ship request's KV to a decode instance
    RECV_KV = auto()        # coord -> decode engine: receive KV then admit + generate
    KV_READ_DONE = auto()   # decode engine -> coord: one-sided read drained (release credit + KV)
    RELEASE_KV = auto()     # coord -> prefill engine: release the request's pinned KV blocks


class UnknownHeaderError(Exception):
    """A signal with an unrecognized header was received by the coordinator."""

    def __init__(self, header):
        super().__init__(f"specialize for {header}.")
