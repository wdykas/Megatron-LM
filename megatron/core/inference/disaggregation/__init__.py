# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated prefill->decode inference.

A request is prefilled on one engine and decoded on another. The KV cache is
handed off between them; only *control* flows through the shared
DataParallelInferenceCoordinator, the KV *bytes* move engine->engine via the
transport backend (see ``transfer_backends/``). Two backend families:

* **Push** (two-sided, NCCL): the prefill ships the KV to the decode.
* **Pull** (one-sided, NIXL): the decode READs the KV straight out of the
  prefill's registered buffer; the prefill blocks stay pinned until the read
  completes.

Control messages are the ``Headers`` in ``megatron/core/inference/headers.py``.
The coordinator stays transport-agnostic -- it branches only on whether a
handoff descriptor rode along on PREFILL_DONE, never on what it contains.

Push flow (4 headers)::

    REGISTER_ROLE  engine->coord   role + KV layout (is_pull=False)
    PREFILL_DONE   prefill->coord  prefill finished, KV staged (no handoff)
    SEND_KV        coord->prefill  ship request's KV to the chosen decode
    RECV_KV        coord->decode   receive KV, admit, generate

Pull flow (5 headers -- the minimum for the one-sided path; SEND_KV is unused
because the prefill publishes its KV up front and does nothing more)::

    REGISTER_ROLE  engine->coord   role + KV layout, flags is_pull=True so the
                                   coordinator applies flow control
    PREFILL_DONE   prefill->coord  finished + handoff payload (per-rank READ
                                   descriptors: block ids, buffer geometry)
    RECV_KV        coord->decode   relays the handoff; decode does the READ
    KV_READ_DONE   decode->coord   READ drained -> free an outstanding slot, pin is
                                   now safe to release
    RELEASE_KV     coord->prefill  unpin the request's KV blocks

Flow control (pull only): PREFILL_DONE consumes an outstanding slot per pull-prefill
instance; the decode's KV_READ_DONE returns it. This bounds how many hand-offs
(and thus pinned blocks) can be outstanding at once.
"""
