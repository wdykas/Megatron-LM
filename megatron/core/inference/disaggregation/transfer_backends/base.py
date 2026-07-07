# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KV transfer backend registry.

Backends are selected explicitly by the launcher configuration (e.g.
``--disagg-kv-transport-backend``), never from the environment.
"""

from __future__ import annotations

from typing import Any

KVTransportBackend = Any


def construct_kv_transfer_backend_class(name: str) -> KVTransportBackend:
    """Return the backend class registered under ``name``."""

    normalized = name.lower().replace("_", "-")
    if normalized == "nixl":
        from .nixl import NixlTransferBackend

        return NixlTransferBackend
    raise ValueError("Unsupported KV transfer backend %r; expected 'nixl'." % name)
