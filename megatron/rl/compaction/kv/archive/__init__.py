# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Eviction as demotion: the CPU/remote span store + negative-cache trigger."""

from .negative_cache import KVArchive
from .transfer import PinnedCpuStore, NixlStore, build_span_store
