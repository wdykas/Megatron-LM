# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared range math for the KV / Mamba reshard planners."""

from __future__ import annotations

from typing import Optional, Tuple


def intersect(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Overlap of two half-open ``[lo, hi)`` ranges, or ``None`` if disjoint."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo < hi else None
