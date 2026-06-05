# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Process-group building for RL inference models.

The implementation lives in ``megatron.core`` so non-RL frameworks can use it
too (core must not import from ``megatron.rl``). This re-exports it for
backward-compatible ``megatron.rl.parallel_utils`` imports.
"""

from megatron.core.inference.shards import build_inference_pg_collection

__all__ = ["build_inference_pg_collection"]
