# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import Literal


@dataclass
class PomdpConfig:
    enabled: bool = False

    # "record_only": record traces; actor still sees full context.
    # "shadow": build compact context but do not act on it.
    # "live": actor uses compact context.
    mode: Literal["record_only", "shadow", "live"] = "record_only"

    recent_tail_steps: int = 2
    max_observation_chars: int = 8000

    store_raw_trace: bool = True
    store_actor_input_tokens: bool = True

    # KV compaction — only used when a kv_algorithm is wired into the recorder.
    kv_budget_ratio: float = 0.5   # fraction of KV positions to retain per step

    trace_dir: str = "/tmp/pomdp_traces"
