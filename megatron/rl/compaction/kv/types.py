# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class KVMask:
    """Specifies which KV cache positions to retain after compaction.

    ``retained_positions`` is a sorted list of context indices to keep.
    ``total_positions`` is the context length before pruning.

    Duck-typed as a belief: ``to_context_str()`` lets ContextBuilder treat a
    KVMask the same way it treats a TextBelief.
    """

    run_id: str
    step_id: int
    retained_positions: list[int]
    total_positions: int
    strategy: str
    created_at_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def retention_ratio(self) -> float:
        if self.total_positions == 0:
            return 1.0
        return len(self.retained_positions) / self.total_positions

    def to_context_str(self) -> str:
        return (
            f"[KV mask: {len(self.retained_positions)}/{self.total_positions} positions "
            f"retained ({self.retention_ratio():.1%}) via {self.strategy}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KVMask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
