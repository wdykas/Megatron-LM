# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import BaseModel

from ..__init__ import Request


class LLMChatMessage(BaseModel):
    role: str
    content: str
    # Exact serving tokens of the generation this assistant message came from.
    # The chat endpoint's prevent-retokenization patch REQUIRES them on the
    # last assistant message: it splices these tokens into the prompt instead
    # of retokenizing, preserving prefix-cache hits and token identity
    # (self-compaction summary requests use this).
    prompt_token_ids: list[int] | None = None
    generation_token_ids: list[int] | None = None


class InferenceRequest(Request):
    prompt: list[LLMChatMessage]
    tools: list[dict] | None = None
    # Position of this request within its GRPO group (set by group_rollout).
    # Enables deterministic within-group treatment splits (e.g. the
    # kv_compact arm) instead of per-request Bernoulli draws.
    rollout_index: int | None = None


def deterministic_split_arm(index: int, fraction: float) -> bool:
    """Deterministic treatment assignment for group member ``index``.

    Bresenham-style stratification: exactly ``round(n * fraction)`` of any
    first-n members are treated, without knowing the group size — e.g. at
    fraction 0.5 arms alternate treated/control. Replaces a Bernoulli draw
    when the caller knows the member index.
    """
    import math
    return math.floor((index + 1) * fraction) > math.floor(index * fraction)


class InferenceResponse(BaseModel):
    """The minimum required response for an inference interface."""

    response: LLMChatMessage
    raw_text: str | None = None
    token_ids: list[int] | None = None
    prompt_length: int | None = None
    logprobs: list[float] | None = None
    policy_staleness: list[int]
    kv_cache_staleness: list[int]
    completed_at_step: int
    num_evictions: int
    # split-group: which compaction arm this rollout ran on (None = split off).
    kv_compacted: bool | None = None
    # 'length' when generation stopped at the token cap (drives self-compaction
    # segmentation), 'stop' on a natural finish. None for interfaces that
    # don't report it.
    finish_reason: str | None = None
