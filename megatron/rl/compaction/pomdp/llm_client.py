# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Adapter from the RL ``InferenceInterface`` to the simple LLM-client shape.

The text-level compaction components in this package (``LLMAlgorithm`` and any
model-driven ``CompactionTrigger``) talk to the policy through a minimal
contract::

    class LLMClient(Protocol):
        async def complete(self, prompt: str) -> str: ...

``InferenceInterfaceClient`` wraps a ``megatron.rl.inference.InferenceInterface``
so the same compaction code can run against the live policy during rollouts.

The wrapper is deliberately dependency-light: it passes the prompt as a plain
string to ``prepare_request`` (which wraps it into a user-role chat message),
so this module imports nothing from torch/megatron at import time.
"""

from __future__ import annotations

from typing import Any


class InferenceInterfaceClient:
    """Expose an ``InferenceInterface`` via ``async def complete(prompt) -> str``.

    Args:
        interface: any object with ``prepare_request(prompt, generation_args)``
            and ``async agenerate(request) -> response`` (response carries
            ``response.content``) — i.e. the RL ``InferenceInterface`` contract.
        generation_args: generation args forwarded to ``prepare_request``
            (e.g. a ``GenericGenerationArgs``). Required by the interface.
        system_prompt: optional text prepended to every prompt.
    """

    def __init__(
        self,
        interface: Any,
        generation_args: Any,
        system_prompt: str | None = None,
    ) -> None:
        self._interface = interface
        self._generation_args = generation_args
        self._system_prompt = system_prompt

    async def complete(self, prompt: str) -> str:
        text = prompt
        if self._system_prompt:
            text = f"{self._system_prompt}\n\n{prompt}"
        request = self._interface.prepare_request(text, self._generation_args)
        response = await self._interface.agenerate(request)
        return response.response.content
