# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph capture wrapper around :class:`CompiledRoute`.

The third and final layer of compile work for layer-kind disagg:

1. :class:`CompiledRoute` made the per-layer Python loop static so
   Dynamo can trace cleanly.
2. ``transport_ops`` registered the NVSHMEM primitives as
   ``torch.library.custom_op`` so the transport calls participate
   in the graph instead of breaking it.
3. **This module** wraps a :class:`CompiledRoute` in a CUDA graph
   so the entire per-shard forward (kernel launches + transport
   puts/waits + reductions) replays from a captured graph instead
   of re-issuing Python-mediated kernel launches every step.

CUDA graphs eliminate per-kernel launch overhead — the dominant
cost in small-batch decode (batch=1 token per request) where the
GPU itself is under-utilized and the bottleneck is CPU-side
dispatch. For prefill, graphs don't help directly (shape varies
per request); for decode, they're the big win.

Requirements that callers must honor:

- **Fixed shape per captured graph.** Hidden tensor shape (and
  dtype) must be the same on every replay. Variable batch size or
  sequence-position-dependent shape requires multiple captured
  graphs (one per shape bucket) or padding to a common shape.
- **All ops on the activation stream.** The transport backend's
  ``stream()`` returns the CUDA stream used for puts / waits;
  capture runs against that stream so transport participates in
  the graph naturally.
- **Stable input/output tensor identities.** CUDA graph replay
  re-runs against the *same* memory buffers used at capture time.
  Callers provide the input buffer once; subsequent steps
  ``copy_`` new data into it before ``replay()``.
- **Pre-warmed backend.** The capture call must follow at least
  one normal forward through the same CompiledRoute so any
  lazy-allocated transport slots / fwd flags are in place.
  ``warmup_steps=1`` in :meth:`capture` handles this for callers.

Scope (v1):

- One graph per ``CompiledRoute``; recapture on layout or shape
  changes.
- No multi-stream graphs (transport + compute share a stream).
  Multi-stream graph capture is a follow-up if profiling shows
  transport/compute serialization is hurting.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from megatron.core.inference.compiled_route import CompiledRoute


class CompiledRouteGraph:
    """CUDA-graph-captured replay of a :class:`CompiledRoute.run`.

    Typical usage on the decode hot path::

        cr = CompiledRoute(route=route, ..., use_torch_ops=True, device=cuda)
        crg = CompiledRouteGraph(cr, hidden_shape=(1, hidden_dim),
                                 dtype=torch.float16, device=cuda)
        crg.capture(layer_callables=callables)
        for step in range(num_steps):
            new_hidden = embed_token(token_at(step))
            out = crg.replay(new_hidden)  # cheap — no kernel-launch overhead
            ...

    Args:
        compiled_route: A constructed :class:`CompiledRoute` instance.
            Must have ``use_torch_ops=True`` if the route has any
            cross-shard hops — otherwise the transport calls
            graph-break under Dynamo and the capture won't see them.
            (CompiledRoute itself never traces under Dynamo; the
            graph capture mechanism here is independent of
            torch.compile.)
        hidden_shape: Shape of the input hidden tensor. Must match
            the shape ``CompiledRoute`` was constructed with — the
            graph wrapper enforces this for safety.
        dtype: Dtype of the input hidden tensor.
        device: CUDA device the graph runs on.
        warmup_steps: Number of un-captured forward passes before
            capture, to materialize lazy buffers + populate any
            backend slot pools. Default 1.
    """

    def __init__(
        self,
        compiled_route: CompiledRoute,
        hidden_shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        *,
        warmup_steps: int = 1,
    ) -> None:
        assert tuple(hidden_shape) == compiled_route._hidden_shape, (
            f"CompiledRouteGraph: hidden_shape={tuple(hidden_shape)} "
            f"must match the CompiledRoute's shape "
            f"({compiled_route._hidden_shape})."
        )
        assert dtype == compiled_route._hidden_dtype, (
            f"CompiledRouteGraph: dtype={dtype} must match "
            f"CompiledRoute's dtype ({compiled_route._hidden_dtype})."
        )
        self._cr = compiled_route
        self._hidden_shape = tuple(hidden_shape)
        self._dtype = dtype
        self._device = device
        self._warmup_steps = warmup_steps

        # Persistent input buffer — replay copies new data into this
        # tensor and the graph reads from it. Allocated up-front so
        # capture has a stable pointer.
        self._input_buffer = torch.empty(
            self._hidden_shape, dtype=dtype, device=device
        )
        # Output is allocated lazily during capture (depends on
        # whether the final action is SEND or LOCAL — see ``capture``).
        self._output_buffer: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured: bool = False

    # ---- introspection ---------------------------------------------------

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def input_buffer(self) -> torch.Tensor:
        """Persistent input tensor — ``replay`` copies caller data
        into this before invoking the graph. Exposed for callers who
        want to write directly (e.g. fused embed+copy)."""
        return self._input_buffer

    # ---- capture / replay -----------------------------------------------

    def capture(
        self,
        layer_callables: Sequence[Callable[[torch.Tensor], torch.Tensor]],
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Capture a CUDA graph of one ``compiled_route.run`` execution.

        Performs ``warmup_steps`` regular forward passes first so any
        lazily-allocated backend state (NVSHMEM slot pools, etc.)
        gets materialized before capture begins — captured graphs
        can't allocate.

        Args:
            layer_callables: As passed to ``CompiledRoute.run``.
            stream: CUDA stream to capture against. Defaults to the
                transport backend's activation stream (so transport
                puts / waits and compute share a single stream and
                participate in the same graph).
        """
        if stream is None:
            from megatron.core.inference.transport_backend import (
                get_activation_transport_backend,
            )

            backend = get_activation_transport_backend()
            stream = backend.stream() or torch.cuda.current_stream()

        # Warmup outside capture so lazy buffers materialize.
        for _ in range(self._warmup_steps):
            self._cr.run(self._input_buffer, layer_callables=layer_callables)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=stream):
            out = self._cr.run(
                self._input_buffer, layer_callables=layer_callables
            )
            # SEND-only finale → ``out`` is None on the standard
            # dispatcher. Store a 0-element sentinel so ``replay``
            # has something tensor-valued to return. Exit-shard
            # finale → ``out`` is a tensor we keep as the replay
            # output.
            if out is None:
                self._output_buffer = torch.empty(
                    0, dtype=self._dtype, device=self._device
                )
            else:
                self._output_buffer = out
        self._captured = True

    def replay(self, hidden_in: torch.Tensor) -> torch.Tensor:
        """Copy ``hidden_in`` into the persistent input buffer and
        replay the captured graph.

        ``hidden_in`` must match ``hidden_shape`` + ``dtype`` from
        the constructor. Returns the captured output tensor — either
        the exit-shard's post-layer hidden state or a 0-element
        sentinel for non-exit shards (where the final action is
        SEND and there's no local hidden left).
        """
        assert self._captured, (
            "CompiledRouteGraph.replay called before capture(). "
            "Call capture(layer_callables) first."
        )
        assert tuple(hidden_in.shape) == self._hidden_shape
        assert hidden_in.dtype == self._dtype
        self._input_buffer.copy_(hidden_in)
        self._graph.replay()
        assert self._output_buffer is not None  # set in capture()
        return self._output_buffer
