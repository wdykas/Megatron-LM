# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Static-action specialization of a route for ``torch.compile``.

The default :class:`RouteDispatcher` is a per-layer interpreter: each
call to ``dispatch_layer(layer_idx, ...)`` walks a dict-keyed plan,
branches on action kind, and may invoke the transport backend.
Inductor / Dynamo can't trace through that cleanly — the dict lookup
is data-dependent, the action branches are Python-level, and the
backend calls are opaque C extensions. The result: graph-breaks at
every layer.

This module provides :class:`CompiledRoute`, an alternate execution
model that **pre-resolves every per-layer action** at construction.
Given a fixed ``(route, my_shard_idx, hidden_shape)`` the action
sequence is fully static — there's no per-layer control flow keyed
on runtime tensor state. The resulting :meth:`CompiledRoute.run`
method is a flat sequence of:

    1. Optionally receive (at hop entry).
    2. Optionally substitute a cached hidden (cache hit).
    3. Call the user-supplied ``layer_callables[layer_idx]``.
    4. Optionally send (at hop exit).
    5. Optionally populate the cache (entry shard, first pass).

Wrap the result in ``@torch.compile`` and the loop unrolls into one
fused graph per shard's portion of the forward pass. Transport calls
still graph-break for now (they're not registered torch ops yet — a
follow-up); but the *Python overhead* between layers vanishes, which
is the dominant cost in small-batch decode.

Scope (v1):

- Single-request execution. Batched per-request route diversity
  uses the standard dispatcher (compile boundaries cost more than
  they save when each request has different actions).
- Receive / send call through ``get_activation_transport_backend()``
  for back-compat with mocks. A future commit will register them as
  ``torch.library.custom_op`` so Dynamo sees through them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from megatron.core.inference.route_dispatcher import _build_layer_plan
from megatron.core.inference.transport_backend import (
    get_activation_transport_backend,
)
from megatron.rl.inference.route_planner import Route


class _ActionKind(Enum):
    """Static action emitted by the route compiler for one layer.

    Skipped (not-my-request) layers do NOT appear in the action list
    — :meth:`CompiledRoute.run` only iterates owned layers, so the
    "skip everyone else" behavior is implicit.
    """

    RECV = auto()       # Hop entry: receive (+ optional reduce) one or more inbound activations.
    LOCAL = auto()      # Owned layer's local compute (``layer_callables[layer_idx]``).
    SEND = auto()       # Hop exit: send the outbound activation to peer PE(s).
    CACHE_HIT = auto()  # Layer covered by the prefix activation cache — skip the local compute.
    CACHE_BOUNDARY = auto()  # First non-cached layer — substitute cached hidden as input.


@dataclass(frozen=True)
class _Action:
    """One step in the compiled action list.

    Attributes:
        kind: The action class.
        layer_idx: Original global layer index this action belongs to.
            ``LOCAL`` / ``CACHE_HIT`` / ``CACHE_BOUNDARY`` actions key
            into ``layer_callables[layer_idx]``.
        pes: For ``RECV`` and ``SEND``, the resolved peer PE list.
            Empty for non-transport actions.
        reduce_op: For ``RECV`` with multiple PEs, the reduction
            kernel (``"sum"`` / ``"mean"``). ``None`` for single-input.
    """

    kind: _ActionKind
    layer_idx: int
    pes: Tuple[int, ...] = ()
    reduce_op: Optional[str] = None


class CompiledRoute:
    """Statically-specialized per-shard forward driver.

    Builds the same per-layer plan the standard
    :class:`RouteDispatcher` uses, then *flattens* it into a tuple of
    :class:`_Action` records covering only the layers this shard owns.
    The :meth:`run` method iterates that tuple — no dict lookups, no
    "is this my layer?" check per step, no Python-level branching
    keyed on tensor state.

    Args:
        route: The request's route.
        my_shard_idx: This shard's index in the layout.
        my_pe: This rank's NVSHMEM PE id (== global rank).
        my_tp_offset: This rank's TP offset within its shard.
        shard_tp: Per-shard TP size, indexed by shard idx.
        shard_rank_offset: Per-shard base rank, indexed by shard idx.
        hidden_shape: Activation tensor shape.
        hidden_dtype: dtype of the activation tensor.
        cache: Optional :class:`PrefixActivationCache`. When set + this
            rank is the entry shard, a lookup at construction time
            sets the cache skip depth; the action list emits
            ``CACHE_HIT`` for skipped layers and ``CACHE_BOUNDARY``
            for the first uncached one.
        prompt_tokens: Prompt token sequence — required when ``cache``
            is set (it's the cache key).
        single_rep: Hetero-TP single-rep mode.
    """

    def __init__(
        self,
        route: Route,
        my_shard_idx: int,
        my_pe: int,
        my_tp_offset: int,
        shard_tp: Sequence[int],
        shard_rank_offset: Sequence[int],
        hidden_shape: Tuple[int, ...],
        hidden_dtype: torch.dtype,
        *,
        cache: Optional[object] = None,
        prompt_tokens: Optional[Sequence[int]] = None,
        single_rep: bool = False,
    ) -> None:
        self._route = route
        self._my_shard = my_shard_idx
        self._my_pe = my_pe
        self._hidden_shape = tuple(hidden_shape)
        self._hidden_dtype = hidden_dtype
        nelem = 1
        for d in self._hidden_shape:
            nelem *= d
        self._payload_nbytes = (
            nelem * torch.empty((), dtype=hidden_dtype).element_size()
        )

        plan = _build_layer_plan(
            route=route,
            my_shard_idx=my_shard_idx,
            my_tp_offset=my_tp_offset,
            shard_tp=list(shard_tp),
            shard_rank_offset=list(shard_rank_offset),
            single_rep=single_rep,
        )

        # Cache lookup at construction — drives the CACHE_HIT /
        # CACHE_BOUNDARY action emission below.
        self._cache = cache
        self._prompt_tokens: Optional[Tuple[int, ...]] = (
            tuple(prompt_tokens) if prompt_tokens is not None else None
        )
        skip_depth = 0
        self._cache_starting_hidden: Optional[torch.Tensor] = None
        if (
            cache is not None
            and self._prompt_tokens is not None
            and my_shard_idx == route.entry_shard
        ):
            hit = cache.lookup(self._prompt_tokens)
            if hit is not None:
                skip_depth = hit.skip_depth
                self._cache_starting_hidden = hit.starting_hidden
        self._cache_skip_depth = skip_depth

        # Flatten the plan into the action list. We walk every layer
        # the route covers; NOT_MY_REQUEST layers (plan[li] is None)
        # don't produce actions — they're implicit "skip this step"
        # in run()'s caller.
        owned_layers: List[int] = sorted(
            li for li, p in plan.items() if p is not None
        )
        actions: List[_Action] = []
        for li in owned_layers:
            p = plan[li]
            assert p is not None  # for type narrowing
            # Cache hit: skip the local compute entirely.
            if skip_depth > 0 and li < skip_depth:
                actions.append(_Action(kind=_ActionKind.CACHE_HIT, layer_idx=li))
                continue
            # Cache boundary: substitute cached hidden for the inbound.
            is_boundary = (skip_depth > 0 and li == skip_depth)
            # Receive at hop entry (skipped at cache boundary — the
            # cached hidden replaces the inbound).
            if p.receive_from_pes and not is_boundary:
                actions.append(
                    _Action(
                        kind=_ActionKind.RECV,
                        layer_idx=li,
                        pes=p.receive_from_pes,
                        reduce_op=p.reduce_op,
                    )
                )
            actions.append(
                _Action(
                    kind=(
                        _ActionKind.CACHE_BOUNDARY if is_boundary
                        else _ActionKind.LOCAL
                    ),
                    layer_idx=li,
                )
            )
            if p.send_to_pes:
                actions.append(
                    _Action(
                        kind=_ActionKind.SEND,
                        layer_idx=li,
                        pes=p.send_to_pes,
                    )
                )
        self._actions: Tuple[_Action, ...] = tuple(actions)

    # ---- introspection ---------------------------------------------------

    @property
    def actions(self) -> Tuple[_Action, ...]:
        """Read-only view of the compiled action list. Useful for
        tests and debugging — e.g. counting RECV/SEND/CACHE_HIT
        actions on a given shard."""
        return self._actions

    @property
    def owned_layers(self) -> Tuple[int, ...]:
        """Layer indices this shard owns under the compiled route, in
        ascending order."""
        return tuple(sorted({a.layer_idx for a in self._actions}))

    @property
    def cache_skip_depth(self) -> int:
        return self._cache_skip_depth

    # ---- execution -------------------------------------------------------

    def run(
        self,
        hidden_in: Optional[torch.Tensor],
        layer_callables: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Execute the compiled action sequence and return the final
        hidden state this shard produces.

        ``hidden_in`` is the embedding output (entry shard) or
        ``None`` (non-entry shards — the first ``RECV`` action
        produces the initial hidden). The exit shard's return is the
        pre-lm-head hidden state; intermediate shards return
        ``None`` (the last action is a ``SEND``).

        ``layer_callables`` is indexed by *global* layer index — this
        method ignores entries the action list doesn't reach, so a
        sparse list (only owned layers populated) works fine.

        torch.compile note: the for-loop over a fixed-length tuple
        unrolls cleanly under Dynamo. Transport calls (``RECV`` /
        ``SEND``) currently graph-break because the backend isn't
        registered as a custom torch op — that's the next layer of
        compile work.
        """
        backend = get_activation_transport_backend()
        hidden = hidden_in
        for action in self._actions:
            if action.kind == _ActionKind.RECV:
                hidden = self._do_recv(backend, action)
            elif action.kind == _ActionKind.CACHE_HIT:
                # No local compute, no transport — the cache covered
                # this layer's output on a prior prefill. The next
                # action is either another CACHE_HIT (still inside the
                # cached region) or a CACHE_BOUNDARY (which will
                # substitute the cached hidden).
                continue
            elif action.kind == _ActionKind.CACHE_BOUNDARY:
                hidden = self._cache_starting_hidden
                hidden = layer_callables[action.layer_idx](hidden)
                self._maybe_populate(action.layer_idx, hidden)
            elif action.kind == _ActionKind.LOCAL:
                assert hidden is not None, (
                    "CompiledRoute.run: LOCAL action with no inbound "
                    "hidden — route lacks a preceding RECV / "
                    "CACHE_BOUNDARY or the caller didn't pass "
                    "hidden_in for an entry-shard run."
                )
                hidden = layer_callables[action.layer_idx](hidden)
                self._maybe_populate(action.layer_idx, hidden)
            elif action.kind == _ActionKind.SEND:
                assert hidden is not None
                self._do_send(backend, action, hidden)
                hidden = None  # post-send the activation lives on the peer
        return hidden

    # ---- internals -------------------------------------------------------

    def _do_recv(self, backend, action: _Action) -> torch.Tensor:
        if len(action.pes) == 1:
            return backend.receive_hidden(
                my_pe=self._my_pe,
                src_pe=action.pes[0],
                hidden_shape=self._hidden_shape,
                hidden_dtype=self._hidden_dtype,
                payload_nbytes=self._payload_nbytes,
            )
        # Multi-input: receive each branch then reduce.
        first = backend.receive_hidden(
            my_pe=self._my_pe,
            src_pe=action.pes[0],
            hidden_shape=self._hidden_shape,
            hidden_dtype=self._hidden_dtype,
            payload_nbytes=self._payload_nbytes,
        )
        for src_pe in action.pes[1:]:
            extra = backend.receive_hidden(
                my_pe=self._my_pe,
                src_pe=src_pe,
                hidden_shape=self._hidden_shape,
                hidden_dtype=self._hidden_dtype,
                payload_nbytes=self._payload_nbytes,
            )
            first.add_(extra)
        if action.reduce_op == "mean":
            first.div_(len(action.pes))
        elif action.reduce_op not in ("sum", None):
            raise AssertionError(
                f"CompiledRoute: unsupported reduce_op={action.reduce_op!r}"
            )
        return first

    def _do_send(self, backend, action: _Action, hidden: torch.Tensor) -> None:
        for dst_pe in action.pes:
            backend.send_hidden(
                my_pe=self._my_pe,
                dst_pe=dst_pe,
                hidden=hidden,
                payload_nbytes=self._payload_nbytes,
            )

    def _maybe_populate(self, layer_idx: int, hidden: torch.Tensor) -> None:
        if (
            self._cache is not None
            and self._prompt_tokens is not None
            and self._my_shard == self._route.entry_shard
        ):
            self._cache.store(
                prompt_tokens=self._prompt_tokens,
                prefix_len=len(self._prompt_tokens),
                layer_idx=layer_idx,
                hidden=hidden,
            )
