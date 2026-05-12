# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer dispatcher for layer-kind-disaggregated forward passes.

Combines :class:`Route` with the :mod:`activation_transport`
primitives. The model's forward loop calls
:meth:`RouteDispatcher.dispatch_layer` once per layer; the dispatcher
decides whether to run the layer locally, receive an activation from
a peer shard, or send an outgoing activation.

Dispatch is **stateless** given ``(route, my_shard_idx, layer_idx)``
— the dispatcher is just an indexed query plus the I/O calls. There
is no walker / cursor.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, Optional, Tuple

import torch

from megatron.core.inference import activation_transport as at
from megatron.rl.inference.route_planner import Route


@dataclass(frozen=True)
class _LayerPlan:
    """Per-layer action precomputed at dispatcher init.

    ``receive_from``/``send_to`` are peer shard indices when this layer
    is the entry/exit of the hop and a predecessor/successor hop exists,
    otherwise ``None``. A layer with a ``_LayerPlan`` is always owned by
    this shard; not-owned layers are stored as ``None`` in the plan map.
    """

    receive_from: Optional[int]
    send_to: Optional[int]


class LayerAction(Enum):
    """What happened at a layer this shard dispatched on.

    Returned for caller introspection / tests. The model's forward
    loop does not need to branch on the action — it just threads
    ``hidden`` through every layer (``None`` between a SEND and the
    next RECEIVE is handled automatically by NOT_MY_REQUEST
    pass-through). The shard's ``post_process`` flag (gated on
    ``own_lm_head`` via partial-model construction) decides whether
    the final LM head runs.
    """

    # Layer ran locally, possibly after receiving an inbound activation.
    LOCAL = auto()
    # Activation sent to peer shard. ``hidden`` returned as ``None``;
    # the loop continues with NOT_MY_REQUEST until the next hop where
    # this shard RECEIVEs again (or until the loop ends).
    SEND = auto()
    # Past the end of the request's route (rare during a normal
    # forward pass since the loop only iterates ``num_layers``).
    DONE = auto()
    # This layer's hop belongs to another shard; ``hidden`` flows
    # through unchanged.
    NOT_MY_REQUEST = auto()


class RouteDispatcher:
    """Stateless per-layer router for disaggregated forward passes.

    One instance per in-flight disagg request. The dispatcher owns:

    - The :class:`Route` (read-only).
    - A precomputed ``layer_idx → RouteHop`` index so each
      ``dispatch_layer`` call is O(1).
    - The activation transport configuration (lane assignment,
      payload size).

    No per-layer cursor or "inside hop" flag — every call to
    :meth:`dispatch_layer` reads the action straight from the route.

    Arguments:
        route: The request's pre-computed route plan.
        my_shard_idx: The current shard's index in the layout.
        my_pe: This shard's NVSHMEM PE id.
        shard_to_pe: Function ``shard_idx → pe_id`` resolving a peer
            shard to its NVSHMEM PE.
        hidden_shape: Activation tensor shape (e.g., ``(batch, hidden)``).
        hidden_dtype: dtype of the activation tensor.
        stream: Optional CUDA stream override; defaults to
            :func:`activation_transport.activation_stream`.
    """

    def __init__(
        self,
        route: Route,
        my_shard_idx: int,
        my_pe: int,
        shard_to_pe: Callable[[int], int],
        hidden_shape: Tuple[int, ...],
        hidden_dtype: torch.dtype,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        self._route = route
        self._my_shard = my_shard_idx
        self._my_pe = my_pe
        self._shard_to_pe = shard_to_pe
        self._hidden_shape = tuple(hidden_shape)
        self._hidden_dtype = hidden_dtype
        nelem = 1
        for d in self._hidden_shape:
            nelem *= d
        self._payload_nbytes = nelem * torch.empty((), dtype=hidden_dtype).element_size()
        self._stream = stream

        # Precompute layer_idx → _LayerPlan so dispatch_layer is a flat
        # dict lookup. ``None`` means "another shard owns this layer";
        # a missing key means "past the route's last layer (DONE)".
        self._plan: Dict[int, Optional[_LayerPlan]] = {}
        last_hop_pos = len(route.hops) - 1
        for hop_pos, hop in enumerate(route.hops):
            if hop.shard_idx != my_shard_idx:
                for li in hop.layer_indices:
                    self._plan[li] = None
                continue
            receive_from = (
                route.hops[hop_pos - 1].shard_idx if hop_pos > 0 else None
            )
            send_to = (
                route.hops[hop_pos + 1].shard_idx
                if hop_pos < last_hop_pos
                else None
            )
            for i, li in enumerate(hop.layer_indices):
                self._plan[li] = _LayerPlan(
                    receive_from=receive_from if i == 0 else None,
                    send_to=send_to if i == len(hop.layer_indices) - 1 else None,
                )

    # ---- public API -------------------------------------------------------

    def is_entry_shard(self) -> bool:
        return self._route.entry_shard == self._my_shard

    def is_exit_shard(self) -> bool:
        return self._route.exit_shard == self._my_shard

    def dispatch_layer(
        self,
        layer_idx: int,
        hidden: Optional[torch.Tensor],
        run_local: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], LayerAction]:
        """Make progress on ``layer_idx`` for this request.

        Returns ``(hidden_out, action)``. The model's forward loop
        threads ``hidden_out`` (possibly ``None`` between SEND and
        the next RECEIVE) through every layer; no caller-side branch
        on action is needed.
        """
        if layer_idx not in self._plan:
            return hidden, LayerAction.DONE
        plan = self._plan[layer_idx]
        if plan is None:
            return hidden, LayerAction.NOT_MY_REQUEST
        if plan.receive_from is not None:
            hidden = self._receive(plan.receive_from)
        hidden = run_local(hidden)
        if plan.send_to is not None:
            self._send(plan.send_to, hidden)
            return None, LayerAction.SEND
        return hidden, LayerAction.LOCAL

    # ---- transport internals ----------------------------------------------

    def _send(self, dst_shard: int, hidden: torch.Tensor) -> None:
        at.send_hidden(
            my_pe=self._my_pe,
            dst_pe=self._shard_to_pe(dst_shard),
            hidden=hidden,
            payload_nbytes=self._payload_nbytes,
            stream=self._stream,
        )

    def _receive(self, src_shard: int) -> torch.Tensor:
        return at.receive_hidden(
            my_pe=self._my_pe,
            src_pe=self._shard_to_pe(src_shard),
            hidden_shape=self._hidden_shape,
            hidden_dtype=self._hidden_dtype,
            payload_nbytes=self._payload_nbytes,
            stream=self._stream,
        )
