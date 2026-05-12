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

from enum import Enum, auto
from typing import Callable, Optional, Tuple

import torch

from megatron.core.inference import activation_transport as at
from megatron.rl.inference.route_planner import Route


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

        # Precompute layer_idx → (hop, hop_position) so dispatch_layer is
        # an O(1) dict lookup instead of an O(num_hops) linear scan.
        self._layer_to_hop: dict = {}
        for hop_pos, hop in enumerate(route.hops):
            for li in hop.layer_indices:
                self._layer_to_hop[li] = (hop, hop_pos)
        self._last_hop_pos = len(route.hops) - 1

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
        entry = self._layer_to_hop.get(layer_idx)
        if entry is None:
            # Past the route's last layer (or before its first, which
            # only happens on degenerate layouts).
            return hidden, LayerAction.DONE

        hop, hop_pos = entry
        if hop.shard_idx != self._my_shard:
            return hidden, LayerAction.NOT_MY_REQUEST

        # I own this layer. RECEIVE first if this is the hop's entry
        # layer AND the hop has a predecessor shard.
        if layer_idx == hop.layer_indices[0] and hop_pos > 0:
            hidden = self._receive(self._route.hops[hop_pos - 1].shard_idx)

        hidden = run_local(hidden)

        # SEND if this is the hop's exit layer AND another hop follows
        # (on any shard — for revisits to the same shard we still send).
        if layer_idx == hop.layer_indices[-1] and hop_pos < self._last_hop_pos:
            next_hop = self._route.hops[hop_pos + 1]
            self._send(next_hop.shard_idx, hidden)
            return None, LayerAction.SEND

        return hidden, LayerAction.LOCAL

    # ---- transport internals ----------------------------------------------

    def _send(self, dst_shard: int, hidden: torch.Tensor) -> None:
        dst_pe = self._shard_to_pe(dst_shard)
        lane = at.lane_for(self._my_pe, dst_pe)
        slot = at.next_send_slot(lane)
        sym = at.activation_slot(slot)
        s = self._stream or at.activation_stream()
        with torch.cuda.stream(s):
            view = (
                sym[: self._payload_nbytes]
                .view(self._hidden_dtype)
                .reshape(hidden.shape)
            )
            view.copy_(hidden)
        at.put_activation(
            slot, dst_pe=dst_pe, nbytes=self._payload_nbytes, stream=s
        )

    def _receive(self, src_shard: int) -> torch.Tensor:
        src_pe = self._shard_to_pe(src_shard)
        lane = at.lane_for(src_pe, self._my_pe)
        slot = at.next_recv_slot(lane)
        s = self._stream or at.activation_stream()
        at.wait_activation(slot, stream=s)
        s.synchronize()
        sym = at.activation_slot(slot)
        view = (
            sym[: self._payload_nbytes]
            .view(self._hidden_dtype)
            .reshape(self._hidden_shape)
        )
        out = view.clone()
        at.ack_activation(slot, src_pe=src_pe, stream=s)
        return out
