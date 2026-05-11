# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""High-level disagg forward-pass dispatcher.

Combines :class:`route_walker.RouteWalker` with the
:mod:`activation_transport` primitives so the model's forward pass
becomes a thin per-layer call:

.. code-block:: python

    for li, layer in enumerate(self.layers):
        hidden, action = dispatcher.dispatch_layer(li, hidden, layer.forward)
        if action is LayerAction.SEND:
            break  # request suspends; resumes later when activation returns
        if action is LayerAction.DONE:
            break

The dispatcher owns:

- The :class:`RouteWalker` (per-request state).
- The lane assignment (which NVSHMEM lane to use for each peer shard).
- The gather/scatter of the activation tensor into the symmetric slot.

Everything that's NOT the model's per-layer compute lives here, so the
model.forward integration is a single conditional + call.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from megatron.core.inference import activation_transport as at
from megatron.core.inference.route_walker import (
    LayerAction,
    LayerDecision,
    RouteWalker,
)
from megatron.rl.inference.route_planner import Route


class RouteDispatcher:
    """Per-request forward-pass driver.

    One instance per in-flight request. Reads the route, walks through
    layers, and at each layer either runs it locally, sends the hidden
    state, or waits for an inbound activation.

    Arguments:
        route: The request's pre-computed route plan.
        my_shard_idx: The current shard's index in the layout.
        my_pe: This shard's NVSHMEM PE id.
        shard_to_pe: Function ``shard_idx -> pe_id`` resolving a peer
            shard to its NVSHMEM PE (needed for the lane lookup).
        hidden_shape: Expected ``(batch, hidden_dim)`` (or equivalent)
            shape of the activation tensor that flows between shards.
            Used to compute the per-op byte count and reshape the
            scattered tensor on the receive side.
        hidden_dtype: dtype of the activation tensor (so the symmetric
            slot's ``uint8`` view can be reinterpreted correctly).
        stream: Optional CUDA stream override; defaults to
            :func:`activation_transport.activation_stream`.

    The dispatcher does not own the model or any per-layer state — it
    only mediates between the walker's decision and the NVSHMEM ops.
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
        self._walker = RouteWalker(route, my_shard_idx=my_shard_idx)
        self._my_pe = my_pe
        self._shard_to_pe = shard_to_pe
        self._hidden_shape = tuple(hidden_shape)
        self._hidden_dtype = hidden_dtype
        # Bytes-per-activation = product(shape) * elem_size.
        nelem = 1
        for d in self._hidden_shape:
            nelem *= d
        self._payload_nbytes = nelem * torch.empty((), dtype=hidden_dtype).element_size()
        self._stream = stream

    # ---- public API -------------------------------------------------------

    def is_done(self) -> bool:
        return self._walker.is_done()

    def is_entry_shard(self) -> bool:
        return self._walker.is_entry()

    def is_exit_shard(self) -> bool:
        return self._walker.is_exit()

    def dispatch_layer(
        self,
        layer_idx: int,
        hidden: Optional[torch.Tensor],
        run_local_layer: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], LayerAction]:
        """Make progress on the request at ``layer_idx``.

        Args:
            layer_idx: The model's current layer index.
            hidden: Current hidden state on this shard, or ``None`` if
                this shard has not yet received an activation.
            run_local_layer: Function applied for LOCAL layers; called
                with ``hidden`` and expected to return the post-layer
                hidden state. Typically ``self.layers[layer_idx].forward``.

        Returns:
            Tuple ``(hidden_out, action)`` where ``action`` is the
            :class:`LayerAction` the walker emitted. Callers should:

            - ``LOCAL`` / ``RECEIVE`` then LOCAL: take ``hidden_out``
              and proceed.
            - ``SEND``: request has been suspended; ``hidden_out`` is
              ``None``. Break out of the per-layer loop on this shard.
            - ``DONE``: route exhausted; for the exit shard, the
              caller should now run the LM head on the final
              ``hidden_out``.
            - ``NOT_MY_REQUEST``: skip layer; ``hidden_out`` unchanged.
        """
        dec = self._walker.before_layer(layer_idx)

        if dec.action is LayerAction.LOCAL:
            assert hidden is not None, (
                f"layer {layer_idx} LOCAL but no inbound hidden state — "
                "did dispatcher miss a RECEIVE?"
            )
            return run_local_layer(hidden), LayerAction.LOCAL

        if dec.action is LayerAction.RECEIVE:
            received = self._receive(dec.peer_shard)
            self._walker.after_receive()
            # Now run the actual layer compute locally on the received
            # activation.
            return run_local_layer(received), LayerAction.RECEIVE

        if dec.action is LayerAction.SEND:
            assert hidden is not None, (
                f"layer {layer_idx} SEND but no hidden state to send"
            )
            self._send(dec.peer_shard, hidden)
            self._walker.after_send()
            return None, LayerAction.SEND

        if dec.action is LayerAction.DONE:
            return hidden, LayerAction.DONE

        # NOT_MY_REQUEST
        return hidden, LayerAction.NOT_MY_REQUEST

    def maybe_send_final(
        self, hidden: torch.Tensor, final_layer_idx: int
    ) -> bool:
        """After the model's last real layer, the walker may still owe a
        terminal SEND if this shard's last hop ended at the final layer
        and the next hop is on another shard.

        Call this after the per-layer loop with ``layer_idx ==
        num_layers``; returns ``True`` if a send happened (request now
        suspended), ``False`` if the walker is already done."""
        dec = self._walker.before_layer(final_layer_idx)
        if dec.action is LayerAction.SEND:
            self._send(dec.peer_shard, hidden)
            self._walker.after_send()
            return True
        return False

    # ---- internals --------------------------------------------------------

    def _send(self, dst_shard: int, hidden: torch.Tensor) -> None:
        """Gather ``hidden`` into the next available slot for this
        shard→dst_shard lane and put + signal it onto the destination."""
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
        """Wait for the next inbound activation from ``src_shard`` on
        the lane, scatter it into a fresh tensor, ack back, and return
        the activation. Stream-synchronous on the activation stream
        before returning so callers see a host-visible tensor."""
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
