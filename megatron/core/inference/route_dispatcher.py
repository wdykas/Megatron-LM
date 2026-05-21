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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from megatron.core.inference import activation_transport as at
from megatron.rl.inference.route_planner import Route


@dataclass(frozen=True)
class _LayerPlan:
    """Per-layer action precomputed at dispatcher init.

    ``receive_from_pe`` is the NVSHMEM PE this rank waits on at the
    hop's entry layer (``None`` if there's no predecessor hop or if
    this rank isn't a receiver under the TP topology). ``send_to_pes``
    is the tuple of PEs this rank puts to at the hop's exit layer
    (empty if no successor hop or this rank isn't a sender). A layer
    with a ``_LayerPlan`` is always owned by this shard; not-owned
    layers are stored as ``None`` in the plan map.

    Multi-peer ``send_to_pes`` supports hetero-TP between disagg
    shards: a tp=2 → tp=4 hop has each src rank fan out to 2 dst ranks
    (since the residual stream is TP-replicated, sending the same
    hidden state to multiple peers is correct).
    """

    receive_from_pe: Optional[int]
    send_to_pes: Tuple[int, ...]


def tp_pair_routing(
    src_tp: int, dst_tp: int, *, single_rep: bool = False
) -> List[Tuple[int, int]]:
    """Return the ``(src_tp_offset, dst_tp_offset)`` exchange pairs for
    one cross-shard hop between a src shard of TP ``src_tp`` and a dst
    shard of TP ``dst_tp``.

    The residual stream entering / leaving a block is TP-replicated
    (every TP rank holds the same ``(batch, hidden)``), so we don't need
    a slice / concat dance — only correct fan-out / stride so every dst
    rank receives the full hidden state from exactly one src rank.

    - **Matched TP** (``src_tp == dst_tp``): 1-to-1 by offset.
    - **Dst larger, divides** (``dst_tp = k * src_tp``): each src rank
      fans out to ``k`` consecutive dst ranks. Every dst rank receives.
    - **Src larger, divides** (``src_tp = k * dst_tp``): only every
      ``k``-th src rank sends; the others would be sending the same
      data to the same dst, wasting slots. Every dst rank receives from
      exactly one src rank.
    - **Non-divisible**: rejected with a clear error. A non-divisible
      ratio would need slice / concat redistribution within each block,
      not a primitive of v1.

    ``single_rep=True`` collapses the table to a single ``(0, 0)`` pair
    — only TP-0 of src sends to TP-0 of dst over NVSHMEM, and dst-side
    TP-internal broadcast distributes the hidden state to its other
    TP ranks. Cuts cross-shard bandwidth by ``max(src_tp, dst_tp)``
    at the cost of one NCCL broadcast per dst shard per hop. Win for
    multi-node disagg where inter-node BW is the bottleneck.
    """
    if single_rep:
        return [(0, 0)]
    if src_tp == dst_tp:
        return [(k, k) for k in range(src_tp)]
    if dst_tp > src_tp:
        if dst_tp % src_tp != 0:
            raise AssertionError(
                f"hetero-TP between disagg shards requires divisibility: "
                f"src_tp={src_tp}, dst_tp={dst_tp} (dst must be a multiple "
                f"of src)."
            )
        fanout = dst_tp // src_tp
        return [
            (k, k * fanout + i) for k in range(src_tp) for i in range(fanout)
        ]
    if src_tp % dst_tp != 0:
        raise AssertionError(
            f"hetero-TP between disagg shards requires divisibility: "
            f"src_tp={src_tp}, dst_tp={dst_tp} (src must be a multiple "
            f"of dst)."
        )
    stride = src_tp // dst_tp
    return [(k * stride, k) for k in range(dst_tp)]


def _build_layer_plan(
    route: Route,
    my_shard_idx: int,
    my_tp_offset: int,
    shard_tp: List[int],
    shard_rank_offset: List[int],
    *,
    single_rep: bool = False,
) -> Dict[int, Optional[_LayerPlan]]:
    """Resolve every layer's dispatch action against the given TP
    topology. ``None`` value means another shard owns the layer; an
    absent key means past the route's last layer (DONE).

    The hop-level routing tables (built via :func:`tp_pair_routing`)
    apply only to the hop's entry / exit layer; middle layers of a
    multi-layer hop have no inbound / outbound transport.

    ``single_rep=True``: only TP-0 sends/receives via NVSHMEM. Other
    TP ranks of the dst shard get ``receive_from_pe`` set to TP-0
    of src so :meth:`RouteDispatcher._receive` knows to participate
    in the dst-side TP broadcast (TP-0 as src) instead of doing its
    own NVSHMEM recv. ``send_to_pes`` is empty on non-TP-0 ranks.
    """
    my_tp = shard_tp[my_shard_idx]
    plan: Dict[int, Optional[_LayerPlan]] = {}
    last_hop_pos = len(route.hops) - 1
    for hop_pos, hop in enumerate(route.hops):
        if hop.shard_idx != my_shard_idx:
            for li in hop.layer_indices:
                plan[li] = None
            continue
        # Resolve inbound source PE under the prior hop's TP topology,
        # outbound destination PE list under the next hop's TP topology.
        receive_from_pe: Optional[int] = None
        if hop_pos > 0:
            prev = route.hops[hop_pos - 1]
            if single_rep:
                # All dst ranks "receive from" TP-0 of src; non-TP-0
                # ranks participate in the dst-side TP broadcast rather
                # than calling NVSHMEM recv directly. The flag lives
                # in the dispatcher's _receive based on my_tp_offset.
                receive_from_pe = shard_rank_offset[prev.shard_idx]
            else:
                srcs_for_me = [
                    s for (s, d) in tp_pair_routing(shard_tp[prev.shard_idx], my_tp)
                    if d == my_tp_offset
                ]
                if srcs_for_me:
                    receive_from_pe = shard_rank_offset[prev.shard_idx] + srcs_for_me[0]
        send_to_pes: Tuple[int, ...] = ()
        if hop_pos < last_hop_pos:
            nxt = route.hops[hop_pos + 1]
            if single_rep:
                # Only TP-0 of src actually sends. Other TP ranks
                # have the same hidden state but skip the NVSHMEM put.
                if my_tp_offset == 0:
                    send_to_pes = (shard_rank_offset[nxt.shard_idx],)
            else:
                dsts_for_me = [
                    d for (s, d) in tp_pair_routing(my_tp, shard_tp[nxt.shard_idx])
                    if s == my_tp_offset
                ]
                send_to_pes = tuple(
                    shard_rank_offset[nxt.shard_idx] + d for d in dsts_for_me
                )
        last_layer_in_hop = len(hop.layer_indices) - 1
        for i, li in enumerate(hop.layer_indices):
            plan[li] = _LayerPlan(
                receive_from_pe=(receive_from_pe if i == 0 else None),
                send_to_pes=(send_to_pes if i == last_layer_in_hop else ()),
            )
    return plan


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
    - A precomputed ``layer_idx → _LayerPlan`` map so each
      :meth:`dispatch_layer` call is O(1) — see :class:`_LayerPlan`.
    - The activation transport configuration (payload size, stream).

    No per-layer cursor or "inside hop" flag — every call to
    :meth:`dispatch_layer` reads the action straight from the
    precomputed plan.

    Arguments:
        route: The request's pre-computed route plan.
        my_shard_idx: The current shard's index in the layout.
        my_pe: This shard's NVSHMEM PE id (equal to global rank under
            standard NVSHMEM init).
        my_tp_offset: This rank's TP offset within its shard
            (``my_pe - my_shard.rank_offset`` in the common layout).
        shard_tp: Per-shard TP size, indexed by shard idx. Used to
            compute hetero-TP fanout / stride at init.
        shard_rank_offset: Per-shard base rank, indexed by shard idx.
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
        my_tp_offset: int,
        shard_tp: Sequence[int],
        shard_rank_offset: Sequence[int],
        hidden_shape: Tuple[int, ...],
        hidden_dtype: torch.dtype,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        tp_group: object = None,
        single_rep: bool = False,
    ) -> None:
        self._route = route
        self._my_shard = my_shard_idx
        self._my_pe = my_pe
        self._my_tp_offset = my_tp_offset
        self._tp_rank_zero_pe = my_pe - my_tp_offset
        self._tp_group = tp_group
        self._single_rep = single_rep
        self._hidden_shape = tuple(hidden_shape)
        self._hidden_dtype = hidden_dtype
        nelem = 1
        for d in self._hidden_shape:
            nelem *= d
        self._payload_nbytes = nelem * torch.empty((), dtype=hidden_dtype).element_size()
        self._stream = stream

        self._plan = _build_layer_plan(
            route=route,
            my_shard_idx=my_shard_idx,
            my_tp_offset=my_tp_offset,
            shard_tp=list(shard_tp),
            shard_rank_offset=list(shard_rank_offset),
            single_rep=single_rep,
        )

    # ---- public API -------------------------------------------------------

    def is_entry_shard(self) -> bool:
        return self._route.entry_shard == self._my_shard

    def is_exit_shard(self) -> bool:
        return self._route.exit_shard == self._my_shard

    def participating_shards(self) -> List[int]:
        """Sorted list of shard indices this request's route visits.
        Used by the engine's release path to address the right
        coord-side fan-out — same set the route was originally
        fanned to."""
        return sorted({h.shard_idx for h in self._route.hops})

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
        if plan.receive_from_pe is not None:
            hidden = self._receive(plan.receive_from_pe)
        hidden = run_local(hidden)
        if plan.send_to_pes:
            self._send(plan.send_to_pes, hidden)
            return None, LayerAction.SEND
        return hidden, LayerAction.LOCAL

    # ---- transport internals ----------------------------------------------

    def _send(self, dst_pes: Tuple[int, ...], hidden: torch.Tensor) -> None:
        for dst_pe in dst_pes:
            at.send_hidden(
                my_pe=self._my_pe,
                dst_pe=dst_pe,
                hidden=hidden,
                payload_nbytes=self._payload_nbytes,
                stream=self._stream,
            )

    def _receive(self, src_pe: int) -> torch.Tensor:
        if self._single_rep and self._tp_group is not None:
            import torch.distributed as _dist

            # Single-rep mode: TP-0 of dst pulls the hidden state via
            # NVSHMEM from TP-0 of src; then we broadcast to the rest
            # of this shard's TP group. Other TP ranks skip the
            # NVSHMEM recv and just join the broadcast as dst.
            if self._my_tp_offset == 0:
                out = at.receive_hidden(
                    my_pe=self._my_pe,
                    src_pe=src_pe,
                    hidden_shape=self._hidden_shape,
                    hidden_dtype=self._hidden_dtype,
                    payload_nbytes=self._payload_nbytes,
                    stream=self._stream,
                )
            else:
                out = torch.empty(
                    self._hidden_shape,
                    dtype=self._hidden_dtype,
                    device=torch.cuda.current_device(),
                )
            _dist.broadcast(out, src=self._tp_rank_zero_pe, group=self._tp_group)
            return out
        return at.receive_hidden(
            my_pe=self._my_pe,
            src_pe=src_pe,
            hidden_shape=self._hidden_shape,
            hidden_dtype=self._hidden_dtype,
            payload_nbytes=self._payload_nbytes,
            stream=self._stream,
        )
