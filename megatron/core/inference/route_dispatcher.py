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

from megatron.core.inference.transport_backend import (
    get_activation_transport_backend,
)
from megatron.rl.inference.route_planner import Route


@dataclass(frozen=True)
class _LayerPlan:
    """Per-layer action precomputed at dispatcher init.

    ``receive_from_pes`` lists the PEs this rank waits on at the hop's
    entry layer:

    - 0 entries: no predecessor (entry hop) or this rank isn't a
      receiver under the TP topology — the dispatcher just uses the
      inbound ``hidden`` from the caller.
    - 1 entry: linear inbound; the dispatcher does a single
      ``backend.receive_hidden`` to overwrite ``hidden``.
    - >1 entries: DAG merge point; the dispatcher receives one
      activation per predecessor and reduces them via :attr:`reduce_op`.

    ``send_to_pes`` is the flat tuple of PEs this rank puts the
    outbound activation to at the hop's exit layer. The flatness is
    intentional — it combines TWO orthogonal sources of fan-out:

    - **Hetero-TP fan-out**: a tp=2 → tp=4 hop has each src rank send
      to 2 dst ranks (the residual stream is TP-replicated so sending
      the same hidden state to multiple peers is correct).
    - **DAG fan-out**: a hop with ``parallel_succs`` sends the same
      hidden state to multiple downstream shards in parallel.

    The dispatcher's ``_send`` loop doesn't need to distinguish the two
    — both are "ship this hidden to N peers" operations under
    NVSHMEM's stream-ordered put_signal.

    A layer with a ``_LayerPlan`` is always owned by this shard;
    not-owned layers are stored as ``None`` in the plan map.
    """

    receive_from_pes: Tuple[int, ...]
    send_to_pes: Tuple[int, ...]
    reduce_op: Optional[str] = None


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


def _resolve_inbound_pe(
    src_shard_idx: int,
    my_tp_offset: int,
    my_tp: int,
    shard_tp: List[int],
    shard_rank_offset: List[int],
    *,
    single_rep: bool,
) -> Optional[int]:
    """One predecessor's PE under hetero-TP. ``None`` if this rank
    doesn't receive from that predecessor under the routing table."""
    if single_rep:
        return shard_rank_offset[src_shard_idx]
    srcs_for_me = [
        s for (s, d) in tp_pair_routing(shard_tp[src_shard_idx], my_tp)
        if d == my_tp_offset
    ]
    if not srcs_for_me:
        return None
    return shard_rank_offset[src_shard_idx] + srcs_for_me[0]


def _resolve_outbound_pes(
    dst_shard_idx: int,
    my_tp_offset: int,
    my_tp: int,
    shard_tp: List[int],
    shard_rank_offset: List[int],
    *,
    single_rep: bool,
) -> Tuple[int, ...]:
    """One successor's PE list under hetero-TP. Empty when this rank
    doesn't send to that successor (e.g. non-TP-0 under single-rep)."""
    if single_rep:
        if my_tp_offset != 0:
            return ()
        return (shard_rank_offset[dst_shard_idx],)
    dsts_for_me = [
        d for (s, d) in tp_pair_routing(my_tp, shard_tp[dst_shard_idx])
        if s == my_tp_offset
    ]
    return tuple(shard_rank_offset[dst_shard_idx] + d for d in dsts_for_me)


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

    DAG hops (:attr:`RouteHop.parallel_succs` /
    :attr:`RouteHop.reduce_from`) add extra predecessor / successor
    edges on top of the linear ``hop_pos ± 1`` chain. The PE list for
    each edge is computed independently against that edge's hetero-TP
    relationship; the lists are concatenated into a single flat
    ``send_to_pes`` / ``receive_from_pes`` per layer plan so the
    dispatcher's transport loop stays straightforward.

    ``single_rep=True``: only TP-0 sends/receives via NVSHMEM. Other
    TP ranks of the dst shard get ``receive_from_pes`` set to TP-0
    of src so :meth:`RouteDispatcher._receive` knows to participate
    in the dst-side TP broadcast (TP-0 as src) instead of doing its
    own NVSHMEM recv. ``send_to_pes`` is empty on non-TP-0 ranks.
    """
    my_tp = shard_tp[my_shard_idx]
    plan: Dict[int, Optional[_LayerPlan]] = {}
    for hop_pos, hop in enumerate(route.hops):
        if hop.shard_idx != my_shard_idx:
            for li in hop.layer_indices:
                plan[li] = None
            continue

        # Inbound: union of linear predecessor + DAG reduce_from edges.
        inbound_pes: List[int] = []
        for pred_pos in route.predecessors(hop_pos):
            prev = route.hops[pred_pos]
            pe = _resolve_inbound_pe(
                prev.shard_idx,
                my_tp_offset,
                my_tp,
                shard_tp,
                shard_rank_offset,
                single_rep=single_rep,
            )
            if pe is not None:
                inbound_pes.append(pe)
        reduce_op = hop.reduce_op if len(inbound_pes) > 1 else None

        # Outbound: union of linear successor + DAG parallel_succs edges.
        outbound_pes: List[int] = []
        for succ_pos in route.successors(hop_pos):
            nxt = route.hops[succ_pos]
            outbound_pes.extend(
                _resolve_outbound_pes(
                    nxt.shard_idx,
                    my_tp_offset,
                    my_tp,
                    shard_tp,
                    shard_rank_offset,
                    single_rep=single_rep,
                )
            )

        last_layer_in_hop = len(hop.layer_indices) - 1
        for i, li in enumerate(hop.layer_indices):
            plan[li] = _LayerPlan(
                receive_from_pes=(tuple(inbound_pes) if i == 0 else ()),
                send_to_pes=(tuple(outbound_pes) if i == last_layer_in_hop else ()),
                reduce_op=(reduce_op if i == 0 else None),
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
        if plan.receive_from_pes:
            hidden = self._receive_and_maybe_reduce(
                plan.receive_from_pes, plan.reduce_op
            )
        hidden = run_local(hidden)
        if plan.send_to_pes:
            self._send(plan.send_to_pes, hidden)
            return None, LayerAction.SEND
        return hidden, LayerAction.LOCAL

    # ---- transport internals ----------------------------------------------

    def _send(self, dst_pes: Tuple[int, ...], hidden: torch.Tensor) -> None:
        backend = get_activation_transport_backend()
        for dst_pe in dst_pes:
            backend.send_hidden(
                my_pe=self._my_pe,
                dst_pe=dst_pe,
                hidden=hidden,
                payload_nbytes=self._payload_nbytes,
                stream=self._stream,
            )

    def _receive(self, src_pe: int) -> torch.Tensor:
        backend = get_activation_transport_backend()
        if self._single_rep and self._tp_group is not None:
            import torch.distributed as _dist

            # Single-rep mode: TP-0 of dst pulls the hidden state via
            # the transport from TP-0 of src; then we broadcast to the
            # rest of this shard's TP group. Other TP ranks skip the
            # transport recv and just join the broadcast as dst.
            if self._my_tp_offset == 0:
                out = backend.receive_hidden(
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
        return backend.receive_hidden(
            my_pe=self._my_pe,
            src_pe=src_pe,
            hidden_shape=self._hidden_shape,
            hidden_dtype=self._hidden_dtype,
            payload_nbytes=self._payload_nbytes,
            stream=self._stream,
        )

    def _receive_and_maybe_reduce(
        self,
        src_pes: Tuple[int, ...],
        reduce_op: Optional[str],
    ) -> torch.Tensor:
        """Receive one tensor per predecessor PE and combine them.

        Single-input fast path: identical to ``_receive`` (no reduce).
        Multi-input: pull each branch's activation via the transport,
        then reduce per ``reduce_op``:

        - ``"sum"``: in-place accumulate into the first branch's tensor.
        - ``"mean"``: sum + divide by branch count.

        The recv order matches ``src_pes`` order — branches with
        independent payloads land in independent slots so a slow branch
        doesn't head-of-line-block the others (the transport's
        ``signal_wait`` is per-(my_pe, src_pe) lane). The reduce
        happens on the transport stream after all branches arrive.
        """
        if len(src_pes) == 1:
            return self._receive(src_pes[0])
        # Each branch lands in its own buffer (the backend's
        # receive_hidden returns a fresh tensor cloned out of the slot,
        # so we can hold all branches simultaneously without worrying
        # about slot recycling). First branch goes through the standard
        # _receive helper to keep single-rep / TP-broadcast semantics
        # intact for the head branch; subsequent branches bypass it
        # since the dst-side broadcast already populated TP-replication.
        accum = self._receive(src_pes[0])
        for src_pe in src_pes[1:]:
            extra = self._receive(src_pe)
            accum.add_(extra)
        if reduce_op == "mean":
            accum.div_(len(src_pes))
        elif reduce_op not in ("sum", None):
            raise AssertionError(
                f"_receive_and_maybe_reduce: unsupported reduce_op={reduce_op!r}; "
                f"expected 'sum' or 'mean'."
            )
        return accum
