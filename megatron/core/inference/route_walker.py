# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-request forward-pass state machine for disagg routing.

Engine-side helper that the inference engine consults at each layer
during a request's forward pass. Encapsulates the "am I supposed to run
this layer, send the activation onward, or wait for an inbound
activation?" decision so the engine code stays linear and the
state-tracking lives in one testable place.

The walker is created once per (request, shard) at request-submit time
and consulted layer-by-layer during ``DynamicInferenceEngine.step``.
It does *not* talk to NVSHMEM directly — the engine calls
``activation_transport.put_activation`` / ``wait_activation`` based on
what the walker returns.

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from megatron.rl.inference.route_planner import Route


class LayerAction(Enum):
    """What the engine should do at a given layer for this request."""

    # This shard owns this layer. Compute the layer locally; the walker
    # does no inter-shard work for this step.
    LOCAL = auto()

    # This shard is the entry point and just finished its hop. Send
    # the produced activation to the next shard in the route and
    # suspend the request (no further compute on this shard until the
    # route returns here or finishes).
    SEND = auto()

    # This shard's next hop needs an activation from the previous shard.
    # Wait on activation_transport, then proceed to compute this hop's
    # layers (which will trigger LOCAL on subsequent layers).
    RECEIVE = auto()

    # The route is exhausted; the request's forward pass is done on
    # this shard. (For the exit shard, this is when the LM head has
    # been applied.)
    DONE = auto()

    # This shard isn't on the request's route at all — skip the
    # request entirely on this engine's step.
    NOT_MY_REQUEST = auto()


@dataclass(frozen=True)
class LayerDecision:
    """Result of :meth:`RouteWalker.before_layer`.

    Attributes:
        action: What the engine should do for this (request, layer) pair.
        peer_shard: For SEND, the destination shard's index; for
            RECEIVE, the source shard's index. ``None`` for the other
            actions.
        hop_layers: For SEND / LOCAL within a hop, the full
            ``layer_indices`` of the current hop (so the engine can
            batch-run them all in one inner loop and call SEND once).
            ``None`` otherwise.
    """

    action: LayerAction
    peer_shard: Optional[int] = None
    hop_layers: Optional[tuple] = None


# Singletons for actions that carry no peer/hop_layers payload. Hot-
# path callers (the per-layer dispatch loop) hit ``NOT_MY_REQUEST`` and
# ``DONE`` most often; returning the same frozen instance avoids
# allocating a fresh dataclass per (request, layer) tuple. LOCAL is
# returned by the dispatcher (not the walker) with run_local output, so
# no singleton there.
_DECISION_DONE = LayerDecision(action=LayerAction.DONE)
_DECISION_NOT_MY_REQUEST = LayerDecision(action=LayerAction.NOT_MY_REQUEST)


class RouteWalker:
    """Per-request, per-shard layer-by-layer state machine.

    Lifecycle:
        walker = RouteWalker(route, my_shard_idx=...)
        for layer_idx in range(num_layers):
            decision = walker.before_layer(layer_idx)
            if decision.action is LayerAction.LOCAL:
                ...compute...
            elif decision.action is LayerAction.SEND:
                put_activation(...)
                walker.after_send()
                break  # request suspends on this shard
            elif decision.action is LayerAction.RECEIVE:
                hidden = wait_activation(...)
                walker.after_receive()
                # continue: next layer will be LOCAL
            elif decision.action is LayerAction.DONE:
                break
            elif decision.action is LayerAction.NOT_MY_REQUEST:
                break

    The walker is intentionally pure: it never touches process groups,
    NVSHMEM, or the model. The engine wires those in around it.
    """

    def __init__(self, route: Route, my_shard_idx: int):
        self._route = route
        self._my_shard = my_shard_idx
        # Which hops in the route belong to my shard, by position.
        self._my_hop_positions = tuple(
            i for i, h in enumerate(route.hops) if h.shard_idx == my_shard_idx
        )
        # Current cursor into my hops list. Advances on after_receive
        # (entering a hop) and after_send (leaving a hop).
        self._cursor = 0
        # Within the current hop, the next layer expected. Compared
        # against the engine's layer_idx to detect SEND boundaries.
        self._next_local_layer_idx: Optional[int] = None
        # True if we've consumed the entry hop's RECEIVE (or, for the
        # entry shard which doesn't RECEIVE, the initial state).
        self._inside_hop = False
        # True after the final hop's last layer ran on this shard.
        self._exhausted = False

        if self._my_hop_positions:
            first_hop = route.hops[self._my_hop_positions[0]]
            # If the first hop my shard runs is the entry hop, we're
            # already "inside" it (input came from embeddings, not
            # from another shard). Otherwise, we'll RECEIVE first.
            if first_hop.src_shard is None:
                self._inside_hop = True
                self._next_local_layer_idx = first_hop.layer_indices[0]

    # ---- core decision API --------------------------------------------------

    def before_layer(self, layer_idx: int) -> LayerDecision:
        """Decide what the engine should do for ``layer_idx``.

        ``layer_idx`` is the global model-order layer index. The walker
        compares it against its cursor into the route to produce the
        right action.
        """
        if self._exhausted:
            return _DECISION_DONE
        if not self._my_hop_positions:
            return _DECISION_NOT_MY_REQUEST

        current_hop_pos = self._my_hop_positions[self._cursor]
        current_hop = self._route.hops[current_hop_pos]

        if not self._inside_hop:
            return self._decide_before_hop_entry(
                layer_idx, current_hop_pos, current_hop
            )
        return self._decide_inside_hop(layer_idx, current_hop_pos, current_hop)

    def _decide_before_hop_entry(
        self, layer_idx: int, hop_pos: int, hop
    ) -> LayerDecision:
        """We haven't consumed this hop's RECEIVE yet. RECEIVE on the
        hop's first layer; NOT_MY_REQUEST for earlier layers."""
        if layer_idx < hop.layer_indices[0]:
            return _DECISION_NOT_MY_REQUEST
        assert layer_idx == hop.layer_indices[0], (
            f"RouteWalker out of sync: layer_idx={layer_idx} but hop "
            f"{hop_pos} starts at {hop.layer_indices[0]} and we "
            f"haven't received yet."
        )
        assert hop.src_shard is not None, (
            f"Hop {hop_pos} has no src_shard but the walker is waiting "
            f"to RECEIVE into it; route is malformed."
        )
        return LayerDecision(
            LayerAction.RECEIVE,
            peer_shard=hop.src_shard,
            hop_layers=hop.layer_indices,
        )

    def _decide_inside_hop(
        self, layer_idx: int, hop_pos: int, hop
    ) -> LayerDecision:
        """Inside a hop: LOCAL while inside its range, SEND past it
        (to the next shard), DONE if this is the final hop."""
        if layer_idx < hop.layer_indices[0]:
            return _DECISION_NOT_MY_REQUEST
        if layer_idx <= hop.layer_indices[-1]:
            return LayerDecision(
                LayerAction.LOCAL, hop_layers=hop.layer_indices
            )
        # Past the last layer of this hop.
        is_last_hop_for_me = self._cursor == len(self._my_hop_positions) - 1
        is_route_end = hop_pos == len(self._route.hops) - 1
        if is_last_hop_for_me and is_route_end:
            self._exhausted = True
            return _DECISION_DONE
        next_route_pos = hop_pos + 1
        assert next_route_pos < len(self._route.hops), (
            "Walker has another local hop but the route ended; route "
            "was likely truncated or malformed."
        )
        next_hop = self._route.hops[next_route_pos]
        return LayerDecision(
            LayerAction.SEND,
            peer_shard=next_hop.shard_idx,
            hop_layers=hop.layer_indices,
        )

    # ---- state transitions ------------------------------------------------

    def after_receive(self) -> None:
        """Engine: I just completed the wait_activation for the current
        hop. Mark the hop as entered so the next ``before_layer`` calls
        can return LOCAL until the hop's layers are done."""
        assert not self._inside_hop, (
            "after_receive called twice without an intervening send "
            "or DONE."
        )
        self._inside_hop = True
        current_hop = self._route.hops[self._my_hop_positions[self._cursor]]
        self._next_local_layer_idx = current_hop.layer_indices[0]

    def after_send(self) -> None:
        """Engine: I just completed the put_activation. Advance the
        cursor to the next hop my shard owns; the engine should not
        emit further compute for this request until it sees this
        shard again (or the request comes back via RECEIVE)."""
        assert self._inside_hop, (
            "after_send called outside a hop; route walker mis-stepped."
        )
        self._inside_hop = False
        self._next_local_layer_idx = None
        self._cursor += 1
        if self._cursor >= len(self._my_hop_positions):
            # This shard has no more hops in the route. The request is
            # done from this shard's perspective.
            self._exhausted = True

    def is_done(self) -> bool:
        return self._exhausted

    def is_entry(self) -> bool:
        """Whether this shard is the request's entry shard."""
        return (
            bool(self._my_hop_positions)
            and self._my_hop_positions[0] == 0
        )

    def is_exit(self) -> bool:
        """Whether this shard is the request's exit shard (the one
        that produces the final logits / next token)."""
        return (
            bool(self._my_hop_positions)
            and self._my_hop_positions[-1] == len(self._route.hops) - 1
        )

    # ---- introspection ----------------------------------------------------

    def remaining_hops(self) -> int:
        if self._exhausted:
            return 0
        return len(self._my_hop_positions) - self._cursor

    def current_hop_layers(self) -> Optional[tuple]:
        """Layer indices in the hop we're currently running, or ``None``
        if we're between hops or done."""
        if self._exhausted or not self._inside_hop:
            return None
        pos = self._my_hop_positions[self._cursor]
        return self._route.hops[pos].layer_indices
