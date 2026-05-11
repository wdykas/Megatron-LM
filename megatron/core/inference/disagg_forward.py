# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-layer dispatch helper for layer-kind disaggregation.

The model's forward calls :func:`maybe_dispatch_layer` once per layer.
For collocated requests it's a passthrough to ``run_local(hidden)``.
For disagg requests it consults the request's
:class:`RouteDispatcher` and returns the action the caller should
take (``LOCAL`` / ``RECEIVE`` / ``SEND`` / ``DONE`` / ``NOT_MY_REQUEST``).
Callers break out of the layer loop on ``SEND`` or ``DONE``.

"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from megatron.core.inference.route_walker import LayerAction


# Module-level sentinel: a falsy hidden return paired with ``SEND``
# tells the caller "request suspended; stop iterating layers."
_SEND_ACTIONS = frozenset({LayerAction.SEND, LayerAction.DONE})


def should_stop_layer_loop(action: LayerAction) -> bool:
    """Whether the model's per-layer loop should break on this action."""
    return action in _SEND_ACTIONS


def maybe_dispatch_layer(
    engine: Any,
    request_id: Optional[int],
    layer_idx: int,
    hidden: Optional[torch.Tensor],
    run_local: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[Optional[torch.Tensor], LayerAction]:
    """Per-layer hook the model.forward calls instead of layer.forward.

    Behavior:

    - ``request_id is None`` or no dispatcher → passthrough; returns
      ``(run_local(hidden), LayerAction.LOCAL)``. Equivalent to no
      disagg for the call site.
    - Dispatcher present → delegate to
      :meth:`RouteDispatcher.dispatch_layer`; return its
      ``(hidden_out, action)`` tuple unchanged.

    The model's per-layer loop should ``break`` when
    :func:`should_stop_layer_loop` returns ``True`` on the action
    (``SEND`` or ``DONE``). On ``NOT_MY_REQUEST`` ``hidden`` flows
    through unchanged.
    """
    if request_id is None:
        return run_local(hidden), LayerAction.LOCAL
    dispatcher = engine.get_route_dispatcher(request_id) if engine is not None else None
    if dispatcher is None:
        return run_local(hidden), LayerAction.LOCAL

    hidden_out, action = dispatcher.dispatch_layer(layer_idx, hidden, run_local)
    if action is LayerAction.NOT_MY_REQUEST:
        return hidden, action
    return hidden_out, action


def finalize_dispatch(
    engine: Any,
    request_id: Optional[int],
    final_layer_idx: int,
    hidden: torch.Tensor,
) -> bool:
    """Call after the per-layer loop to flush any terminal SEND the
    dispatcher still owes (e.g., the entry shard's last hop ended on
    the model's final layer).

    Returns ``True`` if a send happened (request suspended; the caller
    should skip the LM head), ``False`` otherwise.
    """
    if request_id is None or engine is None:
        return False
    dispatcher = engine.get_route_dispatcher(request_id)
    if dispatcher is None or dispatcher.is_done():
        return False
    return dispatcher.maybe_send_final(hidden, final_layer_idx)
