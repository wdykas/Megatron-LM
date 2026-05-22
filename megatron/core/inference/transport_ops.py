# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Registered ``torch.library.custom_op`` wrappers around the
activation transport backend.

The :class:`RouteDispatcher` and :class:`CompiledRoute` call into
:func:`get_activation_transport_backend` directly, which works fine
for the standard interpreted path. Under ``torch.compile``, however,
those direct calls are opaque C extensions and Dynamo graph-breaks at
every hop boundary — so a per-shard forward made of (say) two RECVs
and two SENDs costs four graph breaks, plus a Python re-entry per
break.

This module wraps the two transport primitives as registered torch
ops with fake / meta kernels:

- ``disagg::send_hidden`` (side effect: ships the tensor; no return).
- ``disagg::receive_hidden`` (side effect: blocks on incoming; returns
  a fresh tensor whose shape is fully known from the inputs).

Once these are registered, Dynamo records the ops directly into the
graph instead of breaking. The actual NVSHMEM / mock backend call
happens at op-execution time via :func:`get_activation_transport_backend`,
so the pluggable-backend story is preserved.

The fake kernels are the contract that Dynamo trusts:

- ``send_hidden``'s fake is a no-op — no output, no side effect that
  affects the local rank's tensor state.
- ``receive_hidden``'s fake constructs an ``empty`` tensor of the
  declared shape + dtype + device. Inductor / TorchDynamo treat
  this as the producer for downstream graph nodes that consume the
  received hidden state.
"""

from __future__ import annotations

from typing import List, Optional

import torch

# Importing get_activation_transport_backend at module level would
# create a circular import (transport_backend → activation_transport
# → transport_ops). Defer the import to the op's body.


# ---- send_hidden ---------------------------------------------------------


@torch.library.custom_op("disagg::send_hidden", mutates_args=("hidden",))
def send_hidden(
    hidden: torch.Tensor,
    my_pe: int,
    dst_pe: int,
    payload_nbytes: int,
) -> None:
    """Ship ``hidden`` to ``dst_pe`` via the active backend.

    Side-effecting at the transport level (ships data over NVSHMEM)
    but local-tensor-state-preserving: the input ``hidden`` is only
    *read*. ``mutates_args=("hidden",)`` is a deliberate over-declaration
    so Dynamo doesn't DCE the call — it must keep send_hidden in the
    graph since "hidden" is now considered mutated. In the standard
    dispatch pattern ``hidden`` has no downstream consumers after a
    SEND (the dispatcher sets ``hidden = None``), so the
    pessimization this implies is harmless.
    """
    from megatron.core.inference.transport_backend import (
        get_activation_transport_backend,
    )

    backend = get_activation_transport_backend()
    backend.send_hidden(
        my_pe=my_pe,
        dst_pe=dst_pe,
        hidden=hidden,
        payload_nbytes=payload_nbytes,
    )


@send_hidden.register_fake
def _send_hidden_fake(
    hidden: torch.Tensor,
    my_pe: int,
    dst_pe: int,
    payload_nbytes: int,
) -> None:
    # No-op for Dynamo's purposes — the op has no return and no
    # locally-visible side effect on the input tensor.
    return None


# ---- receive_hidden ------------------------------------------------------


@torch.library.custom_op("disagg::receive_hidden", mutates_args=())
def receive_hidden(
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    my_pe: int,
    src_pe: int,
    payload_nbytes: int,
) -> torch.Tensor:
    """Receive a hidden activation from ``src_pe`` via the active
    backend. Blocks on the transport's signal_wait for the
    corresponding lane.

    The output's shape / dtype / device are passed in explicitly
    because the backend can't decide them at runtime — they're
    properties of the route, known at dispatcher / compile time.
    The fake kernel uses these to construct an ``empty`` tensor so
    Dynamo can wire downstream graph nodes.
    """
    from megatron.core.inference.transport_backend import (
        get_activation_transport_backend,
    )

    backend = get_activation_transport_backend()
    out = backend.receive_hidden(
        my_pe=my_pe,
        src_pe=src_pe,
        hidden_shape=tuple(shape),
        hidden_dtype=dtype,
        payload_nbytes=payload_nbytes,
    )
    # Ensure the returned tensor lives on the declared device — some
    # backends (e.g. CPU mocks under test) return CPU tensors that
    # need to move to the route's declared device for downstream
    # consumers to be happy.
    if out.device != device:
        out = out.to(device)
    return out


@receive_hidden.register_fake
def _receive_hidden_fake(
    shape: List[int],
    dtype: torch.dtype,
    device: torch.device,
    my_pe: int,
    src_pe: int,
    payload_nbytes: int,
) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


# ---- thin convenience wrappers ------------------------------------------
#
# CompiledRoute (and any future opt-in caller) imports these wrappers
# rather than the registered op names so a future re-organization
# (e.g. moving the op registration into a C++ extension) doesn't
# require touching the call sites.


def send_via_op(
    hidden: torch.Tensor,
    *,
    my_pe: int,
    dst_pe: int,
    payload_nbytes: int,
) -> None:
    """Call ``disagg::send_hidden`` as a registered torch op."""
    send_hidden(hidden, my_pe, dst_pe, payload_nbytes)


def receive_via_op(
    *,
    hidden_shape,
    hidden_dtype: torch.dtype,
    device: Optional[torch.device],
    my_pe: int,
    src_pe: int,
    payload_nbytes: int,
) -> torch.Tensor:
    """Call ``disagg::receive_hidden`` as a registered torch op.

    When ``device`` is ``None`` the receiver's current CUDA device is
    used — matches the un-compiled path's behavior under
    :class:`RouteDispatcher`.
    """
    if device is None:
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    return receive_hidden(
        list(hidden_shape),
        hidden_dtype,
        device,
        my_pe,
        src_pe,
        payload_nbytes,
    )
