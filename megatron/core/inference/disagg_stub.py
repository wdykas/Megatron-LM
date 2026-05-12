# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer-stub primitives for layer-kind-disaggregated inference.

A disagg shard runs only the layer kinds it owns; for the rest, the
shard installs a zero-parameter pass-through at the matching position
in the ``ModuleList`` so block indexing remains valid. The route
dispatcher short-circuits to activation transport before the stub is
ever called; the ``IdentityLayer.forward`` body is defensive.

Kept in :mod:`megatron.core.inference` (not in the shared model files)
because the concept is purely inference-time — training never produces a
:data:`DISAGG_STUB_MARKER` in any layer pattern.
"""

from torch import nn


# Single-char marker placed in ``layer_type_list`` at positions a
# disagg shard does not own. Disjoint from every entry in
# ``LayerSymbols.VALID_LAYERS`` so the factory loop distinguishes a
# stub from a real layer kind.
DISAGG_STUB_MARKER = "_"


class IdentityLayer(nn.Module):
    """Zero-parameter pass-through used in place of non-owned layers."""

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states
