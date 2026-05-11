# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Partial-model construction primitives for layer-kind disaggregation.

When a shard runs only a subset of layer kinds (M / * / E / D / -),
allocating weights for the non-owned blocks wastes memory. The model
factory needs to know:

- Which global layer indices to materialize (real blocks with weights).
- Which global layer indices to stub (zero-parameter pass-throughs that
  keep ``ModuleList`` indexing dense so the forward loop is unchanged).
- Whether the embedding/lm-head should be built on this shard.

This module supplies the descriptor (:class:`PartialModelOwnership`)
that callers compute from their :class:`InferenceShard` once at engine
build time, and the :class:`IdentityLayer` stub the factory plugs in
for non-owned positions.

The actual surgery into the model factories
(``GPTModel`` / hybrid model construction, checkpoint loading) is
deeper engine-side work that consumes these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from megatron.core.inference.shards import InferenceShard

# Single-char marker placed in ``layer_type_list`` at positions a disagg
# shard does not own. Disjoint from every entry in
# ``Symbols.VALID_LAYERS`` so the model factory can distinguish a stub
# from a real layer kind.
STUB_MARKER = "_"


class IdentityLayer(nn.Module):
    """Zero-parameter pass-through used in place of non-owned layers.

    Layers a disagg shard doesn't own are replaced by an
    :class:`IdentityLayer` in the model's ``ModuleList`` so the forward
    loop's per-index access stays valid. The engine's forward-pass
    router (see :mod:`route_walker`) prevents the identity layer from
    ever actually being called for the request — it short-circuits to
    activation_transport. But a defense-in-depth pass-through ensures
    that if a request *does* reach an identity layer for any reason
    (e.g., a unit test that exercises the bare module), the hidden
    state propagates through unchanged.

    No parameters, no buffers. Allocates effectively zero VRAM.
    """

    def __init__(self) -> None:
        super().__init__()
        # Defensive marker so the engine can sanity-check at request-
        # routing time that any layer the walker decides to RECEIVE for
        # is in fact a stub (not a real layer that was accidentally
        # materialized).
        self._is_identity_stub = True

    def forward(
        self, hidden_states: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return hidden_states

    def extra_repr(self) -> str:
        return "IdentityLayer(stub for disagg-routed layer)"


@dataclass(frozen=True)
class PartialModelOwnership:
    """What pieces of the model this shard should materialize.

    Attributes:
        layer_indices: Global layer indices the shard owns. Always
            non-empty for a shard that participates in the forward
            pass. Sparse / non-contiguous in general.
        own_embedding: Whether the shard owns the embedding layer
            (token + position). True iff this shard owns layer 0.
            Disagg shards that aren't the entry shard pass embeddings
            transparently as part of the inbound activation.
        own_lm_head: Whether the shard owns the LM head + final norm.
            True iff this shard owns the last layer.
        num_layers_total: Total layer count of the model (length of
            ``layer_type_list``). Useful for the factory loop:
            ``for i in range(num_layers_total): if i in layer_indices:
            real_block() else: IdentityLayer()``.
    """

    layer_indices: Tuple[int, ...]
    own_embedding: bool
    own_lm_head: bool
    num_layers_total: int

    def owns(self, layer_idx: int) -> bool:
        return layer_idx in self.layer_indices

    def num_owned_layers(self) -> int:
        return len(self.layer_indices)

    def memory_savings_ratio(self) -> float:
        """Fraction of the layer space this shard avoids materializing.

        Reported in the bench / logging path so the "no wasted memory"
        claim is verifiable from the run output."""
        if self.num_layers_total == 0:
            return 0.0
        return 1.0 - (len(self.layer_indices) / self.num_layers_total)


def ownership_for_shard(
    shard: InferenceShard,
    layer_type_list: Tuple[str, ...],
) -> PartialModelOwnership:
    """Compute a :class:`PartialModelOwnership` from a shard + the model's
    per-block kind pattern.

    For shards without an explicit ``kinds`` restriction (collocated /
    hetero-PP back-compat), every layer in ``layer_type_list`` is
    treated as owned — i.e. this shard builds the full model. The
    embedding and LM-head ownership flags are always set in that case;
    the existing pipeline-parallelism gating elsewhere in Megatron
    handles which PP stage builds them.

    For disagg shards (``kinds`` populated): only the matching layer
    indices are owned. ``own_embedding`` ties to owning layer 0;
    ``own_lm_head`` ties to owning the last layer.
    """
    n_layers = len(layer_type_list)
    assert n_layers > 0, "layer_type_list must be non-empty."
    if shard.layer_indices is None:
        # Non-disagg shard: full ownership.
        return PartialModelOwnership(
            layer_indices=tuple(range(n_layers)),
            own_embedding=True,
            own_lm_head=True,
            num_layers_total=n_layers,
        )
    return PartialModelOwnership(
        layer_indices=tuple(shard.layer_indices),
        own_embedding=0 in shard.layer_indices,
        own_lm_head=(n_layers - 1) in shard.layer_indices,
        num_layers_total=n_layers,
    )


def filter_layer_pattern(
    pattern: str,
    ownership: PartialModelOwnership,
    stub_marker: str = STUB_MARKER,
) -> str:
    """Produce a layer pattern where non-owned positions are replaced
    with ``stub_marker``.

    Returned string has the same length as ``pattern``. The hybrid
    model construction path consumes the filtered pattern and
    instantiates an :class:`IdentityLayer` at every ``stub_marker``
    position.

    Args:
        pattern: Per-block kind symbols (e.g., ``"M*M*EE*M-"``).
        ownership: This shard's ownership descriptor.
        stub_marker: Single-character marker for non-owned positions.
            Defaults to ``"_"``. MUST NOT collide with any symbol in
            :class:`megatron.core.models.hybrid.hybrid_layer_allocation.Symbols`
            — caller responsibility.

    Returns:
        Filtered pattern, same length as input.
    """
    assert ownership.num_layers_total == len(pattern), (
        f"pattern length {len(pattern)} != ownership.num_layers_total "
        f"{ownership.num_layers_total}; they must agree."
    )
    return "".join(
        c if i in ownership.layer_indices else stub_marker
        for i, c in enumerate(pattern)
    )


def select_owned_layer_state(
    state_dict_keys: list,
    ownership: PartialModelOwnership,
    layer_key_prefix: str = "decoder.layers.",
) -> list:
    """Filter a checkpoint's state-dict keys to those for owned layers.

    Used by the checkpoint loader hook: when loading, skip parameter
    tensors for layers this shard doesn't materialize.

    The current Megatron checkpoint loader iterates per-layer
    directories and assumes contiguous layer offsets per PP stage. For
    disagg shards with non-contiguous ownership, the loader must be
    extended to consult this filter. Implementation lives outside this
    module; the helper here is the pure-Python decision function.

    Args:
        state_dict_keys: Full list of dotted keys in a checkpoint.
        ownership: This shard's descriptor.
        layer_key_prefix: How the checkpoint keys layer state. Default
            matches Megatron's convention; pass an alternative for
            other model classes.

    Returns:
        Subset of ``state_dict_keys`` belonging to either:
          - a layer this shard owns, OR
          - a parameter that isn't layer-indexed at all (embeddings,
            LM head, etc.). Embeddings and LM head are gated separately
            by ``ownership.own_embedding`` / ``own_lm_head`` at the
            module-construction level — they're not filtered here
            because the keys don't carry the layer offset.
    """
    owned_set = set(ownership.layer_indices)
    out: list = []
    for k in state_dict_keys:
        if not k.startswith(layer_key_prefix):
            out.append(k)
            continue
        # Parse the layer index from the next dotted segment.
        rest = k[len(layer_key_prefix):]
        idx_str = rest.split(".", 1)[0]
        try:
            layer_idx = int(idx_str)
        except ValueError:
            # Key doesn't have an integer right after the prefix; treat
            # as non-layer-indexed (won't be filtered).
            out.append(k)
            continue
        if layer_idx in owned_set:
            out.append(k)
    return out
