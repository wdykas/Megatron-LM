# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Integration test: partial-model ownership → filtered layer pattern →
disagg stub marker recognized by ``HybridStack``.

This test verifies the *pipeline* without constructing a real HybridStack
(which needs distributed init + real model config). Specifically:

- ``filter_layer_pattern`` outputs use the same stub marker
  (``DISAGG_STUB_MARKER``) that ``hybrid_block`` recognizes.
- The marker is the only thing connecting the two modules; if one
  changes, this test catches the drift.
- The stub class (``IdentityLayer``) and the marker symbol both come
  from the partial_model module, so the hybrid_block import is the
  single integration seam.
"""

from megatron.core.inference.partial_model import (
    IdentityLayer,
    PartialModelOwnership,
    filter_layer_pattern,
)
from megatron.core.models.hybrid.hybrid_block import (
    DISAGG_STUB_MARKER,
    IdentityLayer as HybridImportedIdentityLayer,
)


def test_marker_symbol_is_underscore():
    """The stub marker must be a single character that doesn't collide
    with any valid LayerSymbol (M / * / D / E / - / G). Using ``_``
    avoids the collision; if the design ever changes the marker, this
    test forces an explicit update."""
    assert DISAGG_STUB_MARKER == "_"
    # Make sure it's a single character.
    assert isinstance(DISAGG_STUB_MARKER, str) and len(DISAGG_STUB_MARKER) == 1


def test_marker_is_disjoint_from_layer_symbols():
    """The stub marker must NOT match any real layer kind, or the
    hybrid stack would silently build a real layer at a stub
    position."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    assert DISAGG_STUB_MARKER not in Symbols.VALID_LAYERS, (
        f"DISAGG_STUB_MARKER={DISAGG_STUB_MARKER!r} collides with a "
        f"valid layer symbol; pick a different marker."
    )


def test_hybrid_block_imports_identity_layer_from_partial_model():
    """The HybridStack module imports its IdentityLayer from the
    partial_model module — single source of truth."""
    assert HybridImportedIdentityLayer is IdentityLayer


def test_filter_pattern_output_uses_marker_hybrid_recognizes():
    """The pattern that comes out of ``filter_layer_pattern`` must use
    the exact same marker character that ``hybrid_block`` recognizes."""
    pattern = "M*E"
    own = PartialModelOwnership(
        layer_indices=(0,),
        own_embedding=True,
        own_lm_head=False,
        num_layers_total=3,
    )
    filtered = filter_layer_pattern(pattern, own)
    # filtered[1] and filtered[2] should be DISAGG_STUB_MARKER.
    assert filtered[0] == "M"
    assert filtered[1] == DISAGG_STUB_MARKER
    assert filtered[2] == DISAGG_STUB_MARKER


def test_identity_layer_can_substitute_in_module_list():
    """The stub class must be a real nn.Module so it slots into a
    standard ModuleList without errors."""
    import torch.nn as nn

    layers = nn.ModuleList([IdentityLayer(), IdentityLayer()])
    assert len(layers) == 2
    # Iterating works; stubs are real Modules.
    for layer in layers:
        assert isinstance(layer, nn.Module)
        assert hasattr(layer, "_is_identity_stub")
