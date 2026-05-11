# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for partial-model construction primitives."""

import pytest
import torch

from megatron.core.inference.partial_model import (
    IdentityLayer,
    PartialModelOwnership,
    filter_layer_pattern,
    ownership_for_shard,
    select_owned_layer_state,
)
from megatron.core.inference.shards import InferenceShard


def _shard(index=0, kinds=None, layer_indices=None):
    spec = {"tp": 1, "pp": 1, "ep": 1, "expt_tp": 1, "dp": 1}
    if kinds is not None:
        spec["kinds"] = tuple(kinds)
    return InferenceShard(
        index=index,
        spec=spec,
        rank_offset=0,
        world_size=1,
        pg_collection=None,
        kinds=tuple(kinds) if kinds is not None else None,
        layer_indices=tuple(layer_indices) if layer_indices is not None else None,
    )


# ---- IdentityLayer ----------------------------------------------------------


def test_identity_layer_zero_parameters():
    layer = IdentityLayer()
    assert sum(p.numel() for p in layer.parameters()) == 0
    assert sum(b.numel() for b in layer.buffers()) == 0


def test_identity_layer_returns_input_unchanged():
    layer = IdentityLayer()
    x = torch.randn(2, 3, 4)
    y = layer(x)
    # Same tensor object — no copy, no shape change.
    assert y is x


def test_identity_layer_accepts_arbitrary_kwargs():
    """Real layers in Megatron take all kinds of extras (attention
    masks, position ids, etc.). The stub must accept them without
    crashing in case it's called by mistake."""
    layer = IdentityLayer()
    x = torch.zeros(1, 1, 4)
    y = layer(x, attention_mask=None, position_ids=torch.arange(1))
    assert y is x


def test_identity_layer_has_stub_marker():
    """Defense-in-depth marker so the engine can sanity-check."""
    layer = IdentityLayer()
    assert getattr(layer, "_is_identity_stub", False) is True


# ---- PartialModelOwnership --------------------------------------------------


def test_ownership_non_disagg_shard_owns_everything():
    """Shard without kinds= owns the full model."""
    shard = _shard()
    pattern = ("M", "*", "M", "*", "E")
    own = ownership_for_shard(shard, pattern)
    assert own.layer_indices == (0, 1, 2, 3, 4)
    assert own.own_embedding is True
    assert own.own_lm_head is True
    assert own.num_layers_total == 5
    assert own.num_owned_layers() == 5
    assert own.memory_savings_ratio() == 0.0


def test_ownership_disagg_shard_owns_subset():
    """Disagg mamba shard owns only the M-positions."""
    shard = _shard(kinds=("M",), layer_indices=(0, 2))
    pattern = ("M", "*", "M", "*", "E")
    own = ownership_for_shard(shard, pattern)
    assert own.layer_indices == (0, 2)
    assert own.own_embedding is True  # owns layer 0
    assert own.own_lm_head is False  # last layer (E=4) not owned
    assert own.num_owned_layers() == 2
    # 5 total, 2 owned -> 60% savings.
    assert own.memory_savings_ratio() == pytest.approx(0.6)


def test_ownership_lm_head_only_on_last_layer_owner():
    pattern = ("M", "*", "M", "*", "E")
    # Expert shard owns just the final block — gets the LM head.
    shard = _shard(kinds=("E",), layer_indices=(4,))
    own = ownership_for_shard(shard, pattern)
    assert own.own_embedding is False
    assert own.own_lm_head is True


def test_ownership_middle_shard_owns_neither_endpoint():
    pattern = ("M", "*", "M", "*", "E")
    # Attention shard sits in the middle.
    shard = _shard(kinds=("*",), layer_indices=(1, 3))
    own = ownership_for_shard(shard, pattern)
    assert own.own_embedding is False
    assert own.own_lm_head is False


def test_ownership_owns_predicate():
    shard = _shard(kinds=("M",), layer_indices=(0, 2, 7))
    pattern = tuple("MMMMMMMM")
    own = ownership_for_shard(shard, pattern)
    assert own.owns(0)
    assert not own.owns(1)
    assert own.owns(7)


# ---- filter_layer_pattern --------------------------------------------------


def test_filter_pattern_replaces_unowned_with_stub_marker():
    pattern = "M*M*EE-"
    own = PartialModelOwnership(
        layer_indices=(0, 2),  # only first two M's
        own_embedding=True,
        own_lm_head=False,
        num_layers_total=7,
    )
    filtered = filter_layer_pattern(pattern, own)
    # Same length, owned positions preserve their symbol, others become "_".
    assert filtered == "M_M____"
    assert len(filtered) == len(pattern)


def test_filter_pattern_custom_marker():
    pattern = "M*M*"
    own = PartialModelOwnership(
        layer_indices=(1, 3),
        own_embedding=False,
        own_lm_head=True,
        num_layers_total=4,
    )
    assert filter_layer_pattern(pattern, own, stub_marker="x") == "x*x*"


def test_filter_pattern_length_mismatch_raises():
    own = PartialModelOwnership(
        layer_indices=(0,),
        own_embedding=True,
        own_lm_head=True,
        num_layers_total=2,
    )
    with pytest.raises(AssertionError, match="must agree"):
        filter_layer_pattern("MMM", own)


# ---- select_owned_layer_state ----------------------------------------------


def test_state_filter_keeps_owned_layer_keys():
    own = PartialModelOwnership(
        layer_indices=(0, 2),
        own_embedding=True,
        own_lm_head=True,
        num_layers_total=4,
    )
    keys = [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.attn.q_proj.weight",
        "decoder.layers.1.attn.q_proj.weight",
        "decoder.layers.2.attn.q_proj.weight",
        "decoder.layers.3.attn.q_proj.weight",
        "output_layer.weight",
    ]
    out = select_owned_layer_state(keys, own)
    # Embedding + output kept (not layer-indexed).
    # Only layers 0 and 2 of the layer-indexed keys retained.
    assert out == [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.attn.q_proj.weight",
        "decoder.layers.2.attn.q_proj.weight",
        "output_layer.weight",
    ]


def test_state_filter_non_integer_layer_index_passes_through():
    """If a key starts with the layer prefix but has a non-integer next
    segment, treat it as non-layer-indexed (don't filter)."""
    own = PartialModelOwnership(
        layer_indices=(1,),
        own_embedding=True,
        own_lm_head=True,
        num_layers_total=4,
    )
    keys = [
        "decoder.layers.weird.weight",  # not "decoder.layers.<int>"
        "decoder.layers.0.x",
        "decoder.layers.1.x",
    ]
    out = select_owned_layer_state(keys, own)
    assert "decoder.layers.weird.weight" in out  # passed through
    assert "decoder.layers.0.x" not in out
    assert "decoder.layers.1.x" in out


def test_state_filter_custom_prefix():
    own = PartialModelOwnership(
        layer_indices=(0,),
        own_embedding=True,
        own_lm_head=True,
        num_layers_total=2,
    )
    keys = ["my_model.blocks.0.w", "my_model.blocks.1.w"]
    out = select_owned_layer_state(keys, own, layer_key_prefix="my_model.blocks.")
    assert out == ["my_model.blocks.0.w"]
