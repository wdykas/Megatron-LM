# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for delta re-rotation of cached RoPE keys (rope_mode='renumber').

The load-bearing check: our delta rotation must match Megatron's OWN rope
application exactly — keys rotated by RotaryEmbedding/apply_rotary_pos_emb at
positions m, delta-rotated to m', must equal the raw keys rotated directly at
m'. Conventions (NeoX halves vs interleaved, partial rotary) come from the
same modules the model uses.
"""

import pytest
import torch

from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.rl.compaction.kv.megatron_hook import MegatronInferenceHook
from megatron.rl.compaction.kv.rope import delta_rotate_keys

from tests.unit_tests.rl.compaction.test_megatron_hook import _make_context


def _megatron_rotate(raw, positions, rotary, interleaved):
    """Rotate raw (T, H, D) keys at the given positions via Megatron's own path."""
    cfg = TransformerConfig(
        num_layers=1, hidden_size=raw.shape[-1], num_attention_heads=1,
        apply_rope_fusion=False, rotary_interleaved=interleaved,
    )
    freqs = rotary(int(positions.max().item()) + 1)          # (max, 1, 1, rot)
    t = raw.unsqueeze(1)                                     # (T, B=1, H, D) sbhd
    return apply_rotary_pos_emb(t, freqs[positions], config=cfg).squeeze(1)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rotary_percent", [1.0, 0.5])
def test_delta_rotation_matches_megatron_rope(interleaved, rotary_percent):
    torch.manual_seed(0)
    T, H, D = 12, 2, 16
    rotary = RotaryEmbedding(
        kv_channels=D, rotary_percent=rotary_percent,
        rotary_interleaved=interleaved, rotary_base=10000,
    )
    raw = torch.randn(T, H, D, device="cuda", dtype=torch.float32)
    old_pos = torch.arange(17, 17 + T, device="cuda")        # scattered original
    new_pos = torch.arange(T, device="cuda")                 # renumbered 0..T-1

    stored = _megatron_rotate(raw, old_pos, rotary, interleaved)    # what the cache holds
    want = _megatron_rotate(raw, new_pos, rotary, interleaved)      # ground truth at new pos

    got = delta_rotate_keys(
        stored.unsqueeze(0),                                 # (L=1, T, H, D)
        old_pos, new_pos, rotary.inv_freq, interleaved=interleaved,
    ).squeeze(0)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_delta_rotation_group_property():
    """Composing 0->m then m->m' equals 0->m' directly."""
    torch.manual_seed(1)
    L, T, H, D = 3, 8, 1, 32
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    raw = torch.randn(L, T, H, D, device="cuda")
    zero = torch.zeros(T, dtype=torch.long, device="cuda")
    m = torch.randint(0, 500, (T,), device="cuda")
    m2 = torch.randint(0, 500, (T,), device="cuda")
    via = delta_rotate_keys(
        delta_rotate_keys(raw, zero, m, inv_freq), m, m2, inv_freq)
    direct = delta_rotate_keys(raw, zero, m2, inv_freq)
    torch.testing.assert_close(via, direct, atol=1e-4, rtol=1e-4)


def test_identity_when_positions_unchanged():
    inv_freq = torch.rand(8, device="cuda")
    k = torch.randn(2, 5, 1, 16, device="cuda")
    pos = torch.arange(5, device="cuda")
    torch.testing.assert_close(delta_rotate_keys(k, pos, pos, inv_freq), k)


def test_shape_validation():
    inv_freq = torch.rand(8, device="cuda")
    k = torch.randn(2, 5, 1, 16, device="cuda")
    with pytest.raises(ValueError, match="positions"):
        delta_rotate_keys(k, torch.arange(5, device="cuda"),
                          torch.arange(4, device="cuda"), inv_freq)
    with pytest.raises(ValueError, match="token dim"):
        delta_rotate_keys(k, torch.arange(4, device="cuda"),
                          torch.arange(4, device="cuda"), inv_freq)
    with pytest.raises(ValueError, match="rot_dim"):
        delta_rotate_keys(k, torch.arange(5, device="cuda"),
                          torch.arange(5, device="cuda"),
                          torch.rand(9, device="cuda"))


class _StubEngine:
    """Minimal engine stub for LiveKVCompactor ctor guard tests."""

    def __init__(self, pos_emb, rotary=None):
        from types import SimpleNamespace
        # One fake attention layer so archive/snapkv modes can register Q hooks.
        attn = SimpleNamespace(core_attention=object(),
                               flash_decode_and_prefill=lambda *a, **k: None)
        model = SimpleNamespace(
            position_embedding_type=pos_emb,
            decoder=SimpleNamespace(layers=[SimpleNamespace(self_attention=attn)]),
        )
        if rotary is not None:
            model.rotary_pos_emb = rotary
        self.context = _make_context()
        self.controller = SimpleNamespace(
            inference_wrapped_model=SimpleNamespace(model=model))


class TestRopeModeGuards:
    def _build(self, pos_emb, rope_mode=None, rotary=None, strategy="streaming_llm"):
        from megatron.rl.compaction.kv.live import LiveKVCompactor
        return LiveKVCompactor(
            _StubEngine(pos_emb, rotary), strategy=strategy,
            budget_ratio=0.5, rope_mode=rope_mode)

    def test_rope_model_requires_a_mode(self):
        with pytest.raises(NotImplementedError, match="rope_mode"):
            self._build("rope")

    def test_none_pe_model_rejects_a_mode(self):
        with pytest.raises(ValueError, match="no positional embedding"):
            self._build("none", rope_mode="logical")

    def test_belief_still_under_rope_unsupported(self):
        with pytest.raises(NotImplementedError, match="belief_still"):
            self._build("rope", rope_mode="logical", strategy="belief_still")

    def test_renumber_pulls_model_inv_freq(self):
        from types import SimpleNamespace
        rotary = SimpleNamespace(
            inv_freq=torch.rand(8, device="cuda"), rotary_interleaved=True)
        comp = self._build("rope", rope_mode="renumber", rotary=rotary)
        assert comp._inv_freq is rotary.inv_freq and comp._rope_interleaved

    def test_renumber_without_rotary_module_raises(self):
        with pytest.raises(RuntimeError, match="inv_freq"):
            self._build("rope", rope_mode="renumber")

    def test_logical_mode_builds_on_rope_model(self):
        comp = self._build("rope", rope_mode="logical")
        assert comp.rope_mode == "logical" and comp._inv_freq is None


class TestPrefetchTrigger:
    """The static-band + rising-trend prefetch predicate (D3)."""

    def _comp(self, prefetch_margin=None, prefetch_horizon=None):
        from megatron.rl.compaction.kv.live import LiveKVCompactor
        return LiveKVCompactor(
            _StubEngine("none"), strategy="streaming_llm", budget_ratio=0.5,
            archive=True, retrieval_margin=-3.0,
            prefetch_margin=prefetch_margin, prefetch_horizon=prefetch_horizon)

    def test_static_band(self):
        comp = self._comp(prefetch_margin=-5.0)
        assert comp._should_prefetch(-4.0, None)
        assert not comp._should_prefetch(-5.5, None)

    def test_rising_trend_predicts_crossing(self):
        comp = self._comp(prefetch_horizon=3)
        # -6 -> -4: slope +2, predicted -4 + 3*2 = +2 > -3 -> stage.
        assert comp._should_prefetch(-4.0, -6.0)
        # falling margin never stages.
        assert not comp._should_prefetch(-4.0, -3.5)
        # rising but too slow for the horizon: -4 + 3*0.1 = -3.7 <= -3.
        assert not comp._should_prefetch(-4.0, -4.1)
        # no history yet.
        assert not comp._should_prefetch(-4.0, None)

    def test_either_trigger_fires(self):
        comp = self._comp(prefetch_margin=-4.5, prefetch_horizon=2)
        assert comp._should_prefetch(-4.0, -3.9)   # band, despite falling trend
        assert comp._should_prefetch(-5.0, -4.0) is False  # below band, falling
        assert comp._should_prefetch(-5.0, -6.5)   # below band, trend predicts

    def test_horizon_validation(self):
        import pytest as _pytest
        with _pytest.raises(ValueError, match="prefetch_horizon"):
            self._comp(prefetch_horizon=0)


class TestOverwriteKeys:
    def test_round_trip_keys_only(self):
        ctx = _make_context(n_layers=2, n_heads=1, d_head=4, block_size=4, seq_len=6)
        hook = MegatronInferenceHook(ctx)
        _, v_before = hook.get_kv_for_request(0)
        new_k = torch.randn(2, 6, 1, 4)
        hook.overwrite_keys_for_request(0, new_k)
        k_after, v_after = hook.get_kv_for_request(0)
        torch.testing.assert_close(k_after, new_k.to(k_after))
        torch.testing.assert_close(v_after, v_before)       # values untouched

    def test_wrong_length_raises(self):
        ctx = _make_context(n_layers=1, n_heads=1, d_head=1, block_size=4, seq_len=6)
        hook = MegatronInferenceHook(ctx)
        with pytest.raises(RuntimeError, match="KV length"):
            hook.overwrite_keys_for_request(0, torch.zeros(1, 3, 1, 1))
