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

from megatron.rl.compaction.kv.serving.megatron_hook import MegatronInferenceHook
from megatron.rl.compaction.kv.serving.rope import delta_rotate_keys

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
        from megatron.rl.compaction.kv.serving.live import LiveKVCompactor
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


class TestAlphaCusumTrigger:
    """The α̂/CUSUM retrieval trigger (scale-free, per-span novelty)."""

    def _comp(self, **kw):
        from megatron.rl.compaction.kv.serving.live import LiveKVCompactor
        return LiveKVCompactor(
            _StubEngine("none"), strategy="snapkv", budget_ratio=0.5,
            archive=True, **kw)

    def test_defaults_are_scale_free_fractions(self):
        comp = self._comp()
        assert 0.0 < comp.retrieval_alpha < 1.0
        assert comp.retrieval_cusum > 0.0

    def test_archive_object_constructed(self):
        """archive=True must build the KVArchive — with it None, every
        +archive eval silently no-ops (measured: no retrievals key in stats,
        empty flywheel) while all tests stay green."""
        comp = self._comp()
        assert comp._archive is not None
        assert comp._prefetch_stream is not None
        assert "retrievals" in comp.stats()

    def test_alpha_validation(self):
        import pytest as _pytest
        with _pytest.raises(ValueError, match="retrieval_alpha"):
            self._comp(retrieval_alpha=1.5)
        with _pytest.raises(ValueError, match="retrieval_cusum"):
            self._comp(retrieval_cusum=0.0)

    def test_cusum_fires_on_novel_persistent_span_not_chronic(self):
        """A span sitting at a HIGH but constant alpha never fires (its EMA
        baseline absorbs it); a span that jumps from ~0 fires within a few
        steps — the exact false-positive mode measured under mass eviction."""
        comp = self._comp(retrieval_alpha=0.9)   # disable the fast path
        state = {}
        def step(sid, a):
            b, S = state.get(sid, (a, 0.0))
            S = max(0.0, S + a - b - comp.cusum_drift)
            b = comp.ema_decay * b + (1.0 - comp.ema_decay) * a
            state[sid] = (b, S)
            return S
        # chronically hot filler: alpha 0.4 every step -> never crosses
        for _ in range(50):
            S_hot = step(1, 0.4)
        assert S_hot < comp.retrieval_cusum
        # novel needle: idle at 0.01 for 20 steps, then 0.25 sustained
        for _ in range(20):
            step(2, 0.01)
        fired_at = None
        for t in range(10):
            if step(2, 0.25) >= comp.retrieval_cusum:
                fired_at = t
                break
        assert fired_at is not None and fired_at <= 5


class TestBudgetAnneal:
    """Budget anneal: linear schedule (RL loop)."""

    def _comp(self, **kw):
        from megatron.rl.compaction.kv.serving.live import LiveKVCompactor
        return LiveKVCompactor(_StubEngine("none"), strategy="streaming_llm",
                               budget_ratio=0.8, **kw)

    def test_linear_schedule(self):
        comp = self._comp(budget_final=0.2, budget_anneal_iters=100)
        assert comp.schedule_ratio(0) == pytest.approx(0.8)
        assert comp.schedule_ratio(50) == pytest.approx(0.5)
        assert comp.schedule_ratio(100) == pytest.approx(0.2)
        assert comp.schedule_ratio(500) == pytest.approx(0.2)   # clamped

    def test_flags_must_be_set_together(self):
        with pytest.raises(ValueError, match="set together"):
            self._comp(budget_final=0.2)
        with pytest.raises(ValueError, match="set together"):
            self._comp(budget_anneal_iters=10)

    def test_final_and_iters_validated(self):
        with pytest.raises(ValueError, match="budget_final"):
            self._comp(budget_final=1.5, budget_anneal_iters=10)
        with pytest.raises(ValueError, match="anneal_iters"):
            self._comp(budget_final=0.2, budget_anneal_iters=0)


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
