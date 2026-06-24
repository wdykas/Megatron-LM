# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for TrajectoryBuilder, TinyLMAdapter, and CompactionEvaluator."""

import pytest
import torch

from megatron.rl.compaction.learned.training.data import TrajectoryBuilder, PipelineConfig
from .tiny_lm import TinyLM, TinyLMAdapter
from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe
from megatron.rl.compaction.learned.serving.eval import CompactionEvaluator, EvalConfig
from megatron.rl.compaction.learned.models.belief import BeliefUpdater
from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB = 64
N_LAYERS = 2
D_MODEL = 32
N_HEADS = 4
CHUNK_SIZE = 16


def _tiny_adapter():
    model = TinyLM(vocab_size=VOCAB, n_layers=N_LAYERS, d_model=D_MODEL, n_heads=N_HEADS)
    return TinyLMAdapter(model)


def _updater():
    cfg = PerceiverConfig(d_kv=D_MODEL, n_heads=N_HEADS, n_compress=4, n_attn_layers=N_LAYERS)
    return BeliefUpdater(PerceiverCompactor(cfg))


def _tokens(B=1, L=64):
    return torch.randint(0, VOCAB, (B, L))


# ---------------------------------------------------------------------------
# TinyLM
# ---------------------------------------------------------------------------

class TestTinyLM:
    def test_forward_shape(self):
        model = TinyLM(VOCAB, N_LAYERS, D_MODEL, N_HEADS)
        tokens = _tokens(B=2, L=10)
        logits, kv = model(tokens)
        assert logits.shape == (2, 10, VOCAB)
        assert len(kv) == N_LAYERS
        assert kv[0][0].shape == (2, 10, D_MODEL)

    def test_kv_cache_prefix(self):
        model = TinyLM(VOCAB, N_LAYERS, D_MODEL, N_HEADS)
        tokens = _tokens(B=1, L=8)
        _, past_kv = model(tokens)
        # Run with prefix
        next_tok = _tokens(B=1, L=4)
        logits2, new_kv = model(next_tok, past_kv=past_kv)
        assert logits2.shape == (1, 4, VOCAB)
        # KV should now include both original and new tokens
        assert new_kv[0][0].shape[1] == 8 + 4

    def test_logits_change_with_prefix(self):
        """Conditioning on past KV should change logits vs no prefix."""
        torch.manual_seed(7)
        model = TinyLM(VOCAB, N_LAYERS, D_MODEL, N_HEADS)
        tokens = _tokens(B=1, L=8)
        _, past_kv = model(tokens)
        q = _tokens(B=1, L=4)
        logits_with, _ = model(q, past_kv=past_kv)
        logits_without, _ = model(q)
        assert not torch.allclose(logits_with, logits_without)


# ---------------------------------------------------------------------------
# TinyLMAdapter
# ---------------------------------------------------------------------------

class TestTinyLMAdapter:
    def test_prefill_shape(self):
        adapter = _tiny_adapter()
        tokens = _tokens(B=1, L=16)
        kv, logits = adapter.prefill(tokens, kv_prefix=None, logical_start=0)
        assert len(kv) == N_LAYERS
        k, v = kv[0]
        assert k.shape == (1, 16, D_MODEL)
        assert logits.shape == (1, 16, VOCAB)

    def test_prefill_with_prefix_returns_chunk_kv_only(self):
        adapter = _tiny_adapter()
        tokens = _tokens(B=1, L=16)
        kv_pre, _ = adapter.prefill(tokens, kv_prefix=None, logical_start=0)
        # Chunk of 8 with the 16-token prefix
        tokens2 = _tokens(B=1, L=8)
        kv2, logits2 = adapter.prefill(tokens2, kv_prefix=kv_pre, logical_start=16)
        # Should return only the new 8-token KV
        k2, v2 = kv2[0]
        assert k2.shape == (1, 8, D_MODEL)
        assert logits2.shape == (1, 8, VOCAB)

    def test_student_logits_shape(self):
        adapter = _tiny_adapter()
        tokens = _tokens(B=1, L=8)
        kv, _ = adapter.prefill(tokens, kv_prefix=None, logical_start=0)
        q = _tokens(B=1, L=4)
        logits = adapter.student_logits(q, kv, logical_kv_start=0, logical_query_start=8)
        assert logits.shape == (1, 4, VOCAB)

    def test_make_student_fn(self):
        adapter = _tiny_adapter()
        tokens = _tokens(B=1, L=8)
        kv, _ = adapter.prefill(tokens, kv_prefix=None, logical_start=0)
        fn = adapter.make_student_fn()
        q = _tokens(B=1, L=4)
        logits = fn(q, kv)
        assert logits.shape == (1, 4, VOCAB)

    def test_satisfies_protocol(self):
        from megatron.rl.compaction.learned.training.data import FrozenModelAdapter
        adapter = _tiny_adapter()
        assert isinstance(adapter, FrozenModelAdapter)


# ---------------------------------------------------------------------------
# TrajectoryBuilder
# ---------------------------------------------------------------------------

class TestTrajectoryBuilder:
    def test_returns_trajectory(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(
            chunk_size=CHUNK_SIZE,
            probe_stride=1,
            probe_query_len=8,
        ))
        tokens = _tokens(B=1, L=64)
        traj = builder.build(tokens, adapter)
        assert isinstance(traj, Trajectory)

    def test_n_chunks(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(chunk_size=16, probe_stride=1, probe_query_len=4))
        tokens = _tokens(B=1, L=64)
        traj = builder.build(tokens, adapter)
        assert traj.n_chunks == 4   # 64 / 16

    def test_chunks_have_kv(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(chunk_size=16, probe_stride=1, probe_query_len=4))
        tokens = _tokens(B=1, L=64)
        traj = builder.build(tokens, adapter)
        keys, vals = traj.chunks[0]
        assert len(keys) == N_LAYERS
        assert keys[0].shape[1] == 16   # T_chunk = 16

    def test_probes_at_each_chunk(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(
            chunk_size=16, probe_stride=1, probe_query_len=8
        ))
        tokens = _tokens(B=1, L=80)
        traj = builder.build(tokens, adapter)
        # probe_stride=1 → every chunk has a probe (except possibly last)
        n_probes = sum(len(p) for p in traj.probes_by_chunk.values())
        assert n_probes >= 1

    def test_probe_stride_2(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(
            chunk_size=16, probe_stride=2, probe_query_len=8
        ))
        tokens = _tokens(B=1, L=80)
        traj = builder.build(tokens, adapter)
        # Only even-indexed chunks get probes
        for chunk_idx in traj.probes_by_chunk:
            assert chunk_idx % 2 == 0

    def test_teacher_logits_shape(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(
            chunk_size=16, probe_stride=1, probe_query_len=8
        ))
        tokens = _tokens(B=1, L=64)
        traj = builder.build(tokens, adapter)
        probe_chunk = next(iter(traj.probes_by_chunk.values()))
        probe = probe_chunk[0]
        assert probe.teacher_logits.shape[0] == 1
        assert probe.teacher_logits.shape[-1] == VOCAB

    def test_max_probes_respected(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(
            chunk_size=8, probe_stride=1, probe_query_len=4, max_probes=2
        ))
        tokens = _tokens(B=1, L=64)
        traj = builder.build(tokens, adapter)
        total = sum(len(p) for p in traj.probes_by_chunk.values())
        assert total <= 2

    def test_batch_size_2(self):
        adapter = _tiny_adapter()
        builder = TrajectoryBuilder(PipelineConfig(chunk_size=16, probe_stride=2, probe_query_len=8))
        tokens = _tokens(B=2, L=64)
        traj = builder.build(tokens, adapter)
        assert traj.n_chunks == 4
        keys, _ = traj.chunks[0]
        assert keys[0].shape[0] == 2   # batch dim


# ---------------------------------------------------------------------------
# CompactionEvaluator
# ---------------------------------------------------------------------------

class TestCompactionEvaluator:
    def test_perplexity_returns_dict(self):
        adapter = _tiny_adapter()
        evaluator = CompactionEvaluator(adapter, EvalConfig(chunk_size=16))
        tokens = _tokens(B=1, L=64)
        result = evaluator.perplexity(tokens, _updater())
        assert "ppl_full" in result
        assert "ppl_compact" in result
        assert "ppl_ratio" in result

    def test_perplexity_values_positive(self):
        adapter = _tiny_adapter()
        evaluator = CompactionEvaluator(adapter, EvalConfig(chunk_size=16))
        tokens = _tokens(B=1, L=64)
        result = evaluator.perplexity(tokens, _updater())
        assert result["ppl_full"] > 0.0
        assert result["ppl_compact"] > 0.0
        assert result["ppl_ratio"] > 0.0

    def test_perplexity_1d_tokens(self):
        """Accept 1-D token tensor."""
        adapter = _tiny_adapter()
        evaluator = CompactionEvaluator(adapter, EvalConfig(chunk_size=16))
        tokens = _tokens(B=1, L=64).squeeze(0)    # (64,)
        result = evaluator.perplexity(tokens, _updater())
        assert result["ppl_ratio"] > 0.0

    def test_needle_in_haystack(self):
        adapter = _tiny_adapter()
        evaluator = CompactionEvaluator(adapter, EvalConfig(chunk_size=16, max_new_tokens=4))
        haystack = _tokens(B=1, L=64)
        query   = _tokens(B=1, L=4)
        answer  = _tokens(B=1, L=4)
        result  = evaluator.needle_in_haystack(haystack, query, answer, _updater())
        assert "exact_match" in result
        assert "token_overlap" in result
        assert 0.0 <= result["exact_match"] <= 1.0
        assert 0.0 <= result["token_overlap"] <= 1.0

    def test_perplexity_eval(self):
        adapter = _tiny_adapter()
        tokens  = _tokens(B=1, L=64)
        result  = CompactionEvaluator(adapter, EvalConfig(chunk_size=16)).perplexity(tokens, _updater())
        assert "ppl_full" in result
