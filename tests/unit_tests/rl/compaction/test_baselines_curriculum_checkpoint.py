# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for H2O, StreamingLLM, and checkpoint I/O."""

import os
import tempfile
import pytest
import numpy as np
import torch

from megatron.rl.compaction.kv import (
    H2OAccumulator,
    StreamingLLMCompressor,
    CompactionResult,
)
from megatron.rl.compaction.kv.benchmark import KVCompactionBenchmark
from megatron.rl.compaction.learned.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_optimizer_state,
    CheckpointMeta,
)
from megatron.rl.compaction.learned.models.compactor import PerceiverConfig, PerceiverCompactor
from megatron.rl.compaction.learned.models.belief import GatedUpdaterConfig, GatedRecurrentUpdater


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _kv_np(T=32, d=16):
    """Return random K, V, Q as torch float32 tensors (name kept for backward compat)."""
    K = torch.randn(T, d)
    V = torch.randn(T, d)
    Q = torch.randn(4, d)
    return K, V, Q


def _kv_torch(n_layers=2, B=1, T=16, d=16):
    keys   = [torch.randn(B, T, d) for _ in range(n_layers)]
    values = [torch.randn(B, T, d) for _ in range(n_layers)]
    return keys, values


# ===========================================================================
# H2OAccumulator  (paper-faithful true H2O — the eval baseline)
# ===========================================================================

class TestH2OAccumulator:
    def test_recent_window_always_kept(self):
        """The recent window (last n_recent positions) is always retained."""
        h2o = H2OAccumulator(recent_ratio=0.5)
        K, V, Q = _kv_np(T=40)
        r = h2o.compress(K, V, 10, ref_queries=Q, run_id="r", step_id=0)
        assert isinstance(r, CompactionResult)
        assert len(r.retained_positions) == 10
        # recent_ratio=0.5, budget=10 -> 5 recent = last 5 positions [35..39]
        assert all(pos in r.retained_positions for pos in range(35, 40))
        assert r.retained_positions == sorted(r.retained_positions)

    def test_heavy_hitters_selected(self):
        """Heavy hitters (high accumulated attention) outside recent are kept."""
        # recent_ratio=0 -> pure heavy-hitter selection by accumulated score.
        h2o = H2OAccumulator(recent_ratio=0.0)
        K, V, _ = _kv_np(T=20)
        for _ in range(3):
            w = torch.zeros(20)
            w[5] = 1.0
            w[11] = 0.8
            h2o.update(w)
        r = h2o.compress(K, V, 6, run_id="r", step_id=0)
        assert 5 in r.retained_positions
        assert 11 in r.retained_positions

    def test_recent_plus_heavy_split(self):
        """Budget splits into recent window + heavy hitters from the rest."""
        h2o = H2OAccumulator(recent_ratio=0.5)
        K, V, _ = _kv_np(T=10)
        scores = torch.arange(10, dtype=torch.float32)  # position 9 highest, but it's recent
        r = h2o.compress(K, V, 4, accumulated_scores=scores, run_id="r", step_id=0)
        # budget=4, 2 recent = [8,9]; 2 heavy from [0..7] by score = [6,7]
        assert r.retained_positions == [6, 7, 8, 9]

    def test_requires_scores(self):
        """Without update(), accumulated_scores, or ref_queries it must error."""
        h2o = H2OAccumulator()
        K, V, _ = _kv_np()
        with pytest.raises(RuntimeError):
            h2o.compress(K, V, 8, run_id="r", step_id=0)

    def test_no_value_fitting(self):
        """Paper-exact: retained values are the originals, bias is zero."""
        h2o = H2OAccumulator(recent_ratio=0.0)
        K, V, _ = _kv_np(T=10)
        r = h2o.compress(K, V, 4, accumulated_scores=torch.arange(10.0), run_id="r", step_id=0)
        assert torch.allclose(r.compacted_values, V[r.retained_positions])
        assert (r.bias == 0).all()

    def test_reset_clears_state(self):
        h2o = H2OAccumulator()
        h2o.update(torch.ones(8))
        h2o.reset()
        K, V, _ = _kv_np(T=8)
        with pytest.raises(RuntimeError):
            h2o.compress(K, V, 4, run_id="r", step_id=0)

    def test_strategy_string(self):
        assert "h2o_recent" in H2OAccumulator().strategy

    def test_in_benchmark(self):
        bench = KVCompactionBenchmark()
        K, V, Q = _kv_np(T=40)
        results = bench.run(
            compressors={"h2o": H2OAccumulator()},
            keys=K, values=V, ref_queries=Q, eval_queries=Q, budget=10,
        )
        assert len(results) == 1
        assert results[0].algorithm == "h2o"


# ===========================================================================
# StreamingLLMCompressor
# ===========================================================================

class TestStreamingLLMCompressor:
    def test_result_type(self):
        slm = StreamingLLMCompressor()
        K, V, Q = _kv_np()
        r = slm.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert isinstance(r, CompactionResult)

    def test_budget_respected(self):
        slm = StreamingLLMCompressor(n_sink=4)
        K, V, Q = _kv_np(T=50)
        r = slm.compress(K, V, 12, ref_queries=Q, run_id="r", step_id=0)
        assert len(r.retained_positions) == 12

    def test_sink_positions_always_first(self):
        slm = StreamingLLMCompressor(n_sink=4)
        K, V, Q = _kv_np(T=30)
        r = slm.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert r.retained_positions[:4] == list(range(4))

    def test_recent_window_is_contiguous_at_end(self):
        slm = StreamingLLMCompressor(n_sink=4, fit_bias=False, fit_values=False)
        K, V, Q = _kv_np(T=30)
        r = slm.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        # Last 4 positions should be the most recent 4 tokens
        assert r.retained_positions[-4:] == list(range(26, 30))

    def test_positions_sorted(self):
        slm = StreamingLLMCompressor(n_sink=2)
        K, V, Q = _kv_np(T=20)
        r = slm.compress(K, V, 6, ref_queries=Q, run_id="r", step_id=0)
        assert r.retained_positions == sorted(r.retained_positions)

    def test_budget_larger_than_T(self):
        slm = StreamingLLMCompressor(n_sink=2)
        K, V, Q = _kv_np(T=5)
        r = slm.compress(K, V, 20, ref_queries=Q, run_id="r", step_id=0)
        assert len(r.retained_positions) == 5

    def test_strategy_string(self):
        slm = StreamingLLMCompressor(n_sink=8)
        assert "streaming" in slm.strategy
        assert "sink8" in slm.strategy

    def test_no_query_dependency(self):
        """StreamingLLM result is the same regardless of ref_queries."""
        slm = StreamingLLMCompressor(n_sink=4, fit_bias=False, fit_values=False)
        K, V, _ = _kv_np(T=30)
        Q1 = torch.randn(4, 16)
        Q2 = torch.randn(4, 16)
        r1 = slm.compress(K, V, 8, ref_queries=Q1, run_id="r", step_id=0)
        r2 = slm.compress(K, V, 8, ref_queries=Q2, run_id="r", step_id=0)
        assert r1.retained_positions == r2.retained_positions

    def test_in_benchmark(self):
        bench = KVCompactionBenchmark()
        K, V, Q = _kv_np(T=40)
        results = bench.run(
            compressors={"streaming": StreamingLLMCompressor(n_sink=4)},
            keys=K, values=V, ref_queries=Q, eval_queries=Q, budget=10,
        )
        assert len(results) == 1
        assert results[0].algorithm == "streaming"

    def test_all_methods_in_benchmark(self):
        """All five compressors can be compared in the same benchmark run."""
        from megatron.rl.compaction.kv import TopKCompressor, OMPCompressor
        bench = KVCompactionBenchmark()
        K, V, Q = _kv_np(T=50)
        results = bench.run(
            compressors={
                "topk":      TopKCompressor(),
                "omp":       OMPCompressor(),
                "h2o":       H2OAccumulator(),
                "streaming": StreamingLLMCompressor(n_sink=4),
            },
            keys=K, values=V, ref_queries=Q, eval_queries=Q, budget=12,
        )
        algorithms = {r.algorithm for r in results}
        assert algorithms == {"topk", "omp", "h2o", "streaming"}


@pytest.mark.skip(
    reason="checkpoint.py uses collective Megatron dist_checkpointing whose async "
    "writer spawns a multiprocessing Manager that re-imports the entry module — it "
    "needs a real multi-rank torchrun launch with an __main__ guard, which plain "
    "single-process pytest does not provide. The save/load/resume round-trip is "
    "validated end-to-end by scripts/_optc_online_smoke.py under torchrun.",
)
class TestCheckpointIO:
    def _perceiver(self):
        cfg = PerceiverConfig(d_kv=16, n_heads=2, n_compress=4, n_attn_layers=2)
        return PerceiverCompactor(cfg)

    def _gated(self):
        cfg = GatedUpdaterConfig(d_kv=16, n_heads=2, n_compress=4, n_attn_layers=2)
        return GatedRecurrentUpdater(cfg)

    def test_save_perceiver(self):
        model = self._perceiver()
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            save_checkpoint(model, f.name)
            assert os.path.exists(f.name)

    def test_load_perceiver(self):
        model = self._perceiver()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path, step=42, metadata={"val_ppl": 9.1})
            loaded, meta = load_checkpoint(path)
            assert isinstance(loaded, PerceiverCompactor)
            assert meta.step == 42
            assert meta.metadata["val_ppl"] == 9.1
            assert meta.model_type == "perceiver"
        finally:
            os.unlink(path)

    def test_weights_preserved_perceiver(self):
        """Loaded model has identical weights to saved model."""
        model = self._perceiver()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path)
            loaded, _ = load_checkpoint(path)
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
                assert torch.allclose(p1, p2), f"Parameter {n1} differs after load"
        finally:
            os.unlink(path)

    def test_save_load_gated(self):
        model = self._gated()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path, step=100)
            loaded, meta = load_checkpoint(path)
            assert isinstance(loaded, GatedRecurrentUpdater)
            assert meta.step == 100
        finally:
            os.unlink(path)

    def test_weights_preserved_gated(self):
        model = self._gated()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path)
            loaded, _ = load_checkpoint(path)
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded.named_parameters()):
                assert torch.allclose(p1, p2), f"Parameter {n1} differs after load"
        finally:
            os.unlink(path)

    def test_config_preserved(self):
        cfg = PerceiverConfig(d_kv=32, n_heads=4, n_compress=8, n_attn_layers=3)
        model = PerceiverCompactor(cfg)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path)
            loaded, meta = load_checkpoint(path)
            assert loaded.cfg.d_kv == 32
            assert loaded.cfg.n_compress == 8
            assert loaded.cfg.n_attn_layers == 3
        finally:
            os.unlink(path)

    def test_optimizer_save_load(self):
        model = self._perceiver()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Do one optimizer step so state is non-trivial
        fake_loss = sum(p.sum() for p in model.parameters())
        fake_loss.backward()
        opt.step()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path, optimizer=opt)
            opt2 = torch.optim.Adam(self._perceiver().parameters(), lr=1e-3)
            load_optimizer_state(path, opt2)
            # State groups should match
            assert len(opt2.state) == len(opt.state)
        finally:
            os.unlink(path)


    def test_unsupported_model_raises(self):
        import torch.nn as nn
        model = nn.Linear(4, 4)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            with pytest.raises(TypeError, match="PerceiverCompactor"):
                save_checkpoint(model, f.name)

    def test_parent_dirs_created(self):
        model = self._perceiver()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "subdir", "nested", "ckpt.pt")
            save_checkpoint(model, path)
            assert os.path.exists(path)

    def test_checkpoint_meta_dataclass(self):
        meta = CheckpointMeta(
            model_type="perceiver",
            config={"d_model": 16},
            step=5,
            metadata={"foo": "bar"},
        )
        assert meta.model_type == "perceiver"
        assert meta.step == 5
