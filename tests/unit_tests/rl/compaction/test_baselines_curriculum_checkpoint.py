# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for H2O, StreamingLLM, CurriculumScheduler, and checkpoint I/O."""

import os
import tempfile
import pytest
import numpy as np
import torch

from megatron.rl.compaction.kv import (
    H2OProxyCompressor,
    StreamingLLMCompressor,
    CompactionResult,
)
from megatron.rl.compaction.kv.benchmark import KVCompactionBenchmark
from megatron.rl.compaction.learned.training.curriculum import (
    CurriculumStage,
    CurriculumScheduler,
)
from megatron.rl.compaction.learned.training.losses import CompactorLossWeights
from megatron.rl.compaction.learned.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_optimizer_state,
    load_scheduler_state,
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
# H2OProxyCompressor
# ===========================================================================

class TestH2OProxyCompressor:
    def test_result_type(self):
        h2o = H2OProxyCompressor()
        K, V, Q = _kv_np()
        r = h2o.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert isinstance(r, CompactionResult)

    def test_budget_respected(self):
        h2o = H2OProxyCompressor(n_sink=2)
        K, V, Q = _kv_np(T=40)
        r = h2o.compress(K, V, 10, ref_queries=Q, run_id="r", step_id=0)
        assert len(r.retained_positions) == 10

    def test_sink_positions_always_included(self):
        h2o = H2OProxyCompressor(n_sink=4)
        K, V, Q = _kv_np(T=30)
        r = h2o.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert all(pos in r.retained_positions for pos in range(4))

    def test_positions_sorted(self):
        h2o = H2OProxyCompressor(n_sink=2)
        K, V, Q = _kv_np(T=20)
        r = h2o.compress(K, V, 6, ref_queries=Q, run_id="r", step_id=0)
        assert r.retained_positions == sorted(r.retained_positions)

    def test_budget_larger_than_T(self):
        h2o = H2OProxyCompressor(n_sink=2)
        K, V, Q = _kv_np(T=5)
        r = h2o.compress(K, V, 20, ref_queries=Q, run_id="r", step_id=0)
        assert len(r.retained_positions) == 5

    def test_no_bias_option(self):
        h2o = H2OProxyCompressor(n_sink=2, fit_bias=False, fit_values=False)
        K, V, Q = _kv_np()
        r = h2o.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert (r.bias == 0).all()

    def test_strategy_string(self):
        h2o = H2OProxyCompressor(n_sink=4)
        assert "h2o" in h2o.strategy
        assert "sink4" in h2o.strategy

    def test_n_sink_zero(self):
        h2o = H2OProxyCompressor(n_sink=0, fit_bias=False, fit_values=False)
        K, V, Q = _kv_np()
        r = h2o.compress(K, V, 8, ref_queries=Q, run_id="r", step_id=0)
        assert len(r.retained_positions) == 8

    def test_in_benchmark(self):
        """H2O plugs into KVCompactionBenchmark without error."""
        bench = KVCompactionBenchmark()
        K, V, Q = _kv_np(T=40)
        results = bench.run(
            compressors={"h2o": H2OProxyCompressor(n_sink=2)},
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
                "h2o":       H2OProxyCompressor(n_sink=4),
                "streaming": StreamingLLMCompressor(n_sink=4),
            },
            keys=K, values=V, ref_queries=Q, eval_queries=Q, budget=12,
        )
        algorithms = {r.algorithm for r in results}
        assert algorithms == {"topk", "omp", "h2o", "streaming"}


# ===========================================================================
# CurriculumScheduler
# ===========================================================================

class TestCurriculumScheduler:
    def _make(self, steps_per_stage=10):
        return CurriculumScheduler.default_4stage(steps_per_stage)

    def test_initial_stage_zero(self):
        sched = self._make()
        assert sched.stage_idx == 0

    def test_first_step_returns_stage0_weights(self):
        sched = self._make(steps_per_stage=5)
        w = sched.step()
        assert w.teacher_kl == 1.0
        assert w.future_kl == 0.0   # stage 0: still_warmup

    def test_advances_stage(self):
        sched = self._make(steps_per_stage=3)
        for _ in range(3):
            sched.step()
        assert sched.stage_idx == 1   # now in stage 1 (belief_still)

    def test_stage_1_has_future_kl(self):
        sched = self._make(steps_per_stage=2)
        for _ in range(2):
            sched.step()   # exhaust stage 0
        w = sched.step()   # first step of stage 1
        assert w.future_kl > 0.0

    def test_all_4_stages_reached(self):
        sched = self._make(steps_per_stage=1)
        stage_names = set()
        for _ in range(10):
            stage_names.add(sched.stage_name)
            sched.step()
        assert "still_warmup" in stage_names
        assert "belief_still" in stage_names
        assert "retrieval_focus" in stage_names
        assert "value_directed" in stage_names

    def test_last_stage_does_not_end(self):
        """After all stages are exhausted, stays in the last one."""
        sched = self._make(steps_per_stage=2)
        for _ in range(100):
            sched.step()
        assert sched.stage_idx == 3   # stays in last stage

    def test_stage_4_has_path_consistency(self):
        sched = self._make(steps_per_stage=1)
        for _ in range(3):
            sched.step()   # skip stages 0-2
        w = sched.step()
        assert w.path_consistency > 0.0

    def test_total_steps_counter(self):
        sched = self._make()
        for _ in range(7):
            sched.step()
        assert sched.total_steps == 7

    def test_steps_in_stage(self):
        sched = self._make(steps_per_stage=5)
        for _ in range(7):
            sched.step()
        assert sched.steps_in_stage == 2   # 7 steps: 5 in stage 0, 2 in stage 1

    def test_stage_progress(self):
        sched = self._make(steps_per_stage=10)
        for _ in range(5):
            sched.step()
        assert sched.stage_progress == pytest.approx(0.5)

    def test_custom_stages(self):
        stages = [
            CurriculumStage("a", 3, CompactorLossWeights(teacher_kl=2.0)),
            CurriculumStage("b", 3, CompactorLossWeights(teacher_kl=0.5)),
        ]
        sched = CurriculumScheduler(stages)
        w = sched.step()
        assert w.teacher_kl == 2.0
        for _ in range(3):
            sched.step()
        w = sched.step()
        assert w.teacher_kl == 0.5

    def test_empty_stages_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            CurriculumScheduler([])

    def test_state_dict_roundtrip(self):
        sched = self._make(steps_per_stage=5)
        for _ in range(7):
            sched.step()
        state = sched.state_dict()
        sched2 = self._make(steps_per_stage=5)
        sched2.load_state_dict(state)
        assert sched2.total_steps == 7
        assert sched2.stage_idx == sched.stage_idx


# ===========================================================================
# Checkpoint I/O
# ===========================================================================

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

    def test_scheduler_save_load(self):
        sched = CurriculumScheduler.default_4stage(steps_per_stage=5)
        for _ in range(7):
            sched.step()
        model = self._perceiver()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_checkpoint(model, path, scheduler=sched)
            sched2 = CurriculumScheduler.default_4stage(steps_per_stage=5)
            load_scheduler_state(path, sched2)
            assert sched2.total_steps == 7
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
