# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the learned heavy-hitter oracle ."""

import pytest
import torch

from megatron.rl.compaction.kv import build_kv_compressor
from megatron.rl.compaction.kv.selectors.oracle import (
    LearnedOracleScorer,
    OracleCompressor,
    OracleScorerConfig,
    fit_oracle_scorer,
    load_oracle_scorer,
    save_oracle_scorer,
    token_level_oracle,
)

L, P, H, D, HQ, Q = 2, 48, 1, 16, 4, 24


def _capture(seed, heavy_dir=None):
    """Synthetic (keys, queries) with a TOKEN-level planted structure.

    One heavy mask shared across layers (the oracle aggregates layers, and the
    paged cache evicts token-level): heavy tokens carry heavy_dir in every
    layer, queries point at it — their mass becomes predictable from content.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    mask = torch.rand(P, device="cuda", generator=g) < 0.33 if heavy_dir is not None else None
    keys, queries = [], []
    for _ in range(L):
        k = torch.randn(P, H * D, device="cuda", generator=g)
        if heavy_dir is not None:
            k[mask] += heavy_dir * (3.0 + torch.rand(int(mask.sum()), 1,
                                                     device="cuda", generator=g))
        q = torch.randn(Q, HQ, D, device="cuda", generator=g) * 0.3
        if heavy_dir is not None:
            q += heavy_dir.view(H, D).mean(0) * 2.0
        keys.append(k)
        queries.append(q)
    return (keys, queries) if heavy_dir is None else (keys, queries, mask)


class TestTokenLevelOracle:
    def test_shape_and_mass_conservation(self):
        keys, queries = _capture(0)
        oracle = token_level_oracle(keys, queries)
        assert oracle.shape == (P,)
        # Softmax rows sum to 1: total mass = n_layers * n_groups * Q * group.
        assert torch.isclose(oracle.sum(),
                             torch.tensor(float(L * H * Q * HQ), device="cuda"))

    def test_heavy_direction_gets_the_mass(self):
        u = torch.randn(1, H * D, device="cuda")
        u = u / u.norm()
        keys, queries, mask = _capture(1, heavy_dir=u)
        oracle = token_level_oracle(keys, queries)
        assert oracle[mask].mean() > 5 * oracle[~mask].mean()


class TestLearnedOracleScorer:
    def _fit(self, n_train=6, epochs=150):
        torch.manual_seed(0)
        u = torch.randn(1, H * D, device="cuda")
        u = u / u.norm()
        scorer = LearnedOracleScorer(
            OracleScorerConfig(d_key=H * D, n_layers=L, hidden=64)).cuda()
        train = [_capture(s, heavy_dir=u)[:2] for s in range(n_train)]
        losses = fit_oracle_scorer(scorer, train, epochs=epochs, lr=1e-3)
        return scorer, u, losses

    def test_fit_learns_the_planted_structure(self):
        scorer, u, losses = self._fit()
        assert losses[-1] < losses[0] * 0.5
        # Held-out capture: predicted ranking must recover the oracle's top set.
        keys, queries, _ = _capture(99, heavy_dir=u)
        oracle = token_level_oracle(keys, queries)
        pred = scorer.score_tokens(
            torch.stack([k.view(P, H, D) for k in keys]))          # (L, P, H, D)
        budget = int(P * 0.4)
        top_pred = set(torch.topk(pred, budget).indices.tolist())
        top_true = set(torch.topk(oracle, budget).indices.tolist())
        recall = len(top_pred & top_true) / budget
        assert recall > 0.75, f"recall@40% {recall}"

    def test_fit_requires_captures(self):
        scorer = LearnedOracleScorer(OracleScorerConfig(d_key=H * D, n_layers=L)).cuda()
        with pytest.raises(ValueError, match="no captures"):
            fit_oracle_scorer(scorer, [])

    def test_feature_validation(self):
        scorer = LearnedOracleScorer(OracleScorerConfig(d_key=H * D, n_layers=L)).cuda()
        with pytest.raises(ValueError, match="key width"):
            scorer.features(torch.zeros(4, H * D + 1, device="cuda"), 0)
        with pytest.raises(ValueError, match="layer_idx"):
            scorer.features(torch.zeros(4, H * D, device="cuda"), L)

    def test_save_load_round_trip(self, tmp_path):
        scorer, _, _ = self._fit(n_train=2, epochs=5)
        path = str(tmp_path / "oracle.pt")
        save_oracle_scorer(scorer, path)
        loaded = load_oracle_scorer(path)
        assert loaded.cfg == scorer.cfg
        k = torch.randn(3, P, H, D, device="cuda")[:L]
        torch.testing.assert_close(loaded.score_tokens(k), scorer.score_tokens(k))


class TestFlywheelRefit:
    def _events(self, tmp_path, u):
        """Two event files: spans aligned with u were restored (label 1)."""
        from megatron.rl.compaction.kv.archive import KVArchive
        g = torch.Generator(device="cuda").manual_seed(0)
        arch = KVArchive(max_span=4, flywheel_dir=str(tmp_path))
        for rid in range(4):
            k = torch.randn(L, 24, H, D, device="cuda", generator=g)
            k[:, 8:12] += u.view(1, 1, H, D) * 4.0        # the "needed" span
            arch.store_evicted(rid, k, k.clone(), list(range(0, 8)))
            # restore the span that contains positions 8..12
            target = next(i for i, sp in enumerate(arch._spans[rid])
                          if sp.positions[0] == 8)
            arch.take(rid, target)
            arch.drop(rid)

    def test_refit_separates_restored_from_unused(self, tmp_path):
        from megatron.rl.compaction.kv.selectors.oracle import fit_scorer_on_flywheel
        torch.manual_seed(0)
        u = torch.randn(H * D, device="cuda")
        u = u / u.norm()
        self._events(tmp_path, u)
        scorer = LearnedOracleScorer(
            OracleScorerConfig(d_key=H * D, n_layers=L, hidden=32)).cuda()
        losses = fit_scorer_on_flywheel(scorer, str(tmp_path), epochs=150, lr=3e-3)
        assert losses[-1] < losses[0] * 0.6
        # Restored-like keys must now outscore unused-like keys.
        g = torch.Generator(device="cuda").manual_seed(9)
        base = torch.randn(16, H * D, device="cuda", generator=g)
        needed = base + u * 4.0
        s_need = scorer(scorer.features(needed, 0)).mean()
        s_rest = scorer(scorer.features(base, 0)).mean()
        assert (s_need - s_rest).item() > 0.5

    def test_refit_requires_both_classes(self, tmp_path):
        from megatron.rl.compaction.kv.archive import KVArchive
        from megatron.rl.compaction.kv.selectors.oracle import fit_scorer_on_flywheel
        arch = KVArchive(max_span=4, flywheel_dir=str(tmp_path))
        k = torch.randn(L, 8, H, D, device="cuda")
        arch.store_evicted(1, k, k.clone(), [0, 1])
        arch.drop(1)                                      # all label 0
        scorer = LearnedOracleScorer(
            OracleScorerConfig(d_key=H * D, n_layers=L, hidden=32)).cuda()
        with pytest.raises(ValueError, match="one class"):
            fit_scorer_on_flywheel(scorer, str(tmp_path), epochs=1)


class TestOracleCompressor:
    def test_protocol(self):
        scorer = LearnedOracleScorer(OracleScorerConfig(d_key=H * D, n_layers=L)).cuda()
        comp = OracleCompressor(scorer)
        k = torch.randn(P, H * D, device="cuda")
        v = torch.randn(P, H * D, device="cuda")
        res = comp.compress(k, v, budget=10)
        assert res.strategy == "learned_oracle"
        assert len(res.retained_positions) == 10
        assert res.retained_positions == sorted(res.retained_positions)
        assert res.compacted_keys.shape == (10, H * D)
        with pytest.raises(ValueError):
            comp.compress(k, v, budget=0)

    def test_factory_demands_a_scorer(self):
        with pytest.raises(ValueError, match="trained scorer"):
            build_kv_compressor("learned_oracle")
