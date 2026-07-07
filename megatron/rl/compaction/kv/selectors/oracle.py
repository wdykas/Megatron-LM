# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Learned heavy-hitter oracle: predict H2O's score without queries.

H2O's true heavy-hitter score — the softmax attention mass a key will
accumulate over future queries — is unobservable live under flash attention.
SnapKV proxies it with the last-W real queries; this module goes one step
further and asks the query-free question: is the oracle largely predictable from the
KEY VECTOR alone (plus position)? On trained-Nano captures the answer was yes
(held-out Spearman 0.97 vs 0.82 for SnapKV's proxy at matched budget), which
makes a query-free live strategy possible: score each prefix key by content,
no Q capture, no eager-prefill requirement, fully CUDA-graph compatible.

Pieces:
- ``token_level_oracle``     — the ground-truth target (accumulated softmax
                               attention, aggregated over layers/KV groups).
- ``LearnedOracleScorer``    — MLP on [key, position/P, layer one-hot] →
                               log1p oracle mass. Megatron/TE modules,
                               replicated (singleton TP), like the compactors.
- ``fit_oracle_scorer``      — offline training loop over captured (K, Q).
- ``OracleCompressor``       — the offline ``KVCompressor`` protocol wrapper.
- ``save/load_oracle_scorer``— plain ``torch.save`` round trip. Deliberate:
                               unlike the online compactor (trained collectively
                               under DP, where rank-0-only saves were broken and
                               dist_checkpointing is required), the scorer is
                               trained OFFLINE in one process and deployed
                               replicated read-only — every rank loads the same
                               file.

Live deployment: ``--kv-compaction-strategy learned_oracle`` with
``--kv-compaction-oracle-checkpoint`` (see ``live.py``).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from ..compressors import CompactionResult, _validate_budget


def token_level_oracle(
    keys: list[torch.Tensor],       # per layer (P, d_local)
    queries: list[torch.Tensor],    # per layer (Q, Hq_local, D)
) -> torch.Tensor:
    """Ground truth: accumulated softmax attention per prefix token, (P,).

    Aggregated over layers and KV groups exactly like live compaction scores
    (the paged cache evicts token-level, so the target is token-level too).
    """
    P = keys[0].shape[0]
    D = queries[0].shape[-1]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(P, device=keys[0].device)
    for k, q in zip(keys, queries):
        k3 = k.view(P, -1, D)
        n_groups = k3.shape[1]
        group = q.shape[1] // n_groups
        for g in range(n_groups):
            qg = q[:, g * group:(g + 1) * group, :].reshape(-1, D).float()
            out += torch.softmax(qg @ k3[:, g, :].float().T * scale, dim=-1).sum(dim=0)
    return out


@dataclass
class OracleScorerConfig:
    """Sizes for the scorer. ``d_key`` is the TP-local per-layer key width
    (H_local * D — e.g. 128 on Nano TP4); features add position + layer one-hot."""

    d_key: int
    n_layers: int
    hidden: int = 256

    @property
    def in_dim(self) -> int:
        return self.d_key + 1 + self.n_layers


class LearnedOracleScorer(torch.nn.Module):
    """MLP: [key, position/P, layer one-hot] → predicted log1p oracle mass.

    Built from the same TE linear wrappers as the compactors, replicated
    (singleton TP group) — every rank scores its own TP-local key slice with
    identical weights.
    """

    def __init__(self, cfg: OracleScorerConfig,
                 params_dtype: torch.dtype = torch.float32,
                 pg_collection=None) -> None:
        from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
        from megatron.rl.compaction.learned.models.compactor import (
            compactor_transformer_config,
        )
        super().__init__()
        self.cfg = cfg
        config = compactor_transformer_config(cfg.hidden, 1, cfg.hidden, params_dtype)
        tp_group = pg_collection.tp if pg_collection is not None else None
        lin = dict(config=config, init_method=config.init_method, gather_output=False,
                   bias=False, skip_bias_add=False, is_expert=False, tp_group=tp_group)
        self.fc1 = TEColumnParallelLinear(cfg.in_dim, cfg.hidden, **lin)
        self.fc2 = TEColumnParallelLinear(cfg.hidden, cfg.hidden, **lin)
        self.fc3 = TEColumnParallelLinear(cfg.hidden, 1, **lin)
        # Feature normalisation, set by fit_oracle_scorer from the train set.
        self.register_buffer("feat_mu", torch.zeros(1, cfg.in_dim))
        self.register_buffer("feat_sd", torch.ones(1, cfg.in_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features (N, in_dim) → predicted log1p oracle mass (N,)."""
        # Normalise in fp32 (the buffers stay fp32 across checkpoint loads),
        # then cast to the TE linears' parameter dtype (bf16 when serving).
        x = (features.float() - self.feat_mu) / self.feat_sd
        x = x.to(next(self.parameters()).dtype)
        x = F.gelu(self.fc1(x)[0])
        x = F.gelu(self.fc2(x)[0])
        return self.fc3(x)[0].squeeze(-1)

    def features(self, keys: torch.Tensor, layer_idx: int, n_positions: int | None = None,
                 ) -> torch.Tensor:
        """Build (S, in_dim) features for one layer's keys (S, d_key)."""
        S, d = keys.shape
        if d != self.cfg.d_key:
            raise ValueError(f"key width {d} != scorer d_key {self.cfg.d_key}")
        if not 0 <= layer_idx < self.cfg.n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.cfg.n_layers})")
        P = n_positions or S
        pos = torch.arange(S, device=keys.device, dtype=torch.float32).unsqueeze(1) / P
        onehot = torch.zeros(S, self.cfg.n_layers, device=keys.device)
        onehot[:, layer_idx] = 1.0
        return torch.cat([keys.float(), pos, onehot], dim=1)

    @torch.no_grad()
    def score_tokens(self, keys: torch.Tensor) -> torch.Tensor:
        """Token-level scores for a request's full KV — (L, S, H, D) → (S,).

        The live-selection entry point: per-layer predictions averaged over
        layers, mirroring how ``token_level_oracle`` aggregates the target.
        """
        L, S, H, D = keys.shape
        feats = torch.cat([
            self.features(keys[li].reshape(S, H * D), li) for li in range(L)
        ])
        return self(feats).view(L, S).float().mean(dim=0)


def fit_oracle_scorer(
    scorer: LearnedOracleScorer,
    captures: list[tuple[list[torch.Tensor], list[torch.Tensor]]],
    epochs: int = 200,
    lr: float = 3e-4,
    batch: int = 65536,
) -> list[float]:
    """Train the scorer on captured (keys_per_layer, queries_per_layer) pairs.

    Targets are ``log1p(token_level_oracle)`` shared across layers (each
    layer's features regress the same token-level target, exactly like the
    validated scorer script). Sets the scorer's normalisation buffers from the
    training set. Returns the per-epoch losses.
    """
    if not captures:
        raise ValueError("fit_oracle_scorer: no captures given")
    xs, ys = [], []
    for keys, queries in captures:
        target = token_level_oracle(keys, queries).log1p()
        for li, k in enumerate(keys):
            xs.append(scorer.features(k, li))
            ys.append(target)
    X = torch.cat(xs)
    y = torch.cat(ys)
    scorer.feat_mu.copy_(X.mean(0, keepdim=True))
    scorer.feat_sd.copy_(X.std(0, keepdim=True).clamp(min=1e-5))

    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(epochs):
        idx = torch.randperm(X.shape[0], device=X.device)[:batch]
        loss = F.mse_loss(scorer(X[idx]).float(), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def fit_scorer_on_flywheel(
    scorer: LearnedOracleScorer,
    flywheel_dir: str,
    epochs: int = 100,
    lr: float = 1e-4,
) -> list[float]:
    """Refit the scorer on retrieval-flywheel events (the self-labeling loop).

    Every archived span that live decoding RESTORED is a proven eviction
    mistake (label 1 — should have been kept); every span still archived when
    its request finished was correctly evicted (label 0). Fine-tunes the
    scorer with BCE on its score output so restored-like keys rank higher —
    the eviction policy learns from its own misses on real traffic. No new
    hyperparameters beyond the optimizer's.
    """
    import os

    files = sorted(
        os.path.join(flywheel_dir, f) for f in os.listdir(flywheel_dir)
        if f.startswith("events_") and f.endswith(".pt"))
    if not files:
        raise ValueError(f"fit_scorer_on_flywheel: no event files in {flywheel_dir}")
    feats, labels = [], []
    for f in files:
        blob = torch.load(f, map_location="cuda", weights_only=True)
        for keys, positions, label in zip(blob["keys"], blob["positions"],
                                          blob["labels"]):
            L, T, H, D = keys.shape
            k = keys.cuda().float()
            for li in range(L):
                feats.append(scorer.features(k[li].reshape(T, H * D), li))
                labels.append(torch.full((T,), float(label), device="cuda"))
    X = torch.cat(feats)
    y = torch.cat(labels)
    if y.min() == y.max():
        raise ValueError(
            "fit_scorer_on_flywheel: events are all one class "
            f"(label {int(y[0].item())}) — need both restored and unused spans.")
    opt = torch.optim.AdamW(scorer.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(scorer(X).float(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def save_oracle_scorer(scorer: LearnedOracleScorer, path: str) -> None:
    """Persist config + weights (plain torch.save — see module docstring)."""
    torch.save({"cfg": asdict(scorer.cfg), "state_dict": scorer.state_dict()}, path)


def load_oracle_scorer(path: str, params_dtype: torch.dtype = torch.float32,
                       pg_collection=None) -> LearnedOracleScorer:
    """Rebuild a scorer from ``save_oracle_scorer`` output."""
    blob = torch.load(path, map_location="cuda", weights_only=True)
    scorer = LearnedOracleScorer(OracleScorerConfig(**blob["cfg"]),
                                 params_dtype=params_dtype,
                                 pg_collection=pg_collection).cuda()
    scorer.load_state_dict(blob["state_dict"])
    return scorer.eval()


class OracleCompressor:
    """Offline ``KVCompressor`` protocol wrapper around a trained scorer.

    Selection-only (like H2O/SnapKV): retain the ``budget`` keys with the
    highest predicted oracle mass; K/V unchanged. ``ref_queries`` is accepted
    and ignored — being query-free is the point.
    """

    def __init__(self, scorer: LearnedOracleScorer, layer_idx: int = 0) -> None:
        self.scorer = scorer
        self.layer_idx = layer_idx

    @property
    def strategy(self) -> str:
        return "learned_oracle"

    def compress(self, keys: torch.Tensor, values: torch.Tensor, budget: int,
                 ref_queries: torch.Tensor | None = None,
                 run_id: str = "", step_id: int = 0) -> CompactionResult:
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)
        with torch.no_grad():
            scores = self.scorer(self.scorer.features(keys, self.layer_idx))
        retained = torch.topk(scores.float(), budget).indices.sort().values.tolist()
        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=retained,
            compacted_keys=keys[retained],
            compacted_values=values[retained],
            bias=None,
            strategy=self.strategy,
            original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )
