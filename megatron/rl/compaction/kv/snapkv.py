# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""SnapKV compressor.

Source: Li et al., 2024 (arXiv:2404.14469).

SnapKV is the flash-attention-compatible heavy-hitter method: instead of
accumulating attention over every decode step (which flash attention never
materialises), it scores prefix keys using only a small **observation window**
of the most-recent query tokens. Those window queries' attention to the prefix
is computed explicitly (a cheap W×T partial attention), pooled over neighbouring
keys (clustering), and the top scorers are retained together with the
observation window itself. Retained K/V are the originals — no fitting.

This is why SnapKV deploys live where paper-exact H2O cannot: it needs attention
for only W queries, computable alongside a flash forward, rather than the full
accumulated attention matrix.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .compressors import (
    CompactionResult,
    _select_recent_plus_heavy,
    _softmax_attention,
    _validate_budget,
)


class SnapKVCompressor:
    """SnapKV: observation-window heavy hitters + the recent window, K/V kept.

    Parameters
    ----------
    obs_window:   Number of most-recent tokens used as the observation window;
                  these queries score the prefix and are themselves always kept.
    pool_kernel:  1-D max-pool kernel over the key axis (clustering, so a whole
                  important span is retained, not isolated tokens). Paper uses 7.

    ``compress`` expects ``ref_queries`` to be the observation-window query
    vectors (the last ``obs_window`` real queries). Retains ``budget`` keys:
    the observation window plus the top pooled-attention prefix keys.
    """

    def __init__(self, obs_window: int = 32, pool_kernel: int = 7) -> None:
        if obs_window < 1:
            raise ValueError(f"obs_window must be >= 1, got {obs_window}")
        if pool_kernel < 1 or pool_kernel % 2 == 0:
            raise ValueError(f"pool_kernel must be a positive odd int, got {pool_kernel}")
        self.obs_window = obs_window
        self.pool_kernel = pool_kernel

    @property
    def strategy(self) -> str:
        return f"snapkv_w{self.obs_window}_p{self.pool_kernel}"

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        budget: int,
        ref_queries: torch.Tensor | None = None,
        run_id: str = "",
        step_id: int = 0,
    ) -> CompactionResult:
        if ref_queries is None:
            raise ValueError(
                "SnapKVCompressor needs ref_queries (the observation-window queries) "
                "to score prefix keys."
            )
        t0 = time.perf_counter()
        T = keys.shape[0]
        budget = _validate_budget(budget, T)

        # CAUSAL attention of the observation-window queries over all keys
        # (official SnapKV masks the window x window block), summed, then
        # max-pooled over neighbouring PREFIX keys only — the official kernel
        # drops the window columns before pooling; pooling across the boundary
        # leaks the window keys' large scores into the last pool_kernel//2
        # prefix positions and silently spends budget there every time.
        scores = _softmax_attention(ref_queries, keys, causal_tail=True).sum(dim=0)
        # Official semantics: the observation WINDOW KV is retained
        # unconditionally (window_size tokens), independent of how many query
        # rows the caller supplies; the causal mask handles the query count.
        n_recent = min(self.obs_window, budget, T)
        prefix = scores[: T - n_recent]
        pad = self.pool_kernel // 2
        if prefix.numel():
            pooled_prefix = F.max_pool1d(
                prefix[None, None, :], kernel_size=self.pool_kernel, stride=1, padding=pad
            )[0, 0]
            pooled = torch.cat([pooled_prefix, scores[T - n_recent:]])
        else:
            pooled = scores
        pooled = pooled.to(keys.dtype)

        positions = _select_recent_plus_heavy(
            pooled, T, budget, n_recent=n_recent
        )
        return CompactionResult(
            run_id=run_id, step_id=step_id,
            retained_positions=positions,
            compacted_keys=keys[positions],
            compacted_values=values[positions],
            bias=torch.zeros(len(positions), device=keys.device, dtype=keys.dtype),
            strategy=self.strategy, original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )
