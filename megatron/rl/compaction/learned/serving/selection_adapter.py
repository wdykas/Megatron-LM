# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bridge: selection-based compressors → BeliefMemory-compatible interface.

Selection-based compressors (OMP, TopK, H2O, StreamingLLM) implement the
``KVCompressor`` protocol (torch.Tensor in → CompactionResult).
The ``CompactionEvaluator`` and ``BeliefCompactorTrainer`` expect an object with:

    initial_compress(keys_per_layer, values_per_layer) -> BeliefMemory
    __call__(memory, new_keys, new_values)             -> BeliefMemory

``SelectionBeliefAdapter`` wraps any ``KVCompressor`` to satisfy
this interface so all compaction methods can be compared in the same evaluator.

Proxy queries
-------------
Selection-based methods need reference queries to score key positions.
Since queries are not available at compression time (that's the problem Still
solves), we use the mean of the current keys as a single proxy query.
This approximation is reasonable for H2O/TopK (scoring is robust) and
intentionally crude for OMP (which is query-sensitive); for exact OMP
behaviour supply a ``query_fn`` that returns real queries.

Multi-chunk streaming
---------------------
``SelectionBeliefAdapter.__call__`` concatenates the current BeliefMemory
slots with the new chunk's KV and re-selects from the combined pool.
Pool size is capped by ``max_pool_size`` to bound memory; oldest tokens
are evicted first when the cap is hit.
"""

from __future__ import annotations

from typing import Callable

import torch

from megatron.rl.compaction.kv.compressors import KVCompressor, CompactionResult
from megatron.rl.compaction.learned.models.belief import BeliefMemory


# ---------------------------------------------------------------------------
# SelectionBeliefAdapter
# ---------------------------------------------------------------------------

class SelectionBeliefAdapter:
    """Wraps an KVCompressor as a BeliefUpdater-compatible object.

    Enables using OMP, TopK, H2O, StreamingLLM through CompactionEvaluator
    and BeliefCompactorTrainer for fair behavioral comparison against Still.

    Parameters
    ----------
    compressor:   Any KVCompressor (OMP, TopK, H2O, StreamingLLM…).
    budget:       C — number of compact slots (= PerceiverConfig.n_compress).
    n_layers:     Number of transformer layers.  Must match the KV lists passed
                  to initial_compress / __call__.
    max_pool_size: Maximum tokens accumulated before oldest-first eviction.
                  Prevents unbounded memory growth in streaming eval.
                  Default: 8 × budget.
    query_fn:     Optional callable(keys (T, d)) -> queries (n, d) used to
                  score positions.  Default: mean key as single-row query.
    """

    def __init__(
        self,
        compressor: KVCompressor,
        budget: int,
        n_layers: int,
        max_pool_size: int | None = None,
        query_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.compressor = compressor
        self.budget = budget
        self.n_layers = n_layers
        self.max_pool_size = max_pool_size or budget * 8
        self.query_fn = query_fn or _mean_key_query
        self._step = 0

    # ------------------------------------------------------------------
    # BeliefUpdater-compatible interface
    # ------------------------------------------------------------------

    def initial_compress(
        self,
        keys_per_layer: list[torch.Tensor],    # n_layers × (B, T, d)
        values_per_layer: list[torch.Tensor],
    ) -> BeliefMemory:
        """Compress the first chunk into a BeliefMemory."""
        self._step += 1
        return self._compress_all_layers(keys_per_layer, values_per_layer)

    def __call__(
        self,
        memory: BeliefMemory,
        new_keys: list[torch.Tensor],           # n_layers × (B, T_new, d)
        new_values: list[torch.Tensor],
    ) -> BeliefMemory:
        """Streaming update: concatenate memory + new chunk, re-select."""
        self._step += 1
        combined_keys, combined_values = [], []

        for l in range(memory.n_layers):
            mem_k = memory.keys[l]              # (B, C, d)
            mem_v = memory.values[l]
            new_k = new_keys[l]                 # (B, T_new, d)
            new_v = new_values[l]

            cat_k = torch.cat([mem_k, new_k], dim=1)   # (B, C+T_new, d)
            cat_v = torch.cat([mem_v, new_v], dim=1)

            # Evict oldest tokens if pool exceeds max_pool_size
            T_total = cat_k.shape[1]
            if T_total > self.max_pool_size:
                cat_k = cat_k[:, -self.max_pool_size:, :]
                cat_v = cat_v[:, -self.max_pool_size:, :]

            combined_keys.append(cat_k)
            combined_values.append(cat_v)

        return self._compress_all_layers(combined_keys, combined_values)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress_all_layers(
        self,
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
    ) -> BeliefMemory:
        """Run the selection compressor on each layer independently."""
        all_keys, all_values = [], []

        for l, (k_t, v_t) in enumerate(zip(keys_per_layer, values_per_layer)):
            # k_t: (B, T, d)
            B, T, d = k_t.shape
            budget = min(self.budget, T)

            batch_keys, batch_values = [], []
            for b in range(B):
                k_b = k_t[b].detach().float()   # (T, d)
                v_b = v_t[b].detach().float()

                ref_q = self.query_fn(k_b)       # (n_q, d)

                result: CompactionResult = self.compressor.compress(
                    keys=k_b,
                    values=v_b,
                    budget=budget,
                    ref_queries=ref_q,
                    run_id=f"sel_{l}_{b}",
                    step_id=self._step,
                )

                ck = result.compacted_keys    # (t, d)  t <= budget
                cv = result.compacted_values

                # Pad to fixed budget with zeros so BeliefMemory has uniform shape
                if ck.shape[0] < self.budget:
                    pad = self.budget - ck.shape[0]
                    ck = torch.cat([ck, torch.zeros(pad, d, device=ck.device, dtype=ck.dtype)])
                    cv = torch.cat([cv, torch.zeros(pad, d, device=cv.device, dtype=cv.dtype)])

                batch_keys.append(ck)
                batch_values.append(cv)

            # Stack to (B, budget, d), place on original device
            device = k_t.device
            all_keys.append(torch.stack(batch_keys).to(device))    # (B, C, d)
            all_values.append(torch.stack(batch_values).to(device))

        return BeliefMemory(
            keys=torch.stack(all_keys),      # (n_layers, B, C, d)
            values=torch.stack(all_values),
            step=self._step,
        )


# ---------------------------------------------------------------------------
# Default proxy query function
# ---------------------------------------------------------------------------

def _mean_key_query(keys: torch.Tensor) -> torch.Tensor:
    """Return the column-mean of the key matrix as a single proxy query.

    keys: (T, d)  →  returns (1, d)
    """
    return keys.mean(dim=0, keepdim=True)
