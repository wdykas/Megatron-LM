# KV-Cache Compaction — Selection Algorithms & Live Deployment

Paper-exact KV-cache compression baselines, the shared math they build on, and
the machinery that deploys them inside Megatron's dynamic inference engine.
One file per paper; shared primitives in `compressors.py`; deployment in
`live.py` + `megatron_hook.py`.

## The common interface

Every algorithm implements the `KVCompressor` protocol:

```python
result = compressor.compress(keys, values, budget, ref_queries=None)
# keys/values: (T, d) single-head (or head-flattened) tensors, GPU
# budget:      number of positions to retain (budget < 1 hard-fails;
#              budget > T clamps to T)
# ref_queries: (n, d) query vectors for attention-based scoring/fitting
# -> CompactionResult(retained_positions, compacted_keys, compacted_values,
#                     bias, strategy, original_length, wall_time_s)
```

Build by name with the factory (single seam shared by benchmarks and serving):

```python
from megatron.rl.compaction.kv import build_kv_compressor
comp = build_kv_compressor("snapkv")                 # offline use
comp = build_kv_compressor("h2o", inference=True)    # NotImplementedError, see below
```

## Algorithms

### `h2o.py` — H2OAccumulator (Zhang et al. 2023, arXiv:2306.14048)
Heavy-Hitter Oracle, paper-exact. A token's score is the **accumulated softmax
attention** it has received across queries (`F_score(j) = Σ_i softmax(q_i·K)_j`).
The retained budget splits between the most-recent tokens and the top
accumulated scorers; retained K/V are kept unchanged (H2O selects, never refits).

- `recent_ratio` (default 0.5): fraction of the budget reserved for the recent
  window — the paper's even split.
- Online: call `update(attn_weights)` after each decode step, then `compress()`.
- Offline: pass `ref_queries`; scores become the softmax mass over those queries.
- **Offline-only.** Flash attention never materialises the attention weights the
  accumulated score requires, so `build_kv_compressor("h2o", inference=True)`
  raises with exactly this reason. The flash-compatible live heavy-hitter is
  SnapKV.

### `snapkv.py` — SnapKVCompressor (Li et al. 2024, arXiv:2404.14469)
The deployable heavy-hitter method: score prefix keys using only an
**observation window** — the last `obs_window` real queries — computed as an
explicit small attention (`W×T`), max-pooled over neighbouring keys so whole
important spans survive, plus the window itself. Retained K/V unchanged.

- `obs_window` (32): how many trailing queries score the prefix (they are
  always retained).
- `pool_kernel` (7, odd): neighbour max-pool over the key axis (the paper's
  clustering).

### `attention_matching.py` — TopKCompressor / OMPCompressor (Zweiger et al. 2026)
Attention-*reconstruction* methods: choose keys so the retained set reproduces
the full-cache attention, then refit.

- **TopK**: rank keys by RMS normalised attention weight over `ref_queries`;
  optional NNLS bias fit (`fit_bias`) and OLS value fit (`fit_values`).
- **OMP**: greedy Orthogonal Matching Pursuit on the unnormalised attention
  mass (`keys_per_iter` per round, NNLS refit every `nnls_every` rounds,
  L2-regularised at the end), bias = log of the NNLS weights, then OLS value
  fit. The strongest reconstructor at tight budgets; its per-key bias term has
  no live deployment (paged flash attention cannot add per-key logit offsets),
  so live OMP would be selection+value-fit only — currently offline-only.

### `streaming_llm.py` — StreamingLLMCompressor (Xiao et al. 2023, arXiv:2309.17453)
Positional: keep `n_sink` initial tokens (attention sinks) + the most recent
`budget - n_sink`. Content-blind — the cheap floor every content-aware method
must beat (on RULER NIAH @2k/0.5 it scores EM 0.38 where SnapKV holds 1.00).

### `selectors.py` — AttentionSumScorer / UniformScorer
Simple online selectors sharing the same protocol: attention-sum (or key-norm
proxy when no queries) with a protected recent window, and uniform subsampling.

## Shared primitives (`compressors.py`)

`CompactionResult`, the `KVCompressor` protocol, and the math helpers:
`_softmax_attention` (normalised), `_mass_features` (unnormalised exp),
`_select_recent_plus_heavy` (the H2O/SnapKV selection skeleton),
`_fit_bias` (NNLS), `_fit_values` (ridge OLS), `_validate_budget`.

## Benchmarking (`benchmark.py`)

`KVCompactionBenchmark.run(compressors, keys, values, ref_queries,
eval_queries, budget)` scores every algorithm on the same data with held-out
queries: `output_mse` (attention-output reconstruction) and `mass_error`
(attention-mass error). Keep `ref_queries` and `eval_queries` disjoint or
fitting methods will look better than they generalise.

## Live deployment (`live.py` + `megatron_hook.py`)

`LiveKVCompactor` prunes each request's prompt KV **inside the dynamic
inference engine**, right after its prefill completes. Constraints that shape
it (details in the module docstrings):

- The paged cache shares one block table per request across **all layers and
  heads** → live eviction is token-level; scores aggregate over layers/KV-heads.
- Prefill must run **eagerly** for SnapKV's Q capture → launch with
  `--decode-only-cuda-graphs`.
- Only position-embedding-free attention is supported live (Nemotron Nano's
  hybrid: `--position-embedding-type none`); RoPE models hard-fail (retained
  keys would carry stale rotations).
- `megatron_hook.py` maintains every piece of engine bookkeeping a prune
  touches; note the offset field is the **count of tokens in the last block**
  (engine post-update semantics), and a count on a block boundary is
  represented as full data blocks + a trailing empty current block.

Live strategies: `snapkv`, `streaming_llm`, and `belief_still` (the learned
compactor from `../learned/` — loads a trained checkpoint and replaces the
prompt KV with `n_compress` synthetic tokens).

### Serving flags (`tools/run_dynamic_text_generation_server.py`)

```
--kv-compaction-strategy {snapkv,streaming_llm,belief_still}
--kv-compaction-budget-ratio 0.5        # fraction of prompt tokens kept
--kv-compaction-obs-window 32           # snapkv observation window
--kv-compaction-min-tokens 128          # skip shorter prompts
--kv-compaction-compactor-checkpoint …  # belief_still trained checkpoint
--kv-compaction-n-compress 64           # belief_still synthetic slots
--decode-only-cuda-graphs               # required for snapkv
```

### RL-rollout flags (GRPO loop, `MegatronLocal.launch`)

```
--rl-compaction-enabled --rl-compaction-mode live
--rl-compaction-strategy {snapkv,streaming_llm,belief_still}
--rl-compaction-kv-budget-ratio 0.5
--rl-compaction-n-compress 64
--rl-compaction-compactor-checkpoint …
```

Debugging: `KV_COMPACTION_DEBUG=1` makes the live compactor dump per-step
request bookkeeping (`[kv-dbg]` lines) — diff a compacted trace against a
baseline run to localise any engine-state divergence.
