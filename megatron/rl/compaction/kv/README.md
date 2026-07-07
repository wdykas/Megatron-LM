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

### `oracle.py` — LearnedOracleScorer / OracleCompressor (learned oracle, ours)
The learned heavy-hitter oracle: a small TE MLP on [key vector, position/P,
layer one-hot] trained offline to predict H2O's true accumulated-attention
score — the quantity flash attention makes unobservable live. On trained-Nano
captures it predicts the oracle BETTER than SnapKV's last-32-real-queries
proxy (held-out Spearman 0.97 vs 0.82, recall@40% 0.93 vs 0.75). Query-free by
construction, so the live strategy needs no Q capture, no eager prefill, and
is fully CUDA-graph compatible. Train with `fit_oracle_scorer` on
(keys, queries) captures, persist with `save_oracle_scorer` (plain torch.save
— the scorer is offline-trained and deployed replicated read-only, unlike the
collectively-trained compactor which requires dist_checkpointing), serve with
`--kv-compaction-strategy learned_oracle --kv-compaction-oracle-checkpoint`.
Caveat: v0 signal is from a single text family — retrain on a diverse corpus
before trusting it broadly, and retrain per model/TP layout (the checkpoint
records d_key/n_layers and the loader hard-fails on mismatch).

### `eviction_policy.py` — EvictionPolicy + GRPO trainer (eviction-policy RL, ours)
Eviction as a stochastic policy with exact logprobs: the learned-oracle scorer
architecture (same [key, position, layer] features) emits per-token retain
logits; masks sample Bernoulli(sigmoid(score)) so the set logprob is exact —
the budget is a reward penalty λ·kept_fraction rather than a hard top-k,
which also lets the policy learn prompt-dependent budgets (B3 for free).
Training is offline GRPO (`train_eviction_policy_grpo`): a group of masks per
prompt, within-group reward normalization, REINFORCE. The canonical reward is
negative sufficiency-KL through the real frozen model
(`make_sufficiency_reward` — inject the retained rows via the student
forward, compare with full-cache teacher logits). Reconstruction preserves
what attention looked at; this preserves what the task needed — the selection
gap between the two is the eviction-RL result. A trained policy's `.scorer`
deploys live via `--kv-compaction-oracle-checkpoint` (mind the protected
recent window the live path adds).

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
- Positional encodings: on position-embedding-free models (Nemotron Nano's
  hybrid, `--position-embedding-type none`) pruning is transparent — the cache
  is a set, and slot order carries no meaning. RoPE models must choose a
  `--kv-compaction-rope-mode` (hard-fail otherwise):
  - **`logical`** — positions of record never change. Stored keys keep their
    original (exact) rotations; every step the compactor patches the decode
    tokens' `token_to_pos_ids` (which rotary indexes directly) back to their
    original sequence positions, while cache write slots keep coming from the
    separate `token_to_*` bookkeeping fields. Exact relative geometry — the
    full-cache counterfactual — and archive splice-back needs no rotation.
  - **`renumber`** — StreamingLLM semantics: cache positions are contiguous
    `0..C-1`. Retained keys are delta-rotated to their new positions
    (`rope.py`: RoPE planes compose additively, so a key moves from position m
    to m' by one rotation of m'−m using the model's own
    `rotary_pos_emb.inv_freq`); restored archive spans delta-rotate to the
    cache tail. Positions stay bounded by cache size (beyond-training-window
    generation) at the cost of collapsing the gaps evicted content occupied.
  belief_still stays unsupported under RoPE (synthetic KV has no position
  convention yet).
- `megatron_hook.py` maintains every piece of engine bookkeeping a prune
  touches; note the offset field is the **count of tokens in the last block**
  (engine post-update semantics), and a count on a block boundary is
  represented as full data blocks + a trailing empty current block.

Live strategies: `snapkv`, `streaming_llm`, and `belief_still` (the learned
compactor from `../learned/` — loads a trained checkpoint and replaces the
prompt KV with `n_compress` synthetic tokens).

## CPU archive + negative-cache retrieval (`archive.py`, the archive track)

With `--kv-compaction-archive`, eviction becomes **demotion, not deletion**:
the pruned spans' exact KV tensors move to pinned CPU memory, and a tiny GPU
index of the *evicted* content stays behind — one mean-key centroid per
contiguous span per layer (the "negative cache": an index of what is absent).
Every decode step, the request's query is scored with the same attention math
against retained keys *and* against these centroids; a query that scores
unusually high on a centroid is demonstrably reaching for dropped content. The
winning centroid names the span, `KVArchive.take` pulls its exact KV back from
CPU, and `append_kv_to_request` splices it into the paged cache — one
computation gives both the *when* (trigger) and the *where* (span address).

Trigger: each archived span's centroid attention-mass fraction α̂ — the
estimated share of the current step's attention that wants that span,
normalized against the retained keys' mass (`KVArchive.span_alphas`).
Scale-free in [0, 1): `--kv-compaction-retrieval-alpha` (default 0.2) is the
single-step fast path, and a per-span CUSUM of (α̂ − the span's own EMA
baseline − drift) crossing `--kv-compaction-retrieval-cusum` (default 0.4)
catches novel persistent reaches while chronically hot spans self-absorb
into their baselines. No per-model calibration. Under content-blind mass
eviction (streaming) the signal is weaker — measured and warned at
construction; the archive is most effective with content-aware strategies.

Speculative prefetch: the leading CUSUM candidate is staged (CPU→GPU on a
side stream) at half-threshold, so a later firing splices the staged copy
with no synchronous PCIe stall (`take` logs `prefetched` vs `sync copy`).
A fire on the first decode step is always a sync copy — there was no earlier
step to stage from.

### The retrieval flywheel (self-labeling eviction data)

With `--kv-compaction-flywheel-dir`, the archive logs every finished request's
labelled spans: a span the trigger RESTORED is a proven eviction mistake
(label 1 — the model's own future queries demanded it back); a span still
archived at request end was correctly evicted (label 0). File names rotate
(bounded disk). `fit_scorer_on_flywheel` then fine-tunes the learned scorer
on these events (BCE, no new hyperparameters) — the eviction policy learns
from its own misses on real traffic, and the training distribution IS the
deployment distribution. Archive → safety net → teacher.

### Serving flags (`tools/run_dynamic_text_generation_server.py`)

```
--kv-compaction-strategy {snapkv,streaming_llm,belief_still}
--kv-compaction-budget-ratio 0.5        # fraction of prompt tokens kept
--kv-compaction-obs-window 32           # snapkv observation window
--kv-compaction-min-tokens 128          # skip shorter prompts
--kv-compaction-compactor-checkpoint …  # belief_still trained checkpoint
--kv-compaction-oracle-checkpoint …     # learned_oracle trained scorer
--kv-compaction-n-compress 64           # belief_still synthetic slots
--decode-only-cuda-graphs               # required for snapkv
--kv-compaction-archive                 # CPU archive + retrieval (archive)
--kv-compaction-retrieval-alpha 0.2     # trigger fast path (mass fraction)
--kv-compaction-retrieval-cusum 0.4     # trigger CUSUM threshold h
--kv-compaction-flywheel-dir DIR        # log self-labeling eviction data
--kv-compaction-rope-mode logical       # RoPE models: logical | renumber
                                        # (with archive: --cuda-graph-impl none)
```

### RL-rollout flags (GRPO loop, `MegatronLocal.launch`)

```
--rl-compaction-enabled --rl-compaction-mode live
--rl-compaction-strategy {snapkv,streaming_llm,belief_still}
--rl-compaction-kv-budget-ratio 0.5
--rl-compaction-n-compress 64
--rl-compaction-compactor-checkpoint …
--rl-compaction-split-fraction 0.5      # split-group: P(rollout compacts)
--rl-compaction-budget-final 0.3        # anneal budget to this ...
--rl-compaction-budget-anneal-iters 200 # ... over this many GRPO iterations
```

With a split fraction, each rollout draws its compaction arm Bernoulli(p) —
the control arm decodes over the full cache — and the arm is recorded on the
rollout. Advantages are then normalized within-arm and training logs emit
`kv_compact_reward_gap`, the per-prompt compact-vs-full reward difference.
The same per-request switch is exposed over HTTP: pass `"kv_compact": false`
in a /completions or /chat/completions body to exempt that request — A/B
compaction against one server without restarting.

Debugging: `KV_COMPACTION_DEBUG=1` makes the live compactor dump per-step
request bookkeeping (`[kv-dbg]` lines) — diff a compacted trace against a
baseline run to localise any engine-state divergence.
