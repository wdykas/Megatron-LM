# Learned KV Compaction — STILL / NextLat Compactors

A small Megatron-native compactor that *synthesizes* a compact KV memory
(instead of selecting positions like `../kv/`), trained inside or alongside the
GRPO loop. Built entirely on Megatron/TE modules and Megatron's own optimizer,
DDP, and dist_checkpointing — replicated on every rank, data-parallel across
the world (each rank trains on its own TP-local KV slice; "Option C").

## Architectures (`models/`)

- **`PerceiverCompactor`** (`compactor.py`): single-pass cross-attention
  compressor — C learned latents attend to the full per-layer KV and emit
  (C, d_kv) synthetic keys/values per attention layer. TEColumn/RowParallelLinear
  + TEDotProductAttention with a singleton TP group so it builds replicated.
- **`GatedRecurrentUpdater`** (`belief.py`): recurrent belief memory
  `M_{t+1} = U_θ(M_t, R_t)` with fixed size C — the online model. Per-slot
  gating decides how much of each memory slot updates per chunk; the updater
  itself serves as the NextLat latent-dynamics model (no extra head).
- **`BeliefMemory`**: the compact state, `(keys, values)` of shape
  `(n_layers, B, C, d_kv)`.

Config: `GatedUpdaterConfig(n_compress, n_heads, d_kv, n_attn_layers)` /
`PerceiverConfig(...)` — `d_kv` and `n_attn_layers` are sized from the actual
captured KV (TP-local), never hardcoded.

## Training objectives (`training/losses.py`)

Weighted terms in `CompactorLossWeights`, combined per-chunk by
`train_compactor_trajectory` (`training/training.py`, truncated-BPTT over
chunks). **Defaults are minimal: only `teacher_kl` is on.** The recommended
ladder is (1) `teacher_kl` alone — zero extra hyperparameters, the floor;
(2) + one `future_latent` weight — the direct NextLat form (`dynamics` and
`future_kv_reconstruction` are its proxies, the consistency terms are one
idea expressed three ways); (3) decomposed proxies only to diagnose a
failure of (2). Terms that lose the sweep get deleted, not left at zero:

| term | signal |
|---|---|
| `kv_reconstruction` | attention-output MSE of compact vs full KV on probe queries (default self-supervised signal) |
| `teacher_kl` (STILL) | CE/KL through the differentiable student forward — the frozen LLM run with compact KV injected (`capture/student_forward.py`) |
| `dynamics` (NextLat) | roll the updater forward and match the next memory (head-free latent dynamics) |
| `future_kv_reconstruction` (NextLat) | old memory must answer queries from the *future* chunk (belief sufficiency) |
| `future_horizon_kl` (NextLat, offline-only) | position-weighted teacher KL, γ<1 upweights later positions |
| `future_latent` (NextLat, offline-only) | SmoothL1 between compact-KV and full-KV **final hidden states** over probe tokens — the strongest NextLat form (direct future hidden-state matching) |
| `consistency` / `merged_chunk_prob` | sequential belief must match one-pass compression of merged chunks (path independence) |

Value-directed weighting (`training/value_directed.py`): scales each probe's
loss by clipped GRPO advantage — RL-trains-compaction in its weakest form and
the mandatory baseline for eviction RL.

## Online integration (`online.py`)

Runs inside GRPO collection once per iteration, on ALL ranks (collective):

1. `build_compactor_trajectories` — replays broadcast rollout sequences through
   one collective forward; each rank captures its own TP-local KV slice
   (`capture/kv_capture.py`) and builds `Trajectory` objects. Also the single
   disk-save path when `--rl-compaction-trajectory-dir` is set.
2. `init_compactor_from_kv` — lazy, sized from the captured dims; loads
   `--rl-compaction-compactor-checkpoint` if given (collective dist-ckpt load).
3. `attach_compactor_optimizer` — mcore DDP over the world DP group +
   `get_megatron_optimizer` (BF16 params, FP32 masters). The compactor always
   has its OWN optimizer (sharing the LLM's corrupts it).
4. `maybe_train_compactor` — trains on the iteration's trajectories, steps,
   checkpoints every `--rl-compaction-compactor-checkpoint-every` iterations
   into `--rl-compaction-compactor-checkpoint-dir` (dist_checkpointing dirs).

Failure policy: hard-fail everywhere except the one documented collective-skip
(a failed capture forward all-reduces a flag so every TP rank skips that
sequence together — the NCCL-deadlock guard).

## CLI flags (training)

```
--rl-compaction-enabled
--rl-compaction-compactor-train           # online compactor training on
--rl-compaction-n-compress 64             # memory slots C
--rl-compaction-chunk-size 256            # capture chunk size
--rl-compaction-compactor-lr 3e-4
--rl-compaction-compactor-checkpoint DIR      # resume/init weights
--rl-compaction-compactor-checkpoint-dir DIR  # periodic saves
--rl-compaction-compactor-checkpoint-every 100
# objective toggles
--rl-compaction-compactor-teacher-kl          # STILL (student CE) mode
--rl-compaction-compactor-dynamics 0.3
--rl-compaction-compactor-future-kv-reconstruction 0.5
--rl-compaction-compactor-merged-chunk-prob 0.5
--rl-compaction-compactor-use-future-accuracy-weight
--rl-compaction-compactor-future-horizon-kl 0.3      # offline pipeline only
--rl-compaction-compactor-future-horizon-gamma 0.8   # γ<1 required with the above
--rl-compaction-compactor-future-latent 0.5          # NextLat, offline pipeline only
# value-directed weighting
--rl-compaction-compactor-advantage-clip 5.0
--rl-compaction-compactor-advantage-min-weight 0.1
--rl-compaction-compactor-use-teacher-logprob        # weight by teacher confidence
```

Canonical pipelines (launch scripts live outside the repo): vd_frozen (frozen
LLM + advantage-weighted recon), vd_joint (LLM trains too), still_online
(teacher-KL). NextLat terms layer onto vd_frozen.

## Evaluating a trained compactor

- **Live**: serve with `--kv-compaction-strategy belief_still
  --kv-compaction-compactor-checkpoint <dir>` (see `../kv/README.md`) — the
  checkpoint drops into the same RULER/LongBench grid as the selection
  baselines.
- **Offline**: `probes.py` — `sufficiency_kl(model, query_tokens, compact_kv,
  teacher_logits)` gives per-position `D_KL(π(·|full) ‖ π(·|compact))` through
  the real model; the primary quality metric at matched budget, the reward
  proxy for RL-trained eviction, and the label source for retrieval triggers.

## Capture utilities (`capture/`)

- `kv_capture.capture_kv_from_forward(model, tokens, position_ids)` — per-layer
  TP-local K/V from a collective forward (hooks on `core_attention`; raises on
  every failure mode).
- `student_forward.student_outputs(model, response_tokens, compact_kv)` — the
  differentiable frozen-model forward with each attention layer's context
  replaced by compact KV; returns `StudentOutput(logits, hidden)` (hidden =
  final decoder output, the future-latent target space); gradients flow only
  into the compactor. `teacher_outputs(model, tokens)` is the capture side:
  the same forward with the REAL context, no grad — store the results on
  `TrainingProbe.teacher_logits` / `teacher_hidden`.
