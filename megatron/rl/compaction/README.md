# KV / Context Compaction for Megatron-RL

Research stack for compressing the KV cache of a Megatron LLM during RL
training and inference. Megatron-only, GPU-only, hard-fail-by-default; every
baseline is paper-exact (each algorithm file cites its paper).

## Package map

| package | what lives there | README |
|---|---|---|
| `kv/` | Selection baselines (H2O, SnapKV, OMP, TopK, StreamingLLM), the shared attention math, the paged-cache hook, live in-engine compaction, benchmark harness, `build_kv_compressor` factory | [kv/README.md](kv/README.md) |
| `learned/` | Learned compactors (Perceiver, gated recurrent updater), STILL/NextLat training objectives, online GRPO integration, sufficiency-KL probes, KV/student capture | [learned/README.md](learned/README.md) |
| `pomdp/` | Text-level rollout recording as POMDP transitions (belief states, triggers, digest algorithms, trace stores). Heavily unit-tested research scaffolding for the text-compaction track; not wired into the live RL path. | module docstrings |

## The two deployment surfaces

**Inference server** (`tools/run_dynamic_text_generation_server.py`): live
post-prefill compaction of every request's prompt KV inside the dynamic
engine — `--kv-compaction-strategy {snapkv,streaming_llm,belief_still}` with a
budget ratio, or a trained learned-compactor checkpoint. Used by the
RULER/LongBench eval grids.

**GRPO rollout loop** (`megatron/rl/inference/megatron.py`): the same live
compactor attaches to the in-process engine with
`--rl-compaction-enabled --rl-compaction-mode live`, so policies train while
decoding over compacted caches. `record_only` mode instead saves KV
trajectories for offline compactor training
(`--rl-compaction-trajectory-dir`), and `--rl-compaction-compactor-train`
trains the learned compactor online alongside GRPO.

## Ground rules encoded in this package

- **Paper-exactness with stated limits**: H2O's accumulated-attention score is
  unobservable under flash attention → H2O is offline-only and the factory
  says so; SnapKV is the flash-compatible live heavy-hitter; OMP's per-key
  bias has no paged-attention deployment.
- **Live compaction is token-level**: the paged cache shares one block table
  per request across all layers and heads.
- **No soft failures**: misconfigurations and impossible states raise; the one
  deliberate skip (failed capture forward) is collective and documented.
- **Everything TP/PP/DP-clean**: the learned compactor is replicated with a
  world-DP optimizer; captures are TP-local by design; checkpoints use
  Megatron dist_checkpointing.
