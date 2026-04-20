# Heterogeneous Inference Shards — Work-in-Progress Handoff

**Audience:** the next Claude session (or any engineer) continuing this branch.
**Status:** groups + model build + refit + multi-shard serving are implemented and
unit-tested. Full end-to-end RL rollout with a real model is **not** yet verified.

## 1. What the user wants

Run RL rollouts where different data-parallel (DP) replicas of the inference
model run with **different parallelism sizes**. E.g., shard 0 serves with
TP=8,EP=8, shard 1 serves with TP=4,EP=4 — backed by the same training model,
hosting different request workloads and reachable to each other for future
cross-shard communication.

User priorities, in order:
1. Actually serve inference with heterogeneous parallelism (not just refit).
2. Clean code — this lands in `main` eventually.
3. DP groups must be "reachable" to each other: HTTP, ZMQ, and torch
   collectives across shards.

## 2. What exists today (before this branch)

Megatron-LM already had the plumbing for **one** inference model at a different
parallelism than the training model:

- `--rl-inference-{tensor,pipeline,expert,expert-tensor}-model-parallel-size`
  CLI flags.
- `megatron/rl/parallel_utils.py::build_inference_pg_collection()` — creates a
  single `ProcessGroupCollection` via `HyperCommGrid` with `rank_offset`
  support.
- `megatron/training/training.py:~1037` builds one inference model that all
  ranks participate in (same TP/PP/EP for every DP replica).
- `megatron/rl/inference/megatron.py::MegatronLocal` — one engine +
  `DataParallelInferenceCoordinator` + `start_text_gen_server` on port 8294.
- Refit path in `megatron/core/resharding/refit.py` already supports
  heterogeneous src/dst via `src_rank_offset`, `dst_rank_offset`,
  `None` placeholders for non-participating ranks, and a cache keyed by
  `(rank, src_config, dst_config)`.

The **only** thing that was truly missing was extending the single-inference
path to N shards.

## 3. Architecture

```
                 ┌──────────────────────────────────────────┐
                 │   Training world (all ranks, WORLD)      │
                 │   initialize_model_parallel(...)         │
                 └──────────────────────────────────────────┘
                                      │
                 parses args.rl_inference_shards_parsed
                                      │
                                      ▼
        build_inference_pg_collections_for_shards(...)
                 returns List[InferenceShard]
                 (every rank calls this; ranks outside a
                  shard get pg_collection=None)
                                      │
       set_inference_shards(...)  ────┤
                                      │
                ┌─────────────────────┼──────────────────────┐
                │                     │                      │
            Shard 0              Shard 1                 Shard N-1
        rank_offset=0         rank_offset=S0       rank_offset=S0+S1+...
        TP=a, EP=b, ...       TP=c, EP=d, ...      ...
        pg_collection          pg_collection         pg_collection
                │                     │                      │
         GPTModel(pgc=shard0)  GPTModel(pgc=shard1)   GPTModel(...)
                │                     │                      │
        DynamicInferenceEngine  DynamicInferenceEngine  ...
                │                     │                      │
        DPCoordinator (auto-port, spawned by shard's rank 0)
                │                     │                      │
        HTTP server @ base_port+0  @ base_port+1  @ base_port+(N-1)
                │                     │                      │
                └─────────────┬───────┴──────────────────────┘
                              │
                              ▼
              MegatronLocalMulti (one instance per rank)
                 ├── _shards: full list visible on every rank
                 ├── _my_shard_index, _my_engine, _my_client
                 ├── _openai_clients[N]: HTTP clients per shard
                 ├── base_generate() → round-robin across shards
                 ├── resume/suspend/kill → fan out to local engine
                 └── shard_urls(), shard_coordinator_addrs() → introspection
```

Reachability implemented at three layers:

1. **HTTP** — `shard.http_url` all-gathered to every rank.
2. **ZMQ** — `shard.coordinator_addr` all-gathered to every rank; any process
   can open a DEALER socket to any shard's `DataParallelInferenceCoordinator`.
3. **Torch collectives** — `build_cross_shard_group([i, j])` returns a
   `ProcessGroup` spanning the union of shards `i` and `j`'s ranks.

## 4. Files touched

### `megatron/training/arguments.py`

- Added `--rl-inference-shards` CLI (string, semicolon-separated spec list).
  Example: `"tp=8,pp=1,ep=8,dp=1;tp=4,pp=1,ep=4,dp=2"`.
- Parser in `validate_args` produces `args.rl_inference_shards_parsed`
  (list of dicts with defaults applied: `expt_tp` defaults to `tp`;
  everything else defaults to 1).
- Mutual exclusion with the scalar `--rl-inference-*-parallel-size` flags.
- Validation: sum of `tp*pp*dp` across shards must be ≤ `world_size`;
  `expt_tp*ep*pp` must divide each shard's world so the expert grid is valid.

### `megatron/core/inference/shards.py` (framework-agnostic primitives)

Lives in `core` so any framework (NeMo-RL, verl, OpenRLHF-on-megatron, …) can
consume it without pulling in Megatron-LM's RL stack. No RL dependency.

- `InferenceShard` (dataclass) — per-shard state with reachability fields
  (`coordinator_addr`, `http_url`), `ranks()` and `owns_rank()` helpers.
- `build_inference_pg_collection(world_size, tp_size, pp_size, cp_size,
  ep_size, expt_tp_size, use_tp_pp_dp_mapping, rank_offset)` — one
  `ProcessGroupCollection` for a contiguous rank window.
- `build_inference_pg_collections_for_shards(total_world_size, shards,
  use_tp_pp_dp_mapping)` → `List[InferenceShard]`. Loops
  `build_inference_pg_collection` with each shard's `rank_offset`; every rank
  must call this (all `dist.new_group` calls are world-collective).
- `build_cross_shard_group(shards, shard_indices)` →
  `Optional[ProcessGroup]`. Takes the shard list explicitly (no registry).

### `megatron/core/resharding/refit.py` (extended)

- `swap_model_weights_across_shards(src_model, target_model, shards,
  refit_method)` — drives `swap_model_weights` once per shard. Per call, the
  rank passes `target_model=target_model` only for the shard it belongs to,
  else `target_model=None`. Reuses the existing refit plan cache
  (`_PlanCacheKey` keys on src/dst config tuples so per-shard caches don't
  collide). Takes the shard list explicitly.

### `megatron/rl/parallel_utils.py` (RL convenience layer)

Now a thin layer on top of the core primitives:

- Back-compat re-exports of `InferenceShard`,
  `build_inference_pg_collection`, `build_inference_pg_collections_for_shards`.
- `set_inference_shards(list) / get_inference_shards() / get_my_inference_shard()`
  — module-level registry so refit and serving callsites can see the layout
  without threading it through every signature. **RL-specific; frameworks
  wanting to avoid globals should thread the shard list through their own
  state and use the core primitives directly.**
- `build_cross_shard_group(shard_indices)` — registry-reading wrapper that
  forwards to the core function.
- `swap_weights_across_shards(src_model, inference_model, refit_method)` —
  registry-reading wrapper around the core
  `swap_model_weights_across_shards`. Falls back to a single
  `swap_model_weights` call when no shards are registered.

### `megatron/training/training.py`

The `if args.perform_rl_step:` block at ~L1037 was restructured:

- Accepts **either** the legacy scalar flags (builds a single-shard list) or
  `args.rl_inference_shards_parsed` (list of dicts).
- Calls `build_inference_pg_collections_for_shards` once.
- Registers the shards via `set_inference_shards(shards)`.
- Each rank builds **one** inference model using its own shard's
  `pg_collection` (loops but `break`s after finding its owner).
- Model alloc context (UVM / torch_memory_saver) is the same as before.
- `inference_model` variable is still a single list-wrapped model — downstream
  code that reads `inference_model[0]` is unchanged.

### `megatron/training/training.py` and `megatron/rl/rl_utils.py`

Three `swap_model_weights(model, inference_model, args.refit_method)` call
sites were retargeted to `swap_weights_across_shards(...)` so refit works
for both single-shard (legacy) and multi-shard paths.

### `megatron/rl/inference/multi_shard.py` (new)

`MegatronLocalMulti(InferenceServer, ReturnsTokens, ReturnsRaw)` — pydantic
model mirroring `MegatronLocal`'s shape. Private attrs hold the full shard
table and rank-local state. Public methods:

- `@classmethod async launch(model, shards, host, base_port, verbose)` — per
  rank: builds engine (if this rank owns a shard), calls
  `start_listening_to_data_parallel_coordinator(inference_coordinator_port=None)`
  (auto-port to avoid cross-shard collisions), all-gathers the coordinator
  address, starts the HTTP server on the shard's rank 0 at
  `base_port + shard.index`, all-gathers HTTP ports, builds an
  `AsyncOpenAI` client per shard.
- `async base_generate(request)` — round-robin over reachable clients under
  `self._route_lock`.
- `set_generation_epoch(epoch)` — local-client-only fan-out.
- `async resume() / suspend() / kill()` — each rank drives its own engine;
  HTTP clients are closed everywhere on kill.
- `shard_urls() / shard_coordinator_addrs()` — introspection for debugging.

### `megatron/rl/rl_utils.py::get_inference_interface`

Dispatches: if `get_inference_shards()` is not None → `MegatronLocalMulti.launch`;
else → legacy `MegatronLocal.launch`. When shards is a list of length 1, Multi
is still used — this path is a no-op-routing case but exercises the same plumbing.

### Tests — `tests/unit_tests/rl/test_inference_shards.py`

Run with:

```
python3 -m torch.distributed.run --nproc-per-node=4 --master_port=29500 \
    -m pytest tests/unit_tests/rl/test_inference_shards.py -v
```

Nine tests covering parser, process-group construction, cross-shard
reachability, refit, routing, and cache invariants:

1. `test_parse_rl_inference_shards_string` — CLI parser invariants.
2. `test_build_shards_basic` — pg_collection membership correct on 2 shards.
3. `test_shard_rank_partition_explicit` — exact TP/DP rank lists per shard.
4. `test_shards_with_idle_ranks` — ranks outside every shard see `pg_collection=None` on all shards.
   *Note:* arg validation now rejects layouts that don't cover the full
   world, so this test exercises the primitive directly — user-facing
   configs still require full coverage.
5. `test_cross_shard_group_broadcast` — NCCL broadcast across DP replicas
   via `build_cross_shard_group`.
6. `test_shard_url_exchange_logic` — all-gather-object pattern for
   coordinator addresses produces a consistent table on every rank.
7. `test_heterogeneous_refit_end_to_end` — TP=4 training → shard 0 (TP=2)
   + shard 1 (TP=1); `swap_weights_across_shards` drives per-shard refit;
   verifies logits match the source model on every rank (atol=5e-4).
8. `test_cross_shard_group_cache_reuses_groups` — repeated calls for the
   same shard union return the *same* cached `ProcessGroup` object; order
   of the argument list doesn't matter; deregistering shards flushes the
   cache.
9. `test_multi_shard_routing_picks_fast_shard` — non-distributed;
   constructs a `MegatronLocalMulti` in isolation, seeds per-shard latency
   samples, and asserts the weighted picker (a) round-robins during
   cold-start, (b) prefers the fast shard once samples exist, and (c)
   switches back when `in_flight` makes the fast shard saturated.

A regression pass of `tests/unit_tests/resharding/test_model_swap.py`
(~50 parametrizations) also passes with these changes in place.

## 5. What was explicitly **not** tested

- `MegatronLocalMulti.launch` has never been exercised end-to-end — spinning
  up the engine requires a loaded checkpoint, tokenizer, configured KV cache,
  and a live OpenAI client. No automated test covers this path.
- `base_generate` routing with a real response flow (request → HTTP → engine →
  tokens) is not tested; only the routing state machine is correct by
  construction.
- `resume`/`suspend`/`kill` lifecycle orderings in multi-shard mode are
  inherited from `MegatronLocal` per-shard but the concurrent behaviour when
  multiple shards go through the cycle simultaneously has not been stress
  tested.
- Prefix-cache-aware routing across shards does **not** exist.
  `DataParallelInferenceCoordinator` still does prefix-affinity routing
  within a shard's DP group, but `MegatronLocalMulti` routes round-robin
  across shards. If the user enables
  `inference_dynamic_batching_enable_prefix_caching`, expect worse cache
  hit rates than the single-shard path.
- Refit performance with N shards is linear in N (plan build + scatter per
  shard). Benchmark before scaling past ~4 shards.

## 6. How to actually try it end-to-end

The first real test should be with an existing GRPO recipe. Locate a config
that already uses `--rl-inference-tensor-model-parallel-size` (e.g.
`tests/functional_tests/test_cases/gpt/gpt_grpo_basic_function/model_config.yaml`
or `gpt_grpo_tp1tp2_pp1_dp8_583m_throughputtest`), and replace the scalar
flags with `--rl-inference-shards`. Rules:

- Sum of `tp_i * pp_i * dp_i` across shards must equal training world size
  (for collocated mode) or less than it (ranks outside shards are idle).
- `expt_tp * ep * pp` must divide each shard's world.

Example: 8-GPU job, TP=4 training →

```yaml
  --rl-inference-shards: "tp=4,ep=1,dp=1;tp=2,ep=1,dp=2"
```

gives shard 0 on ranks 0–3 (TP=4), shard 1 on ranks 4–7 (TP=2, DP=2).
Launch the recipe; rollouts should fan out across the two HTTP servers on
ports 8294 and 8295. Watch for:

- `Building 2 heterogeneous RL inference shard(s):` in training output.
- Two coordinator processes at distinct ports in process list.
- Two HTTP endpoints responding to `curl`.
- Refit time scaling linearly with shard count.

## 7. Known rough edges to look at next

1. **`MegatronLocalMulti.launch` importing `start_text_gen_server`**: the
   module uses process-level globals (`_SERVER_PROCESSES`, `_SHARED_SOCKET`)
   that prevent calling it twice from the same rank. Currently safe because
   each shard's HTTP server runs on that shard's rank 0 (distinct processes).
   If anyone refactors, preserve this invariant.

2. ~~Arg validation tautology~~ — **resolved.** The expert-decomposition
   check was rewritten as a plain `shard_world % (expt_tp * ep * pp) == 0`
   divisibility assertion, and the world-coverage check was tightened from
   `<= world_size` to `== world_size` so idle ranks can't silently skip the
   collective refit loop (see §7.3).

3. **Idle ranks are currently rejected at arg validation time.** Allowing
   them requires routing `target_model=None` through the refit for every
   shard on ranks that own none — the collective needs every rank present.
   Until the refit call-sites (`swap_weights_across_shards` in
   `training.py` and `rl_utils.get_environment_rollouts`) unconditionally
   run when shards are registered (instead of gating on `inference_model is
   not None`), keep the equality assertion. The arg-validation error
   message documents this.

4. ~~`verify_model_weights_swap` is shard-broken~~ — re-audited. The
   comparison is intrinsically per-rank (no WORLD gather; each model's
   `runtime_gather_output=True` gathers within its own TP group). The
   function now no-ops when `inference_model is None` and documents the
   heterogeneous-shard semantics in its docstring. If idle ranks are
   re-enabled in the future (see §7.3) the no-op guard means it continues
   to work without modification.

5. ~~No weighted routing~~ — **resolved.** `MegatronLocalMulti.base_generate`
   maintains a per-shard sliding window of recent response latencies and an
   in-flight request counter; once every reachable shard has at least one
   sample it picks the shard maximizing `1 / (median_latency * (1 +
   in_flight))`. `shard_routing_stats()` returns the current per-shard
   telemetry for debugging. The cold-start fallback is still round-robin.

6. ~~Cross-shard collectives are single-use~~ — **resolved.**
   `build_cross_shard_group` is now cached by `frozenset(ranks)`. Miss is
   world-collective (unchanged); hit is rank-local.
   `clear_cross_shard_group_cache()` flushes on teardown and is invoked
   automatically by `set_inference_shards(None)`.

7. ~~`mpu.get_*` leaks under `megatron/core/inference/`~~ — audited (see
   commit). Only one forward/serving-path leak was found and fixed:
   `run_mcore_engine.py` used `mpu.is_pipeline_first_stage()` for its
   post-processing gate, which would read the training PP group on a
   heterogeneous shard. Replaced with `is_pipeline_first_stage(engine.
   controller.pp_group)`. Other `mpu.*` reads in `shards.py` are init-only
   fallbacks (intentional, for callers that want training defaults) and
   the wrappers under `model_inference_wrappers/` already plumb
   `self.pp_group` from `pg_collection`.

## 8. Minimal reproducer for the serving path

Once you have a real model + tokenizer set up:

```python
# Inside a script that runs after training.py has called set_inference_shards
import asyncio
from megatron.rl.inference.multi_shard import MegatronLocalMulti
from megatron.rl.parallel_utils import get_inference_shards, get_my_inference_shard

shards = get_inference_shards()
my = get_my_inference_shard()
model_for_me = inference_model[0] if my is not None else None

loop = asyncio.new_event_loop()
interface = loop.run_until_complete(
    MegatronLocalMulti.launch(
        model_for_me, shards=shards, host='0.0.0.0', base_port=8294,
    )
)
# Inspect reachability table:
print("URLs:     ", interface.shard_urls())
print("ZMQ addrs:", interface.shard_coordinator_addrs())

# Drive a single request on rank 0:
import torch.distributed as dist
if dist.get_rank() == 0:
    from megatron.rl.inference.inference_interface import InferenceRequest, LLMChatMessage
    req = InferenceRequest(
        prompt=[LLMChatMessage(role='user', content='Hello')],
        generation_args=...,
    )
    resp = loop.run_until_complete(interface.base_generate(req))
    print(resp.raw_text)

loop.run_until_complete(interface.kill())
```

## 9. Design decisions worth preserving

- **World-collective launch**: every rank must call `launch` simultaneously
  — `all_gather_object` assumes this. Don't make shard launch lazy/async
  per-shard without redesigning the exchange.
- **Shard table mutation during launch**: `launch` writes
  `coordinator_addr` and `http_url` into the shared `InferenceShard`
  instances. This is safe because all ranks see the same instances via
  `set_inference_shards`. If you ever serialize shards, be aware.
- **Auto-port for coordinators**: fixed port 41521 (as in the single-shard
  path) cannot work here. `inference_coordinator_port=None` was a
  deliberate choice.
- **Global rank 0 drives lifecycle**: pause/resume/stop/shutdown are issued
  through rank 0's `_lifecycle_clients` (one `InferenceClient` per shard).
  Non-driver ranks only `wait_until` on their local engine state. Do not
  re-introduce per-shard-rank_offset drive — shard 1's rank_offset would
  then pause its shard before rank 0 finishes routing requests to it.
- **Full world coverage required**: arg validation rejects shard layouts
  that don't partition the full world. Supporting idle ranks requires
  running the collective refit even on ranks outside every shard (see §7.3);
  not supported today.
- **Refit cache key**: relies on `(rank, src_config, dst_config,
  num_experts)` tuple uniqueness. The multi-shard driver loops over shards
  and each iteration's `dst_config` is either this rank's shard (one
  unique config) or `None` (cache key has `dst_config=None`, so shared
  across all "not in this shard" iterations — correct because the plan
  body for a non-participating rank is degenerate).

## 10. If you're picking this up cold

1. Read `megatron/core/inference/shards.py` — the framework-agnostic primitives.
2. Read `megatron/core/resharding/refit.py::swap_model_weights_across_shards`
   — the per-shard refit loop.
3. Read `megatron/rl/parallel_utils.py` — thin convenience layer (registry
   + wrappers).
4. Read `megatron/rl/inference/multi_shard.py` — serving fanout (~300 lines).
5. Run the unit test file to confirm your env is healthy.
6. Look at `megatron/training/training.py:~1037` for how shards get
   registered from CLI args.
7. Try the minimal reproducer in §8 with your smallest available model.
8. Measure refit time on two shards vs one; then enable weighted routing
   if needed.

Good luck.
