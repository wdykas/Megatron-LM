# Variant B without replication: state migration plan

## Goal

Replace the current **request replication** mode (every rank handles every request, every rank holds every mamba state) with **request ownership + state migration**: each request is owned by one EP rank, mamba state lives only on the owner, and the global view used by Variant B's skip-AG MoE path is produced on demand by an NVLS multicast all-gather of the owners' mamba outputs.

This removes the three structural problems with replication identified in the EP-scaling analysis:

1. Mamba/router/shared compute is no longer ×EP redundant.
2. Mamba state memory is O(total_requests) across the cluster instead of O(EP × total_requests).
3. Opt B-1/B-2's "G ≥ EP" cliff is gone — the owner-rank already has the full per-token state, so the in_proj / shared computations naturally partition by ownership rather than by `G/EP` slice arithmetic.

The model behavior must stay identical to the replicated path; only the work distribution changes.

## Where the gains come from (and where they don't)

The Variant-B skip-AG MoE path requires the `[G_total, H]` global view to exist on every rank at MoE entry. There are two ways to produce it:

1. **Replication** (v13): every rank holds every request and runs the same upstream compute, so the global view already exists by virtue of redundant work.
2. **Partitioned state with layer-level data-flow restructure** (this plan): each rank only holds its own requests, but we *manufacture* the global view via NVLS all-gathers — once at HybridStack entry (embedding output `[G_local, H]` → `[G_total, H]`), and once per mamba layer (mamba's per-rank output `[G_local, H]` → `[G_total, H]`). All non-mamba layers see `[G_total, H]` consistently, so MoE skip-AG works, residual_add / norm shapes match, and the bda kernel is happy.

Per-step kernel sequence under partitioned mode:

```
embedding   [G_local, H]
   │
   ▼
HybridStack initial AG          (1 NVLS AG per step)
   │
   ▼
[G_total, H]
   │
   ├─► mamba layer:  slice→[G_local,H] → in_proj → conv → ssm → out_proj → AG → [G_total,H]
   │
   ▼
[G_total, H]   (residual / bda matches; norm sees G_total)
   │
   ├─► MoE layer:  skip-AG → experts → AR → [G_total, H]
   │
   ▼
... (alternating)
   │
   ▼
final norm    [G_total, H]
   │
   ▼
slice to [G_local, H] for engine's per-rank output
```

Where this **wins** over default main:
- **High EP + large batch where mamba scan is throughput-bound**: per-rank mamba on `G_local = G_total / EP` tokens beats default's also-per-rank mamba PLUS unlocks skip-AG savings on every MoE layer. At small batches where mamba is latency-bound (b ≤ 8 on GB200), the wins shrink to noise.
- **Cluster memory at high EP**: state is per-rank (matches default) — partitioned doesn't *gain* memory headroom over default, but unlike replication it doesn't lose any either.
- **Per-step skip-AG kernel-count savings**: same as v13 (no symm-mem copies in skip-AG dispatch, valid_tokens=G written directly, AR-instead-of-RSV). Roughly +5-10% over default at any batch size, scaling slightly better with EP.

Where this **does not win**:
- **Imbalanced workloads** (`b < EP` or `b % EP != 0` with non-round-robin dispatch): the AG-everywhere data flow requires every rank to participate at the same shape every step. v1 requires the coordinator to round-robin within the replication group when `inference_partitioned_state=True`. Imbalanced batches must fall back to default mode (turn the flag off).
- **Pure prefill steps**: prefill stays on its owner; the partitioned global-view AG only fires under decode-only conditions (mixed batches fall through to the standard path).

## Cost vs default main per layer pair

| Step | Default | Partitioned (this plan) |
|------|---------|-------------------------|
| Mamba compute | per-rank `G_local` | per-rank `G_local` (same) |
| Mamba AG | none | `G_total` bytes/rank (NVLS multimem) |
| MoE entry | AGV `G_total` bytes/rank | skip-AG (free) |
| MoE combine | RSV `G_local` bytes/rank | AR `G_total` bytes/rank |
| Bytes/rank/pair | `G_total + G_local` | `G_total + G_total` |

At small G the difference is barrier-overhead-bound (plain AG and AR have the same fixed barrier cost). At larger G the AR's bigger byte transfer becomes a real cost — but the kernel-count savings (v13-class) plus mamba's throughput-bound regime savings dominate.

Plus a **fixed +1 NVLS AG per step** for the initial embedding-output gather (~5μs).

## Non-goals

- **Heterogeneous EP**: assume all EP ranks have identical compute and weights. Cross-DP-cluster routing is unchanged.
- **Prefill rebalancing**: prefill always runs on the rank that received the request from the coordinator. A request's owner is fixed for prefill, picked by load-balancing at submission time.
- **Multi-tenancy of state**: state slots are 1:1 per request on their owner. No replicated copies elsewhere.
- **Backwards compatibility with old engine clients**: the engine config flag is the contract.

## Architecture overview

```
                          ┌─────────────────────────────────┐
                          │      Coordinator (broker)       │
                          │  picks owner_rank per request   │
                          └────────────┬────────────────────┘
                                       │ (single dispatch, no broadcast)
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  ┌──────────┐                   ┌──────────┐                   ┌──────────┐
  │ Rank 0   │                   │ Rank 1   │                   │ Rank N-1 │
  │ ────────-│                   │ ────────-│                   │ ────────-│
  │ owns r₀, │                   │ owns r₁, │                   │ owns r_k │
  │  r₂, ... │                   │  r₃, ... │                   │   ...    │
  │          │                   │          │                   │          │
  │ mamba    │                   │ mamba    │                   │ mamba    │
  │ state[]  │                   │ state[]  │                   │ state[]  │
  │ (only    │                   │ (only    │                   │ (only    │
  │  owned)  │                   │  owned)  │                   │  owned)  │
  └────┬─────┘                   └────┬─────┘                   └────┬─────┘
       │                              │                              │
       │  per-step decode pipeline:   │                              │
       │                              │                              │
       │  hidden_states[owned]        │                              │
       │  ──→ mamba forward (own)     │                              │
       │  ──→ NVLS AG ───────────►  global hidden_states [G,H]  ◄────┤
       │  ──→ skip-AG MoE                                             │
       │  ──→ AR combine                                              │
```

**Key invariant**: between layers in a segment, every rank holds the same `[G, H]` global view. This makes Variant B's skip-AG / AR-instead-of-RSV continue to work unchanged. The difference vs. replication is *how* the global view is produced — by mamba's all-gather of the owners' outputs, instead of by every rank computing the same thing locally.

## Key invariants & how they're maintained

| Invariant | Maintenance |
|---|---|
| Each request has exactly one mamba state copy | Coordinator picks owner at submission; engine never copies the state to peers. |
| Owner is fixed for the request's lifetime | No mid-request migration in v1. (See "Future: dynamic rebalancing" below.) |
| Mamba forward output is consistent across ranks before MoE | NVLS multimem AG inside the segment runtime, after mamba's `out_proj`. |
| MoE expert weights remain partitioned by EP rank | No change to expert sharding. |
| State memory per rank scales with `max_requests / EP` | Cache size is `max_requests / EP * state_per_request` per rank, not `max_requests * state_per_request` as today. |

## Phases

### Phase 1: Coordinator dispatch — single-owner, no broadcast

**Goal**: the coordinator routes each new request to exactly one EP rank instead of broadcasting it to all `replication_group_size` peers.

**Changes**:
- `data_parallel_inference_coordinator.py`: gate the broadcast loop behind `replicate_requests=True`. Add a new mode `replicate_requests=False, partition_within_cluster=True` that selects a single owner via the existing prefix-cache + load-balancing logic.
- `dynamic_engine.py`: pass through the new mode to the coordinator. Add an `InferenceConfig.inference_partitioned_state` flag that turns on the no-replication path end-to-end.
- Existing `inference_replicate_requests` stays available as a compatibility flag; the new flag is mutually exclusive.

**Test**: with `inference_partitioned_state=True`, submit N requests, verify exactly one engine receives each by `request_id`.

### Phase 2: Per-rank mamba state cache

**Goal**: each rank's state cache is sized for `max_requests / EP` slots, holding only the requests it owns. The slot layout and lookup logic stays per-rank.

**Changes**:
- `dynamic_context.py`: add `local_max_requests = max_requests // ep_size` (with a small overprovision margin for load imbalance, e.g. 1.25×). Allocate `mamba_conv_states` and `mamba_ssm_states` at this reduced size.
- `mamba_metadata.py`: `batch_indices_decode` is already per-rank in concept — under partitioning it now indexes only into the rank's local slots. The kernel calls don't change.
- Request → state slot mapping: `(owner_rank, local_slot_idx)`. The coordinator already has the rank-id; the engine assigns local slots from a per-rank free list.

**Test**: with EP=4 and `max_requests=16`, each rank's state cache is 4 (or 5 with overprovision) slots. Submit 16 requests, verify the cache fills evenly and no rank exceeds its quota.

**Risk**: load imbalance — if the coordinator's load balancer concentrates requests on one rank, that rank fills up while others are empty. Mitigation: overprovision by 25% (already done in many production engines); add a feedback signal to the coordinator from each rank's free-slot count.

### Phase 3: Mamba forward — per-rank own-set + NVLS AG

**Goal**: each rank runs mamba on its owned requests' hidden_states, then all-gathers into the global view used by the next-segment MoE.

This is where the bulk of the latency win comes from. At EP=4, `b=8`, this drops mamba forward wall-clock from ~50-80μs/layer to ~15-25μs/layer (the per-rank work shrinks 4×, the AG adds ~5μs).

**Changes**:
- `mamba_mixer.py`:
  - Replace `_dynamic_inference_local`'s "compute on full hidden_states" with "compute on own subset, AG result".
  - Input slicing: incoming `hidden_states` arrives as `[G_total, H]` global view (from previous segment's AR or initial layernorm). Each rank takes only its `[G_local, H]` rows — i.e., the rows of requests it owns.
  - `in_proj`, conv1d, ssm_decode, out_proj: all run on `[G_local, *]`. The existing Opt B-2 (`_maybe_in_proj_on_local_slice`) is now a no-op — the input already arrives sliced.
  - Output: `[G_local, H]` is multimem-AG'd to `[G_total, H]` via the existing `multimem_all_gather` helper. This becomes the input to the next layer (or the next segment's MoE).
- `dynamic_context.py`: maintain a precomputed `(rank → list of owned request slot indices)` map per step. The mamba mixer reads its rank's slice from this. Index reuse across layers: same map for every mamba layer in the step.

**Skip-AG of MoE remains correct**: after mamba's AG, every rank has the full `[G_total, H]`. The MoE token_dispatch sees `_segment_input_is_global=True` and skips the AG, exactly as today.

**State write semantics**: only the owner writes its state slots (`conv_state[owned_indices]`, `ssm_state[owned_indices]`). Peers' state is unchanged because they never had it. No cross-rank state coherence to maintain.

**Test**: prefill 4 requests on EP=4, run 5 decode steps. Compare token-by-token output against the replicated baseline. They should match within the existing `1-prompt fp-noise tolerance`.

### Phase 4: Shared expert and router under partitioning

**Goal**: shared expert and router are now naturally per-rank. Each rank computes them only on its owned `[G_local, H]`, and the AG that produces the post-mamba global view also produces what they need.

**Changes**:
- `moe_layer.py`: under `inference_partitioned_state=True`, the shared expert input is the rank's owned slice (`[G_local, H]`). The combine path's existing AR (Opt B-1's AR-fold) sums each rank's contribution into the right global rows, exactly like in replication mode — but now without the redundant compute.
- Router: each rank runs the router only on its own `[G_local, H]`. Routing decisions are local to the owner, then the routing_map is sized at `[G_total, topk]` after the natural AG. (Side note: under replication, the router was technically running on the global view on every rank; under partitioning it runs on just the local slice.)

**Test**: the routing_map seen by the experts kernel must match the replicated-path map. Verify identical `routing_map` tensors (after AG) for the same inputs.

### Phase 5: Inter-segment global-view production

**Goal**: at the start of each attention-bounded segment, every rank starts with the global view `[G_total, H]` (because the previous segment ended with an AR-combine).

The previous segment's AR-combine already produces `[G_total, H]` on every rank, so this works as today. The mamba in_proj / out_proj AG happens *within* the segment, between mamba layers (or between mamba and MoE). The first mamba layer of a segment slices `[G_total, H]` to `[G_local, H]` using the owner map.

**Changes**: minor — make sure the `_segment_input_is_global` flag transitions correctly when entering/leaving a segment. Add an explicit "segment entry: slice to local" step at the first mamba layer.

### Phase 6: Prefill compatibility

**Goal**: prefill stays on the request's owner rank (no migration during prefill). After prefill completes, the request's mamba state is initialized on its owner; subsequent decode steps proceed as Phase 3.

**Changes**:
- `mamba_mixer.py`: prefill path runs as today — on the owner rank, with the owner's local slot. No changes required because prefill was always per-request anyway.
- Mixed prefill+decode steps: the decode-only Variant B fast path is conditional on prefill being absent (or running in a separate sub-step). The current `fast_path_active` gate suffices.

### Phase 7: Bench and correctness validation

**Bench plan**:
- EP=4, batch ∈ {1, 4, 8, 16, 64}: expected gain at b=4–16 from removing mamba redundancy.
- EP=8 (synthetic, if available): replication mode should regress vs partitioned at all batch sizes.
- Compare partitioned vs current v13 replication on the same config — partitioned should be ≥ v13 at all sizes, with bigger gain at b ≥ EP.

**Correctness plan**:
- `correctness_diff.py` against the baseline on 50 prompts, decode-only, OSL=128.
- Token-by-token match against replicated-mode v13 on the same prompts.
- Long-context test (e.g., 4096-token prefill, 64-token decode) to verify mamba state correctness over multiple decode steps.

## Effort estimate

| Phase | LOC | Risk | Time |
|---|---|---|---|
| 1: Coordinator single-dispatch | ~80 | low | 0.5 day |
| 2: Per-rank state cache | ~150 | medium (sizing, free-list) | 1 day |
| 3: Mamba forward per-rank + AG | ~200 | medium-high (correctness across layers) | 1.5 days |
| 4: Shared/router under partition | ~50 | low (mostly removing dead code) | 0.5 day |
| 5: Segment entry slicing | ~30 | low | 0.5 day |
| 6: Prefill compatibility | ~20 | low | 0.5 day |
| 7: Bench + correctness | — | medium (debugging fp-noise) | 1 day |
| **Total** | **~530 LOC** | — | **~5.5 days** |

## The hang we hit (and what's needed to ship Phase 3)

Phase 3 (per-rank mamba forward + NVLS AG + initial AG at HybridStack entry) was **implemented and runs cleanly through CUDA graph warmup**, but hangs at the first real serving request. Root cause:

**The hang is not in our code — it's in how the engine coordinates per-rank work counts.**

When `b=4` requests arrive simultaneously and the coordinator dispatches 1-per-rank, the bench works. When dispatch is uneven (e.g. 2 to rank 0, 1 to rank 1, 1 to rank 2, 0 to rank 3 — possible due to prefix-cache scoring tiebreaking), partitioned mamba's `multimem_all_gather` deadlocks because ranks declare different `[G_local, H]` shapes.

The real fix is **engine-level per-step workload synchronization**. Three concrete pieces, in priority order:

1. **Coordinator round-robin under `partitioned_state=True`** (~30 LOC, contained). Replace `get_best_data_parallel_rank` with strict round-robin within the replication group. Eliminates dispatch-time imbalance for steady state. Brittle if requests finish at different times mid-batch — that's why piece 2 is needed.

2. **Per-step `decode_req_count` synchronization** (~50 LOC). At engine step start, do a single `all_reduce_max(decode_req_count)`. Pad each rank's effective batch to the max via padding rows (zero hidden_states for padding) gated by `valid_tokens`. Eliminates *any* runtime imbalance. Cost: ~1-2μs per step for a one-int collective.

3. **Pre-allocate symm-mem buffers at engine init** (~20 LOC). Currently `ep_init_ag` and `ep_mamba_out` are lazily allocated on first call. The first call is during CUDA graph warmup, which works empirically but is risky. Move allocation to the engine's setup phase (alongside the dispatcher's `allocate_buffers`). Lazy allocation has been an empirical pain across other Variant-B paths too.

With those three pieces in place, Phase 3 (the in-tree mamba changes that we currently revert) ships cleanly and gives the EP-scaling wins documented above.

**What's actually committed today**: pieces of state migration that do *not* require per-step engine coordination —
- Coordinator's single-owner dispatch (mode flag + load-balanced path)
- Engine plumbing for the flag
- Context recognition + `fast_path_active` gate

The mamba forward + initial AG (Phase 3 proper) is documented as ready-to-implement in `mamba_mixer.py::_maybe_partitioned_forward` and `hybrid_block.py::_partitioned_initial_all_gather` (the design we wrote and reverted) — completes once the engine-level synchronization above is in place.

## Risks and mitigations

**Risk: mamba state slot exhaustion under load imbalance.**
A coordinator that biases toward prefix-cache-hit rank can fill one rank's cache while others are 25% empty.
*Mitigation*: overprovision local slots by 25–50% relative to `max_requests / EP`. Add a backpressure signal to the coordinator when a rank's free-slot count drops below a threshold.

**Risk: AG latency dominates at b<EP.**
At b=2 with EP=4, only 2 ranks have any work; the AG still runs across all 4. The AG cost is mostly latency (~5μs barrier), so the relative overhead is high.
*Mitigation*: at b<EP, fall back to replication for that step (the gate is per-step via `fast_path_active`). The crossover where partitioning wins vs replication is around b≈EP — handle both, pick at step setup.

**Risk: numerical noise from different summation order.**
Replicated path: each rank computes the same sum locally. Partitioned + AG path: sum order differs (depends on which rank produced which row). Fp-add is non-associative, so token-level outputs may differ by ~1 ULP per step, accumulating over decode.
*Mitigation*: `correctness_diff.py` against replicated-mode v13 token-by-token. If the diff exceeds the `1-prompt fp-noise tolerance` over more than ~1 in 50 prompts, accumulate in fp32 inside the AG (slower but eliminates the source).

**Risk: prefix caching coordination breaks.**
The coordinator currently picks owner via prefix-cache affinity. With state migration, the per-prefix routing must be deterministic by request prefix to keep cache locality.
*Mitigation*: this is exactly how the coordinator's load balancer already works — no change. The prefix-cache mapping just becomes the *only* dispatch decision (instead of "primary, then broadcast to peers").

**Risk: speculative decoding edge cases.**
Spec decoding runs `K+1` tokens per request per step. The owner's per-rank slice has shape `[G_local × (K+1), H]`, and the AG must reassemble `[G_total × (K+1), H]`.
*Mitigation*: this mostly Just Works because the owner-set partitions cleanly along the leading dim. Verify with a spec-decoding bench at K=2 once Phase 3 lands.

## Future: dynamic rebalancing (out of scope for v1)

When request lifetimes differ wildly across ranks (some finish at step 100, some at step 1000), the cluster utilization eventually skews. v1 doesn't address this. v2 would add:

1. A periodic rebalancing trigger (e.g. when free-slot delta exceeds a threshold).
2. A request-state migration collective: pack `(conv_state, ssm_state)` for the migrating request, NVLS multicast-broadcast to the new owner, free the old slot.
3. Coordinator's request → rank map updated to point to the new owner.

The migration cost per request is ~80KB (one request's mamba state across all layers), so an NVLS broadcast is ~0.1μs. Cheap, but rebalancing logic adds complexity. Defer.

## Rollback / feature flag

The new mode is gated by `InferenceConfig.inference_partitioned_state`. Defaulting to `False` keeps current replicated-mode v13 behavior unchanged. Engines without the flag plumbed through fall back to the existing replication path. This is a strict superset of capabilities — old configs continue to work.
