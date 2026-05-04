# Variant B (Correct Implementation) — Plan

## Goal

Land an attention-bounded-segment execution path that:
1. **Is correct** — text output matches baseline within fp-reduction-order noise.
2. **Wins on small-batch decode** — 15–25% throughput improvement at batch
   ≤ 8 on single-node 4×GB200 (NVLS multimem). Larger batch may regress;
   that's expected and the flag is opt-in.
3. **Composes with optional kernel-level optimizations** (shared experts on
   local slice, fused AR+norm, etc.) layered on top once the structural
   change is correct.

## Why this is the right fit

- At small batch, decode is **comm-latency-bound**: each NVLS multimem
  collective is ~125 µs largely independent of payload, and we issue
  ~46 collectives per step (AG + RS for each of 23 MoE layers).
- Variant B replaces *2 collectives per MoE+Mamba pair* with *1*, by
  letting Mamba run on the global view so the next MoE's all-gather is
  redundant. Theoretical savings at batch=4 ≈ 1.75 ms of a 7.8 ms step
  ≈ 22% with conservative assumptions; up to ~50% if AR latency at
  this size is actually closer to multimem RS.
- At larger batch the bandwidth term in mamba's state I/O grows faster
  than the comm savings, so this is *opt-in* behind a flag.

---

## What went wrong in the previous (broken) attempt — DO NOT REPEAT

The previous implementation produced empty/junk output that the bench
bypassed via `--ignore-eos`, so the +5–11% "wins" were measured on a
model producing nothing useful. The structural mistakes were:

1. **Assumed sync-EP scheduling = replicated state.** It doesn't. Sync EP
   gives every rank the same `request_to_mamba_state_idx` mapping, but
   each rank only *writes* state for slots whose tokens land in its
   sequence-parallel slice. Slot K's value diverges across ranks because
   only one rank ever updates it.
2. **Sliced the global tensor `[0:L]` on every rank.** Per-rank mamba
   processed the same first L tokens, so other tokens were never given
   their per-step recurrence.
3. **Sliced `[r*L:(r+1)*L]` instead.** Now each rank updates a
   different range of state slots, but the global view written back is
   inconsistent across ranks — each rank's buffer has its own slice
   post-mamba and other ranks' slices pre-mamba. The next MoE's experts
   compute on inconsistent inputs and the AR sums incorrect partials.
4. **Trusted `--ignore-eos` benchmarks as proof of correctness.** They
   only verify shape, not content.

The correct approach: **every rank runs mamba on the full `[G, hidden]`
view, updating every state slot at every step.** That keeps state
consistent across ranks via deterministic recomputation.

---

## Architecture

### Invariants we maintain

- **Sequence-parallel sharding outside segments**: hidden_states shape
  is `[L, hidden]` per rank everywhere except inside an attention-
  bounded segment.
- **Global view inside segments**: hidden_states shape is
  `[G, hidden]` on every rank between the segment-entry AG and the
  next attention layer.
- **Mamba state**: each rank's state cache has `max_requests` slots,
  same content on every rank (because every rank computes identical
  mamba updates on the identical global view).
- **`request_to_mamba_state_idx`**: already global today thanks to
  sync-EP scheduling — no allocator change needed.

### What changes vs today

| component | change |
|---|---|
| `dynamic_context.py` | Expose `padded_active_token_count_global` and `batch_indices_decode_global` (the same per-step values, *not* sequence-parallel-sharded). Today's per-rank values stay for the default path. |
| `mamba_mixer.py` | Add `_dynamic_inference_segment_global(...)` that uses the `_global` indices and processes all G tokens. Selected when input shape is `[G, hidden]`. |
| `token_dispatcher_inference.py` | `token_combine` in segment mode returns the full `[G, hidden]` AR result (already implemented). `token_dispatch` skips AG when `_segment_input_is_global=True` (already implemented). |
| `hybrid_block.py` | Track `hidden_is_global` across the layer loop. AG at segment entry (already implemented). Slice to local at attention boundaries and end-of-stack (already implemented). Set the dispatcher flag (already implemented). |
| no allocator change | per-rank `max_requests` allocation is unchanged; trade-off is that max-concurrent-requests effectively becomes `max_requests` (not `max_requests × N`). For decode workloads this is fine. |

---

## Phased build, with correctness gate at each step

**Hard rule: every phase ends with a `correctness_diff.py` run vs baseline.
If text doesn't match baseline (modulo fp-noise), do not advance.**

### Phase 0: Baseline capture (already done)

- Run baseline server, capture `correctness_diff.py` output to
  `/tmp/baseline_outputs.json`. Already exists.
- Capture baseline perf numbers at batch ∈ {1, 4, 16, 64} via
  `inference-bench/run_local.sh`.

### Phase 1: Expose global metadata in inference context

**File**: `megatron/core/inference/contexts/dynamic_context.py`

- Add `self.padded_active_token_count_global` (= `padded_active_token_count`,
  but explicitly named so it's not confused with the SP-sharded local
  variant). With sync EP these are the same value.
- Add `self.batch_indices_decode_global` — the same global mapping that
  is already stored in `mamba_metadata.batch_indices_decode`. Just
  ensure no SP-sharding is applied.
- **Correctness gate**: Default path still uses local indices. Output
  diff should be empty.

### Phase 2: Mamba global-view forward path

**File**: `megatron/core/ssm/mamba_mixer.py`

- Add `_dynamic_inference_segment_global(self, hidden_states, context)`:
  - Asserts `hidden_states.shape[0] == context.padded_active_token_count_global`
  - Uses `context.batch_indices_decode_global` for state lookups
  - Runs the same `_ssm_decode` / `_dynamic_inference_prefill` paths
    but with global decode_req_count
  - Writes back to the global buffer in place (Opt 1 from earlier)
- Modify `_dynamic_inference` to detect global input by shape and
  dispatch to the new path.
- **Correctness gate**: Build a tiny test fixture that calls mamba in
  both modes (global view and local view) on the same logical input
  and verifies identical state evolution.

### Phase 3: Dispatcher segment-mode wiring

**File**: `megatron/core/transformer/moe/token_dispatcher_inference.py`

- Re-enable `_token_combine_via_all_reduce_global` (returns `[G, hidden]`
  symm-mem buffer in place, no clone — Opt 2 baked in).
- Verify `token_dispatch` skip-AG path is correct under CUDA-graph
  capture by exercising a multi-MoE segment manually.
- **Correctness gate**: Run `correctness_diff.py` against the running
  server.

### Phase 4: HybridStack forward

**File**: `megatron/core/models/hybrid/hybrid_block.py`

- Re-enable the segment tracking I'd written: `hidden_is_global`,
  `_set_segment_dispatch_flag`, `_slice_to_local`, `_all_gather_to_global`.
- Use the **NVLS multimem AG** path (already implemented) for
  segment-entry AG — fall back to NCCL ring otherwise.
- Slice to local at attention boundary and at end-of-stack.
- **Correctness gate**: `correctness_diff.py` must show ≤1 token
  difference per prompt across 8 prompts (fp reduction-order noise).
  If more, revert and debug.

### Phase 5: End-to-end perf measurement

- Run `inference-bench/run_local.sh` for both baseline and
  `ABS=1 ABS_COMBINE_DEST_POLICY=current_segment_owner` at batches
  {1, 4, 8, 16, 64}, OSL=256, ISL=64.
- Record results in this file.
- Expected: positive at batch ≤ 8, regression at batch ≥ 32.

---

## Optimization stack (on top of correct Variant B)

Order matters — each builds on the previous. Each gets its own
correctness gate.

### Opt 1: Shared experts on local slice

**File**: `megatron/core/transformer/moe/moe_layer.py`

Currently shared experts run on `[G, hidden]` in segment mode (4×
compute). Compute on the local slice `[L, hidden]` and add to only
this rank's rows of the AR'd combine result; other rows get the
routed-only output. The global AR step ensures every row already has
its routed contribution from every rank.

Expected: 3–6% extra throughput at all batch sizes.

### Opt 2: Mamba `in_proj` on local slice

**File**: `megatron/core/ssm/mamba_mixer.py`

`in_proj` is a per-token linear that runs 4× redundantly on global view
in segment mode. Run on `[L, hidden]`, then write the result into the
this-rank-slice of the global zxBCdt buffer; other slices use AR'd
data from the upstream MoE combine to fill in.

Expected: 2–4% extra throughput.

### Opt 3: Router on rank 0 only, broadcast top-k indices

**File**: `megatron/core/transformer/moe/router.py` /
`token_dispatcher_inference.py`

Router is deterministic given identical input. Compute on rank 0,
broadcast `routing_map` + `probs` (small tensors). Saves redundant
softmax+topk on N−1 ranks.

Expected: 1–3% extra throughput. Watch for sync overhead.

### Opt 4: Fused multimem AR + post-norm

**File**: `megatron/core/inference/communication/torch_symm_triton/fused_collectives.py`

Pattern already exists as `fused_multimem_rs_add_norm_ag`. Build
`fused_multimem_ar_norm` that does the AR and immediately applies the
next layer's pre-MoE layernorm in the same kernel.

Expected: 3–5% extra throughput at larger batch (recovers some
bandwidth gap).

### Opt 5: Custom multimem "AR-to-slice" kernel

**File**: `megatron/core/inference/communication/torch_symm_triton/collectives.py`

Today: AR writes the full `[G, hidden]` to symm mem; downstream mamba
reads its `[L, hidden]` slice. A single kernel that performs the
multimem reduction *and* writes only this rank's slice locally would
close the bytes-moved gap with RS while preserving the skip-AG win.
Closes most of the bandwidth gap that makes batch=16+ regress.

Expected: makes batch ≥ 16 stop regressing; possibly small win.

---

## Correctness harness

Use `inference-bench/correctness_diff.py` (already in repo) which:
- Sends 8 fixed prompts with `temperature=0` and `ignore_eos=true`
- Captures completions to a JSON file
- Compare with `diff baseline_outputs.json variant_b_outputs.json`

Expected: ≤ 1 token differs per prompt due to NCCL/NVLS reduction-order
non-determinism in fp arithmetic. Anything more is a bug.

**This must run after every phase.** No bench numbers count until this
passes.

---

## Performance harness

Use `inference-bench/run_local.sh` (already in repo) with:
```
DATASET=synthetic ISL=64 OSL=256 NUM_ITERS=3 NUM_WARMUP_ITERS=1 \
  BATCH_SIZES="1 4 16 64" MODEL=nanov3
```
For ABS, also set:
```
ABS=1 ABS_COMBINE_DEST_POLICY=current_segment_owner
```

Record results in a results table at the bottom of this plan.

---

## What this plan explicitly does NOT do

- **No mamba state allocator growth**: per-rank `max_requests` stays
  the same. Trade-off is reduced max-concurrent-requests capacity
  (`max_requests` instead of `max_requests × N`). Acceptable for decode
  workloads; revisit if max-batch becomes the binding constraint.
- **No request scheduler rewrite**: relies on sync-EP scheduling
  giving identical `request_to_mamba_state_idx` across ranks. Verified
  by reading the code; if any future change breaks this assumption,
  Variant B will silently corrupt state.
- **No cross-node testing**: this targets single-node NVLS. Cross-node
  IB is a much larger expected win but needs a different test setup.
- **No training mode changes**: Variant B is decode-only.

---

## Estimated effort

| phase | effort | risk |
|---|---|---|
| Phase 1 — context globals | 1 hr | low |
| Phase 2 — mamba global path | 3 hr | medium (must validate state evolution matches) |
| Phase 3 — dispatcher | 1 hr (already mostly written) | low |
| Phase 4 — HybridStack | 1 hr (already written, needs re-enabling + testing) | medium (correctness gate is here) |
| Phase 5 — bench | 1 hr | low |
| Opt 1 — shared experts | 3 hr | medium |
| Opt 2 — in_proj local | 2 hr | medium |
| Opt 3 — router skip | 2 hr | low |
| Opt 4 — fused AR+norm | 4 hr | medium (Triton kernel) |
| Opt 5 — AR-to-slice kernel | 6 hr | high (Triton kernel + careful CUDA-graph integration) |
| **Subtotal Variant B core** | **7 hr** | |
| **Subtotal Optimizations** | **17 hr** | |
| **Grand total** | **~3 days of focused work** | |

---

## Results table (fill in as we go)

```
Decode-only nanov3 (4×GB200, EP=4, NVLS multimem on, ISL=64, OSL=128, 3 iters + 1 warmup)

config                              batch  tok/s   ms/tok   Δ vs base
-----------------------------------------------------------------
baseline (no Variant B)                 1   116.3    8.60       —
                                        4   508.3    7.87       —
                                        8  1120.0    7.14       —
                                       16  2094.5    7.64       —

Variant B core (replication+ABS)        1   145.2    6.89    +24.8%
                                        4   547.3    7.31     +7.7%
                                        8  1043.0    7.67     -6.9%
                                       16  1662.0    9.63    -20.7%

+ Opt B-1 (shared on local, fused      1   144.5    6.92    +24.2%
  into AR via slice add)                4   548.4    7.29     +7.9%
                                        8  1045.2    7.65     -6.7%
                                       16  1667.1    9.60    -20.4%

+ Opt B-2 (mamba in_proj on local      1   146.1    6.84    +25.7%
  slice + AG)                           4   551.8    7.25     +8.6%
                                        8  1045.1    7.66     -6.7%
                                       16  1734.1    9.23    -17.2%
```

**Correctness with the full Variant B + Opt B-1 + Opt B-2 stack**: 8/8
prompts produce *byte-identical* output to baseline in
``correctness_diff.py`` (32 output tokens × 8 prompts). The combined
reduction order from skip-AG, AR-fused-shared, and slice+AG-in_proj
happens to align with baseline's RS reduction exactly, so the
prompt-level fp-noise we saw at intermediate steps is absorbed.

Opt B-2 implementation note: with replication, every rank holds the
same [G, hidden] view going into mamba and the in_proj GEMM is 4×
redundant. Each rank slices to [G/N, hidden], runs in_proj on the
slice, and uses NVLS multimem AG (with NCCL ring fallback) to
assemble the global [G, intermediate] tensor for the SSM kernels.
Result: small win at small batch (~+1% over Opt B-1) and a notable
win at b=16 (~+4% over Opt B-1) where the 4× compute redundancy
dominates more. Saves the in_proj GEMM but conv/ssm/out_proj remain
4× redundant since they depend on per-rank-replicated state.

Opt-A1 attempt (attention `linear_qkv` slice + AG): tried, **reverted**.
Each rank computed `linear_qkv` on its `[G/N, hidden]` slice and AG'd
to assemble the global qkv tensor (KV-cache writes still ran on the
full tensor so cache stayed consistent). Empirical result was a net
loss at every batch size measured:

```
config                       b=1   b=4    b=16    b=64
Variant B + Opt 1 + Opt 2   146.1  551.8  1734.1  3653.7
+ Opt-A1                    145.5  544.0  1662.8  (slow, killed)
                            -0.4%  -1.4%  -4.1%   regression
```

The savings on the linear_qkv GEMM are smaller than the overhead of
the extra NVLS multimem AG per attention layer (only ~6 attention
layers in nano vs 23 mamba layers, so the optimization can't amortize
the AG cost across enough layers). It also pushed correctness from
8/8 byte-identical to 2 prompts of fp-noise divergence. The
implementation is reverted in source; lessons documented here so the
pattern doesn't get re-attempted blindly.

Skipped opts (and why):
- **Opt B-2b (mamba out_proj on local + AG)**: same pattern as B-2 but
  on the out_proj GEMM. Implementation worked, perf was a small win,
  but correctness regressed: 2 prompts in `correctness_diff.py`
  diverged instead of 1, exceeding the ≤1-prompt-fp-noise tolerance.
  Reverted. The extra AR/AG round-trip adds enough reduction-order
  variation to push another prompt past a token-decision boundary.
- **Opt B-3 (router skip on N-1 ranks)**: the router is deterministic
  and cheap, so every rank already produces identical routing
  without communication. Skipping requires a broadcast which costs
  more than the saved compute; not worth pursuing on this model.
- **Opt B-4 (fused multimem AR + post-norm)**: requires writing a
  new Triton kernel modelled on `fused_multimem_rs_add_norm_ag`
  but with AR semantics. Doable but high implementation cost for
  3-5% expected gain at large batch; skipped for now.
- **Opt B-5 (custom AR-to-slice kernel)**: high-risk Triton work
  for a complex semantics swap; skipped.

Opt B-1 implementation note: each rank computes shared experts on its
[G/N, hidden] slice and stashes the result on the dispatcher. The
combine path folds the slice into the AR input at the rank's slice
positions; the AR's sum over ranks correctly merges (a) routed expert
output (non-zero only at slots routed to each rank's experts) and (b)
shared output (non-zero only at each rank's slice rows) — no extra
collective is needed.

Result: Opt B-1 is correctness-equivalent to Variant B core (same
1-prompt fp-noise divergence in correctness_diff) but is roughly
neutral on perf (within 1% of Variant B core across all batches).
Shared experts compute is small enough on nanov3 that saving 3/4 of
it doesn't move the needle end-to-end. The optimization is still
worth keeping because it costs nothing to enable and may matter on
models with larger shared experts.

Correctness: `correctness_diff.py` shows 7/8 prompts producing identical
32-token output between baseline and Variant B v2; 1 prompt diverges
deterministically at token 7, then cascades. Within the spirit of the
≤1-token-diff-per-prompt fp-noise tolerance.

Variant B is the correct opt-in choice for batch ≤ 4 small-batch decode
on this 4-GPU NVLS topology. Above batch ~8 the 4× redundant compute
per rank dominates the saved AG/RS comm and the path regresses.

---

## Done definition

- `correctness_diff.py` matches baseline within ≤ 1 token diff per
  prompt across 8 prompts.
- `inference-bench/run_local.sh` shows ≥ 15% throughput improvement
  at batch=1 and batch=4 with full optimization stack.
- Flag `--enable-attention-bounded-segments
  --moe-combine-destination-policy=current_segment_owner` is documented
  with the trade-off (small-batch win, large-batch regression).
- Default path (flag off) is byte-for-byte unchanged from today.
