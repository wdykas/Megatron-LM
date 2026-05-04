# Decode-only Variant B: load-balanced prefill + replicated decode

This document explains the data-flow patterns at play in the Mamba+MoE
inference path: the **default (baseline) pattern**, the **always-on
Variant B pattern** (replicated compute), and the **decode-only Variant
B pattern** we are building. It explains *why* replication is required
to unlock the Variant B optimization, and where the design wins and
loses.

The setup discussed below assumes a single model copy with `EP=4 TP=1
PP=1 --sequence-parallel`, hybrid Mamba+MoE blocks (e.g. nanov3), and
inference under the dynamic-batching engine.

---

## 1. The MoE layer's two collectives

Every MoE layer in baseline mode does **two collectives** on the
expert-parallel (EP) group: one before the experts and one after.

```
  per-rank input              per-rank input
  [G/EP, hidden]              [G/EP, hidden]
        │                            │
        ▼                            ▼
   ┌──────────┐                 ┌──────────┐
   │ rank 0   │                 │ rank 1   │       (one rank per
   └──────────┘                 └──────────┘        EP shard)
        │                            │
        └─────────► AllGather ◄──────┘
                   (gather all to global view)
                          │
                          ▼
                   [G, hidden]              ← every rank now has
                          │                   the global view
                  alltoall dispatch
                  (route tokens to
                   rank holding the
                   chosen expert)
                          │
                  experts compute
                  (each rank: only
                   its local experts'
                   contributions)
                          │
                  alltoall combine
                          │
                          ▼
                   [G, hidden]              ← partial: only this
                          │                   rank's experts'
                          │                   contributions for
                          │                   each token
                          ▼
                   ReduceScatter
                   (sum partials AND
                    scatter back to
                    per-rank slices)
                          │
                          ▼
            [G/EP, hidden] per rank          ← per-rank slices
```

So baseline does **AllGather + ReduceScatter** = **2 collectives** per
MoE layer.

The reason for both: tokens enter sharded across the SP/EP dimension
(one slice per rank), the expert dispatch needs the *global* view of
tokens (because the chosen expert can be on any rank), and after the
experts run we want to scatter the result back to the per-rank slice
layout for the next layer.

---

## 2. Variant B's idea: skip the AllGather, replace RS with AR

**Variant B's win** is structural: it eliminates the AllGather *if* the
input to MoE is already in the global `[G, hidden]` form on every
rank, *and* it replaces ReduceScatter with AllReduce so the output
stays in `[G, hidden]` form on every rank — feeding the next MoE layer
which can also skip its AllGather.

```
  per-rank input              per-rank input
  [G, hidden] (full)          [G, hidden] (full)        ← KEY: every
        │                            │                    rank already
        │  ─── skip AG ───           │                    has the global
        │                            │                    view
        ▼                            ▼
                  alltoall dispatch
                  (each rank picks
                   tokens for its
                   local experts
                   from its [G,H])
                          │
                  experts compute
                          │
                  alltoall combine
                          │
                          ▼
                   [G, hidden]              ← partial per rank
                          │
                          ▼
                    AllReduce
                    (sum across EP,
                     stays full-shape
                     — no scatter)
                          │
                          ▼
                   [G, hidden] (full)        ← every rank has
                                               full reduced view,
                                               feeds next MoE's
                                               skipped AG
```

So Variant B does **only AllReduce** = **1 collective** per MoE layer.
At low batch where collective latency floors dominate (10–20 µs per
launch regardless of bytes), saving 1 collective × ~26 MoE layers
= ~250–500 µs per step is a *significant* fraction of a decode step.

---

## 3. Why replicated compute is required for Variant B

The "input is already global on every rank" property — which is what
lets the AllGather be skipped — has only **one** algorithmic source:

> **Replicated upstream compute.** Every rank ran the same mamba /
> router / in_proj on the same input, so they each independently
> arrived at the same `[G, hidden]`.

There is no third option:

| how to make input "already global"        | cost                  |
|--------------------------------------------|----------------------|
| 1. every rank computed it independently    | redundant compute    |
| 2. an explicit AllGather just happened     | the collective we're trying to skip |
| 3. an explicit broadcast from one rank     | another collective, same cost as (2) |

Option (1) is "replication." It's the only one that adds *no* extra
collective. So Variant B's collective-saving win is **inseparable
from compute replication** — they're tied algorithmically.

This is why current Variant B has every rank doing the same compute
on the same `[G, hidden]` input from end to end. The wasted compute
is the price paid to skip the AllGather.

---

## 4. The trade-off: when is replication a good deal?

Replication is **N× compute** to save **1 collective** per layer. The
math depends on whether compute is large or small relative to the
collective.

```
┌──────────────────┬─────────────────┬──────────────────┐
│                  │ prefill         │ decode           │
├──────────────────┼─────────────────┼──────────────────┤
│ tokens/request   │ many (ISL)      │ 1 per step       │
│ compute / step   │ LARGE           │ small            │
│ collective cost  │ AG/RS scale     │ AG/RS dominated  │
│ at b=1           │ with tokens     │ by latency floor │
├──────────────────┼─────────────────┼──────────────────┤
│ replication cost │ HUGE            │ tiny             │
│ AG savings       │ small fraction  │ big fraction     │
├──────────────────┼─────────────────┼──────────────────┤
│ verdict          │ BAD trade       │ GOOD trade       │
└──────────────────┴─────────────────┴──────────────────┘
```

So the optimal strategy is:

- **Prefill: load-balanced (no replication)**. Take the AG/RS pair —
  collectives are cheap relative to the heavy mamba/in_proj compute.
- **Decode: replicated**. Pay the small redundant compute, take the
  big collective savings.

This is the **decode-only Variant B** design.

---

## 5. The default (baseline) pattern in DP=1 EP=4

When the coordinator uses load-balanced submission (`replicate=False`),
each request is sent to **one** of the 4 EP ranks' engine processes.
The other 3 ranks have no work for that request and call
`dummy_forward` to keep the cooperative collectives in sync.

```
   coordinator
       │
   load-balance
       │
   ┌───┴────────────────┐
   │                    │
   ▼                    ▼
 engine on rank 0    engine on rank 1, 2, 3
 ────────────────    ────────────────────────
 has request X       no work
 schedules it        dummy_forward
 runs forward        (zero-tensor input
                      to keep AG/RS
                      shapes aligned)
       │                    │
       └──── cooperative collectives ────┘
       (AG before MoE concatenates per-rank
        slices into the global view; RS
        scatters back)
```

At b=4 with one request per rank: each rank does its own request's
prefill, no replication waste. AG-before-MoE concatenates the per-rank
slices into the global view, MoE does its thing, RS scatters back.
Two collectives per MoE layer, but no redundant compute.

---

## 6. The decode-only Variant B pattern we want

Per phase, the data flow looks different:

### Prefill phase (per request)

Each rank does *its own* requests' prefill. State for each request
lives on its **owner rank only**.

```
  request X  (load-balanced to rank 0)
       │
       ▼
   rank 0 prefill            ranks 1,2,3 dummy_forward
   ─────────────             ─────────────────────────
   mamba on X's tokens       (no real work)
   updates X's mamba state
   at slot S on rank 0
```

The AG/RS pair around MoE handles the per-rank-different prefill
inputs naturally. **No replication waste.**

### Boundary (M2 state migration)

When request X finishes its prefill, owner broadcasts its mamba state
at slot S to all peers in the EP group:

```
  rank 0 (owner)          rank 1, 2, 3 (peers)
  ──────────────          ────────────────────
  conv_state[:, S]   ──┐   (empty / stale at slot S)
  ssm_state[:, S]    ──┤
                       │
                  M2 broadcast
                       │
                       ▼
                  every rank has
                  conv_state[:, S]
                  ssm_state[:, S]
                  fully populated
```

This is what `migrate_mamba_state` does — one fused
broadcast across all per-layer state in a single collective.

### Decode phase

Now every rank has the request's mamba state. The decode token (sampled
from the prefill output) is broadcast to all ranks. From here on,
every rank runs the *same* compute on the *same* token:

```
  rank 0          rank 1          rank 2          rank 3
  ──────          ──────          ──────          ──────
  mamba on        mamba on        mamba on        mamba on
  token T         token T         token T         token T
  (replicated     (replicated     (replicated     (replicated
   compute)        compute)        compute)        compute)
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                          │
                  All 4 ranks have
                  identical [G, hidden]
                  input to MoE
                          │
                          ▼
                  SKIP AllGather
                  (already global)
                          │
                  alltoall dispatch
                  experts compute
                  alltoall combine
                          │
                          ▼
                  AllReduce (returns
                  [G, hidden] global)
                          │
                          ▼
                  next MoE layer's
                  input is [G, hidden]
                  → it also skips AG
```

This is **1 collective per MoE layer** (just the AR), at the cost of
~tiny redundant decode compute. **Big collective savings, small
compute waste.**

---

## 7. The architectural blocker that allgatherv removed

Before PR #4258 (`siddharth/all-gather-v-dispatcher`), the inference
dispatcher's AllGather kernel was **fixed-size**: every rank had to
contribute the same number of tokens. That forced one of two scheduling
patterns to be used end-to-end:

- Replicated: every rank has the full `[G, hidden]` (sizes equal,
  trivially)
- SP-sharded: every rank has `[G/EP, hidden]` (sizes equal because of
  the SP scatter)

What we **could not do**: have rank 0 contribute `[64, hidden]` and
rank 1 contribute `[0, hidden]`. The fixed-size AG kernel would crash.
This made "load-balanced prefill within an EP cluster" impossible at
the dispatcher level — and so the prefill→decode mode switch we want
was structurally blocked.

PR #4258 introduces `NVLSAllGatherVDispatcher`, which uses
`multimem_all_gatherv_3tensor` — variable-size all-gather. Per-rank
contributions can differ, no padding, no shape coordination. With it:

- Prefill can be load-balanced (each rank contributes its actual
  active token count, possibly 0 if it owns nothing this step)
- Decode can be replicated (every rank contributes `G` global tokens)
- The dispatcher works with both layouts, switchable per step

That's the unblock. The decode-only Variant B design becomes
implementable.

---

## 8. The full picture: load-balanced prefill + replicated decode

Putting it all together, here's the lifecycle of a single request in
the new design:

```
  t = 0  request X submitted to coordinator
         coordinator broadcasts metadata to all ranks
         (cheap, ZMQ-level), every rank pre-allocates
         the same mamba slot S deterministically.
         Only owner rank queues X for prefill compute.
            │
            ▼
  prefill steps (rank 0 only does work for X; others dummy_forward
  but contribute their own owned requests' work to the
  cooperative collectives)
            │
            │  rank 0:                 ranks 1-3:
            │    mamba(X)              (their own owned reqs)
            │    AG/RS at MoE          (cooperative)
            │    update slot S
            │
            ▼
  prefill complete for X (after last chunk)
            │
            ▼
  M2 broadcast: owner rank 0 sends state at slot S
                to all peers; M2 collective on EP group.
                After M2, every rank has X's state at slot S.
            │
            ▼
  promotion sync: all_gather "newly promoted" event;
                  every rank adds X to its decode queue
                  in canonical (sort) order.
            │
            ▼
  decode steps (every rank has X in its active context;
  replicated compute, Variant B fast-path engages):
            │
            │  every rank:
            │    mamba(X) on full [G, H]
            │    SKIP AG before MoE
            │    AR-instead-of-RS at combine
            │    every rank has next-layer input as [G, H]
            │    → next MoE skips its AG too
            │
            ▼
  request X finishes
  free its slot S on every rank
```

---

## 9. Why this is strictly better than the current Variant B

Current Variant B pays replication cost end-to-end:

```
  prefill: replicated     →  4× redundant LARGE compute (BAD)
  decode:  replicated     →  4× redundant TINY compute (FINE)
                          AG savings × 26 MoE layers (BIG WIN)
```

Net: low-batch decode wins big, prefill is wasteful, high-batch loses
because the replication waste scales with token count.

Decode-only Variant B:

```
  prefill: load-balanced  →  no replication waste (GOOD)
                          AG/RS cost = baseline (acceptable since
                                                 prefill collectives
                                                 are bandwidth-dominated)
  decode:  replicated     →  4× redundant TINY compute (FINE)
                          AG savings × 26 MoE layers (BIG WIN)
```

Net: best of both — prefill at baseline efficiency, decode at Variant
B efficiency. The mode switch happens once per request at its
prefill→decode boundary, and the cost is one M2 broadcast.

---

## 10. Where each component lives in the codebase

| component                                    | file                                                       |
|----------------------------------------------|------------------------------------------------------------|
| Per-step `fast_path_active` gate (M1)        | `megatron/core/inference/contexts/dynamic_context.py`      |
| `migrate_mamba_state` collective (M2)        | `megatron/core/inference/contexts/dynamic_context.py`      |
| Decode-only mode config + CLI (M3)           | `InferenceConfig.inference_decode_only_variant_b` + arg    |
| CUDA-stream prefetch on migration (M4)       | `migrate_mamba_state` accepts a stream argument            |
| Skip-AG / AR-instead-of-RS                   | `NVLSAllGatherVDispatcher._segment_input_is_global`        |
| Variable-size collectives (PR #4258)         | `megatron/core/inference/communication/torch_symm_triton/variable_collectives.py` |
| Engine ownership filter (M5a, in progress)   | `megatron/core/inference/engines/dynamic_engine.py`        |
| Promotion sync at boundary (M5b/c, in prog.) | `megatron/core/inference/engines/dynamic_engine.py`        |

---

## Summary

- The MoE layer needs the global token view to dispatch to experts. It
  gets there via either an AllGather (baseline) or by every rank
  having the global view already (Variant B's skip-AG trick).
- The "every rank has the global view" property requires replicated
  compute upstream. That's algorithmically inseparable.
- Replication is N× compute to save 1 collective per layer. Profitable
  at decode (small compute), unprofitable at prefill (large compute).
- The optimal design is **load-balanced prefill, replicated decode**.
  The mode switch happens at the prefill→decode boundary per request
  via M2 mamba-state migration.
- Allgatherv (PR #4258) removes the dispatcher's fixed-size shape
  requirement, making per-rank-different active contexts (needed
  during prefill) a first-class supported configuration.
- The full implementation is M1–M5: per-step gate (M1), state migration
  collective (M2), decode-only mode wiring (M3), prefetch (M4),
  ownership-filtered scheduling + promotion sync (M5).
