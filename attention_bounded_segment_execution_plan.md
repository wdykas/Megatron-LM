# Attention-Bounded Segment Execution for Mamba+MoE Inference

## Goal

Implement an exact, non-approximate inference runtime optimization for hybrid Mamba+MoE models with sparse attention layers.

The core idea:

```text
Do not gather MoE outputs back to the original request/KV owner after every MoE layer.

Instead:
  - canonicalize only at attention layers;
  - between attention layers, run Mamba+MoE blocks inside a chosen execution island;
  - keep/move exact Mamba recurrent state to that island;
  - combine MoE outputs back to the segment island, not necessarily the original KV owner.
```

This is intended for models with many Mamba+MoE layers and relatively few attention layers, such as Nemotron-style hybrid MoE models.

The output must match the baseline model up to ordinary floating-point/kernel-order differences. No KV approximation, no Mamba-state approximation, no expert approximation.

---

## Key Idea

Standard inference does this:

```text
attention/KV owner
  -> Mamba
  -> MoE dispatch to expert ranks
  -> MoE combine back to original owner
  -> Mamba
  -> MoE dispatch
  -> MoE combine back to original owner
  -> ...
```

The proposed runtime does this:

```text
attention/KV owner
  -> enter Mamba/MoE segment island
  -> Mamba
  -> MoE dispatch
  -> combine back to segment island
  -> Mamba
  -> MoE dispatch
  -> combine back to segment island
  -> ...
  -> return to attention/KV owner only before next attention layer
```

This optimization should only be enabled for regions between attention layers. Attention remains anchored to the canonical KV owner.

---

## Theoretical Win Condition

For a segment between two attention layers:

```text
Attention -> [Mamba, MoE, Mamba, MoE, ...] -> Attention
```

The new runtime is better when:

```text
saved repeated MoE combine-back / dispatch cost
>
cost to enter/exit the segment island
+ Mamba recurrent state movement cost
+ extra scheduling/metadata overhead
```

More concretely, this is promising when:

```text
1. There are multiple MoE layers between attention layers.
2. Expert parallelism spans multiple GPUs/nodes.
3. MoE communication is a bottleneck.
4. Routing has locality across the segment.
5. The same request remains in the segment island for several layers/tokens.
6. Mamba state is cheaper to move/keep local than repeatedly returning hidden states to the original owner.
7. Decode is the bottleneck.
```

This is probably not worth it when:

```text
1. Every MoE is immediately followed by attention.
2. All EP traffic is NVSwitch-local and cheap.
3. Routing is uniformly random.
4. Dynamic ownership destroys batching/CUDA graph performance.
5. Mamba state movement is more expensive than the saved MoE traffic.
```

---

## Required Invariants

The implementation must preserve exact model semantics.

Do not:

```text
- drop KV pages;
- approximate attention;
- compress Mamba state;
- approximate Mamba state;
- use a different logical expert;
- skip experts;
- change top-k routing results;
- change sampling semantics;
- change request ordering in a way that changes RNG unless explicitly controlled.
```

Allowed differences:

```text
- normal floating-point ordering differences;
- different communication reduction order;
- different deterministic-but-not-bitwise-identical grouped GEMM layout;
- different CUDA kernel numerics within expected tolerance.
```

---

## Main Abstraction Change

Currently, MoE combine usually assumes:

```python
return_rank[token] = original_owner[token]
```

Change this to:

```python
return_rank[token] = next_owner[token]
```

where `next_owner` may be:

```text
- original request/KV owner, for baseline behavior;
- current segment island rank;
- next Mamba-state owner;
- chosen aggregation rank for the current segment;
- attention/KV owner, if the next layer is attention.
```

The first milestone should set:

```python
next_owner = original_owner
```

and reproduce baseline exactly. Then enable non-baseline destinations behind a feature flag.

---

## New Concepts to Add

### 1. Layer Regions / Attention-Bounded Segments

Parse the model layer list and identify regions:

```python
@dataclass
class Segment:
    start_layer: int
    end_layer: int
    layers: list[int]
    has_attention_boundary_before: bool
    has_attention_boundary_after: bool
```

Example:

```text
Attention
  Segment 0: Mamba, MoE, Mamba, MoE, Mamba, MoE
Attention
  Segment 1: Mamba, MoE, Mamba, MoE
Attention
```

Only enable dynamic ownership inside segments without attention.

At attention layers, force canonicalization:

```python
current_owner[token] = attention_owner[request_id]
```

---

### 2. Token Owner

Track where the current hidden state lives:

```python
token_owner[token_id] -> rank
```

For decode, there is often one active token per request, so request-level ownership may be enough:

```python
request_current_owner[request_id] -> rank
```

---

### 3. Attention / KV Owner

Keep KV ownership stable:

```python
attention_owner[request_id] -> rank
```

At attention layers:

```python
send hidden state to attention_owner[request_id]
run attention there
append/read KV there
```

Do not migrate KV for the first implementation.

---

### 4. Mamba State Owner

Track exact recurrent state ownership:

```python
mamba_state_owner[request_id, layer_id] -> rank
```

Before running a Mamba layer:

```python
ensure_mamba_state_local(request_id, layer_id, target_rank)
```

After running the Mamba layer:

```python
mamba_state_owner[request_id, layer_id] = target_rank
```

This is exact state movement, not approximation.

---

### 5. Segment Island

Choose an execution island for each request and segment:

```python
segment_owner[request_id, segment_id] -> rank or rank_group
```

For MVP, choose one rank per request per segment.

Later, choose a small rank group.

---

## Implementation Stages

### Stage 0: Instrumentation Only

Before changing execution, collect stats.

Add logging for:

```python
expert_counts[layer_id, expert_id]
source_to_expert_counts[layer_id, source_rank, expert_id]
expert_transition_counts[layer_id, expert_id, next_layer_expert_id]
moe_dispatch_bytes[layer_id]
moe_combine_bytes[layer_id]
moe_dispatch_time[layer_id]
moe_combine_time[layer_id]
mamba_state_bytes[layer_id]
attention_boundary_frequency
per_request_generation_length
per_request_segment_expert_signature
```

Also log per-layer owner assumptions:

```python
current_owner_before_layer
current_owner_after_layer
attention_owner
```

Output a JSON or tensor dump that can be analyzed offline.

Success criterion:

```text
Able to estimate whether segment execution could save communication before implementing it.
```

---

### Stage 1: Make MoE Combine Destination Configurable

Find the MoE inference token dispatcher/combine path.

Current logical behavior:

```python
dispatch hidden states to expert owners
run local experts
combine outputs back to source/original owners
unpermute to original layout
```

Change metadata to include:

```python
combine_dest_rank[token]
combine_dest_offset[token]
logical_token_id[token]
request_id[token]
```

Add config:

```yaml
inference_moe_combine_destination: original_owner | next_owner
```

Initially:

```python
combine_dest_rank = original_owner
```

so baseline is preserved.

Success criterion:

```text
With combine_destination=original_owner, outputs and performance match baseline.
```

---

### Stage 2: Explicit Token / Request Ownership Tracking

Add:

```python
current_owner[request_id]
attention_owner[request_id]
```

For baseline:

```python
current_owner[request_id] = attention_owner[request_id]
```

after every layer.

Then add an experimental path where after MoE:

```python
current_owner[request_id] = combine_dest_rank
```

but still use `original_owner` as the destination.

Success criterion:

```text
Ownership metadata tracks baseline behavior exactly without changing execution.
```

---

### Stage 3: Attention-Boundary Canonicalization

Add a function:

```python
canonicalize_to_attention_owner(request_ids, hidden_states)
```

This sends active hidden states to:

```python
attention_owner[request_id]
```

Call this before every attention layer.

For baseline, this should be a no-op because hidden states are already there.

Success criterion:

```text
Explicit canonicalization exists and baseline still works.
```

---

### Stage 4: Mamba State Ownership and Movement

Add a state directory:

```python
mamba_state_owner[request_id, layer_id]
```

Add APIs:

```python
ensure_mamba_state_local(request_id, layer_id, rank)
move_mamba_state(request_id, layer_id, src_rank, dst_rank)
batch_move_mamba_states(request_layer_pairs, dst_rank)
```

For the first implementation, state movement may be simple blocking point-to-point or all-to-all. Optimize later.

At Mamba layer execution:

```python
target_rank = current_owner[request_id]
ensure_mamba_state_local(request_id, layer_id, target_rank)
run_mamba_layer_on(target_rank)
mamba_state_owner[request_id, layer_id] = target_rank
```

Success criterion:

```text
Can move exact Mamba state and run a Mamba layer on a non-original owner.
```

---

### Stage 5: Static Segment Island Mode

Add config:

```yaml
enable_attention_bounded_segments: true
segment_owner_policy: original_owner | fixed_rank | local_expert_affinity
```

For first nontrivial test:

```python
segment_owner[request_id, segment_id] = fixed_rank
```

Execution:

```text
At segment start:
    move hidden to segment_owner
    current_owner = segment_owner

Inside segment:
    Mamba runs on current_owner
    MoE dispatches from current_owner
    MoE combines back to current_owner

At next attention:
    canonicalize hidden back to attention_owner
```

This already removes repeated MoE combine-to-original-owner inside the segment.

Success criterion:

```text
For a segment with multiple Mamba+MoE blocks, hidden state stays on segment_owner until attention.
Outputs match baseline within tolerance.
```

---

### Stage 6: Expert-Affinity Segment Owner Policy

Use logged routing stats to choose better segment owners.

For each segment, estimate:

```python
cost(request, segment_owner) =
    enter_cost(attention_owner, segment_owner)
  + sum_moe_layers expected_dispatch_combine_cost(segment_owner, experts_used)
  + mamba_state_movement_cost(segment_owner)
  + exit_cost(segment_owner, next_attention_owner)
```

Pick:

```python
segment_owner = argmin_rank cost(request, rank)
```

For online decode, use one of:

```text
1. Previous token's routing in this segment.
2. Prefix/prompt routing stats.
3. Router replay if available.
4. EMA of request's expert usage.
5. Static global expert hotness.
```

Start simple:

```python
segment_owner = rank with best local coverage of the segment's hottest experts
```

Success criterion:

```text
Segment owner policy reduces measured MoE communication or tail latency versus original_owner.
```

---

### Stage 7: Support Rank Groups / Execution Islands

Instead of one rank, choose a small rank group:

```python
segment_island[request_id, segment_id] -> list[rank]
```

Inside the island:

```text
- Mamba state lives on one rank or is sharded within the island.
- Experts are preferentially chosen from replicas inside the island.
- MoE combine returns to the island aggregation rank.
```

Config:

```yaml
segment_island_size: 1 | 2 | 4 | 8
prefer_same_node_island: true
```

Success criterion:

```text
Can keep segment execution mostly within a node/NVLink island and avoid cross-node traffic.
```

---

### Stage 8: Hot Expert Replication

Add exact redundant expert replicas for inference.

Maintain:

```python
expert_locations[layer_id, expert_id] -> list[rank]
```

At dispatch time, choose physical replica:

```python
replica = argmin(
    network_cost[current_owner, replica_rank]
  + queue_depth[replica_rank]
  + combine_cost[replica_rank, combine_dest_rank]
)
```

This does not change the logical expert. It only chooses among identical physical copies.

Add config:

```yaml
enable_expert_replicas: true
num_redundant_experts: N
expert_replica_policy: hottest | source_affinity | segment_affinity
```

Success criterion:

```text
Hot expert stragglers reduce without changing outputs beyond numeric tolerance.
```

---

## Correctness Tests

### 1. Baseline Equivalence

Run with all features enabled but policies set to baseline:

```yaml
combine_destination: original_owner
segment_owner_policy: original_owner
enable_expert_replicas: false
```

Expected:

```text
Exact same token ownership as baseline.
Same outputs within tolerance.
Same sampled tokens if deterministic RNG and kernels are used.
```

---

### 2. One-Segment Forced Island

Create a tiny model pattern:

```text
Attention -> Mamba -> MoE -> Mamba -> MoE -> Attention
```

Run:

```yaml
segment_owner_policy: fixed_rank
```

Check:

```text
Output logits match baseline within tolerance.
Mamba states match after canonicalization.
Request ownership returns to KV owner before attention.
```

---

### 3. Mamba State Movement Test

Move Mamba state from rank A to rank B and run one step.

Compare with baseline where state stayed on A.

Expected:

```text
Same output within tolerance.
Same final recurrent state within tolerance.
```

---

### 4. MoE Combine Destination Test

For top-k MoE, combine expert outputs on a non-original rank.

Compare to baseline.

Expected:

```text
Same combined hidden state within tolerance.
```

---

### 5. Attention Boundary Test

Force hidden state to live on a non-KV rank before attention.

Then canonicalize.

Expected:

```text
Attention sees identical hidden state and KV.
Output matches baseline.
```

---

## Benchmark Plan

Measure these separately:

```text
1. End-to-end decode tokens/sec.
2. p50/p95/p99 per-token latency.
3. p50/p95/p99 request completion time.
4. MoE dispatch time.
5. MoE combine time.
6. Mamba state movement time.
7. Attention canonicalization time.
8. Cross-node bytes.
9. NVLink-local bytes.
10. Expert load imbalance.
11. Segment owner churn.
12. CUDA graph hit rate.
```

Important comparisons:

```text
A. Baseline standard MoE.
B. Baseline + instrumentation only.
C. Static segment owner.
D. Expert-affinity segment owner.
E. Segment owner + hot expert replicas.
F. Segment owner + rank-group islands.
```

For RL, also measure:

```text
1. rollout batch wall time;
2. p95/p99 group completion time;
3. trainer idle time;
4. long-tail request completion time;
5. time until enough completed rollout groups are available for training.
```

---

## Expected Win Profile

This should help most when:

```text
- model has many Mamba+MoE blocks between attention layers;
- decode is the bottleneck;
- EP crosses nodes;
- top-k is large;
- expert routing is skewed or segment-local;
- the runtime currently combines every MoE output back to the KV owner;
- Mamba state is cheap enough to move or keep local.
```

It may not help when:

```text
- single-node only;
- attention appears after every MoE;
- MoE traffic is already latent-compressed and not the bottleneck;
- weight bandwidth dominates;
- CUDA graph fragmentation offsets communication savings;
- segment owner changes too frequently.
```

---

## Implementation Notes for Megatron/vLLM-Like Runtimes

Search the codebase for these concepts:

```text
MoE token dispatcher
MoE token combine
all-to-all dispatch
all-to-all combine
expert parallel group
dynamic inference context
Mamba inference state
Mamba state cache
KV cache owner
request scheduler
active request pool
CUDA graph inference path
```

The first code change should be surgical:

```text
Add explicit combine_dest_rank metadata to MoE combine.
```

Do not begin by rewriting attention or KV cache.

The second code change should be:

```text
Add current_owner/request_owner metadata and make attention canonicalization explicit.
```

The third code change should be:

```text
Allow Mamba recurrent state movement between owners.
```

Only after those work should segment placement policies be added.

---

## Suggested Feature Flags

Use flags so every component can be ablated:

```yaml
enable_dynamic_token_ownership: false
enable_attention_bounded_segments: false
enable_mamba_state_migration: false
enable_non_original_moe_combine: false
enable_segment_owner_policy: false
enable_expert_replica_selection: false

segment_owner_policy: original_owner
# choices:
#   original_owner
#   fixed_rank
#   same_node_as_attention_owner
#   hottest_expert_affinity
#   measured_cost_model

moe_combine_destination_policy: original_owner
# choices:
#   original_owner
#   current_segment_owner
#   next_mamba_owner
#   cost_model

attention_boundary_policy: force_canonicalize
```

---

## Minimal Viable Prototype

The MVP should implement only this:

```text
1. Identify attention-bounded segments.
2. Add combine_dest_rank to MoE combine.
3. Track current_owner and attention_owner.
4. Add explicit canonicalization before attention.
5. For a chosen segment, force current_owner to a fixed rank.
6. Move exact Mamba state to that rank.
7. Combine all MoE outputs in the segment back to that rank.
8. Return to attention_owner before attention.
9. Compare output and latency to baseline.
```

Do not implement hot expert replication in the MVP. That is stage two.

---

## Decode-Step Pseudocode

```python
def run_decode_step(requests):
    for layer in model.layers:
        if layer.type == "attention":
            # Attention is the hard boundary.
            canonicalize_hidden_to_attention_owner(requests)
            run_attention_layer(requests)
            for req in requests:
                req.current_owner = req.attention_owner

        elif layer.type == "mamba":
            for req in requests:
                target = req.current_owner
                ensure_mamba_state_local(req.id, layer.id, target)
            run_mamba_layer_on_current_owners(requests, layer)
            for req in requests:
                mamba_state_owner[req.id, layer.id] = req.current_owner

        elif layer.type == "moe":
            for req in requests:
                if inside_attention_bounded_segment(layer):
                    combine_dest = segment_owner[req.id, segment_id(layer)]
                else:
                    combine_dest = req.attention_owner

                req.next_owner = combine_dest

            routed = route_tokens(requests, layer)

            dispatch_plan = build_dispatch_plan(
                routed_tokens=routed,
                source_owner={req.id: req.current_owner for req in requests},
                expert_locations=expert_locations[layer.id],
                combine_dest={req.id: req.next_owner for req in requests},
            )

            outputs = moe_dispatch_compute_combine(dispatch_plan)

            for req in requests:
                req.current_owner = req.next_owner
                req.hidden = outputs[req.id]
```

---

## Cost-Model Pseudocode

```python
def choose_segment_owner(req, segment):
    candidates = available_ranks_or_islands()

    best_rank = None
    best_cost = float("inf")

    for rank in candidates:
        cost = 0.0

        # Enter segment.
        cost += network_cost(req.current_owner, rank) * hidden_bytes

        # Mamba state cost.
        for layer in segment.mamba_layers:
            src = mamba_state_owner[req.id, layer.id]
            if src != rank:
                cost += network_cost(src, rank) * mamba_state_bytes(layer)

        # MoE dispatch/combine cost from this rank.
        for layer in segment.moe_layers:
            predicted_experts = predict_experts(req, layer)
            for expert, weight in predicted_experts:
                replica_rank = choose_best_replica_for_cost_only(
                    expert,
                    source_rank=rank,
                    combine_dest=rank,
                )
                cost += weight * (
                    network_cost(rank, replica_rank) * routed_hidden_bytes(layer)
                    + network_cost(replica_rank, rank) * routed_hidden_bytes(layer)
                )

        # Exit segment to next attention owner.
        cost += network_cost(rank, req.attention_owner) * hidden_bytes

        if cost < best_cost:
            best_cost = cost
            best_rank = rank

    return best_rank
```

---

## Success Criteria

The implementation is successful if, for Mamba+MoE-heavy models:

```text
1. Logits match baseline within expected numeric tolerance.
2. Mamba recurrent states remain exact within tolerance.
3. Attention always runs with the correct KV state.
4. MoE combine-back traffic to original owner is reduced.
5. p95/p99 decode latency improves.
6. RL rollout long-tail completion time improves.
7. CUDA graph hit rate is not destroyed.
```

The most important benchmark for RL is:

```text
time for enough complete rollout groups to start training
```

not just raw tokens/sec.

---

## Final Design Summary

Implement **attention-bounded segment execution**:

```text
- KV stays anchored at attention owners.
- Tokens may move only inside non-attention Mamba+MoE segments.
- Mamba state moves exactly with the segment execution.
- MoE combine destination becomes configurable.
- At attention layers, hidden states canonicalize back to the KV owner.
- Later, add expert-affinity segment placement and hot expert replicas.
```

This is the most plausible exact optimization for Nemotron-style Mamba+MoE models because the architecture has many Mamba/MoE layers and relatively few attention layers.
