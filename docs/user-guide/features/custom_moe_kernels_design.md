# Optimal custom-kernel design for Mamba+MoE inference (Triton)

This document is a from-scratch design for what an *optimal* set of
custom kernels would look like for low-batch inference of hybrid
Mamba+MoE models with EP (expert parallelism). Goal: eliminate the
AllGather-before-MoE without paying replicated-compute cost, fold the
dispatch/expert/combine pipeline into a small number of fused kernels,
and exploit NVSHMEM one-sided semantics on NVLink-domain hardware
(NVL72 / Hopper / Blackwell).

**Implementation language: Triton (Python DSL).** This matches the
existing codebase pattern: `megatron/core/inference/communication/
torch_symm_triton/collectives.py` and `variable_collectives.py`
already implement NVLS multimem kernels in Triton. We extend that
pattern to NVSHMEM one-sided ops.

---

## 1. Goals, non-goals, success metric

**Goals.**
- At low batch (b=1..8), eliminate the AG-before-MoE without
  introducing replicated compute upstream.
- Hide cross-rank data movement behind compute via fine-grain
  producer/consumer overlap inside a kernel and across consecutive
  layers.
- One CUDA-graph-capturable pipeline per (model, batch size).
- Match or beat the post-merge baseline at every batch size we care
  about; aim for ≥+15% over post-merge baseline at b=1.

**Non-goals.**
- Replacing the existing AGV+RSV path entirely. The new kernels are
  an alternative path selected when the workload favors them.
- Cross-node optimization. Single NVLink domain only.
- Training. Inference-only.

**Success metric.**
- Decode tok/s on nanov3 with EP=4 at b=1, 4, 16, 64. Headline:
  post-merge baseline at 137.3 / 596.9 / ~1700 / ~6000; target
  ≥+15% / +5% / parity / parity.
- Byte-identical or fp-noise-equivalent output vs the AGV path on
  `correctness_diff.py`'s 8 fixed prompts at greedy decoding.

---

## 2. The fundamental architectural choice

The AG-before-MoE exists because **the alltoall dispatch needs a
globally-known list of (token, destination-expert) pairs** before it
can route tokens to their owning experts. Three ways to get that:

1. **Replicate compute** — every rank ran the same router. Free
   global view. (Today's Variant B.)
2. **Local routing + explicit gather** — per-rank routing, then AG.
   (Today's baseline / PR #4258 AGV path.)
3. **Direct push** — each rank computes routing for ITS tokens, then
   *writes* tokens directly into the destination rank's expert input
   buffer via NVSHMEM put. No AG, no alltoall. **This is the
   custom-kernel path.**

---

## 3. Library / DSL choices

**Triton** — the only Python DSL with the right combination here:
- Native `tl.load` / `tl.store` with `mask=` for arbitrary memory
  patterns
- `tl.atomic_add`, `tl.atomic_cas` for the slot-reservation atomics
  on the destination rank's symm-mem counter
- `tl.async_copy` (Triton's async-copy primitive backed by Hopper
  TMA) for staged loads
- `tl.extra.cuda.libdevice` for low-level intrinsics
- Already used in the codebase for `multimem_*` kernels — same toolchain

**`torch._C._distributed_c10d._SymmetricMemory`** (PyTorch's symm-mem
API) — provides multicast pointers and direct peer-pointer access.
The existing `multimem_*` Triton kernels in `torch_symm_triton/`
read from these multicast pointers. We extend the same pattern to
issue **NVSHMEM put-equivalent** ops: write through the per-peer
pointers obtained from the symm-mem handle.

```python
# Existing pattern in collectives.py:
@triton.jit
def _multimem_kernel(out_ptr, in_ptr, multicast_ptr, ...):
    ...
    # multimem.st_release writes to ALL ranks via multicast pointer
    tl.extra.cuda.multimem_st(multicast_ptr + offset, val)
```

For our direct-push kernels, instead of multicast we use **per-peer
pointers** obtained from the symm-mem handle's `peer_buffer_at(rank)`
API, treated as ordinary device pointers. Triton can `tl.store` to
those pointers — that's the put.

**`torch.ops.cutlass`** for grouped GEMM. CUTLASS exposes
grouped-GEMM-with-Python-epilogue via `torch.ops.cutlass.grouped_mm`.
For the K2 expert kernel we use a custom Triton epilogue that does
the NVSHMEM atomic_add to combine buffers.

**`cuda.bindings.runtime`** (the official CUDA Python bindings) for
graph capture, stream creation, event recording.

**`cuda::pipeline` equivalents in Triton** via Triton's `tl.cuda
.async_copy` and `tl.cuda.barrier_arrive` (experimental in newer
Triton releases).

---

## 4. The kernel suite

Five custom Triton kernels:

| # | kernel name                       | replaces                                              | new behavior                                              |
|---|-----------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| K1| `fused_route_and_push`            | router + AG + alltoall(dispatch) + permute            | local router + direct symm-mem put to peer buffers        |
| K2| `nvshmem_grouped_moe_gemm`        | dispatch_postprocess + experts + combine_preprocess   | grouped GEMM + NVSHMEM atomic-add into combine buffers     |
| K3| `multimem_combine_reduce`         | combine alltoall + RS / AR                            | (rolled into K2's atomic_add — no separate kernel needed) |
| K4| `mamba_dispatch_overlap`          | mamba SSM + (gap) + K1                                | mamba SSM streams output into K1 input via shared buffer  |
| K5| `layer_to_layer_pipeline`         | sequential layers                                     | next layer's K4 starts on per-token completion signal     |

---

### 4.1 K1: `fused_route_and_push`

**Purpose.** Take this rank's local tokens, compute their top-k
expert assignments, and push each token directly to the rank that
owns each of its assigned experts.

**Symmetric memory layout** (allocated once at model init via
`SymmetricMemoryManager.get_buffer`):

```
expert_input_buffer:   bf16  [num_local_experts, max_per_expert, H]
expert_input_probs:    f32   [num_local_experts, max_per_expert]
expert_input_src_meta: int32 [num_local_experts, max_per_expert, 2]   # (src_rank, src_token_id)
expert_slot_counters:  int32 [num_local_experts]                       # atomics
```

**Triton kernel.**

```python
import triton
import triton.language as tl

@triton.jit
def fused_route_and_push_kernel(
    # Local inputs (this rank only)
    my_hidden_ptr,           # *bf16  [G_local, H]
    my_router_logits_ptr,    # *f32   [G_local, num_experts]
    G_local: tl.constexpr,
    H: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    MAX_PER_EXPERT: tl.constexpr,
    EP_SIZE: tl.constexpr,
    MY_RANK: tl.constexpr,

    # Per-peer pointers obtained from symm-mem handle.peer_buffer_at(rank).
    # Passed as flat int64 arrays of length EP_SIZE; Triton casts to typed
    # pointers via tl.cast / tl.advance.
    peer_expert_buf_ptrs,    # *int64 [EP_SIZE]
    peer_probs_ptrs,         # *int64 [EP_SIZE]
    peer_src_meta_ptrs,      # *int64 [EP_SIZE]
    peer_slot_ctr_ptrs,      # *int64 [EP_SIZE]
):
    # One program instance per local token.
    tok = tl.program_id(0)
    if tok >= G_local:
        return

    # Load my hidden state once into registers via TMA-style async copy.
    h_off = tok * H + tl.arange(0, H)
    hidden = tl.load(my_hidden_ptr + h_off)            # [H] bf16

    # Compute top-k from router logits. For NUM_EXPERTS small (~32 in
    # nanov3) we can just sort in-register or use repeated argmax.
    logits_off = tok * NUM_EXPERTS + tl.arange(0, NUM_EXPERTS)
    logits = tl.load(my_router_logits_ptr + logits_off)   # [NUM_EXPERTS] f32

    # In-place top-k via softmax + iterative argmax (small NUM_EXPERTS).
    # ... (Triton implementation of top-k selection on register array)
    top_k_experts, top_k_probs = top_k_select(logits, TOP_K)

    # For each chosen expert: route to its rank.
    for k in tl.static_range(TOP_K):
        dst_expert = top_k_experts[k]
        dst_rank = dst_expert // NUM_LOCAL_EXPERTS
        local_e = dst_expert % NUM_LOCAL_EXPERTS

        # Reserve a slot on dst_rank's counter for local_e.
        dst_ctr_ptr = tl.load(peer_slot_ctr_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
        slot = tl.atomic_add(dst_ctr_ptr + local_e, 1, sem='relaxed', scope='sys')

        # Compute the destination row offset.
        dst_offset = local_e * MAX_PER_EXPERT + slot

        # Direct put of the token into dst_rank's expert input buffer.
        dst_buf = tl.load(peer_expert_buf_ptrs + dst_rank).to(tl.pointer_type(tl.bfloat16))
        tl.store(dst_buf + dst_offset * H + tl.arange(0, H), hidden, sem='release', scope='sys')

        # Put the routing weight.
        dst_probs = tl.load(peer_probs_ptrs + dst_rank).to(tl.pointer_type(tl.float32))
        tl.store(dst_probs + dst_offset, top_k_probs[k], sem='release', scope='sys')

        # Put source metadata so K2 knows where to atomic_add the combine result.
        dst_meta = tl.load(peer_src_meta_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
        tl.store(dst_meta + dst_offset * 2 + 0, MY_RANK, sem='release', scope='sys')
        tl.store(dst_meta + dst_offset * 2 + 1, tok, sem='release', scope='sys')
```

**Host-side launch wrapper.**

```python
def fused_route_and_push(
    my_hidden: torch.Tensor,
    my_router_logits: torch.Tensor,
    expert_input_buffer: SymmemTensor,    # symm-mem
    expert_input_probs: SymmemTensor,
    expert_input_src_meta: SymmemTensor,
    expert_slot_counters: SymmemTensor,
    ep_group: ProcessGroup,
):
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    handle = expert_input_buffer.symm_mem_handle

    # Build per-peer pointer arrays. Each entry is the int64 address of
    # the buffer ON THAT RANK, accessible via NVLink P2P.
    peer_buf_ptrs = torch.tensor(
        [handle.peer_buffer_at(r).data_ptr() for r in range(ep_size)],
        dtype=torch.int64, device='cuda',
    )
    peer_probs_ptrs = torch.tensor(... , dtype=torch.int64, device='cuda')
    peer_meta_ptrs  = torch.tensor(... , dtype=torch.int64, device='cuda')
    peer_ctr_ptrs   = torch.tensor(... , dtype=torch.int64, device='cuda')

    # Reset slot counters at the start of each step.
    expert_slot_counters.zero_()

    # Launch one program per local token.
    grid = (my_hidden.shape[0],)
    fused_route_and_push_kernel[grid](
        my_hidden, my_router_logits,
        my_hidden.shape[0], my_hidden.shape[1],
        NUM_EXPERTS=NUM_EXPERTS,
        NUM_LOCAL_EXPERTS=NUM_LOCAL_EXPERTS,
        TOP_K=TOP_K,
        MAX_PER_EXPERT=MAX_PER_EXPERT,
        EP_SIZE=ep_size,
        MY_RANK=my_rank,
        peer_expert_buf_ptrs=peer_buf_ptrs,
        peer_probs_ptrs=peer_probs_ptrs,
        peer_src_meta_ptrs=peer_meta_ptrs,
        peer_slot_ctr_ptrs=peer_ctr_ptrs,
    )

    # System-scope fence so K2 sees the writes.
    handle.barrier()
```

**Why fast.**
- No AG: tokens move directly source→destination via NVLink P2P.
- The atomic_add for slot reservation happens on the destination's
  symm-mem counter, hardware-accelerated.
- TMA-style async loads via `tl.async_copy` keep the kernel busy
  while NVLink writes are in flight.

**Bytes on the wire** (top_k=6, EP=4): each rank sends ~`G_local ×
top_k × H` bytes ≈ `1.5 G × H`, vs AG of `0.75 G × H`. K1 trades
bytes for *latency* — the AG had a structural blocking sync that
vanishes with one-sided puts.

---

### 4.2 K2: `nvshmem_grouped_moe_gemm`

**Purpose.** Read the tokens K1 just delivered, run them through
local experts, apply routing weights, and atomic_add weighted
contributions back to source ranks' combine buffers.

**Approach.** Two-stage Triton kernel:

```python
@triton.jit
def grouped_expert_gemm_kernel(
    # Inputs (local symm-mem buffers populated by K1)
    expert_input_ptr,       # *bf16  [num_local_experts, max_per_expert, H]
    expert_input_probs_ptr, # *f32   [num_local_experts, max_per_expert]
    expert_input_src_meta_ptr,  # *int32 [num_local_experts, max_per_expert, 2]
    expert_slot_counters_ptr,   # *int32 [num_local_experts]

    # Expert weights (replicated on every rank? no — local experts only)
    fc1_weight_ptr,         # *bf16  [num_local_experts, H, intermediate]
    fc2_weight_ptr,         # *bf16  [num_local_experts, intermediate, H]

    # Output: peer combine buffers (symm-mem) for atomic_add
    peer_combine_buf_ptrs,  # *int64 [EP_SIZE]
    peer_combine_visit_ptrs,# *int64 [EP_SIZE]

    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_LOCAL_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
):
    # One program per (expert, M-block).
    expert_id = tl.program_id(0)
    m_block = tl.program_id(1)

    # How many tokens for this expert this step?
    n_tokens = tl.load(expert_slot_counters_ptr + expert_id)
    if m_block * BLOCK_M >= n_tokens:
        return

    # Load M-block of expert input.
    m_off = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < n_tokens
    input_offset = expert_id * MAX_PER_EXPERT * H + m_off[:, None] * H + tl.arange(0, H)[None, :]
    x = tl.load(expert_input_ptr + input_offset, mask=m_mask[:, None])  # [BLOCK_M, H]

    # Load routing weights for these tokens.
    weights = tl.load(expert_input_probs_ptr + expert_id * MAX_PER_EXPERT + m_off, mask=m_mask)

    # Load source metadata.
    src_rank = tl.load(expert_input_src_meta_ptr + ... + 0, mask=m_mask)
    src_token = tl.load(expert_input_src_meta_ptr + ... + 1, mask=m_mask)

    # Two-stage GEMM with SwiGLU activation in between.
    # Stage 1: x @ fc1[expert] → [BLOCK_M, INTERMEDIATE]
    #   (use Triton's matmul, K-loop with tl.dot, accumulate in f32)
    # Activation: SwiGLU
    # Stage 2: act(stage1) @ fc2[expert] → [BLOCK_M, H]
    #   (apply weight in epilogue: out *= weights[:, None])
    out = ... # [BLOCK_M, H] bf16 with weighted output

    # Atomic-add into the source rank's combine buffer at the source
    # token's row. Each token's top_k contributions accumulate via
    # these atomics — no cross-rank reduction needed.
    for i in tl.static_range(BLOCK_M):
        if m_mask[i]:
            dst_buf = tl.load(peer_combine_buf_ptrs + src_rank[i]).to(tl.pointer_type(tl.bfloat16))
            for h in range(0, H, 8):
                tl.atomic_add(
                    dst_buf + src_token[i] * H + h + tl.arange(0, 8),
                    out[i, h:h+8],
                    sem='release', scope='sys'
                )
            # Mark this contribution as arrived for completion detection.
            visit_ptr = tl.load(peer_combine_visit_ptrs + src_rank[i]).to(tl.pointer_type(tl.int32))
            tl.atomic_add(visit_ptr + src_token[i], 1, sem='release', scope='sys')
```

**Why this works.** Each token has top_k = 6 contributions arriving
via atomic_add from up to 6 different (rank, expert) sources. The
atomic_add accumulates them lock-free. The `visit_count` is
incremented per arrival; when it reaches `top_k`, the token is fully
combined and ready for the next layer.

**Why bf16 atomic_add is OK.** Hopper supports `atomic_add` on bf16
via PTX `atom.global.add.relaxed.bf16` (or via fp32 staging — for
combine where we have at most 6 contributions per token, bf16 atomic
is precise enough).

---

### 4.3 K3 — folded into K2

K2's atomic_add into the source rank's combine buffer IS the combine
reduction. No separate K3 kernel.

**Per-token completion detection.** The next layer's K1 spins (or
schedules) on `combine_visit_count[my_token] == top_k` for each of
its tokens before reading them. We implement this with a small
`completion_wait` Triton kernel that's part of K4.

---

### 4.4 K4: `mamba_dispatch_overlap`

**Purpose.** Mamba SSM is recurrent; K1 needs each token to push it.
Today these run sequentially. With this kernel they overlap inside a
single Triton kernel using producer/consumer warp specialization.

**Approach.** Persistent Triton kernel with warp specialization:

```python
@triton.jit
def mamba_dispatch_overlap_kernel(
    # Mamba inputs
    zxBCdt_ptr,              # *bf16  [G_local, intermediate]
    conv_state_ptr,          # *bf16  [num_slots, conv_dim]
    ssm_state_ptr,           # *bf16  [num_slots, num_heads, head_dim, state_dim]
    slot_indices_ptr,        # *int32 [G_local]

    # Outputs of mamba
    mamba_out_ptr,           # *bf16  [G_local, H]

    # ... K1 inputs/outputs as in fused_route_and_push_kernel ...
    my_router_logits_ptr,    # router will run on mamba_out
    peer_expert_buf_ptrs,
    peer_probs_ptrs,
    peer_src_meta_ptrs,
    peer_slot_ctr_ptrs,

    ring_buffer_depth: tl.constexpr,
    ...
):
    # Producer/consumer split via warp specialization.
    # Producer warps: run the SSM recurrence, write mamba_out token-by-
    #   token, raise per-token "ready" flags in shared memory.
    # Consumer warps: wait on flags, when ready compute router top_k
    #   for that token, do the K1 puts.

    if tl.consumer_warp():
        # Consumer side
        for tok in range(G_local):
            # Wait for producer to finish token `tok`.
            wait_token_ready(tok)
            # Read mamba_out[tok] from shared memory ring buffer.
            x = tl.load_smem(ring_buffer + (tok % ring_buffer_depth) * H)
            # Compute router output (top_k from logits) — same as K1.
            top_k_experts, top_k_probs = top_k_select(...)
            # Issue NVSHMEM puts to destination ranks (same as K1 body).
            for k in tl.static_range(TOP_K):
                ... (puts, atomics)
    else:
        # Producer side (mamba SSM)
        for tok in range(G_local):
            # SSM recurrence: y = SSM_step(x, conv_state, ssm_state)
            y = ssm_step(...)
            # Write into shared-memory ring buffer.
            tl.store_smem(ring_buffer + (tok % ring_buffer_depth) * H, y)
            # Also write to mamba_out global memory for downstream layers.
            tl.store(mamba_out_ptr + tok * H + ..., y)
            # Signal consumer.
            signal_token_ready(tok)
```

**Hopper-only feature: warp specialization** is supported in Triton
3.0+ via `@triton.jit(num_warps=...)` and explicit warp branching.
On H100/H200/B200 this maps to async TMA + tensor cores cleanly.

**Net effect.** For T tokens, sequential cost is `T × (t_ssm +
t_dispatch)`; overlapped is `T × max(t_ssm, t_dispatch) + small
startup`. At b=1 dispatch dominates; we recover ~half the dispatch
cost.

---

### 4.5 K5: `layer_to_layer_pipeline` (host-side stream orchestration)

This is **not** a Triton kernel — it's host-side Python code that
launches K4 of layer N+1 on a separate stream, waiting on per-token
completion events from layer N's K2.

```python
def run_decode_step_pipelined(layers, hidden, ...):
    streams = [torch.cuda.Stream() for _ in range(2)]
    events = [torch.cuda.Event() for _ in range(2)]

    for i, layer in enumerate(layers):
        s = streams[i % 2]
        with torch.cuda.stream(s):
            # Wait on previous layer's completion event for this token range
            if i > 0:
                s.wait_event(events[(i-1) % 2])

            # Launch K4 (mamba + dispatch fused) for this layer
            mamba_dispatch_overlap_launch(layer, hidden, ...)

            # Launch K2 (experts + atomic_add combine) for this layer
            grouped_expert_gemm_launch(layer, ...)

            # Record event when this layer's K2 is done
            events[i % 2].record(s)

    # Final sync
    streams[(len(layers)-1) % 2].synchronize()
```

CUDA graphs capture the entire above sequence (events, streams,
kernels) into a single graph that can be replayed per decode step.

---

## 5. End-to-end pipeline

For one decode step at b=1:

```
Stream 0:                                 Stream 1:

K4(layer 0): mamba SSM for tok T
             + push to peers via K1
   │
   ▼
K2(layer 0): grouped GEMM
             + atomic_add to combine_buffer
             + atomic_inc visit_count
   │
   ▼
[combine_visit_count[T] == top_k]
   │                                      ◄── stream 1's K4 of next
   ▼                                          decode step T+1
K4(layer 1): wait on visit_count;             can start as soon as
             read combine_buffer[T];          layer 0's combine done
             SSM + push                       on T from prev step
... layers 2..L-1
   │
   ▼
output sample → broadcast → next step
```

**Properties.**
- Zero AGs, zero RSes. NVSHMEM puts and atomics replace them.
- One symm-mem barrier per layer (K1 → K2 boundary).
- Per-token completion detection via `visit_count` atomic, no host
  collectives.
- Mamba/dispatch overlap inside K4. Layer-to-layer overlap via K5
  multi-stream.

---

## 6. CUDA-graph capture strategy

All Triton kernels above are launched via `@triton.jit` which
compiles to CUDA kernels — graph-capturable as long as we avoid:
- Dynamic shape changes (handled via per-batch graphs)
- Host-side control flow inside the captured region
- `cudaMemset` on tensors (use kernel-based zeroing instead)

Symm-mem allocations live outside the graph. Per-peer pointer arrays
are computed once at model init and stay constant per (graph, step).

For variable token counts, we capture one graph per padded batch
size (matching the existing AGV's per-cuda-graph strategy).

---

## 7. Failure modes and fallbacks

**Imbalanced routing.** If routing temporarily concentrates tokens
on one expert, slot counter overflow. Mitigation: 2× margin on
`MAX_PER_EXPERT` + an overflow-detect Triton kernel that sets a host
flag; on overflow we fall back to AGV+RSV for one step.

**Atomic contention.** At b ≥ 256 atomic_add at combine could
contend. Mitigation: warp-aggregated atomic_add, or fall back to
AGV+RSV at high batch (which wins anyway).

**Slow peer.** Per-token spin-waits could stall if one rank lags.
Mitigation: stream-level event watchdog → fall back to host-
synchronized AGV+RSV.

---

## 8. Testing strategy

- **Microbenchmarks** for each Triton kernel via the existing
  `inference-bench/` harness pattern. Measure bytes-on-the-wire and
  wall time vs AGV/RSV at b=1, 4, 16, 64, 256.
- **Unit tests** in `tests/unit_tests/inference/` for K1, K2, K4.
  Run with synthetic inputs; verify K1+K2+K3 combined matches the
  AGV+experts+RSV reference within bf16 fp-noise.
- **End-to-end correctness**: `correctness_diff.py` against post-
  merge baseline on 8 fixed greedy prompts.
- **Soak**: 24-hour bench at b=1 / 4 / 64 with no crashes / leaks /
  hangs.

---

## 9. Effort & risk estimate

| component                                  | effort     | risk      |
|--------------------------------------------|------------|-----------|
| K1 fused router-push (Triton)              | 3–4 wks    | medium    |
| K2 grouped GEMM with NVSHMEM atomic-add epilogue (Triton + CUTLASS) | 5–6 wks | high (Triton + CUTLASS interop) |
| K4 mamba/dispatch overlap (Triton warp specialization) | 3 wks | medium-high (warp spec is Triton 3.0+) |
| K5 cross-layer software pipeline (Python)  | 1–2 wks    | low       |
| Symm-mem allocation/management Python glue | 1 wk       | low       |
| CUDA-graph capture & fallback paths        | 2 wks      | medium    |
| Microbenchmarks + correctness harness      | 2 wks      | low       |
| Soak / production hardening                | 4 wks      | medium    |
| **Total**                                  | **21–25 wks** | overall medium-high |

For one engineer: ~5–6 months. For two engineers: ~3 months.

---

## 10. Where this lands relative to the merged dispatcher

PR #4258 (NVLS allgatherv) is the *first* generation of custom
inference dispatch kernels written in Triton. It made the AG fast
but kept the AGV → experts → RSV shape. The design above is the
*next* generation: kill AG entirely via direct push, fuse mamba into
the dispatch, pipeline across layers.

Concrete payoff over PR #4258 (estimated):

| batch | post-merge baseline | this design (estimated) | speedup |
|-------|---------------------|--------------------------|---------|
| 1     | 137.3 tok/s         | 165–180                  | ~+20–30% |
| 4     | 596.9 tok/s         | 660–720                  | ~+10–20% |
| 16    | ~1700               | ~1750–1900               | ~+0–10%  |
| 64    | ~6000               | ~6000                    | ~0%      |

Win flattens at high batch where bandwidth dominates and the AGV
bandwidth path is already optimal. The custom-kernel approach pays
for itself most strongly at low batch — the chat/agent serving case.

---

## 11. Why this is *not* on the critical path right now

PR #4258 already delivered ~+16% over old baseline at b=1 by making
AG fast. The custom-kernel design above is another 20–30% over that,
but takes 5–6 months and needs serious Triton + NVSHMEM expertise.
The right ordering:

1. Ship M1+M2+M3+M4 + the merge as the immediate win (already done
   in this branch).
2. Profile post-merge at b=1 to confirm where residual cost is. If
   dispatch+combine collective time is still ≥30% of step time, the
   custom-kernel work pays.
3. Spec K1 in detail, write a single-kernel prototype in Triton,
   microbenchmark vs AGV. If K1 alone wins ≥10% at b=1, commit to the
   full suite.

---

## Summary

Five Triton kernels using NVSHMEM-style direct push/atomic-add
cross-rank communication, CUTLASS for grouped GEMMs (called from
Triton), and `cuda::pipeline` + multi-stream software pipelining for
compute-comm overlap. All in Python, matching the existing
`torch_symm_triton/` pattern.

**Win sources:**
- **K1 + K2**: eliminate AG and RS via NVSHMEM one-sided ops fused
  into routing and expert kernels. Half the collective sync points
  per layer.
- **K4**: mamba SSM and dispatch overlap inside one kernel via warp
  specialization. Recovers ~half the dispatch latency at low batch.
- **K5**: adjacent layers overlap via per-token progress signals.
  Recovers serial layer-transition latency.

**Estimated payoff over post-merge baseline:** +20–30% at b=1,
diminishing to parity at b=64.

**Estimated effort:** 21–25 engineer-weeks (5–6 months for one
engineer, 3 months for a small team).

This is the "what if we go all the way" plan. It is not what we are
shipping today. Ship M1–M4 + the merge first, profile, then decide.
