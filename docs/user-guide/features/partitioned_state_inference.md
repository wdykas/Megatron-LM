# Partitioned-state inference

This document describes the `--inference-partitioned-state` mode: how it
works, what it costs, what it buys, and the design trade-offs that fall
out of running per-rank state alongside CUDA Graphs and NVLS multimem
collectives.

## Background: three inference modes

Megatron's hybrid (Mamba + MoE + attention) inference engine supports
three coordinator/dispatch policies. The choice changes both per-rank
memory consumption and the shape of every collective at every layer.

| Mode | Coordinator policy | Per-rank state | Per-(M,E)-pair collectives |
|---|---|---|---|
| **Default DP** | DP-style load balancing across replication groups | ~1× | 2: AGV-V (dispatch) + RSV-V (combine) |
| **Replicated** (`--inference-replicate-requests`) | Broadcast each request to every rank in a replication group | EP× | 1: skip-AG (input already global) + multimem AR |
| **Partitioned** (`--inference-partitioned-state`) | Pin each request to one rank in its replication group | 1/EP× | 2: publish (per-rank → global) + multimem AR |

Replicated mode buys the v13 throughput win (skip the dispatch AG) at
the cost of duplicating mamba state on every rank. Partitioned mode is
the inverse trade — it saves the per-rank state at the cost of paying
for an explicit publish step before every MoE.

## What partitioned mode is for

Partitioned mode shines in deployments where:

- **Memory is the binding constraint.** With long-context decode and
  large mamba state, replicated mode multiplies state by `EP`. At
  EP=4 with 23 mamba layers, one rank holds 4× the conv + SSM cache it
  would otherwise need. Partitioned mode caps per-rank state at the
  number of requests assigned to that rank.
- **Multi-replica DP serving.** When `DP > EP × TP × PP` (multiple
  model copies behind one coordinator), default DP balancing might
  route a single request to multiple copies, causing state to
  proliferate across copies. Partitioned mode pins each request to one
  copy and keeps state local to that copy.
- **Predictable per-rank state caps.** Independent of load-balancer
  behavior, `max_requests` per rank is bounded by `total_requests / EP`.

Where it is *not* needed:

- **Single-replica deployments with one model copy.** Default DP
  already assigns one request to one rank per replication group, so
  per-rank state is the same as partitioned mode. The flag adds no
  memory savings — only the publish-step cost.

## How requests flow through the system

```
Coordinator                            EP rank 0       EP rank 1       EP rank 2       EP rank 3
───────────                            ─────────       ─────────       ─────────       ─────────
batch [r0, r1, r2, ...]
  │ pin each request to one rank
  ├─────► r0 ────────────────────────► [r0]
  ├─────► r1 ─────────────────────────────────────────► [r1]
  ├─────► r2 ──────────────────────────────────────────────────────────► [r2]
  └─────► r3 ─────────────────────────────────────────────────────────────────────────► [r3]
                                       │               │               │               │
                                       ▼               ▼               ▼               ▼
                                       embedding       embedding       embedding       embedding
                                       │               │               │               │
                                  ┌───►│ mamba_0       │ mamba_0       │ mamba_0       │ mamba_0
                                  │    │ (per-rank     │ (per-rank     │ (per-rank     │ (per-rank
                                  │    │  state)       │  state)       │  state)       │  state)
                                  │    │               │               │               │
                                  │    ▼               ▼               ▼               ▼
                                  │    [local, hidden] [local, hidden] [local, hidden] [local, hidden]
                                  │    │               │               │               │
                                  │    └──── publish_kernel (multimem.st) ─────────────┘
                                  │                    ▼
                                  │                  [G_total, hidden]   ◄── visible on every rank
                                  │                    │
                                  │                    ▼
                                  │                   MoE_0 (router + experts + AR combine)
                                  │                    │
                                  │                    ▼
                                  │                  [G_total, hidden] (post-AR)
                                  │                    │
                                  │             slice back to per-rank
                                  └────────────────────┘
                                       (alternate M, E for 23 pairs)
                                       │
                                       ▼
                                       output for the rank's pinned requests
```

The two state stores at each layer:

- **Mamba state** — conv1d + SSM caches, sized `[max_per_rank_requests,
  state_dim]`. In partitioned mode, only the requests *pinned to this
  rank* have entries here.
- **Symmetric memory buffers** — the AGV / publish / RSV staging. Sized
  `[per_rank_worst_case_token_count × ep_size, hidden]`, identical
  across modes (allocated by `NVLSAllGatherVDispatcher`).

## The publish step

Mamba is per-rank: each rank computes its own pinned tokens through
`in_proj → SSM scan → out_proj`, producing a local `[local_tokens,
hidden]` tensor. The next MoE layer needs the global view of all
ranks' tokens (because routing decisions and expert dispatch depend on
the full token set). So between mamba and MoE the partitioned path
inserts a *publish* kernel.

`hybrid_block.py:_partitioned_pre_moe_agv` calls
`multicast_publish_constexpr` (see
`fused_matmul_multicast.py`), which is a constexpr-specialized Triton
kernel that:

1. Reads each row of the per-rank `[local_tokens, hidden]` tensor in
   128-bit chunks.
2. Issues `multimem.st` to broadcast each chunk to all peers' copies of
   the global symmetric-memory buffer at offset
   `rank * local_tokens × hidden`.
3. Ends with a `symm_mem_sync` cross-rank barrier so that no peer
   reads the global buffer before all peers have finished writing.

After the publish kernel returns, every rank's local view of the
global buffer holds the full `[ep_size × local_tokens, hidden]`
tensor in compact prefix-sum layout (rank 0's contribution at rows
`[0, local_tokens)`, rank 1's at `[local_tokens, 2 × local_tokens)`,
etc.). Slicing to `[G_total = ep_size × local_tokens]` is critical —
otherwise the dispatcher sees the full per-rank-max-padded buffer and
its AR-combine does ~150× wasted work.

The slice-back pattern after MoE is the symmetric:
`hybrid_block.py:_partitioned_slice_to_local` returns `hidden_states[r
* local_tokens : (r + 1) * local_tokens]` for the next mamba layer.

## EP token-count synchronization

This is the load-bearing detail of how partitioned mode coexists with
CUDA Graphs.

CUDA Graphs are captured per `local_tokens` size. The engine maintains
a small set of pre-captured graphs (typically `[1, 2, 4, 8, 16, 32,
64, 128, 256, 624]` or similar) and at each step picks the graph
whose `local_tokens` slot ≥ the current rank's per-rank token count.

In default and replicated modes every rank trivially has the same
`local_tokens` because the dispatch policy is symmetric. In
partitioned mode it is not — different ranks can hold different
numbers of pinned requests, possibly with different sequence lengths
during prefill.

If we let each rank pick its own graph independently:

- Rank 0 has 3 tokens → picks graph[4]
- Rank 1 has 5 tokens → picks graph[8]
- Rank 2 has 1 token → picks graph[2]
- Rank 3 has 2 tokens → picks graph[2]

Now ranks 0 and 1 issue 4-token publish kernels, ranks 2 and 3 issue
2-token publish kernels. The captured graphs hold *different
sequences of NVLS multimem collectives*. The
`multimem.st`/`multimem.ld_reduce` instructions assume every peer
arrives at the matching slot at the matching time — when one rank's
graph has, say, an extra layer of collectives that another rank
skipped, the cross-rank symm-mem barrier deadlocks.

The current fix (`dynamic_context.py:1898`):

```python
sync_ep_token_counts = (
    self._nccl_ep_dispatcher
    or (self._inference_partitioned_state and not self.is_creating_cuda_graphs)
)
```

When the gate is True, `match_ep_token_counts` runs a tiny all-reduce-max
across the EP group on the per-rank token count. Every rank then picks
the graph for that max value. Padding is handled inside the AGV
kernel — CTAs whose `pid >= local_tokens` exit immediately, and the
multimem.st of padded rows writes garbage that is later masked out by
the dispatcher's slice-to-G_total.

The sync is gated off during graph *capture*
(`is_creating_cuda_graphs`) because the capture-time helper
`adjust_batch_dims_for_expert_parallelism` returns `None` (forcing
eager mode) whenever any rank has prefill, which would break the
capture-side assertion `_using_cuda_graph_this_step`. The sync is only
needed at inference replay time when ranks may have divergent
per-rank counts.

## Cost of EP-sync

The sync itself is a single 1-element int32 all-reduce-max. On B200
NVLink that is a sub-microsecond NCCL/multimem op. Within the
captured graph it is essentially free.

The *real* cost is that ranks who could have run with fewer tokens are
forced up to the EP-max, padding their per-rank work. For the publish
kernel that means more CTAs launched than strictly necessary; for the
NVLS AR over the full active region it means slightly more bytes
moved. Both are small at typical batch sizes (the kernels are
launch-overhead-bound, not compute- or bandwidth-bound).

## Can we remove the EP-sync requirement?

**Short answer: no, not without redesigning either the captured graphs
or the multimem collective API.** The sync exists to make every rank
pick the same captured graph; the captured graph commits ranks to a
specific NVLS collective sequence that requires symmetric per-rank
arrival. If we want each rank to dispatch its own graph independently,
the collectives inside those graphs must tolerate ranks who never
issue a particular barrier slot — and `multimem.st` /
`multimem.ld_reduce` semantics with asymmetric per-rank participation
are undefined (and in practice deadlock).

Three plausible designs that would let ranks run different graphs:

### 1. Per-rank padded graphs at a common max

This is the *current* design dressed up. Drop the runtime all-reduce
and replace it with a *static* per-graph maximum: every rank always
picks the graph whose local_tokens slot is ≥ its actual count, but
the chosen slot is pulled from a fixed table keyed on `(global_batch,
ep_rank)`. This still requires every rank to agree on a single
common slot per step.

Net effect on the EP-sync collective: gone.
Net effect on padding: same as today.
Net effect on captured graph count: unchanged.
Engineering effort: low. Useful if profiling shows the all-reduce-max
is non-trivial, which today it is not.

### 2. Asymmetric NVLS collectives

A `multimem.ld_reduce` variant whose per-rank `numel` argument can
differ — meaning rank A reads `numel_A` elements while rank B reads
`numel_B`. This would require a kernel-level guard that exits CTAs on
a per-rank basis without participating in the cross-rank handshake
slot.

PTX `multimem.ld_reduce` and `multimem.st` do not provide this.
Building it ourselves means hand-rolling a barrier where each rank
explicitly signals which slots it will and will not participate in,
and every other rank waits only for the slots that are advertised.
Doable but invasive: every NVLS kernel
(`multimem_all_gather_v`, `multimem_reduce_scatter_v`,
`multimem_all_reduce`, the v33/v34 fused kernels) needs the new barrier
protocol.

Net effect: each rank dispatches its own graph. No padding. No
EP-sync collective.
Engineering effort: high (rewrite of `barrier.py:symm_mem_sync` and
every kernel that uses it). Correctness risk is also high — the
asymmetric-arrival barrier is a new primitive that must be carefully
designed and tested.

### 3. Single dynamic graph with conditional execution

CUDA Graphs in CUDA 12+ support conditional nodes (`cudaGraphAddNode`
with `CU_GRAPH_NODE_TYPE_CONDITIONAL`) that branch at replay time
based on a value in symmetric memory. In principle every rank
dispatches the *same* graph, but the kernel grids inside react to
each rank's `local_tokens` value at replay time.

This is conceptually clean — every rank goes through the same node
sequence so collectives stay symmetric — but PyTorch's CUDA Graph
capture does not yet expose conditional nodes through `torch.cuda.
graph()`. We would have to drop down to raw `cuda.driver.Graph` and
reimplement the inference-engine capture path.

Engineering effort: very high. Probably not worth it given how cheap
the EP-sync currently is.

### Recommendation

Keep the EP-sync as is. The all-reduce-max is sub-microsecond and
within the captured graph essentially free; the padding is small at
typical batch sizes; and the alternatives all require either
significant kernel rewrites (option 2) or out-of-tree CUDA-Graph work
(option 3). If a future profile shows the sync or its padding to be
non-trivial, option 1 is the cheap incremental step (bake the max
slot into a static per-step lookup table) and option 2 is the only
real fix.

## Performance summary

At nano (23-layer hybrid, EP=4) with `--inference-partitioned-state`:

| Batch | Throughput | Per-step time | Per-rank state |
|---|---|---|---|
| 4 | ~485 tok/s | 8.3 ms | 1× (1 request per rank) |
| 16 | ~1480 tok/s | 10.7 ms | 1× (4 requests per rank) |
| 32 | ~2670 tok/s | 12.0 ms | 1× (8 requests per rank) |

vs replicated mode (b=4): ~566 tok/s, EP× state. The replication win
is from skipping the publish step, not from any compute saving.
Partitioned mode chooses the symmetric trade-off: pay one extra
kernel per layer to keep per-rank state low.

The v33 / v34 fused-kernel optimizations
(`ABS_FUSED_AR_RESIDUAL[, _NORM]`) are *partitioned-mode only* — they
fold the post-MoE `bias_dropout_add` and the next layer's
`input_layernorm` into the same Triton kernel as the AR combine,
saving 1-2 kernel launches per (M,E) pair. Microbench-confirmed; e2e
gain at b=16 is ~+1-2% across runs.

## Files

The partitioned-mode plumbing lives in:

- `megatron/core/inference/config.py` — flag definition
- `megatron/training/arguments.py` — CLI plumbing
- `megatron/core/inference/contexts/dynamic_context.py` —
  EP-sync gate (line ~1906) and partitioned-mode bookkeeping
- `megatron/core/inference/data_parallel_inference_coordinator.py` —
  request pinning policy
- `megatron/core/models/hybrid/hybrid_block.py` —
  `_partitioned_pre_moe_agv` (publish call) and
  `_partitioned_slice_to_local` (slice-back)
- `megatron/core/inference/communication/torch_symm_triton/fused_matmul_multicast.py` —
  the publish kernel itself

The session-added optimizations on top of partitioned mode:

- `fused_ar_residual.py` — v33/v34 fused AR + residual + (optional)
  norm kernels
- Hooks in `transformer_layer.py` and `mamba_layer.py` for skip-norm
- Pre-link pass in `hybrid_block.py` that stashes next-layer norm
  weights on each MoE TransformerLayer

## Reproducer

```bash
EXP_NAME=partitioned MODEL=nanov3 BATCH_SIZES=16 OSL=64 \
  ABS_PARTITIONED_STATE=1 \
  NUM_ITERS=10 NUM_WARMUP_ITERS=3 \
  bash inference-bench/run_local.sh
```

With v33 enabled (recommended at b≥16):

```bash
EXP_NAME=partitioned_v33 MODEL=nanov3 BATCH_SIZES=16 OSL=64 \
  ABS_PARTITIONED_STATE=1 \
  ABS_FUSED_AR_RESIDUAL=1 \
  NUM_ITERS=10 NUM_WARMUP_ITERS=3 \
  bash inference-bench/run_local.sh
```

Do not pair with `--inference-replicate-requests` (mutually exclusive
at the coordinator) or with `--inference-decode-only-variant-b`.
