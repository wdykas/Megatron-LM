# Dynamic-Inference Prefix-Cache Audit and Fixes

## Scope and branch layout

This audit covers dynamic-inference KV prefix caching and the durable Mamba
state cache used by hybrid models such as Nemotron.

The changes are split into two independently reviewable branches:

- `fix/prefix-cache-resume-accounting` (`9cc5ff890`) contains the focused
  paused-request resume-accounting fix and its CPU regressions.
- `fix/prefix-cache-audit-followups` contains the other correctness,
  availability, and performance fixes described below. It is based directly on
  `main` and does not include `9cc5ff890`.

The changes were validated with CPU allocator/context tests and synthetic
benchmarks. An end-to-end Nemotron GPU run was prepared, but could not be
started from the audit host because it had no local Slurm commands and the
configured submit proxy rejected SSH authentication.

## Focused resume-accounting fix

### Problem

`DynamicInferenceContext.resume_paused_requests()` constrained a count of
resumable requests using `KVBlockAllocator.total_avail`, which is a count of raw
free blocks:

```python
resume_request_count = min(requests_that_fit, allocator.total_avail)
```

These quantities are not interchangeable. A paused request may need no new
block, and LRU mode may allocate by evicting an unpinned cached block even when
`total_avail == 0`. The old calculation could therefore strand valid requests.

The original upstream fix also summed each paused request's full block count
when checking the active partition. That double-counted blocks shared with an
active request or another request in the same resume batch.

### Change

- [`DynamicInferenceContext.resume_paused_requests()`](../megatron/core/inference/contexts/dynamic_context.py)
  now separates request/token limits, unique active-partition block usage, and
  cumulative new-block demand.
- Shared block IDs already present in the active set or an earlier resumed
  request have zero incremental active-partition cost.
- [`KVBlockAllocator.get_allocatable_block_count()`](../megatron/core/inference/contexts/kv_block_allocator.py)
  reports raw free blocks plus safe LRU-evictable blocks. It subtracts matches
  that are about to be pinned and clamps that reservation so capacity cannot
  become negative.
- Allocation success is checked after scheduling instead of assuming that
  `total_avail` alone guarantees success.

### Why it is needed

This is primarily scheduler correctness and liveness. It prevents requests from
being incorrectly stranded and prevents allocation assertions caused by
inconsistent admission accounting. It also improves throughput by avoiding
unnecessary under-admission. It does not directly change model logits.

Focused regressions live in
`tests/unit_tests/inference/contexts/test_prefix_cache_resume_accounting_cpu.py`
on `fix/prefix-cache-resume-accounting`.

## Follow-up correctness and availability fixes

### Preserve mutable allocator tensors across inference-mode resets

#### KV block free pool

Location:
[`KVBlockAllocator.reset()`](../megatron/core/inference/contexts/kv_block_allocator.py)

The reset path replaced `block_bag` with a new tensor. Engine reset normally
runs under `torch.inference_mode()`, so the replacement became an inference
tensor. A later allocator release outside inference mode failed with:

```text
Inplace update to inference tensor outside InferenceMode
```

The reset now refills the original tensor using `torch.arange(..., out=...)`.
The tensor retains its normal mutability and its storage identity.

#### Durable Mamba prefix-slot pool

Location:
[`MambaSlotAllocator.reset()`](../megatron/core/inference/contexts/mamba_slot_allocator.py)

`free_slots` had the same inference-tensor replacement bug. Reset now refills
the existing tensor instead of replacing it.

#### Live per-request Mamba slots

Location:
[`MambaMetadata.reset()`](../megatron/core/inference/contexts/attention_context/mamba_metadata.py)

`mamba_state_free_slots` also had the same lifetime bug and now uses an in-place
refill.

### Make Mamba durable-slot allocation atomic

Location:
[`MambaSlotAllocator.allocate_slots_batch()`](../megatron/core/inference/contexts/mamba_slot_allocator.py)

The old allocator removed available slots from the free pool before verifying
that the remaining demand could be satisfied by eviction. If eviction then
failed, the already-removed slots were leaked and the allocator mappings no
longer matched `free_count`.

The allocator now:

1. Determines free-slot supply.
2. Computes all safely evictable candidates.
3. Raises `MambaSlotCapacityError` before mutation if total capacity is
   insufficient.
4. Mutates the free pool and mappings only after the full allocation is known
   to fit.

This prevents a recoverable capacity miss from corrupting future allocations.

### Do not abort generation when optional Mamba snapshots cannot be stored

Location:
[`MambaSlotAllocator.commit_intermediate_states()`](../megatron/core/inference/contexts/mamba_slot_allocator.py)

Durable Mamba snapshots are a prefix-cache optimization; live request state is
already stored separately. Previously, durable-slot exhaustion raised from the
post-forward commit path and aborted an otherwise valid generation.

The commit path now catches `MambaSlotCapacityError`, clears the temporary
intermediate-state metadata, and skips that snapshot batch. Generation
continues correctly, with only a future cache-hit opportunity lost.

### Stop prefix matching at the first missing ancestor

Location:
[`DynamicInferenceContext._find_kv_match_count()`](../megatron/core/inference/contexts/dynamic_context.py)

The old matcher searched backward for the deepest hash and assumed every
ancestor remained in the dictionary. If bookkeeping was temporarily
discontinuous, it could find a later hash and then raise `KeyError` while
materializing a missing earlier entry.

The matcher now walks forward and stops at the first miss, matching the method's
longest-consecutive-prefix contract. This converts a corrupting exception into
a safe shorter cache match.

### Cap durable Mamba slots by usable KV blocks

Locations:

- Mamba prefix-cache memory preview in
  [`DynamicInferenceContext`](../megatron/core/inference/contexts/dynamic_context.py)
- [`DynamicInferenceContext._allocate_mamba_cache()`](../megatron/core/inference/contexts/dynamic_context.py)

Each durable Mamba snapshot is keyed by a non-dummy KV block. Allocating more
Mamba slots than usable KV blocks creates GPU-resident slots that can never be
addressed.

Both the memory preview and actual allocation now cap durable slots at:

```python
kv_block_allocator.total_count - 1
```

This keeps reported and actual memory usage consistent and prevents unreachable
GPU allocations.

## Follow-up performance fixes

### Make cold prefix misses proportional to matched-prefix length

Location:
[`DynamicInferenceContext._find_kv_match_count()`](../megatron/core/inference/contexts/dynamic_context.py)

The old implementation copied the full candidate hash slice and scanned it from
the end. A first-block miss was therefore linear in prompt length.

The forward matcher indexes the original list and returns on the first miss. A
cold first-block miss is effectively constant-time while a full hit remains
linear in the number of blocks actually returned.

Synthetic result:

| Prompt blocks | Old reverse matcher | Forward matcher |
| ---: | ---: | ---: |
| 1,000 | 0.0245 ms | 0.00077 ms |
| 10,000 | 0.263 ms | 0.00058 ms |
| 100,000 | 2.65 ms | 0.00058 ms |
| 1,000,000 | 27.4 ms | 0.00061 ms |

### Avoid rebuilding the complete KV prefix forest for small LRU evictions

Location:
[`KVBlockAllocator.evict_lru_blocks()`](../megatron/core/inference/contexts/kv_block_allocator.py)

The old implementation transferred every cached block's timestamp, ID, parent,
and child count into Python lists; built a global-ID dictionary; and heapified
all leaves even when only one block was requested.

The new implementation:

1. Finds cached leaves with tensor operations.
2. Selects at most the `K` oldest initial leaves for a `K`-block eviction.
3. Maintains a small heap ordered by `(timestamp, block_id)`.
4. Exposes a parent only after all of its cached children selected in the peel
   have been removed.

This preserves deterministic descendant-before-parent eviction semantics while
keeping Python work proportional to the requested eviction count.

Synthetic single-block eviction result:

| Cached blocks | Old forest rebuild | Incremental leaf peel |
| ---: | ---: | ---: |
| 1,000 | 0.350 ms | 0.187 ms |
| 10,000 | 2.57 ms | 0.273 ms |
| 100,000 | 17.8 ms | 1.04 ms |

The remaining tensor mask/count scan is still linear in cache size.

### Select Mamba LRU victims without sorting the entire cache

Location:
[`MambaSlotAllocator._evict_lru_slots_batch()`](../megatron/core/inference/contexts/mamba_slot_allocator.py)

The old code called `torch.argsort()` over every candidate even when only a
small number of slots was required. The new code uses
`torch.topk(largest=False, k=num_needed)`.

Synthetic result for selecting the oldest eight slots:

| Candidate slots | Full argsort | `topk(8)` |
| ---: | ---: | ---: |
| 10,000 | 0.334 ms | 0.020 ms |
| 100,000 | 4.51 ms | 0.166 ms |
| 1,000,000 | 52.2 ms | 1.66 ms |

## Tests and benchmarks

### CPU regressions

[`test_prefix_cache_regressions_cpu.py`](../tests/unit_tests/inference/contexts/test_prefix_cache_regressions_cpu.py)
adds coverage for:

- KV free-pool reset under inference mode.
- Durable Mamba free-pool reset under inference mode.
- Live Mamba free-pool reset under inference mode.
- Atomic Mamba capacity failure.
- Non-fatal optional Mamba snapshot exhaustion.
- Safe matching with a missing prefix ancestor.

### Synthetic stress benchmark

[`prefix_cache_stress.py`](../tests/performance_tests/prefix_cache_stress.py)
contains direct synthetic workloads for:

- Long-prompt cold prefix lookup.
- Large cached-prefix forests with small LRU eviction demand.
- Large Mamba slot pools with small victim-selection demand.
- Coordinator shadow-hash-table growth.

Run a short version with:

```bash
python tests/performance_tests/prefix_cache_stress.py --quick
```

The coordinator stress case inserted 800,000 shadow hashes in approximately
1.23 seconds and reached 238 MiB peak traced Python memory.

## Known issues identified but not fixed on these branches

### Prefix cache is not invalidated when RL policy weights change

RL defaults to `--rl-kv-cache-management-mode=persist`. Prefix keys contain
tokens but no model or policy epoch, and `SET_GENERATION_EPOCH` stamps requests
without invalidating cached KV/Mamba activations. A later rollout can therefore
reuse states computed by older weights while reporting the current cache epoch.
This can directly cause inference-versus-training log-probability mismatch.

The safe configuration-level mitigation is:

```text
--rl-kv-cache-management-mode=recompute
```

A complete code fix needs an explicit epoch-aware cache invalidation lifecycle,
including a policy for requests still active at an epoch transition.

### Batch-invariant Mamba execution and prefix restoration

Restarting Mamba execution from a cached recurrent-state boundary can change
scan/reduction grouping. Exact batch-invariant execution does not currently
have a validated Mamba prefix-cache path. A later development branch rejects
this configuration, but that guard is not in the audited `main`.

### Hybrid memory-only prefix caching can rewrite shared KV blocks

When prefix caching is enabled for a hybrid model without durable Mamba cache
capacity, Mamba state cannot be restored and the prompt is recomputed. The
request can still be assigned shared matched KV block IDs, so recomputation may
write into blocks concurrently referenced by another request. This needs a
design decision: either avoid sharing blocks that will be recomputed, or use
copy-on-write ownership.

### Coordinator shadow cache is stale and unbounded

The data-parallel coordinator records hashes assigned to each engine but does
not receive authoritative per-engine eviction/deregistration events. Its
routing view can therefore become stale and its hash table grows without a
capacity bound. Fixing this requires a coordinator/engine protocol change, not
an allocator-local patch.

### GPU numerical parity remains to be validated

CPU tests verify allocator and scheduling invariants, but they cannot prove
Nemotron KV/Mamba numerical parity. The required GPU matrix should compare
prefix caching on/off across:

- Shared and partially shared prompts.
- LRU pressure and repeated eviction.
- Chunked and non-chunked prefill.
- FP32 and BF16 Mamba state storage.
- Policy epoch transitions.
- Batch-invariant mode, which should currently be rejected for Mamba prefix
  caching unless exact parity is implemented.
