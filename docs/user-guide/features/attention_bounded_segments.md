# Attention-Bounded Segments (Variant B inference)

`enable_attention_bounded_segments` is an opt-in inference path for hybrid
Mamba+MoE models (e.g. Nemotron-3-Nano) that lowers small-batch decode
latency by keeping the hidden state in a single global view across each
attention-bounded segment, eliminating the per-MoE-layer all-gather /
reduce-scatter round trip.

The default (flag-off) path is byte-for-byte unchanged.

## When to use it

- **Use** for low-latency, low-concurrency decode (batch ≤ 4 on a
  single 4-GPU NVLS node). At small batch the per-step time is
  dominated by collective latency, and this path replaces the
  AG/RS pair around each MoE layer with a single AR.
- **Don't use** for high-throughput batched serving (batch ≥ 8). The
  path runs the per-rank compute outside MoE on the full global token
  set rather than the per-rank sharded set, and at larger batch the
  4× compute redundancy (on `4×GPU EP=4`) outweighs the saved
  collectives.

Empirical cross-over on Nemotron-3-Nano, 4×GB200, EP=4, NVLS multimem,
ISL=64 / OSL=128:

| batch | baseline   | Variant B   | Δ tok/s |
| ----: | ---------: | ----------: | ------: |
|     1 |  116.3     |  146.1      | **+25.7%** |
|     2 |  246.0     |  278.5      | **+13.2%** |
|     4 |  508.3     |  551.8      |  **+8.6%** |
|     8 | 1,120.0    | 1,045.1     |   −6.7% |
|    16 | 2,094.5    | 1,734.1     |  −17.2% |
|    32 | 3,875.4    | 2,583.9     |  −33.3% |
|    64 | 5,993.4    | 3,653.7     |  −39.0% |

## How it works

A hybrid Mamba+MoE model alternates blocks of Mamba/MoE layers with sparse
attention layers. Call a maximal run of non-attention layers between two
attention layers (or between an attention layer and the model edge) an
**attention-bounded segment**.

In the default flow each MoE layer pays:

1. an all-gather to assemble the per-rank `[L, hidden]` slice into the
   global `[G, hidden]` view experts need, and
2. a reduce-scatter to return the per-rank slice after experts and combine.

When two MoE layers sit in the same segment with only Mamba/MLP/GDN
between them, the AG of the second layer is *redundant* with the RS of
the first. Variant B exploits that:

- **Replicate requests within a model copy**: the data-parallel
  coordinator broadcasts each new request to all ranks inside one
  *replication group* — the set of ranks that together form a single
  model copy (size `EP × TP × PP`) — instead of load-balancing to a
  single rank. Within the group every rank schedules the same set of
  requests, so each rank's Mamba/conv recurrent state stays in sync
  via deterministic recomputation on the shared input.
  
  Across replication groups (i.e. across independent model copies in a
  multi-cluster deployment), the coordinator continues to load-balance,
  so DP throughput scaling is preserved end-to-end. Toggle via
  `--inference-replicate-requests`, surfaced as
  `InferenceConfig.inference_replicate_requests`. Group size is auto-
  computed from the model's parallelism dimensions.
- **Skip the AG, return AR**: with input already in `[G, hidden]` form
  on every rank, `token_dispatch` is a no-op. `token_combine` swaps the
  reduce-scatter for an NVLS multimem all-reduce that returns the full
  `[G, hidden]` view, which feeds directly into the next MoE layer
  without another collective.
- **Canonicalize at boundaries**: at attention layers and at end of
  stack, hidden state is canonicalized back to its per-request KV
  owner. Mamba state is consistent across ranks in segment mode by
  virtue of every rank having executed the same global compute on the
  same global input.

### Optimizations layered on top

Both bit-equivalent (or fp-noise-equivalent) extensions of the core path:

- **Opt B-1 — shared experts on a local slice, folded into the AR.** Each
  rank computes shared experts on its `[G/N, hidden]` slice and stashes
  it on the dispatcher; the combine path adds it into the AR input at
  the rank's slice positions. The all-reduce sums each rank's
  contribution into the matching global row, so shared is already
  applied by the time the AR returns — **no extra collective**.
- **Opt B-2 — Mamba `in_proj` on a local slice + AG.** Slices the
  per-token `in_proj` GEMM to `[G/N, hidden] → [G/N, intermediate]` and
  uses NVLS multimem all-gather to assemble the global
  `[G, intermediate]` for the SSM kernel. Saves `(N-1)/N` of the
  largest GEMM redundancy in the Mamba layer.

## How to enable

```bash
python -m tools.run_dynamic_text_generation_server \
  --enable-attention-bounded-segments \
  --moe-combine-destination-policy current_segment_owner \
  --inference-replicate-requests \
  ...
```

All three CLI flags must be set together. Setting only
`--enable-attention-bounded-segments` (without the policy or
replication flag) keeps behavior at baseline — the runtime tracks
segment metadata but returns
`combine_destination_for_layer == "original_owner"` and the
dispatcher takes the standard reduce-scatter branch.

### Configuration knobs

| flag / env | type | default | effect |
| --- | --- | --- | --- |
| `enable_attention_bounded_segments` | `TransformerConfig: bool` | `False` | Builds the segment runtime. Without the policy below, behavior matches baseline exactly. |
| `moe_combine_destination_policy` | `TransformerConfig: str` | `"original_owner"` | `"current_segment_owner"` switches the dispatcher to the AR path. |
| `inference_replicate_requests` | `InferenceConfig: bool` | `False` | Coordinator broadcasts each request to all DP ranks. Required for Variant B's per-rank state consistency. CLI: `--inference-replicate-requests`. |

## Correctness

`correctness_diff.py` (8 fixed prompts, `temperature=0`, 32 output tokens
each, deterministic greedy decode) shows **8/8 prompts byte-identical**
to baseline output with the full `Variant B + Opt B-1 + Opt B-2` stack
on Nemotron-3-Nano.

Internally the AR/AG paths are bit-equivalent to RS for the per-token
math; the only sources of deviation from baseline are NCCL / NVLS
reduction-tree fp-ordering at the AR step. In practice these stay
sub-LSB on this model and don't propagate into different sampled
tokens. Setups that introduce additional GeMM tile-size variation
(notably any future *aggressive* slicing of `out_proj`, attention
`linear_qkv`, etc. — see "Considered and rejected" below) may push
1-2 prompts past a token-decision boundary; that's the tolerance
threshold the project has been holding to.

## Comparison with the previous (baseline) inference path

| aspect | baseline | Variant B (this path) |
| --- | --- | --- |
| Request distribution | Coordinator load-balances across DP ranks | Coordinator broadcasts to all DP ranks |
| Hidden-state shape, per rank | `[L, hidden]` everywhere (`L = G/N`) | `[G, hidden]` everywhere; canonicalized at attention boundaries |
| MoE dispatch | NVLS AG (or NCCL ring) per layer | Skipped within a segment |
| MoE combine | NVLS RS (or NCCL ring) per layer | NVLS AR returning global view per layer |
| Mamba in_proj | per-rank `[L, hidden]` GEMM (already efficient) | per-rank `[G/N, hidden]` GEMM + NVLS AG (Opt B-2) |
| Shared experts | per-rank `[L, hidden]` GEMM (already efficient) | per-rank `[G/N, hidden]` GEMM, folded into AR (Opt B-1) |
| Per-rank compute outside MoE | sized to `L` (no redundancy) | sized to `G` (4× redundant on `EP=4`) |
| Best at | high-throughput batched serving (batch ≥ 8) | low-latency / low-concurrency decode (batch ≤ 4) |

The trade-off is essentially **comm vs compute**: Variant B trades 4×
per-rank redundant compute outside MoE for one fewer collective per
MoE layer. This pays off at the batch sizes where per-step time is
dominated by collective latency rather than per-token compute.

## Implementation files

| file | role |
| --- | --- |
| `megatron/core/inference/attention_bounded_segments.py` | `Segment` dataclass, `compute_segments`, `SegmentRuntime` (segment topology + ownership metadata). Pure Python, no CUDA. |
| `megatron/core/inference/data_parallel_inference_coordinator.py` | `replicate_requests` constructor parameter (sourced from `InferenceConfig.inference_replicate_requests` via `dynamic_engine.py`'s spawn kwargs) and the broadcast-on-submit path; finished-response dedup so duplicate replies from peer ranks are silently dropped. |
| `megatron/core/inference/communication/torch_symm_triton/collectives.py` | `multimem_all_reduce` Triton kernel + Python wrapper used as a fallback when `torch.ops.symm_mem.multimem_all_reduce_` is unavailable. |
| `megatron/core/transformer/moe/token_dispatcher_inference.py` | `_segment_input_is_global` flag wiring on `InferenceCUDAGraphTokenDispatcher`; `_token_combine_via_all_reduce_global` and `_token_combine_via_all_reduce` paths; shared-experts AR-fold (Opt B-1). |
| `megatron/core/transformer/moe/moe_layer.py` | Opt B-1 stash from `shared_experts_compute`; segment-runtime policy validation in `combine`. |
| `megatron/core/ssm/mamba_mixer.py` | `_maybe_in_proj_on_local_slice` (Opt B-2). |
| `megatron/core/models/hybrid/hybrid_block.py` | Builds the `SegmentRuntime` per stack; toggles `_segment_input_is_global` on each MoE dispatcher when Variant B is active. |
| `megatron/core/transformer/transformer_config.py` | New config fields. |
| `megatron/training/arguments.py` | Auto-generated CLI flags via `ArgumentGroupFactory`. |

### Tests

- `tests/unit_tests/inference/test_attention_bounded_segments.py` —
  20 pure-Python tests covering `compute_segments` (boundaries, MoE/
  stateful indices, GDN handling, DSA-attention boundaries) and
  `SegmentRuntime` (disabled = baseline, enabled reports configured
  policy, ownership state).
- `tests/unit_tests/inference/test_attention_bounded_combine.py` —
  multi-GPU equivalence test (run under
  `torchrun --nproc_per_node=4`): the AR + local-slice combine matches
  the standard RS combine; flagged-but-baseline-policy is identical to
  unflagged.

## Future optimizations

The Variant B path has a fundamental compute-vs-comm trade-off: every
per-token op outside MoE pays `4×` (on `EP=4`) per-rank redundancy
because every rank holds the global view. Future work falls into three
buckets:

### 1. Reducing the per-rank compute redundancy (close the b ≥ 8 regression)

These slice the per-token op to `[G/N, ...]` and rejoin via NVLS AG.
The pattern is the same as Opt B-2; the question is whether the saved
GEMM exceeds the AG cost.

- **Slice the lm_head projection + sample on local slice.** Each rank
  computes logits for `[G/N, hidden]` and samples; AG only the token
  IDs (tiny). Bit-identical: argmax of identical logits gives identical
  tokens, and the slice path produces identical logits up to GeMM
  tile-size noise (which is far smaller than the gap to the runner-up
  token in practice).
- **Slice the pre-Mamba / pre-MoE norm.** Norms are per-token and
  cheap, so the absolute saving is small, but folded into the existing
  Opt B-2 slice path it costs nothing.
- **Mamba `out_proj` on a local slice + AG.** A first attempt
  (`Opt B-2b`) was reverted because GeMM tile-size differences pushed
  correctness from 1-prompt to 2-prompt fp-noise divergence. A
  retry with FP32 accumulation in the slice GEMM may preserve the
  tighter bound; expected gain ≈ 1–3% on Nemotron-3-Nano.

The following were measured to be net-loss on this model and are
documented for posterity rather than as future work:

- Attention `linear_qkv` slice + AG: −0.4 to −4.1% across batches; the
  model has only ~6 attention layers, not enough to amortize the AG
  cost over.
- Attention `linear_proj` slice + AG: same profile, same expected loss.

### 2. Reducing the comm cost itself

- **Fused multimem AR + residual-add + RMS-norm Triton kernel
  (Opt B-4).** Mirrors the existing
  `_multimem_reduce_scatter_residual_add_kernel` but for the AR path:
  fuses the AR with the next layer's pre-norm so only one HBM round
  trip is needed instead of three. **Bit-identical** numerics. On
  Nemotron-3-Nano with full CUDA-graph capture the projected gain is
  small (\~0.02 % at b = 16) because launches are already free under
  graph replay; on workloads with larger hidden_dim or no CUDA graphs
  the gain scales up to the plan's original 3–5% estimate.
- **Custom multimem AR-to-slice kernel (Opt B-5).** Replaces the
  AR-then-slice pattern with a single kernel that performs the
  multimem reduction and writes only this rank's slice locally,
  closing the bytes-moved gap with RS while preserving the skip-AG
  win. Closes much of the high-batch regression but is a substantial
  Triton effort with careful CUDA-graph integration.

### 3. Architectural / scope changes

- **Cross-node Variant B.** All measurements above are on a single
  4-GPU NVLS node where AG/RS are fast (\~125 µs). Cross-node IB
  latency is 10–100× larger, so Variant B's relative win on
  multi-node deployments should be substantially larger. Requires
  separate test infrastructure.
- **Speculative decoding (MTP) composition.** See the dedicated section
  below — Variant B and MTP compose multiplicatively in throughput and
  the integration is mostly already structurally compatible.
- **Runtime adaptive path selection.** The engine could inspect
  decode batch size each step and route to either the Variant B or
  baseline path, using whichever wins for that step. Caps the
  downside at baseline level (no more b ≥ 8 regression) while
  preserving the +25.7% win at b = 1. Requires making the env-var
  gate runtime-toggleable and ensuring both paths can coexist within
  one process.
- **Sub-replication groups (K = 2).** Replicate within rank pairs
  rather than across all `N` ranks: every pair of ranks shares
  requests (Variant B between them) but the model is still sharded
  across pairs (baseline-style). Per-rank compute redundancy is `2×`
  instead of `4×`, with half the comm savings. Trade-off may shift
  the cross-over to a usable middle ground; substantial scheduler
  complexity.
- **FP8 path compatibility.** The Variant B AR path is currently
  bf16-only (NVLS multimem AR eligibility). Adapting for FP8 needs
  the appropriate quantize-on-write semantics in the AR buffer.

## Composing with Multi-Token Prediction (MTP) / speculative decoding

Variant B and MTP / speculative decoding are **structurally compatible**
and **compose multiplicatively** in end-to-end throughput. No new code
is required to enable the combination — the existing MTP infrastructure
(`MultiTokenPredictionBlock`, `is_spec_decode` flag,
`InferenceConfig.num_speculative_tokens`, the dynamic engine's
`_spec_tokens_proposed` / `_spec_tokens_accepted` metrics) lives in code
paths that Variant B never touches.

### Why they compose

- **Variant B** reduces *per-step latency* at small batch by skipping
  the AG/RS pair around each MoE layer (saves ~125 µs × ~23 layers
  per step on 4×GB200 NVLS).
- **MTP** increases *accepted tokens per step* by 2–4× depending on
  acceptance rate, by predicting multiple future tokens conditioned
  on the verified output and accepting any that match.
- Multiplied together: at b = 1 a roughly **3×** end-to-end throughput
  uplift on top of the baseline is plausible, given a model with
  trained MTP heads at typical (~75–85%) acceptance rate.

### Why no Variant-B-specific work is needed

The shape constraint Variant B's slicing optimizations rely on is
`G % N == 0`, where `G` is the global-view token count and `N` is
`ep_size`. Under spec decoding with `K = num_speculative_tokens`, each
of `G` decode requests contributes `K + 1` tokens per step, so the
global token count is `G × (K + 1)`. This is divisible by `N` whenever
`G % N == 0`, so Opt B-2 (Mamba `in_proj` on local slice + AG) and
Opt B-1 (shared experts on local slice + AR fold) operate on the
expanded shape without modification. The subsequent reshape to
`[decode_req_count, K + 1, hidden]` for the SSM kernel works because
the slicing falls on natural `(K + 1)`-token boundaries (each rank's
slice contains exactly `G/N` requests' worth of `(K + 1)`-token
groups, in order).

### What's in scope vs out of scope today

| component | status |
| --- | --- |
| Variant B AR-instead-of-RS combine path under MTP | works as-is — same shape & dispatch flow, just larger leading dim |
| Opt B-1 / B-2 slicing under MTP | works as-is — slicing is along the leading dim, which `G × (K + 1)` divides cleanly |
| Mamba SSM / conv1d_update under MTP | already handles `seq_len = 1 + K`; nothing Variant-B-specific needed |
| Coordinator request replication + spec-token dedup | works as-is — coordinator dedup is keyed on `request_id`, the spec tokens are accepted/rejected internally before the finished response goes out |
| **Distributing the K MTP heads across the N EP ranks** | **not implemented; out of scope for this MR** |

The last row is the more ambitious "Variant B converts wasted compute
into useful work" version sketched in the design notes. The standard
MTP block runs all K heads on every rank with a sequential dependency
(head k+1 conditions on head k's draft). Rewriting that to
EP-parallel — one head per rank — is a multi-week architectural
change to `MultiTokenPredictionBlock`; it's the right *next* MTP
change but isn't part of the Variant B work.

### Validation

> ⚠️ **Testing note**: the perf claims above are theoretical
> compositions of independently measured numbers. They have **not**
> been benchmarked end-to-end on a model with trained MTP heads,
> because the available Nemotron-3-Nano checkpoint used for Variant B
> validation does not include MTP heads. Concrete validation requires:
>
> 1. Running with a checkpoint that has trained MTP heads
>    (`--num-speculative-tokens=K`, `--mtp-num-layers >= 1`).
> 2. End-to-end correctness: token outputs match the no-MTP run
>    after acceptance/rejection, modulo NCCL fp-noise.
> 3. Acceptance rate (`inference/spec_decode_acceptance_rate` metric
>    in the engine).
> 4. Throughput vs baseline (no Variant B, no MTP), Variant B alone,
>    MTP alone, and Variant B + MTP, across `b ∈ {1, 4, 16, 64}`.
>
> This is tracked as a follow-up. The compatibility audit summarized
> above gives confidence that the combination should work, but the
> multiplicative throughput claim is unverified on real hardware.

## Considered and rejected

| variant | result |
| --- | --- |
| Mamba `out_proj` slice + AG (`Opt B-2b`) | Reverted: pushed correctness from 1-prompt to 2-prompt fp-noise divergence under standard bf16 accumulation. Worth retrying with FP32 accumulation in the slice GEMM. |
| Attention `linear_qkv` slice + AG (`Opt A1`) | Reverted: net loss across all batches (−0.4% / −1.4% / −4.1% at b = 1 / 4 / 16). Too few attention layers in nano to amortize the per-layer AG. Same conclusion projected for `linear_proj`. |
| Router skip on N-1 ranks via broadcast | Rejected: router is small and deterministic, so every rank already produces identical routing for free; broadcasting the result costs more than the saved compute. |

These are documented so they don't get re-attempted blindly on the
same model architecture.
