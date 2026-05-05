# L2 cache prefetch for wide-EP MoE inference

## Hypothesis

At wide EP (≥16), the per-rank expert weight footprint shrinks to fit in
L2 cache (~50-80 MB on Hopper/Blackwell). Issuing async reads of expert
weights on a side stream during MoE dispatch lets the subsequent expert
GEMMs hit warm L2 instead of paying HBM latency. Expected gain at decode
+ wide EP: 5-15% step-time reduction.

## What's gating

Only fits in L2 when ``num_local_experts × per_expert_weight_bytes`` is
under ~30-50 MB. For nanov3 (32 experts, ~15 MB/expert):

| EP | local experts | per-rank weight bytes | fits in L2? |
|---:|---:|---:|---|
| 4  | 8 | 120 MB | no |
| 8  | 4 | 60 MB | borderline |
| 16 | 2 | 30 MB | yes |
| 32 | 1 | 15 MB | yes |
| 64 | 0.5 | 7.5 MB | yes |
| 72 | <1 | smaller | yes |

So the experiment is only meaningful at EP ≥ 16. At EP=4 it can only
regress (prefetch can't fit, just thrashes L2).

## Implementation (this branch)

- New file: ``megatron/core/inference/l2_prefetch.py``
  - Triton kernel ``_l2_prefetch_kernel`` reads 1 u32 per cache line with
    ``eviction_policy='evict_last'`` to bias L2 toward keeping the data.
  - ``prefetch_into_l2(tensors, stream)`` enqueues prefetches on a stream.
  - ``collect_expert_weight_tensors(experts)`` walks the TEGroupedMLP
    structure to gather per-expert ``linear_fcN.weightK`` tensors.

- Hook in ``megatron/core/transformer/moe/moe_layer.py``:
  - ``MoELayer.__init__`` creates a side stream when env
    ``L2_PREFETCH_EXPERTS=1``.
  - ``MoELayer.forward`` between ``preprocess`` and ``dispatch``: kicks
    off prefetch on the side stream.
  - Before ``routed_experts_compute``: ``wait_stream`` on the side
    stream so the prefetch is visible to the expert GEMMs.

Default off. Env-gated. Zero touch when ``L2_PREFETCH_EXPERTS=0`` or
unset.

## How to run on slurm (NVL72 / 4-node × 4-GPU = EP=16)

The existing bench infrastructure already supports multi-node submit.

```bash
cd /lustre/.../Megatron-LM

# 4 nodes × 4 GPUs = 16 GPUs total. EP defaults to WORLD_SIZE in
# inference-bench/mcore/mcore_server.sh.
sbatch -N 4 \
    --export=ALL,L2_PREFETCH_EXPERTS=1,EXP_NAME=l2_prefetch_ep16_b16,\
MODEL=nanov3,BATCH_SIZES=16,OSL=64,DATASET=gsm8k \
    inference-bench/submit.sh
```

Then run a baseline (prefetch off) for comparison:

```bash
sbatch -N 4 \
    --export=ALL,L2_PREFETCH_EXPERTS=0,EXP_NAME=baseline_ep16_b16,\
MODEL=nanov3,BATCH_SIZES=16,OSL=64,DATASET=gsm8k \
    inference-bench/submit.sh
```

Compare ``Throughput`` lines in
``inference-bench/experiments/<TS>_l2_prefetch_ep16_b16/benchmark.log``
vs the baseline.

## Wider sweeps to run

Once EP=16 baseline is established, sweep:

- EP=16 vs EP=32 vs EP=72 (extend ``-N`` accordingly; need NVL72-class
  node availability for 72)
- b=8, 16, 32 (decode-batch dependence)
- Combine with FP8 if available (``--inference-grouped-gemm-backend``
  variants)

## Expected outcome guide

- **At EP=16**: marginal (~3-8%) speedup expected. Working set borderline.
- **At EP=32**: clean speedup (~10-15%) expected. Working set comfortably
  in L2.
- **At EP=72**: similar to EP=32. Diminishing returns past this since
  expert work itself becomes a tiny fraction of step time.
- **At EP=4 (this dev VM)**: regression expected. Working set 120MB
  exceeds L2 (50-80MB), prefetch thrashes L2 contents and wastes
  bandwidth.

## Failure modes to watch for

1. **CUDA graph capture crash**: The prefetch kernel must be capturable.
   Triton kernels generally are. If the side-stream wait_stream causes
   issues with graph capture, may need to inline the prefetch into the
   captured graph instead of using a side stream.
2. **L2 thrash from prefetched weights themselves**: if the prefetch
   data is too big it evicts itself. Manifests as no improvement or a
   regression even at EP=16.
3. **Prefetch latency exceeds dispatch latency**: if dispatch is fast
   (low-token batches), the side-stream prefetch can't hide. May need
   to gate on token count.
4. **Eviction by KV cache reads**: subsequent attention layers' KV reads
   may evict prefetched expert weights before the next MoE layer fires.
   Mitigation: prefetch immediately before the consuming MoE rather than
   at step start.

## Next iterations if Phase 1 wins

- **Predicted-expert prefetch**: only prefetch the top-K likely experts
  using a shadow router or "same as last step" predictor. Reduces
  prefetch bandwidth at cost of accuracy.
- **Cross-layer pipeline**: prefetch layer N+1's weights while layer N
  is computing.
- **Persistent expert kernels** (CUTLASS-style): keep weights in L2
  permanently across all decode steps.
