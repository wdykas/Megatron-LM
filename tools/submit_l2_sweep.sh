#!/bin/bash
# Submit a sweep of L2-prefetch experiments at multiple EP sizes.
#
# Usage:
#   bash inference-bench/submit_l2_sweep.sh
#
# Override defaults via env:
#   NODE_COUNTS="4 8 16"          (one slurm job per node count → EP = 4 × N)
#   BATCH_SIZES="16"
#   MODEL=nanov3
#
# Requires sbatch and submit.sh from this directory. Each variant launches
# its own job; they queue independently. Compare benchmark.log Throughput
# lines across the matching exp dirs.
#
# Variants (single env var per variant, so you can post-hoc verify
# from config.env which knob fired):
#
#   off              — baseline, no prefetch (control)
#   experts_all      — prefetch all local-rank experts before MoE dispatch
#   mamba            — prefetch mamba mixer weights before mixer forward
#   experts_all+mamba — both above

set -euo pipefail

NODE_COUNTS="${NODE_COUNTS:-4 8}"   # 4N → EP=4N. 4 nodes = EP=16, 8 nodes = EP=32.
BATCH_SIZES="${BATCH_SIZES:-16}"
OSL="${OSL:-64}"
MODEL="${MODEL:-nanov3}"
DATASET="${DATASET:-gsm8k}"
NUM_ITERS="${NUM_ITERS:-10}"
NUM_WARMUP_ITERS="${NUM_WARMUP_ITERS:-3}"

# Variant table: <name>:<L2_PREFETCH_EXPERTS>:<L2_PREFETCH_MAMBA>
VARIANTS=(
    "off:0:0"
    "experts_all:1:0"
    "mamba:0:1"
    "experts_all+mamba:1:1"
)

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this from a slurm submit/login node."
    exit 1
fi

mkdir -p logs

# Set CHAIN=1 to slurm-chain jobs via --dependency=afterany so each
# job only starts after the previous completes. Useful when QOS limits
# concurrent node counts. Default off — fires them all at once.
CHAIN="${CHAIN:-0}"

JOBIDS=()
PREV_JOBID=""
for N in $NODE_COUNTS; do
    EP=$((N * 4))
    for V in "${VARIANTS[@]}"; do
        IFS=':' read -r NAME EX MA <<< "$V"
        EXP="l2sweep_${NAME}_ep${EP}_b${BATCH_SIZES// /_}"
        echo "=== submitting: nodes=$N EP=$EP variant=$NAME ==="
        DEP_FLAG=""
        if [ "$CHAIN" = "1" ] && [ -n "$PREV_JOBID" ]; then
            DEP_FLAG="--dependency=afterany:$PREV_JOBID"
        fi
        JOBID=$(sbatch --parsable -N "$N" $DEP_FLAG \
            --output=logs/%x-%j.out --error=logs/%x-%j.err \
            --export=ALL,L2_PREFETCH_EXPERTS=$EX,L2_PREFETCH_MAMBA=$MA,\
EXP_NAME=$EXP,MODEL=$MODEL,BATCH_SIZES="$BATCH_SIZES",OSL=$OSL,DATASET=$DATASET,\
NUM_ITERS=$NUM_ITERS,NUM_WARMUP_ITERS=$NUM_WARMUP_ITERS \
            inference-bench/submit.sh)
        echo "  jobid=$JOBID  L2_PREFETCH_EXPERTS=$EX L2_PREFETCH_MAMBA=$MA"
        JOBIDS+=("$JOBID:$EXP")
        PREV_JOBID=$JOBID
    done
done

echo ""
echo "=========================================================="
echo "Submitted ${#JOBIDS[@]} jobs:"
for entry in "${JOBIDS[@]}"; do
    echo "  $entry"
done
echo ""
echo "Watch: squeue --user=\$USER"
echo ""
echo "Aggregate results once complete:"
echo "  for d in inference-bench/experiments/*l2sweep_*; do"
echo "      echo \"\$(basename \$d): \$(grep -E 'Throughput' \$d/benchmark.log)\""
echo "  done"
echo "=========================================================="
