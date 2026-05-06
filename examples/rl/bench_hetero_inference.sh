#!/usr/bin/env bash
# Benchmark heterogeneous inference shard configurations under forced-lag.
#
# Setup: --rl-partial-rollouts plus --rl-num-parallel-generation-batches=L+1
# keeps L+1 generation batches in flight at all times. Slow rollouts in
# any of the L+1 batches push back on the next training step, so the
# tail of per-iteration time directly tracks the tail of rollout
# completion. We run a small fixed number of iterations per config and
# compare median / P95 / P99 / max iteration time across configs.
#
# Hypothesis: a heterogeneous shard layout that splits compute (big TP
# for prefill, parallel DP for decode) plus auto-disagg should reduce
# the tail vs a flat homogeneous layout, by sending each request to the
# shard whose cost profile suits it best.
#
# Each config submits its own slurm job (via run_job.sh → submit_job).
# After all jobs finish, run examples/rl/parse_hetero_bench.py with the
# experiment names this script prints to compare results.
#
# Usage:
#   ./examples/rl/bench_hetero_inference.sh
#
# Env knobs:
#   EXIT_INTERVAL  iterations to run per config (default 20).
#   LAG            staleness L; --rl-num-parallel-generation-batches = L+1 (default 2).
#   CONFIGS        space-separated subset of names (default: all).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

EXIT_INTERVAL=${EXIT_INTERVAL:-20}
LAG=${LAG:-2}
PARALLEL_BATCHES=$((LAG + 1))

# (name | shard_arg | extra CLI flags). The shard arg drives
# ``--rl-inference-shards``; auto-disagg configs append --rl-auto-disagg-*
# flags that MegatronLocalMulti.launch consults.
DISAGG_LENGTH_THRESHOLD=${DISAGG_LENGTH_THRESHOLD:-256}
TAIL_CUT_MIN_TOKENS=${TAIL_CUT_MIN_TOKENS:-128}
declare -a ALL_CONFIGS=(
  "homogeneous_tp4|tp=4,dp=1|"
  "homogeneous_tp2_dp2|tp=2,dp=2|"
  "hetero_balanced|tp=2,dp=1+tp=2,dp=1|"
  "hetero_prefill_decode|tp=2,dp=1+tp=1,dp=2|"
  "hetero_prefill_decode_disagg|tp=2,dp=1+tp=1,dp=2|--rl-auto-disagg-src-shard 0 --rl-auto-disagg-dst-shard 1"
  # Length-aware: same hetero topology as _disagg, but the HTTP
  # endpoint short-circuits prompts shorter than the threshold
  # straight to the decode shard (no migration). Compare against
  # _disagg to see how much migration overhead disappears for short
  # traffic.
  "hetero_disagg_length_aware|tp=2,dp=1+tp=1,dp=2|--rl-auto-disagg-src-shard 0 --rl-auto-disagg-dst-shard 1 --rl-disagg-length-threshold $DISAGG_LENGTH_THRESHOLD"
  # Three-shard tail-cut. Shard 0 = TP=1 prefill, shard 1 = TP=1
  # throughput decode, shard 2 = TP=2 latency-optimized decode for
  # long-tail rollouts. Requests prefill on shard 0, migrate to shard
  # 1 at first token; once on shard 1, requests that exceed
  # ${TAIL_CUT_MIN_TOKENS} tokens are pulled onto shard 2 (faster
  # per-token at TP=2). Compare against ``hetero_prefill_decode_disagg``
  # to see whether routing the slow-completing tail to a higher-TP
  # decode shard reduces p99 / max iter time.
  "hetero_tail_cut|tp=1,dp=1+tp=1,dp=1+tp=2,dp=1|--rl-auto-disagg-src-shard 0 --rl-auto-disagg-dst-shard 1 --rl-tail-cut-dst-shard 2 --rl-tail-cut-min-tokens $TAIL_CUT_MIN_TOKENS"
)

# Optional filter via $CONFIGS env (space-separated names).
if [[ -n "${CONFIGS:-}" ]]; then
    declare -a FILTERED=()
    for name in $CONFIGS; do
        for spec in "${ALL_CONFIGS[@]}"; do
            [[ "${spec%%|*}" == "$name" ]] && FILTERED+=("$spec")
        done
    done
    SELECTED=("${FILTERED[@]}")
else
    SELECTED=("${ALL_CONFIGS[@]}")
fi

echo "Submitting ${#SELECTED[@]} configs at lag L=$LAG (parallel batches = $PARALLEL_BATCHES), exit at iter $EXIT_INTERVAL"

# Where bench artifacts land:
#   $BENCH_OUT_DIR             — run_paths.txt + per-run subdirs
#   $BENCH_OUT_DIR/runs/<EXP>/ — actual training output (logs/,
#                                checkpoints/) for each config
# The default puts everything under your hetero/ tree so it's not
# scattered across the cluster's projects/megatron_rl/runs/ shared
# space. Override via BENCH_OUT_DIR.
BENCH_OUT_DIR=${BENCH_OUT_DIR:-/lustre/fsw/portfolios/llmservice/users/wdykas/code/rl-hetero/bench_hetero_runs}
mkdir -p "$BENCH_OUT_DIR/runs"
RUN_LIST="$BENCH_OUT_DIR/run_paths.txt"
: > "$RUN_LIST"

for spec in "${SELECTED[@]}"; do
    IFS='|' read -r name shard_arg extra_args <<< "$spec"
    EXP_NAME="bench_hetero_${name}_$(date +%s)"

    # SLURM_ACCOUNT defaults to nemotron_sw_pre; override at the
    # script level by exporting SLURM_ACCOUNT before invocation.
    env_prefix=(
        env
        "SLURM_ACCOUNT=nemotron_sw_pre"
        "BASE_RUN_DIR=$BENCH_OUT_DIR/runs"
        "LOG_TO_WANDB=false"
        "TEST_MODE=true"
        "TP=2"
        "NODES_REQUIRED=1"
        "MICRO_BATCH_SIZE=1"
        "GRPO_GROUP_SIZE=4"
        "GRPO_PROMPTS_PER_STEP=4"
        "TRAINING_BATCH_SIZE=4"
        "CHKPT_SAVE_INTERVAL=120"
        "EXIT_INTERVAL=$EXIT_INTERVAL"
        "EVAL_INTERVAL=1000"
        "ENV_CONFIG=$(realpath megatron-rl/examples/rl/environment_configs/countdown.yaml)"
        "EXP_NAME=$EXP_NAME"
    )

    "${env_prefix[@]}" ./run_job.sh -l qwen2p5_3b -- \
        --rl-partial-rollouts \
        --rl-num-parallel-generation-batches "$PARALLEL_BATCHES" \
        --rl-inference-shards "$shard_arg" \
        $extra_args

    # run_job.sh appends ``_$USER`` to the experiment name unless it
    # already ends with the user, so record the actual on-disk name.
    echo "${EXP_NAME}_${USER}" >> "$RUN_LIST"
done

echo
echo "Submitted: $(wc -l < "$RUN_LIST") configs."
echo "Run paths recorded at: $RUN_LIST"
echo "Once jobs finish, parse with:"
echo "  python megatron-rl/examples/rl/parse_hetero_bench.py \$(cat $RUN_LIST | tr '\\n' ' ')"
