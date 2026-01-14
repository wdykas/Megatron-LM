#!/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Llama 3.1 8B Instruct with Collaborative Reasoning
#
# This configuration enables parallel collaborative reasoning
# where multiple rollouts can share information during generation.
#
# Usage:
#   source examples/rl/model_configs/llama3p1_8b_collaborative.sh
#   # Then run your training script

echo "Loading Llama 3.1 8B Instruct with Collaborative Reasoning"

# Load common options
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Model architecture
export HIDDEN_SIZE=4096
export FFN_HIDDEN_SIZE=14336
export NUM_LAYERS=32
export NUM_ATTENTION_HEADS=32
export NUM_QUERY_GROUPS=8
export SEQ_LENGTH=8192
export MAX_POSITION_EMBEDDINGS=131072

# Parallelism defaults (can be overridden)
export TP=${TP:-2}
export PP=${PP:-1}

# Model-specific options
MODEL_OPTIONS="\
    --hidden-size $HIDDEN_SIZE \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-layers $NUM_LAYERS \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --num-query-groups $NUM_QUERY_GROUPS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --group-query-attention \
    --swiglu \
    --disable-bias-linear \
    --normalization RMSNorm \
    --no-position-embedding \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    "

# === Collaborative Reasoning Options ===
COLLAB_OPTIONS="\
    --enable-collaborative-reasoning \
    --collab-memory-dim 1024 \
    --collab-memory-lr-mult 10.0 \
    --collaboration-sync-interval 8 \
    "

# RL training options
RL_OPTIONS="\
    --perform-rl-step \
    --grpo-prompts-per-step 32 \
    --grpo-group-size 4 \
    --grpo-iterations 2 \
    --grpo-kl-beta 0.001 \
    --rl-default-temperature 0.7 \
    "

# Combine all options
export LLAMA_COLLAB_OPTIONS="${COMMON_OPTIONS} ${MODEL_OPTIONS} ${COLLAB_OPTIONS} ${RL_OPTIONS}"

echo "Collaborative reasoning enabled with:"
echo "  - Memory dim: 1024"
echo "  - Sync interval: 8 tokens"
echo "  - LR multiplier: 10x"
