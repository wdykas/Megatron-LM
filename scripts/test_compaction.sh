#!/bin/bash
# ===========================================================================
# test_compaction.sh — Full test suite for KV cache compaction
#
# Usage:
#   bash scripts/test_compaction.sh           # Run everything
#   bash scripts/test_compaction.sh --quick   # Unit tests only (~10s)
#   bash scripts/test_compaction.sh --bench   # Benchmarks only
#   bash scripts/test_compaction.sh --train   # RL training demo only
# ===========================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${REPO_ROOT}/.compaction_test_venv"
TEST_DIR="/tmp/compaction_tests_$$"
REF_DIR="/tmp/compaction_reference"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
RUN_TESTS=true
RUN_BENCH=true
RUN_TRAIN=true
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"

for arg in "$@"; do
    case "$arg" in
        --quick) RUN_BENCH=false; RUN_TRAIN=false ;;
        --bench) RUN_TESTS=false; RUN_TRAIN=false ;;
        --train) RUN_TESTS=false; RUN_BENCH=false ;;
        --gpu=*) GPU_ID="${arg#--gpu=}" ;;
        --help|-h)
            echo "Usage: $0 [--quick|--bench|--train] [--gpu=ID]"
            exit 0
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
setup_env() {
    log "Setting up test environment..."

    if [ ! -f "$VENV/bin/python" ]; then
        log "Creating venv at $VENV"
        python3 -m venv "$VENV"
        "$VENV/bin/pip" install -q --upgrade pip
        "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu129
        "$VENV/bin/pip" install -q pytest numpy packaging click requests
    fi

    PYTHON="$VENV/bin/python"

    # Verify torch + CUDA
    TORCH_INFO=$($PYTHON -c "
import torch
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')" 2>/dev/null)
    log "Environment: $TORCH_INFO"

    if ! echo "$TORCH_INFO" | grep -q "cuda=True"; then
        fail "CUDA not available — cannot run GPU tests"
        exit 1
    fi

    # Clone reference if needed
    if [ ! -d "$REF_DIR/compaction" ]; then
        log "Cloning reference implementation..."
        git clone -q https://github.com/adamzweiger/compaction "$REF_DIR" 2>/dev/null || true
    fi

    # Copy tests to clean dir (avoids parent conftest import issues)
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"
    cp "$REPO_ROOT/tests/unit_tests/inference/compaction"/test_*.py "$TEST_DIR/"
}

cleanup() {
    rm -rf "$TEST_DIR"
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Unit + integration tests
# ---------------------------------------------------------------------------
run_tests() {
    log "Running unit tests (114 tests)..."
    echo ""

    PYTHONPATH="$REPO_ROOT:$REF_DIR" \
    $PYTHON -m pytest "$TEST_DIR/" -v --tb=short 2>&1 | tail -30

    echo ""
    RESULT=$(PYTHONPATH="$REPO_ROOT:$REF_DIR" \
             $PYTHON -m pytest "$TEST_DIR/" -q 2>&1 | tail -1)

    if echo "$RESULT" | grep -q "passed"; then
        pass "$RESULT"
    else
        fail "$RESULT"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# 2. Quality benchmarks at various scales
# ---------------------------------------------------------------------------
run_benchmarks() {
    log "Running compaction quality benchmarks..."
    echo ""

    PYTHONPATH="$REPO_ROOT:$REF_DIR" $PYTHON -c "
import torch
import time
import math

from megatron.core.inference.compaction import (
    am_compact, gather_kv, write_kv, validate_attention_output,
    StreamingClusterCompactor, StreamingCompactorConfig,
)

device = 'cuda'
torch.manual_seed(42)

print('=' * 75)
print('BENCHMARK: AM Compaction Quality vs Compression Ratio')
print('=' * 75)
print()

# Sweep sequence lengths and budgets
for T in [512, 2048, 8192]:
    H, D, R = 8, 128, 32
    K = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    Q = torch.randn(R, H, D, device=device, dtype=torch.bfloat16)

    print(f'Sequence length T={T}, heads={H}, dim={D}, queries={R}')
    print(f'{\"Budget\":>8s} {\"Ratio\":>8s} {\"Method\":>15s} {\"Attn Err\":>10s} {\"Mass Err\":>10s} {\"Time (ms)\":>10s}')
    print('-' * 65)

    for budget in [T // 16, T // 8, T // 4, T // 2]:
        if budget < 4:
            continue
        ratio = T / budget

        for method in ['top_attention', 'omp']:
            torch.cuda.synchronize()
            t0 = time.time()

            result = am_compact(
                K, V, Q, budget, method=method,
                nnls_iters=0,
                omp_keys_per_iter=4,
                omp_refit_every=2,
            )

            torch.cuda.synchronize()
            elapsed_ms = (time.time() - t0) * 1000

            m = validate_attention_output(K, V, result.K_mem, result.V_mem, Q, result.biases)

            print(f'{budget:>8d} {ratio:>7.1f}x {method:>15s} {m.mean_relative_l2:>10.4f} {m.mean_mass_error:>10.4f} {elapsed_ms:>10.1f}')

    print()

print()
print('=' * 75)
print('BENCHMARK: Streaming Compactor vs AM Teacher')
print('=' * 75)
print()

T, H, D = 4096, 8, 128

for M in [128, 256, 512]:
    K = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    V = torch.randn(T, H, D, device=device, dtype=torch.bfloat16)
    Q = torch.randn(32, H, D, device=device, dtype=torch.bfloat16)

    # AM teacher
    torch.cuda.synchronize()
    t0 = time.time()
    teacher = am_compact(K, V, Q, M, method='top_attention', nnls_iters=0)
    torch.cuda.synchronize()
    t_teacher = (time.time() - t0) * 1000
    m_teacher = validate_attention_output(K, V, teacher.K_mem, teacher.V_mem, Q, teacher.biases)

    # Streaming top-1
    config = StreamingCompactorConfig(num_anchors=M, routing='top1')
    student = StreamingClusterCompactor(D, H, config).to(device)
    student.initialize_anchors_from_data(K)

    torch.cuda.synchronize()
    t0 = time.time()
    K_s, V_s = student.compact(K, V)
    torch.cuda.synchronize()
    t_student = (time.time() - t0) * 1000
    m_student = validate_attention_output(K, V, K_s, V_s, Q)

    print(f'T={T}, M={M} ({T//M}x compression):')
    print(f'  AM teacher:   err={m_teacher.mean_relative_l2:.4f}  mass={m_teacher.mean_mass_error:.4f}  time={t_teacher:.1f}ms')
    print(f'  Student top1: err={m_student.mean_relative_l2:.4f}  mass={m_student.mean_mass_error:.4f}  time={t_student:.1f}ms')
    print(f'  Speedup: {t_teacher / max(t_student, 0.01):.1f}x')
    print()

print()
print('=' * 75)
print('BENCHMARK: Paged KV Gather/Write Throughput')
print('=' * 75)
print()

for block_size in [16, 64]:
    num_blocks = 1024
    H, D = 8, 128
    buf = torch.randn(2, 1, num_blocks, block_size, H, D, dtype=torch.bfloat16, device=device)

    for num_read_blocks in [16, 64, 256]:
        if num_read_blocks > num_blocks:
            continue
        ids = torch.arange(num_read_blocks, device=device, dtype=torch.int32)
        T_tokens = num_read_blocks * block_size

        # Warmup
        gather_kv(buf, 0, ids, block_size)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            K, V = gather_kv(buf, 0, ids, block_size)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / 100 * 1000

        bytes_read = 2 * T_tokens * H * D * 2  # bf16
        bw_gb = bytes_read / elapsed / 1e6  # GB/s

        print(f'  block_size={block_size:>3d}  blocks={num_read_blocks:>4d}  tokens={T_tokens:>6d}  gather={elapsed:.3f}ms  BW={bw_gb:.1f} GB/s')

print()
print('=' * 75)
print('BENCHMARK: Memory Savings')
print('=' * 75)
print()

from megatron.core.inference.compaction.validation import compute_memory_savings

for T, M in [(8192, 512), (32768, 1024), (131072, 2048)]:
    s = compute_memory_savings(T, M, num_heads=32, head_dim=128, num_layers=32, dtype_bytes=2)
    print(f'  {T:>7d} -> {M:>5d} tokens ({T//M:>3d}x): save {s[\"saved_bytes\"]/1e9:.1f} GB ({s[\"savings_pct\"]:.1f}%)')

print()
print('All benchmarks complete.')
" 2>&1

    echo ""
    pass "Benchmarks complete"
}

# ---------------------------------------------------------------------------
# 3. RL training demo
# ---------------------------------------------------------------------------
run_training() {
    log "Running RL training demo..."
    echo ""

    PYTHONPATH="$REPO_ROOT" $PYTHON "$REPO_ROOT/examples/compaction/train_compaction_rl.py" 2>&1

    echo ""
    pass "Training demo complete"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  KV Cache Compaction Test Suite"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  CUDA_VISIBLE_DEVICES: $GPU_ID"
echo "=============================================="
echo ""

setup_env

OVERALL_PASS=true

if $RUN_TESTS; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 1: Unit & Integration Tests"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    run_tests || OVERALL_PASS=false
fi

if $RUN_BENCH; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 2: Quality & Performance Benchmarks"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    run_benchmarks || OVERALL_PASS=false
fi

if $RUN_TRAIN; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 3: RL Training Demo"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    run_training || OVERALL_PASS=false
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if $OVERALL_PASS; then
    pass "ALL STAGES PASSED"
else
    fail "SOME STAGES FAILED — check output above"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
