#!/usr/bin/env python3
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#
# Micro-benchmark for batch-invariant kernels:
# - Triton persistent matmul/mean in `batch_invariant_kernels.py`
# - Optional cuTile variants if `cuda.tile` is installed
#
# Usage examples:
#   python tools/bench_batch_invariant_kernels.py --m 8192 --n 8192 --k 8192 --dtype bf16
#   python tools/bench_batch_invariant_kernels.py --mean-shape 8192 4096 --dtype fp16
#
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


# Make sure Megatron-LM is importable when invoked from repo root or tools/
_THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(_THIS_DIR, os.path.pardir)))

from megatron.core.transformer.custom_layers import (  # noqa: E402
    batch_invariant_kernels as bik,
)


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


@dataclass
class Timing:
    name: str
    ms: float


@torch.no_grad()
def time_cuda(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
) -> float:
    """Time a CUDA workload using events; returns milliseconds per iteration."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    # Warmup (includes compilation for Triton/cuda.tile on first call).
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(max(1, iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / max(1, iters)


def _maybe_assert_close(
    ref: torch.Tensor,
    out: torch.Tensor,
    dtype: torch.dtype,
    name: str,
):
    if dtype == torch.float32:
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)
    else:
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def bench_matmul(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    correctness: bool,
    with_bias: bool,
) -> None:
    device = torch.device("cuda")
    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    bias = torch.randn((n,), device=device, dtype=dtype) if with_bias else None

    # Reference (PyTorch)
    def torch_fn():
        if bias is None:
            _ = a @ b
        else:
            _ = a @ b + bias

    # Triton persistent kernel
    def triton_fn():
        _ = bik.matmul_persistent(a, b, bias=bias)

    # cuTile persistent kernel (optional)
    cutile_ok = getattr(bik, "_CUTILE_AVAILABLE", False)

    def cutile_fn():
        _ = bik.matmul_persistent_cutile(a, b, bias=bias)

    timings = []
    timings.append(Timing("torch", time_cuda(torch_fn, warmup=warmup, iters=iters)))
    timings.append(Timing("triton_persistent", time_cuda(triton_fn, warmup=warmup, iters=iters)))
    if cutile_ok:
        timings.append(Timing("cutile_persistent", time_cuda(cutile_fn, warmup=warmup, iters=iters)))

    if correctness:
        ref = (a @ b) if bias is None else (a @ b + bias)
        out_triton = bik.matmul_persistent(a, b, bias=bias)
        _maybe_assert_close(ref, out_triton, dtype, "triton_persistent")
        if cutile_ok:
            out_cutile = bik.matmul_persistent_cutile(a, b, bias=bias)
            _maybe_assert_close(ref, out_cutile, dtype, "cutile_persistent")

    print(f"\n== Matmul: M={m} N={n} K={k} dtype={dtype} bias={with_bias} ==")
    for t in timings:
        print(f"{t.name:>18}: {t.ms:8.3f} ms")


def bench_mean_lastdim(
    shape: Tuple[int, int],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    correctness: bool,
) -> None:
    device = torch.device("cuda")
    x = torch.randn(shape, device=device, dtype=dtype)

    # Reference
    def torch_fn():
        _ = x.mean(dim=-1, keepdim=True)

    # Triton mean_dim (supports arbitrary dim, but we benchmark -1 because it's the hot path here)
    def triton_fn():
        _ = bik.mean_dim(x, dim=-1, keepdim=True)

    cutile_ok = getattr(bik, "_CUTILE_AVAILABLE", False)

    def cutile_fn():
        _ = bik.mean_lastdim_cutile(x, keepdim=True, dtype=dtype)

    timings = []
    timings.append(Timing("torch", time_cuda(torch_fn, warmup=warmup, iters=iters)))
    timings.append(Timing("triton_mean_dim", time_cuda(triton_fn, warmup=warmup, iters=iters)))
    if cutile_ok:
        timings.append(Timing("cutile_mean_lastdim", time_cuda(cutile_fn, warmup=warmup, iters=iters)))

    if correctness:
        ref = x.mean(dim=-1, keepdim=True)
        out_triton = bik.mean_dim(x, dim=-1, keepdim=True)
        _maybe_assert_close(ref, out_triton, dtype, "triton_mean_dim")
        if cutile_ok:
            out_cutile = bik.mean_lastdim_cutile(x, keepdim=True, dtype=dtype)
            _maybe_assert_close(ref, out_cutile, dtype, "cutile_mean_lastdim")

    print(f"\n== Mean(-1): shape={tuple(shape)} dtype={dtype} ==")
    for t in timings:
        print(f"{t.name:>18}: {t.ms:8.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=str, default="bf16", help="bf16|fp16|fp32")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--bias", action="store_true", help="Include bias for matmul benchmark")

    # Matmul shape
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)

    # Mean shape
    parser.add_argument(
        "--mean-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROWS", "COLS"),
        help="If set, benchmark mean over last dim for a 2D tensor of this shape.",
    )

    args = parser.parse_args()
    dtype = _parse_dtype(args.dtype)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    # Keep behavior consistent across runs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    bench_matmul(
        m=args.m,
        n=args.n,
        k=args.k,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
        correctness=args.correctness,
        with_bias=args.bias,
    )
    if args.mean_shape is not None:
        bench_mean_lastdim(
            shape=(args.mean_shape[0], args.mean_shape[1]),
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            correctness=args.correctness,
        )

    cutile_ok = getattr(bik, "_CUTILE_AVAILABLE", False)
    if not cutile_ok:
        print("\n(cuTile backend not available: `import cuda.tile as ct` failed; skipping cuTile timings.)")


if __name__ == "__main__":
    main()


