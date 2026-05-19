# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cute-DSL backend for batch-invariant kernels.

Wraps the (verbatim) TensorRT-LLM cute_dsl kernels in this subpackage with
thin entry points (`mm_cute_dsl`, `addmm_cute_dsl`) that match BIK's mm/addmm
surface. Selected via `TransformerConfig.batch_invariant_kernel_backend =
"cute_dsl"` (default backend is `"deepgemm"`).

JIT-compiled once per dtype and cached; subsequent calls reuse the artifact.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

_LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy availability check.  cute-dsl is only imported on first use so that the
# rest of BIK works on systems without it installed.
# ──────────────────────────────────────────────────────────────────────────────

_CUTE_AVAILABLE: Optional[bool] = None


def _cute_available() -> bool:
    global _CUTE_AVAILABLE
    if _CUTE_AVAILABLE is None:
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401
            import cutlass.torch  # noqa: F401

            _CUTE_AVAILABLE = True
        except Exception as e:
            _LOGGER.warning("cute-dsl not available: %s", e)
            _CUTE_AVAILABLE = False
    return _CUTE_AVAILABLE


# ──────────────────────────────────────────────────────────────────────────────
# Dense GEMM (mm / addmm) via PersistentDenseGemmKernel
# ──────────────────────────────────────────────────────────────────────────────


_DENSE_KERNEL_CACHE: dict = {}


def _get_dense_kernel(dtype: torch.dtype):
    """Return a (possibly cached) JIT-compiled wrapper for the dense GEMM."""
    import cutlass

    from .dense_gemm_persistent import PersistentDenseGemmKernel

    cached = _DENSE_KERNEL_CACHE.get(dtype)
    if cached is not None:
        return cached

    if dtype == torch.bfloat16:
        ab_dtype = cutlass.BFloat16
    elif dtype == torch.float16:
        ab_dtype = cutlass.Float16
    else:
        raise NotImplementedError(
            f"cute_dsl BIK backend supports only bf16/fp16, got {dtype}"
        )

    # Tile/cluster choice mirrors TRT-LLM's bf16 defaults for Blackwell;
    # deterministic output regardless of problem size (no autotuning).
    kernel = PersistentDenseGemmKernel(
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=True,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 1),
        use_tma_store=True,
    )
    _DENSE_KERNEL_CACHE[dtype] = (kernel, ab_dtype)
    return _DENSE_KERNEL_CACHE[dtype]


def mm_cute_dsl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batch-invariant `a @ b` via the cute-dsl persistent dense GEMM.

    Args:
        a: 2-D (M, K) tensor, bf16 or fp16.
        b: 2-D (K, N) tensor, same dtype as `a`.
    Returns:
        c: (M, N) tensor, same dtype as `a`.
    """
    if not _cute_available():
        raise RuntimeError(
            "cute-dsl is not installed; install nvidia-cutlass-dsl to use the "
            "cute_dsl BIK backend."
        )

    assert a.dim() == 2 and b.dim() == 2, "mm expects 2-D tensors"
    assert a.dtype == b.dtype, f"a/b dtype mismatch: {a.dtype} vs {b.dtype}"
    assert a.shape[1] == b.shape[0], "incompatible K dimension"
    assert a.is_cuda and b.is_cuda, "inputs must be on CUDA"

    import cutlass.torch as cutlass_torch

    M, _ = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    kernel, _ = _get_dense_kernel(a.dtype)

    # PersistentDenseGemmKernel expects A=(M,K,B), B=(N,K,B), C=(M,N,B) — all
    # K-major. With batch=1 we add a singleton trailing dim; B comes in as
    # (K,N), so transpose to (N,K).
    a_cute = cutlass_torch.from_dlpack(a.unsqueeze(-1))
    b_cute = cutlass_torch.from_dlpack(b.transpose(0, 1).contiguous().unsqueeze(-1))
    c_cute = cutlass_torch.from_dlpack(c.unsqueeze(-1))

    stream = cutlass_torch.current_stream()
    # Persistent scheduler caps itself at hardware availability; this is just
    # an upper bound hint.
    max_active_clusters = torch.cuda.get_device_properties(a.device).multi_processor_count // 4
    kernel(a_cute, b_cute, c_cute, max_active_clusters, stream)
    return c


def addmm_cute_dsl(
    bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Batch-invariant `bias + a @ b` via the cute-dsl persistent dense GEMM.

    The cute-dsl kernel doesn't support a fused bias add; we add the bias
    after the GEMM. This matches what the BIK Triton kernel does today.
    """
    out = mm_cute_dsl(a, b)
    if bias is not None:
        out = out + bias
    return out


# Grouped GEMM via Sm100BlockScaledContiguousGroupedGemmKernel is vendored in
# this subpackage but not wired up here: the kernel is block-scaled
# (MXFP8/MXFP4/NVF4) and the BIK MoE path is bf16. Grouped GEMM continues to
# use DeepGEMM. See `README.md` for details.
