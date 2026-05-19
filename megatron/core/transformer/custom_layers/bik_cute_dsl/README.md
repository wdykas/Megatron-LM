# bik_cute_dsl

Cute-DSL kernel backend for `batch_invariant_mode`. Selected via
`TransformerConfig.batch_invariant_kernel_backend = "cute_dsl"` (default is
`"deepgemm"`).

## Provenance

The kernel files in this subpackage are **vendored verbatim** from TensorRT-LLM:

- Upstream: https://github.com/NVIDIA/TensorRT-LLM/tree/e9c4b18a/tensorrt_llm/_torch/cute_dsl_kernels/blackwell
- Pinned commit: `e9c4b18afdd8439738b5816adb557afafa34fa14`

Vendored files (copied with no functional changes):
- `dense_gemm_persistent.py` — `PersistentDenseGemmKernel` (SM100 persistent dense bf16/fp16 GEMM)
- `blockscaled_contiguous_grouped_gemm.py` — `Sm100BlockScaledContiguousGroupedGemmKernel` (block-scaled MXFP8/MXFP4/NVF4 grouped GEMM — currently not exposed in BIK, see below)
- `custom_pipeline.py`, `utils.py` — dependencies of the above

## API drift patches

The vendored kernels were written against a newer cute-dsl than 4.4.2 (the
version available at integration time). Two enum constants were replaced with
the string literals the older API expects:

| Original                              | Replaced with        |
|---------------------------------------|----------------------|
| `cute.arch.ProxyKind.async_shared`    | `"async.shared"`     |
| `cute.arch.SharedSpace.shared_cta`    | `"cta"`              |

These are applied directly in the vendored files (search the source for the
new literals). When upstream cute-dsl is bumped past the point where the enum
exists, these can be reverted to use the named constants.

## Files added in this repo (not vendored)

- `bik_cute_backend.py` — thin PyTorch wrappers (`mm_cute_dsl`, `addmm_cute_dsl`)
   that adapt the cute-dsl entry points to BIK's functional surface. Caches
   the compiled kernel by dtype on first call.
- `__init__.py`, this README.

## Scope

- Dense GEMM (`mm` / `addmm`) is the only path currently wired up. The grouped
  GEMM kernel is included for completeness but is **block-scaled (MXFP8/MXFP4/NVF4)**;
  BIK's MoE path is bf16, so the grouped GEMM continues to use DeepGEMM
  `m_grouped_bf16_gemm_nt_contiguous`. Exposing the block-scaled grouped GEMM
  would require adding MXFP quantization to the MoE expert path.

## Verification

The vendored dense GEMM was verified bitwise-deterministic across processes
after the API-drift patches (see the standalone smoke test). DeepGEMM bf16
remains the default backend; cute-dsl is opt-in.
