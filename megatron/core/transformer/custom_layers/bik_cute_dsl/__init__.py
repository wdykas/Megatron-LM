# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Cute-DSL kernel backend for batch-invariant mode.

Vendored TRT-LLM kernels (see `README.md` for provenance + API-drift patches)
wrapped by `bik_cute_backend.mm_cute_dsl` / `addmm_cute_dsl`. Selected via
`TransformerConfig.batch_invariant_kernel_backend = "cute_dsl"` (default is
`"deepgemm"`).
"""
