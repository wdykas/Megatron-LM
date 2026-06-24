# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compaction tests run on GPU.

The compaction models use TransformerEngine (Megatron TE wrappers), which is
CUDA-only — this stack is GPU-only by design. We default every compaction unit
test to the GPU so module construction and forward run on CUDA, in fp32 (the TE
DotProductAttention falls back from flash for fp32, so numerics stay precise and
existing tolerances hold).
"""

import pytest
import torch

# Set the default device to CUDA at import time (before any fixture of any scope
# constructs a model), so module/session-scoped fixtures also build on GPU.
if torch.cuda.is_available():
    torch.set_default_device("cuda")


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("compaction stack is GPU-only (TransformerEngine requires CUDA)")
    yield
