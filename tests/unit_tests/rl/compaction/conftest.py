# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compaction tests run on GPU with Megatron model-parallel state initialized.

The compaction models are built from standard Megatron blocks (``MLP``,
``TEDotProductAttention``, TE column/row-parallel linears), which are CUDA-only and
require ``parallel_state`` to be initialized — exactly like every other Megatron
module unit test. We initialize a single-rank (tp=1, dp=1) model-parallel world for
the whole session, so the compactor builds *replicated* (tp=1) without any manual
process-group plumbing. Tests run on CUDA in fp32 (TE DotProductAttention falls
back from flash for fp32, so numerics stay precise and existing tolerances hold).
"""

import os

import pytest
import torch

# Set the default device to CUDA at import time (before any fixture of any scope
# constructs a model), so module/session-scoped fixtures also build on GPU.
if torch.cuda.is_available():
    torch.set_default_device("cuda")


@pytest.fixture(scope="session", autouse=True)
def _megatron_parallel_state():
    """Initialize a single-rank (tp=1, dp=1) Megatron world for the session."""
    if not torch.cuda.is_available():
        yield
        return
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12399")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(123)
    yield


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("compaction stack is GPU-only (TransformerEngine requires CUDA)")
    yield
