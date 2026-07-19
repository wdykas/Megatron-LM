# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os

import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    deterministic_index_add,
    set_batch_invariant_mode,
)
from megatron.rl.rl_utils import selective_log_softmax


def test_selective_log_softmax_batch_invariant():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    B, S, V = 4, 7, 16
    device = torch.device("cuda")
    logits = torch.randn(B, S, V, dtype=torch.float32, device=device)
    labels = torch.randint(low=0, high=V, size=(B, S), device=device)

    # Randomly permute the batch dimension; a batch-invariant implementation should
    # produce outputs that are identical up to the same permutation.
    perm = torch.randperm(B, device=device)

    with set_batch_invariant_mode(True):
        batch_invariant_logps = selective_log_softmax(logits, labels)  # [B, S]
        batch_invariant_logps_perm = selective_log_softmax(
            logits[perm], labels[perm]
        )  # [B, S] corresponding to permuted batch

    # Undo the permutation on the permuted outputs and compare elementwise.
    # If the kernel is batch invariant, each example's output should not depend
    # on its position in the batch.
    assert torch.equal(batch_invariant_logps, batch_invariant_logps_perm[perm.argsort()])


def test_deterministic_index_add_masks_invalid_nan_rows():
    out = torch.zeros(2, 2, device="cuda", dtype=torch.float32)
    idx = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int64)
    src = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0], [float("nan"), float("nan")]],
        device="cuda",
        dtype=torch.float32,
    )
    valid_mask = torch.tensor([True, True, False], device="cuda")

    deterministic_index_add(out, idx, src, valid_mask=valid_mask)

    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]], device="cuda", dtype=torch.float32)
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
