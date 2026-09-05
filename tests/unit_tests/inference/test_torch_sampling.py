# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import torch

from megatron.core.inference.sampling.torch_sampling import TorchSampling


def test_sample_from_bf16_logits_uses_fp32_probabilities():
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, -0.5, 0.0]], dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(42)

    with mock.patch("torch.multinomial", return_value=torch.zeros((2, 1), dtype=torch.int64)) as fn:
        TorchSampling.sample_from_logits(
            logits, temperature=1.0, top_k=0, top_p=0.0, generator=generator
        )

    probabilities = fn.call_args.args[0]
    assert probabilities.dtype == torch.float32
    torch.testing.assert_close(probabilities, logits.float().softmax(dim=-1))
