# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for sufficiency-KL probes (learned/probes.py) through a real Megatron model."""

import os

import pytest
import torch

from megatron.rl.compaction.learned.probes import sufficiency_kl
from megatron.rl.compaction.learned.training.losses import per_token_kl


class TestKlFromLogits:
    def test_identical_logits_zero_kl(self):
        logits = torch.randn(2, 8, 64, device="cuda")
        kl = per_token_kl(logits, logits.clone())
        assert kl.shape == (2, 8)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

    def test_kl_positive_and_asymmetric(self):
        torch.manual_seed(0)
        p = torch.randn(1, 4, 32, device="cuda")
        q = torch.randn(1, 4, 32, device="cuda")
        kl_pq = per_token_kl(p, q)
        kl_qp = per_token_kl(q, p)
        assert (kl_pq > 0).all()
        assert not torch.allclose(kl_pq, kl_qp)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            per_token_kl(torch.randn(1, 4, 32, device="cuda"),
                           torch.randn(1, 4, 16, device="cuda"))


class TestSufficiencyKlRealModel:
    """End-to-end through a small Megatron GPTModel (GQA + RoPE)."""

    @pytest.fixture(scope="class")
    def model_and_data(self):
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29581")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        from tests.unit_tests.test_utilities import Utils
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)

        cfg = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=16,
            # bf16: the fp32+THD combination has no TE attention backend, and
            # the real stack always runs bf16 anyway.
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
        )
        P, SQ = 96, 32
        S = P + SQ
        vocab = 128
        model = GPTModel(
            config=cfg,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=vocab,
            max_sequence_length=S,
            position_embedding_type="rope",
        ).cuda().eval()

        ids = torch.randint(0, vocab, (1, S), device="cuda")
        pos = torch.arange(S, device="cuda").unsqueeze(0)
        cu = torch.tensor([0, S], dtype=torch.int32, device="cuda")
        psp = PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_kv=cu,
                              max_seqlen_q=S, max_seqlen_kv=S, total_tokens=S)
        with torch.no_grad():
            teacher = model(input_ids=ids, position_ids=pos, attention_mask=None,
                            packed_seq_params=psp)          # (1, S, vocab)

        from megatron.rl.compaction.learned.capture.kv_capture import capture_kv_from_forward
        prefix_ids = ids[:, :P]
        prefix_pos = pos[:, :P]
        keys, vals = capture_kv_from_forward(model, prefix_ids, prefix_pos)
        prefix_kv = [(k.cuda(), v.cuda()) for k, v in zip(keys, vals)]  # (1, P, d_kv) each

        yield {
            "model": model,
            "query_tokens": ids[:, P:],
            "teacher_logits": teacher[:, P:, :],
            "prefix_kv": prefix_kv,
            "P": P,
            "SQ": SQ,
        }
        Utils.destroy_model_parallel()

    def test_eviction_policy_grpo_with_real_reward(self, model_and_data):
        """B1 v0 end-to-end: GRPO steps with sufficiency-KL reward through the
        real model — rewards finite, gradients flow, keeping more helps."""
        from megatron.rl.compaction.kv.eviction_policy import (
            EvictionGRPOConfig, EvictionPolicy, make_sufficiency_reward,
            train_eviction_policy_grpo,
        )
        from megatron.rl.compaction.kv.oracle import OracleScorerConfig
        d = model_and_data
        reward_fn = make_sufficiency_reward(
            d["model"], d["query_tokens"], d["prefix_kv"], d["teacher_logits"])

        # Sanity: the full cache is a better retained set than a sliver.
        P = d["P"]
        full = torch.ones(P, dtype=torch.bool, device="cuda")
        sliver = torch.zeros(P, dtype=torch.bool, device="cuda")
        sliver[:4] = True
        assert reward_fn(full) > reward_fn(sliver)

        d_kv = d["prefix_kv"][0][0].shape[-1]
        keys = [k[0].float() for k, _ in d["prefix_kv"]]        # per layer (P, d_kv)
        policy = EvictionPolicy(
            OracleScorerConfig(d_key=d_kv, n_layers=len(keys), hidden=32)).cuda()
        logs = train_eviction_policy_grpo(
            policy, [(keys, reward_fn)],
            EvictionGRPOConfig(group_size=4, budget_lambda=0.5, lr=1e-3), steps=3)
        assert len(logs) == 3
        assert all(torch.isfinite(torch.tensor(lg["loss"])) for lg in logs)
        assert all(torch.isfinite(torch.tensor(lg["mean_reward"])) for lg in logs)

    def test_student_and_teacher_outputs_capture_hidden(self, model_and_data):
        """C5 plumbing through the real model: hidden capture + gradient flow."""
        from megatron.rl.compaction.learned.capture.student_forward import (
            student_outputs, teacher_outputs,
        )
        from megatron.rl.compaction.learned.training.losses import future_latent_loss
        d = model_and_data
        B, SQ = d["query_tokens"].shape
        H = 128  # hidden_size of the fixture model

        teach = teacher_outputs(d["model"], d["query_tokens"])
        assert teach.logits.shape[:2] == (B, SQ)
        assert teach.hidden.shape == (B, SQ, H)
        assert torch.isfinite(teach.hidden.float()).all()

        compact_kv = [(k.clone().requires_grad_(True), v.clone().requires_grad_(True))
                      for k, v in d["prefix_kv"]]
        stud = student_outputs(d["model"], d["query_tokens"], compact_kv)
        assert stud.hidden.shape == (B, SQ, H)

        loss = future_latent_loss(stud.hidden, teach.hidden)
        assert torch.isfinite(loss)
        loss.backward()
        # Gradient must reach the compact KV through the hidden states.
        assert compact_kv[0][0].grad is not None
        assert compact_kv[0][0].grad.abs().sum() > 0

    def test_future_latent_loss_shape_guard(self):
        from megatron.rl.compaction.learned.training.losses import future_latent_loss
        a = torch.randn(1, 4, 8, device="cuda")
        with pytest.raises(ValueError, match="hidden shape mismatch"):
            future_latent_loss(a, torch.randn(1, 5, 8, device="cuda"))
        # identical hidden -> zero loss
        assert future_latent_loss(a, a.clone()).item() == 0.0

    def test_shape_and_finite(self, model_and_data):
        d = model_and_data
        kl = sufficiency_kl(d["model"], d["query_tokens"], d["prefix_kv"], d["teacher_logits"])
        assert kl.shape == (1, d["SQ"])
        assert torch.isfinite(kl).all()
        assert (kl >= 0).all()

    def test_more_context_is_more_sufficient(self, model_and_data):
        """Full prefix KV must be closer to the teacher than a 4-slot cache."""
        d = model_and_data
        kl_full = sufficiency_kl(
            d["model"], d["query_tokens"], d["prefix_kv"], d["teacher_logits"]
        ).mean()
        tiny_kv = [(k[:, :4, :], v[:, :4, :]) for k, v in d["prefix_kv"]]
        kl_tiny = sufficiency_kl(
            d["model"], d["query_tokens"], tiny_kv, d["teacher_logits"]
        ).mean()
        assert kl_full < kl_tiny, f"full-KV KL {kl_full:.4f} !< tiny-KV KL {kl_tiny:.4f}"

    def test_selection_baseline_pluggable(self, model_and_data):
        """A selection compressor's retained KV drops into the probe directly."""
        from megatron.rl.compaction.kv import StreamingLLMCompressor
        d = model_and_data
        comp = StreamingLLMCompressor(n_sink=4, fit_bias=False, fit_values=False)
        compact_kv = []
        for k, v in d["prefix_kv"]:
            r = comp.compress(k[0], v[0], budget=d["P"] // 2)
            compact_kv.append((r.compacted_keys.unsqueeze(0), r.compacted_values.unsqueeze(0)))
        kl = sufficiency_kl(d["model"], d["query_tokens"], compact_kv, d["teacher_logits"])
        assert torch.isfinite(kl).all()
