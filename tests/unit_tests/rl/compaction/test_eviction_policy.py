# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the GRPO eviction policy (eviction-policy RL, offline v0)."""

import pytest
import torch

from megatron.rl.compaction.kv.selectors.eviction_policy import (
    EvictionGRPOConfig,
    EvictionPolicy,
    train_eviction_policy_grpo,
)
from megatron.rl.compaction.kv.selectors.oracle import OracleScorerConfig

L, S, D = 2, 40, 16


def _keys(seed, important_dir=None, n_important=10):
    """Per-layer (S, D) keys; the first n_important tokens carry a direction."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    keys = []
    for _ in range(L):
        k = torch.randn(S, D, device="cuda", generator=g)
        if important_dir is not None:
            k[:n_important] += important_dir * 3.0
        keys.append(k)
    return keys


class TestEvictionPolicy:
    def test_sample_masks_shapes_and_logprob_grad(self):
        policy = EvictionPolicy(OracleScorerConfig(d_key=D, n_layers=L, hidden=32)).cuda()
        masks, logp = policy.sample_masks(_keys(0), group_size=4)
        assert masks.shape == (4, S) and masks.dtype == torch.bool
        assert logp.shape == (4,) and logp.requires_grad
        logp.sum().backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_logprob_is_exact_bernoulli(self):
        policy = EvictionPolicy(OracleScorerConfig(d_key=D, n_layers=L, hidden=32)).cuda()
        keys = _keys(1)
        torch.manual_seed(0)
        masks, logp = policy.sample_masks(keys, group_size=3)
        probs = torch.sigmoid(policy.token_logits(keys)).clamp(1e-4, 1 - 1e-4)
        want = torch.where(masks, probs.log(), (1 - probs).log()).sum(-1)
        torch.testing.assert_close(logp, want)


class TestGRPOTraining:
    def test_learns_to_keep_important_tokens(self):
        """Reward = recall of planted-important tokens − λ·kept: the policy
        must raise retain-probability on important tokens above the rest."""
        torch.manual_seed(0)
        u = torch.randn(1, D, device="cuda")
        u = u / u.norm()
        n_imp = 10
        prompts = []
        for seed in range(4):
            keys = _keys(seed, important_dir=u, n_important=n_imp)
            def reward_fn(mask, n_imp=n_imp):
                return 2.0 * mask[:n_imp].float().mean().item()
            prompts.append((keys, reward_fn))

        policy = EvictionPolicy(OracleScorerConfig(d_key=D, n_layers=L, hidden=32)).cuda()
        cfg = EvictionGRPOConfig(group_size=8, budget_lambda=1.0, lr=3e-3)
        logs = train_eviction_policy_grpo(policy, prompts, cfg, steps=120)

        held_out = _keys(99, important_dir=u, n_important=n_imp)
        probs = torch.sigmoid(policy.token_logits(held_out))
        p_imp = probs[:n_imp].mean().item()
        p_rest = probs[n_imp:].mean().item()
        assert p_imp > p_rest + 0.2, f"important {p_imp:.3f} vs rest {p_rest:.3f}"
        # Budget pressure: the policy should not keep everything.
        assert logs[-1]["mean_kept_frac"] < 0.9

    def test_config_validation(self):
        policy = EvictionPolicy(OracleScorerConfig(d_key=D, n_layers=L, hidden=32)).cuda()
        with pytest.raises(ValueError, match="no prompts"):
            train_eviction_policy_grpo(policy, [], EvictionGRPOConfig(), steps=1)
        with pytest.raises(ValueError, match="group_size"):
            train_eviction_policy_grpo(
                policy, [(_keys(0), lambda m: 0.0)],
                EvictionGRPOConfig(group_size=1), steps=1)
