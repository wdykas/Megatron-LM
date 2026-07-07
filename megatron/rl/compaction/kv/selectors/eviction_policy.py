# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Eviction policy as a GRPO agent (eviction-policy RL, offline v0).

Reconstruction objectives preserve what attention LOOKED AT; RL preserves what
the TASK NEEDED — the gap between those selections is the eviction-RL result. Here
compaction decisions become actions with exact logprobs:

- The policy is the learned-oracle scorer architectureitecture (``LearnedOracleScorer``: MLP on
  [key, position/P, layer one-hot], aggregated token-level) — same features,
  different training signal. the supervised scorer fits it to imitate H2O's accumulated-attention
  oracle; the RL policy trains it with task reward.
- Action: per-token retain/evict, sampled Bernoulli(sigmoid(score)) — the
  budget enters as a reward penalty λ·kept_fraction rather than a hard top-k,
  so the set logprob is exact (no Gumbel/Plackett-Luce approximations) and the
  policy can learn prompt-dependent budgets (B3's question, for free).
- Reward: task quality of the retained set minus the budget penalty. The
  canonical offline reward is negative sufficiency-KL through the REAL frozen
  model (``make_sufficiency_reward``): inject the selected KV rows via the
  student forward and compare with the full-cache teacher logits.
- Update: GRPO — sample a group of masks per prompt, normalize rewards within
  the group, REINFORCE on the exact mask logprobs.

Offline v0 per the plan: saved captures + sufficiency-KL as reward proxy, no
server. A trained policy deploys live exactly like the learned-oracle scorer (it IS one:
``policy.scorer`` drops into ``--kv-compaction-oracle-checkpoint``) — with the
caveat that live selection keeps a protected recent window while the policy
was free-form; retrain or eval with the same convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .oracle import LearnedOracleScorer, OracleScorerConfig


@dataclass
class EvictionGRPOConfig:
    """Knobs for ``train_eviction_policy_grpo``."""

    group_size: int = 8          # masks sampled per prompt per step (the GRPO group)
    budget_lambda: float = 1.0   # reward penalty per unit of kept fraction
    lr: float = 3e-4
    entropy_bonus: float = 0.0   # optional exploration term on the Bernoulli entropy
    min_prob: float = 1e-4       # clamp for numerically safe logprobs


class EvictionPolicy(torch.nn.Module):
    """Stochastic retain/evict policy over a request's token-level KV."""

    def __init__(self, cfg: OracleScorerConfig, params_dtype: torch.dtype = torch.float32,
                 pg_collection=None) -> None:
        super().__init__()
        self.scorer = LearnedOracleScorer(cfg, params_dtype=params_dtype,
                                          pg_collection=pg_collection)

    def token_logits(self, keys: list[torch.Tensor]) -> torch.Tensor:
        """Per-token retain logits, differentiable — keys: per layer (S, d_key)."""
        S = keys[0].shape[0]
        feats = torch.cat([
            self.scorer.features(k, li) for li, k in enumerate(keys)
        ])
        return self.scorer(feats).view(len(keys), S).float().mean(dim=0)   # (S,)

    def sample_masks(self, keys: list[torch.Tensor], group_size: int,
                     min_prob: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a GRPO group of retain masks with exact logprobs.

        Returns (masks (G, S) bool, logprobs (G,)) — logprobs carry gradient
        into the policy.
        """
        probs = torch.sigmoid(self.token_logits(keys)).clamp(min_prob, 1 - min_prob)
        masks = torch.bernoulli(probs.detach().expand(group_size, -1)).bool()
        logp = (torch.where(masks, probs.log(), (1 - probs).log())).sum(dim=-1)
        return masks, logp


def make_sufficiency_reward(model, query_tokens: torch.Tensor,
                            full_kv: list[tuple[torch.Tensor, torch.Tensor]],
                            teacher_logits: torch.Tensor,
                            gather_logits: bool = False):
    """Reward = −(mean sufficiency-KL of the retained rows), via the real model.

    full_kv: per attention layer (K, V), each (B, S, d_kv) — the captured full
    cache. The returned callable evaluates one retain mask (S,) bool.
    """
    from megatron.rl.compaction.learned.probes import sufficiency_kl

    def reward(mask: torch.Tensor) -> float:
        if not mask.any():
            # An empty cache is maximally insufficient; avoid a 0-length forward.
            return -float(teacher_logits.shape[-1])
        subset = [(k[:, mask], v[:, mask]) for k, v in full_kv]
        with torch.no_grad():
            kl = sufficiency_kl(model, query_tokens, subset, teacher_logits,
                                gather_logits=gather_logits)
        return -kl.mean().item()

    return reward


def train_eviction_policy_grpo(
    policy: EvictionPolicy,
    prompts: list[tuple[list[torch.Tensor], object]],
    cfg: EvictionGRPOConfig,
    steps: int,
) -> list[dict]:
    """Offline GRPO over eviction masks.

    prompts: list of (keys_per_layer, reward_fn) — keys per layer (S, d_key)
    on GPU; reward_fn(mask (S,) bool) → float task reward for the retained
    set (before the budget penalty; see ``make_sufficiency_reward``).

    Each step: pick a prompt round-robin, sample ``group_size`` masks, score
    reward − λ·kept_fraction, normalize within the group (GRPO), REINFORCE on
    the exact mask logprobs. Returns per-step logs (loss, reward, kept_frac).
    """
    if not prompts:
        raise ValueError("train_eviction_policy_grpo: no prompts given")
    if cfg.group_size < 2:
        raise ValueError(f"group_size must be >= 2 for GRPO, got {cfg.group_size}")
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=1e-4)
    logs = []
    for step in range(steps):
        keys, reward_fn = prompts[step % len(prompts)]
        masks, logp = policy.sample_masks(keys, cfg.group_size, cfg.min_prob)
        kept_frac = masks.float().mean(dim=-1)                        # (G,)
        rewards = torch.tensor(
            [reward_fn(m) for m in masks], device=logp.device, dtype=torch.float32,
        ) - cfg.budget_lambda * kept_frac
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        loss = -(adv * logp).mean()
        if cfg.entropy_bonus > 0.0:
            probs = torch.sigmoid(policy.token_logits(keys)).clamp(
                cfg.min_prob, 1 - cfg.min_prob)
            entropy = -(probs * probs.log() + (1 - probs) * (1 - probs).log()).mean()
            loss = loss - cfg.entropy_bonus * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()
        logs.append({
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_kept_frac": kept_frac.mean().item(),
        })
    return logs
