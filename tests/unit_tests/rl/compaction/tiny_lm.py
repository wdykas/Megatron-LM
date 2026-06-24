# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Minimal causal LM for unit tests.

TinyLM and TinyLMAdapter are test-only utilities — they live here rather than
in the production package so production imports stay clean.

Usage in tests:
    from .tiny_lm import TinyLM, TinyLMAdapter
    adapter = TinyLMAdapter(TinyLM(vocab_size=64, n_layers=2, d_model=32, n_heads=4))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.rl.compaction.learned.training.data import CompactKV
from megatron.rl.compaction.learned.training.data import FrozenModelAdapter


class _TinyLMLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out   = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1   = nn.Linear(d_model, d_model * 4, bias=False)
        self.ff2   = nn.Linear(d_model * 4, d_model, bias=False)

    def forward(self, x, past_k=None, past_v=None):
        B, S, d = x.shape
        h, dh = self.n_heads, self.d_head
        q, k, v = self.qkv(self.norm1(x)).split(d, dim=-1)
        k_full = torch.cat([past_k, k], dim=1) if past_k is not None else k
        v_full = torch.cat([past_v, v], dim=1) if past_v is not None else v
        S_total = k_full.shape[1]
        S_past  = S_total - S
        q_mh = q.view(B, S, h, dh).transpose(1, 2)
        k_mh = k_full.view(B, S_total, h, dh).transpose(1, 2)
        v_mh = v_full.view(B, S_total, h, dh).transpose(1, 2)
        scores = (q_mh @ k_mh.transpose(-2, -1)) / math.sqrt(dh)
        mask = torch.ones(S, S_total, dtype=torch.bool, device=x.device)
        for i in range(S):
            mask[i, S_past + i + 1:] = False
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        out_mh = (F.softmax(scores, dim=-1) @ v_mh).transpose(1, 2).contiguous().view(B, S, d)
        x = x + self.out(out_mh)
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x, k_full, v_full


class TinyLM(nn.Module):
    """Minimal causal LM for testing data collection and training loops."""

    def __init__(self, vocab_size: int, n_layers: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self._vocab_size = vocab_size
        self._n_layers   = n_layers
        self._d_model    = d_model
        self._n_heads    = n_heads
        self._d_head     = d_model // n_heads
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.layers  = nn.ModuleList([_TinyLMLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids, past_kv=None):
        x = self.embed(token_ids)
        new_kv = []
        for i, layer in enumerate(self.layers):
            pk = past_kv[i][0] if past_kv is not None else None
            pv = past_kv[i][1] if past_kv is not None else None
            x, k, v = layer(x, pk, pv)
            new_kv.append((k, v))
        return self.lm_head(self.norm(x)), new_kv


class TinyLMAdapter:
    """FrozenModelAdapter wrapping TinyLM (no RoPE — KV is position-free)."""

    def __init__(self, model: TinyLM) -> None:
        self.model = model

    @property
    def n_layers(self) -> int: return self.model._n_layers
    @property
    def d_head(self) -> int: return self.model._d_head
    @property
    def n_heads(self) -> int: return self.model._n_heads
    @property
    def vocab_size(self) -> int: return self.model._vocab_size

    @torch.no_grad()
    def prefill(self, token_ids, kv_prefix, logical_start):
        self.model.eval()
        logits, full_kv = self.model(token_ids, past_kv=kv_prefix)
        if kv_prefix is not None:
            prefix_len = kv_prefix[0][0].shape[1]
            chunk_kv = [(k[:, prefix_len:, :], v[:, prefix_len:, :]) for k, v in full_kv]
        else:
            chunk_kv = full_kv
        return chunk_kv, logits

    def student_logits(self, query_tokens, compact_kv, logical_kv_start, logical_query_start):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(query_tokens, past_kv=compact_kv)
        return logits

    def make_student_fn(self):
        def _fn(query_tokens, compact_kv):
            return self.student_logits(query_tokens, compact_kv, 0, 0)
        return _fn
