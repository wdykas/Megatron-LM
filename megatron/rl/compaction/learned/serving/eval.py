# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Evaluation harness for KV compaction quality.

Three complementary metrics measure whether a compactor preserves what matters:

    1. Perplexity ratio
       ppl_full / ppl_compact — how much worse is the model under compaction.
       A ratio of 1.0 = lossless.  Higher = worse.

    2. Needle-in-haystack accuracy
       Can the model answer a specific factual question after compacting the
       haystack?  Measures verbatim recall / exact retrieval.

    3. QA accuracy
       Open-domain question answering accuracy with compacted context.
       Measures semantic preservation under compression.

All metrics are computed via a FrozenModelAdapter so they are independent of
the underlying framework (Megatron, vLLM, HuggingFace, TinyLM for tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import torch
import torch.nn.functional as F

from megatron.rl.compaction.kv.compressors import CompactionResult
from megatron.rl.compaction.learned.models.compactor import PerceiverCompactor
from megatron.rl.compaction.learned.training.data import CompactKV, FrozenModelAdapter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Settings for CompactionEvaluator.

    Attributes
    ----------
    chunk_size:   Token chunk size used when building the KV cache.
    max_new_tokens: Maximum generation length for QA / needle tasks.
    temperature:  Sampling temperature (0 = greedy decode).
    batch_size:   Batch size for evaluation.
    """

    chunk_size: int = 256
    max_new_tokens: int = 32
    temperature: float = 0.0
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Perplexity metric helpers
# ---------------------------------------------------------------------------

def _token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token NLL from logits (B, S, V) and targets (B, S)."""
    B, S, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, S)
    return nll


def token_perplexity(nll_values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Compute perplexity from per-token NLL values.

    nll_values: (B, S) or (S,)
    mask:       Optional boolean mask, True = include.
    """
    if mask is not None:
        nll_values = nll_values[mask]
    return float(nll_values.mean().exp().item())


# ---------------------------------------------------------------------------
# Compressor interface
# ---------------------------------------------------------------------------

class _CompressorProtocol:
    """Expected interface for any compressor (duck-typed)."""

    def initial_compress(
        self,
        keys_per_layer: list[torch.Tensor],
        values_per_layer: list[torch.Tensor],
    ) -> Any: ...

    def __call__(self, memory: Any, keys: list, values: list) -> Any: ...


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class CompactionEvaluator:
    """Measure perplexity, needle recall, and QA accuracy under compaction.

    All methods return a dict with metric names → float values so results
    can be logged directly to wandb/tensorboard.

    Usage:
        evaluator = CompactionEvaluator(adapter, EvalConfig())
        results   = evaluator.perplexity(token_ids, compressor)
        # results = {"ppl_full": 12.3, "ppl_compact": 14.1, "ppl_ratio": 1.15}
    """

    def __init__(self, adapter: FrozenModelAdapter, config: EvalConfig = EvalConfig()) -> None:
        self.adapter = adapter
        self.cfg = config

    # ------------------------------------------------------------------
    # 1. Perplexity
    # ------------------------------------------------------------------

    def perplexity(
        self,
        token_ids: torch.Tensor,    # (B, total_len) or (total_len,)
        compressor: Any,            # BeliefUpdater / GatedRecurrentUpdater
    ) -> dict[str, float]:
        """Measure token-level perplexity with and without compaction.

        Splits token_ids into chunks, collects KV for each, compresses them
        into a compact memory, and evaluates teacher logits vs student logits
        on the continuation of each chunk.

        Returns: {"ppl_full", "ppl_compact", "ppl_ratio"}.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        B, total_len = token_ids.shape
        chunk_size = self.cfg.chunk_size
        chunk_starts = list(range(0, total_len - 1, chunk_size))

        full_nlls: list[torch.Tensor] = []
        compact_nlls: list[torch.Tensor] = []

        memory = None

        for ci, start in enumerate(chunk_starts):
            end = min(start + chunk_size, total_len - 1)
            if end <= start:
                continue

            chunk_ids = token_ids[:, start:end]          # (B, T)
            target_ids = token_ids[:, start + 1:end + 1] # (B, T) — shifted

            # Full-context teacher logits
            full_kv, full_logits = self.adapter.prefill(chunk_ids, kv_prefix=None, logical_start=start)

            # NLL under full context
            full_nll = _token_nll(full_logits, target_ids)  # (B, T)
            full_nlls.append(full_nll)

            # Build compact memory
            chunk_keys   = [k for k, _ in full_kv]
            chunk_values = [v for _, v in full_kv]

            if memory is None:
                memory = compressor.initial_compress(chunk_keys, chunk_values)
            else:
                memory = compressor(memory, chunk_keys, chunk_values)

            # Student logits via compact memory
            compact_kv: CompactKV = list(zip(memory.keys_list(), memory.values_list()))
            student_logits = self.adapter.student_logits(
                chunk_ids,
                compact_kv,
                logical_kv_start=0,
                logical_query_start=start,
            )
            compact_nll = _token_nll(student_logits, target_ids)  # (B, T)
            compact_nlls.append(compact_nll)

        if not full_nlls:
            return {"ppl_full": float("nan"), "ppl_compact": float("nan"), "ppl_ratio": float("nan")}

        all_full    = torch.cat(full_nlls,    dim=-1)   # (B, total_tokens)
        all_compact = torch.cat(compact_nlls, dim=-1)

        ppl_full    = token_perplexity(all_full)
        ppl_compact = token_perplexity(all_compact)
        ppl_ratio   = ppl_compact / (ppl_full + 1e-8)

        return {"ppl_full": ppl_full, "ppl_compact": ppl_compact, "ppl_ratio": ppl_ratio}

    # ------------------------------------------------------------------
    # 2. Needle-in-haystack
    # ------------------------------------------------------------------

    def needle_in_haystack(
        self,
        haystack_tokens: torch.Tensor,  # (B, H) — long context
        query_tokens: torch.Tensor,      # (B, Q)
        answer_ids: torch.Tensor,        # (B, A) — expected answer token IDs
        compressor: Any,
    ) -> dict[str, float]:
        """Test whether a fact inserted in the haystack survives compaction.

        Collects KV for the haystack in chunks, compresses to compact memory,
        then generates continuations for the query and measures exact-match
        accuracy against answer_ids.

        Returns: {"exact_match", "token_overlap"}.
        """
        B = haystack_tokens.shape[0]
        chunk_size = self.cfg.chunk_size

        # Build compact memory over the haystack
        memory = None
        for start in range(0, haystack_tokens.shape[1], chunk_size):
            end = min(start + chunk_size, haystack_tokens.shape[1])
            chunk = haystack_tokens[:, start:end]
            full_kv, _ = self.adapter.prefill(chunk, kv_prefix=None, logical_start=start)
            keys   = [k for k, _ in full_kv]
            values = [v for _, v in full_kv]
            if memory is None:
                memory = compressor.initial_compress(keys, values)
            else:
                memory = compressor(memory, keys, values)

        compact_kv: CompactKV = list(zip(memory.keys_list(), memory.values_list()))

        # Generate answer tokens (greedy)
        generated = self._greedy_generate(
            query_tokens,
            compact_kv,
            max_new_tokens=self.cfg.max_new_tokens,
        )                                                   # (B, max_new_tokens)

        # Exact match: generated tokens must include answer_ids as a prefix
        A = answer_ids.shape[1]
        gen_prefix = generated[:, :A]                      # (B, A)
        exact_match = float((gen_prefix == answer_ids).all(dim=-1).float().mean().item())

        # Token overlap (F1-style): fraction of answer tokens that appear in generation
        overlap_scores = []
        for b in range(B):
            ans_set = set(answer_ids[b].tolist())
            gen_set = set(generated[b].tolist())
            if not ans_set:
                overlap_scores.append(1.0)
                continue
            overlap = len(ans_set & gen_set) / len(ans_set)
            overlap_scores.append(overlap)
        token_overlap = sum(overlap_scores) / len(overlap_scores)

        return {"exact_match": exact_match, "token_overlap": token_overlap}

    # ------------------------------------------------------------------
    # 3. QA accuracy
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: greedy generation with compact KV prefix
    # ------------------------------------------------------------------

    def _greedy_generate(
        self,
        prompt_tokens: torch.Tensor,     # (B, S_q)
        compact_kv: CompactKV,
        max_new_tokens: int,
    ) -> torch.Tensor:                   # (B, max_new_tokens)
        """Greedy decode max_new_tokens tokens given prompt and compact KV.

        At each step the full history (prompt + generated so far) is fed to
        student_logits so the model conditions on all prior output tokens.
        O(n²) in generation length — acceptable for evaluation.
        """
        generated = []
        history = prompt_tokens          # (B, S_q + steps_so_far)

        for _ in range(max_new_tokens):
            logits = self.adapter.student_logits(
                history,
                compact_kv,
                logical_kv_start=0,
                logical_query_start=0,
            )                            # (B, len(history), vocab)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
            generated.append(next_tok)
            history = torch.cat([history, next_tok], dim=1)

        return torch.cat(generated, dim=1)   # (B, max_new_tokens)


# ---------------------------------------------------------------------------
# Summarise all metrics in one call
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CompactorAdapter (moved from benchmark_adapter.py)
# ---------------------------------------------------------------------------

class CompactorAdapter:
    """Wraps a trained PerceiverCompactor as an KVCompressor.

    The compactor is called in eval mode (no gradient). Inputs may be numpy
    arrays or torch tensors; outputs are always torch tensors to be compatible
    with the PyTorch-based KVCompactionBenchmark.

    Parameters
    ----------
    compactor:  A trained (or randomly initialised) PerceiverCompactor.
    layer_idx:  Which transformer layer to use. Default 0.
    device:     Torch device for inference. Default: cpu.
    """

    def __init__(
        self,
        compactor: PerceiverCompactor,
        layer_idx: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.compactor = compactor
        self.layer_idx = layer_idx
        self.device = torch.device(device)

    @property
    def strategy(self) -> str:
        cfg = self.compactor.cfg
        return (
            f"still_C{cfg.n_compress}_d{cfg.d_kv}"
            + ("_shared" if cfg.share_across_layers else "")
        )

    def compress(
        self,
        keys,        # (T, d) — numpy array or torch.Tensor
        values,      # (T, d) — numpy array or torch.Tensor
        budget: int, # ignored: STILL budget is set by cfg.n_compress
        ref_queries=None,  # (n, d) — not used by STILL (learned queries)
        run_id: str = "",
        step_id: int = 0,
    ) -> CompactionResult:
        """Run the PerceiverCompactor on the provided KV cache."""
        t0 = time.perf_counter()

        # Accept both numpy and torch inputs
        if isinstance(keys, np.ndarray):
            K_t = torch.from_numpy(keys).float()
            V_t = torch.from_numpy(values).float()
        else:
            K_t = keys.float()
            V_t = values.float()

        T, d = K_t.shape
        C = self.compactor.cfg.n_compress

        # Add batch dimension and move to device
        K_in = K_t.unsqueeze(0).to(self.device)    # (1, T, d)
        V_in = V_t.unsqueeze(0).to(self.device)    # (1, T, d)

        self.compactor.eval()
        with torch.no_grad():
            C_k, C_v = self.compactor(K_in, V_in, layer_idx=self.layer_idx)  # (1, C, d)

        ck = C_k.squeeze(0).cpu()   # (C, d) — torch tensor
        cv = C_v.squeeze(0).cpu()   # (C, d) — torch tensor

        return CompactionResult(
            run_id=run_id,
            step_id=step_id,
            retained_positions=list(range(C)),   # synthetic; no literal selection
            compacted_keys=ck,
            compacted_values=cv,
            bias=torch.zeros(C),
            strategy=self.strategy,
            original_length=T,
            wall_time_s=time.perf_counter() - t0,
        )
