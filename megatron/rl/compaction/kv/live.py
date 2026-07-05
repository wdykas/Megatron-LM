# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Live post-prefill KV compaction inside the Megatron dynamic inference engine.

After a request's prefill forward completes (eagerly — decode steps may replay
CUDA graphs, prefill does not when the server runs with
``--decode-only-cuda-graphs``), the compactor scores the request's prompt KV and
prunes the paged cache to a budget before decoding starts.

Deployment constraint (why this is token-level): the paged KV cache shares ONE
block table per request across all layers and heads, so eviction drops a token
position everywhere at once. Scores are therefore aggregated over layers and
KV heads — the SnapKV observation-window attention, summed — and one retained
set is applied per request via ``MegatronInferenceHook.apply_mask_for_request``.

Live strategies:
  snapkv         -- observation-window attention scores (real Q captured from the
                    prefill forward), pooled, recent window kept. Li et al. 2024.
  streaming_llm  -- attention sinks + recent window (positional; no Q needed).

H2O hard-fails live by construction (its accumulated-attention score needs
materialised attention weights that flash never exposes) — use snapkv, or run
H2O offline. See ``build_kv_compressor``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from .compressors import _select_recent_plus_heavy
from .megatron_hook import MegatronInferenceHook

logger = logging.getLogger(__name__)

_LIVE_STRATEGIES = ("snapkv", "streaming_llm")


class LiveKVCompactor:
    """Compacts each request's prompt KV right after its prefill forward.

    Wire-up (see ``tools/run_dynamic_text_generation_server.py``):
        compactor = LiveKVCompactor(engine, strategy="snapkv", budget_ratio=0.5)
    The engine calls ``begin_step`` before each forward and
    ``compact_prefilled_requests`` after it; both are no-ops on decode-only steps.

    Parameters
    ----------
    engine:        DynamicInferenceEngine.
    strategy:      'snapkv' or 'streaming_llm'.
    budget_ratio:  Fraction of prompt tokens to keep (0 < r < 1).
    obs_window:    SnapKV observation window (last W prompt queries score the rest).
    pool_kernel:   SnapKV neighbour max-pool kernel (odd).
    n_sink:        StreamingLLM attention sinks.
    min_tokens:    Skip compaction for prompts shorter than this.
    """

    def __init__(
        self,
        engine: Any,
        strategy: str,
        budget_ratio: float,
        obs_window: int = 32,
        pool_kernel: int = 7,
        n_sink: int = 4,
        min_tokens: int = 128,
    ) -> None:
        if strategy not in _LIVE_STRATEGIES:
            # Route through the factory for the canonical error (h2o explains
            # exactly why it cannot run live).
            from . import build_kv_compressor
            build_kv_compressor(strategy, inference=True)
            raise ValueError(
                f"strategy {strategy!r} resolves offline but has no live "
                f"deployment; live strategies: {_LIVE_STRATEGIES}"
            )
        if not 0.0 < budget_ratio < 1.0:
            raise ValueError(f"budget_ratio must be in (0, 1), got {budget_ratio}")
        self.strategy = strategy
        self.budget_ratio = budget_ratio
        self.obs_window = obs_window
        self.pool_kernel = pool_kernel
        self.n_sink = n_sink
        self.min_tokens = min_tokens

        # Pruning keeps each retained key's cached (post-embedding) value while the
        # engine renumbers positions to the compacted slots. With RoPE the retained
        # keys would carry stale rotations relative to the new positions — that
        # re-rotation is not implemented, so only position-embedding-free attention
        # (e.g. Nemotron Nano's hybrid, where Mamba carries position) is supported.
        model = engine.controller.inference_wrapped_model.model
        pos_emb = getattr(model, "position_embedding_type", "none")
        if pos_emb not in ("none", None):
            raise NotImplementedError(
                f"LiveKVCompactor: model uses position_embedding_type={pos_emb!r}; "
                "pruning would leave retained keys with stale rotary phases. "
                "Re-rotation of cached keys is not implemented yet."
            )

        self._ctx = engine.context
        self._hook = MegatronInferenceHook(self._ctx)
        self._capturing = False
        self._prefill_locals: list[int] = []
        self._cu_q: torch.Tensor | None = None
        self._q_per_layer: list[torch.Tensor] = []
        if strategy == "snapkv":
            self._register_q_hooks(engine)
        self.compacted_requests = 0
        self.tokens_evicted = 0

    # ------------------------------------------------------------------
    # Q capture (snapkv). The dynamic-batching path invokes
    # ``self_attention.flash_decode_and_prefill(q, k, v, ...)`` — a plain method,
    # NOT the ``core_attention`` submodule — so nn.Module hooks never fire on it.
    # We wrap that method on each attention layer instead; the wrapper is a
    # passthrough except when ``begin_step`` armed capture for a prefill step.
    # ------------------------------------------------------------------

    def _register_q_hooks(self, engine: Any) -> None:
        from megatron.rl.compaction.learned.capture.kv_capture import _unwrap_model
        model = _unwrap_model(engine.controller.inference_wrapped_model.model)
        # Hybrid models stub non-attention layers with self_attention=IdentityOp;
        # real attention modules are the ones that own a core_attention (same
        # predicate as kv_capture, so layer ordering matches the paged cache).
        attns = [
            layer.self_attention
            for layer in model.decoder.layers
            if hasattr(layer, "self_attention")
            and hasattr(layer.self_attention, "core_attention")
        ]
        if not attns:
            raise RuntimeError("LiveKVCompactor: model has no attention layers to hook.")

        def _wrap(idx, orig):
            def wrapped(q, k, v, *args, **kwargs):
                if self._capturing:
                    qq = q
                    # (S, B, Hq, D) with B=1 → packed (T, Hq, D); THD stays as-is.
                    if qq.dim() == 4:
                        qq = qq.transpose(0, 1).reshape(-1, qq.shape[-2], qq.shape[-1])
                    self._q_per_layer[idx] = qq.detach()
                return orig(q, k, v, *args, **kwargs)
            return wrapped

        self._q_per_layer = [None] * len(attns)  # type: ignore[list-item]
        for i, attn in enumerate(attns):
            attn.flash_decode_and_prefill = _wrap(i, attn.flash_decode_and_prefill)

    # ------------------------------------------------------------------
    # Engine seam
    # ------------------------------------------------------------------

    def begin_step(self, is_decode_only: bool) -> None:
        """Snapshot which active requests will prefill this step, and arm Q capture.

        Runs pre-forward (after ``schedule_waiting_requests``): the prefill flags
        and per-request query lengths are definitive here, whereas the flags are
        cleared by ``update_requests`` inside the forward call itself.
        """
        self._prefill_locals: list[int] = []
        self._cu_q: torch.Tensor | None = None
        self._capturing = False
        if is_decode_only:
            return
        ctx = self._ctx
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return
        active = slice(ctx.paused_request_count, ctx.total_request_count)
        flags = ctx.request_in_prefill_status_tensor[active]
        self._prefill_locals = torch.nonzero(flags == 1).flatten().tolist()
        if not self._prefill_locals:
            return
        # Packed-Q row boundaries in active-slice order (THD layout).
        qlens = ctx.request_query_lengths[active].to(torch.long)
        self._cu_q = torch.cat([qlens.new_zeros(1), qlens.cumsum(0)])
        if self.strategy == "snapkv":
            self._capturing = True
            self._q_per_layer = [None] * len(self._q_per_layer)  # type: ignore[list-item]

    def compact_prefilled_requests(self, is_decode_only: bool) -> None:
        """Prune the paged KV of every request whose prefill just completed.

        Runs after the step is fully bookkept. ``update_requests`` has already
        precomputed the pending first decode token's placement (position = old
        prompt length) in the ``token_to_*`` tensors, so after pruning we repoint
        those entries at the compacted cache. A chunked-prefill request is
        skipped — its prompt is still streaming in.
        """
        if is_decode_only or not self._prefill_locals:
            return
        self._capturing = False
        cu_q = self._cu_q
        ctx = self._ctx
        BS = ctx.block_size_tokens
        chunked_global = ctx.get_index_of_chunked_prefill_request(safe=True)

        for b_local in self._prefill_locals:
            b_global = ctx.paused_request_count + b_local
            if b_global == chunked_global:
                continue
            k, v = self._hook.get_kv_for_request(b_local)   # (L, S, H, D)
            S = k.shape[1]
            if S < self.min_tokens:
                continue
            budget = max(1, int(S * self.budget_ratio))
            # The pending decode token must land INSIDE the last retained block
            # (a retained count on a block boundary would need a fresh block that
            # update_requests already decided not to allocate).
            if budget % BS == 0:
                budget -= 1
            if budget >= S or budget < 1:
                continue

            if self.strategy == "snapkv":
                q_rows = self._request_q(b_local, cu_q)     # per layer (Tq, Hq, D)
                scores = self._aggregate_snapkv_scores(k, q_rows)
                positions = _select_recent_plus_heavy(
                    scores, S, budget, n_recent=min(self.obs_window, budget)
                )
            else:  # streaming_llm: sinks + recent window, no scores.
                n_sink = min(self.n_sink, budget)
                sinks = list(range(n_sink))
                recent_start = max(n_sink, S - (budget - n_sink))
                positions = sorted(set(sinks + list(range(recent_start, S))))

            self._hook.apply_mask_for_request(b_local, positions)
            self._repoint_pending_token(b_local, b_global, len(positions))
            # Termination compares current length against request_output_lengths
            # (prompt + num_tokens_to_generate, absolute) — shift it down by the
            # evicted count or the request over-generates past its token budget.
            ctx.request_output_lengths[b_global] -= S - len(positions)
            self.compacted_requests += 1
            self.tokens_evicted += S - len(positions)
            logger.info(
                "[kv-compaction] request b_local=%d: %d -> %d tokens (%s, ratio %.2f)",
                b_local, S, len(positions), self.strategy, self.budget_ratio,
            )
        if self.compacted_requests:
            # The prune's gather/scatter kernels run on the default stream, but the
            # next decode step may replay a CUDA graph on its own stream. Without a
            # barrier the replay races the in-flight cache surgery (observed as
            # cudaErrorIllegalAddress; disappears under CUDA_LAUNCH_BLOCKING=1).
            torch.cuda.synchronize()
        self._prefill_locals = []
        self._cu_q = None

    def _repoint_pending_token(self, b_local: int, b_global: int, C: int) -> None:
        """Fix the precomputed placement of the request's pending decode token.

        ``update_requests`` wrote its position/write-slot as if the full prompt
        were still cached. After pruning to C tokens, the token's position, its
        slot within the last block, and its target block must follow the
        compacted cache. Token index = b_local: one pending token per active
        request, in active-slice order (no speculative decoding).
        """
        ctx = self._ctx
        BS = ctx.block_size_tokens
        ctx.token_to_pos_ids[b_local] = C
        ctx.token_to_position_in_request[b_local] = C
        ctx.token_to_local_position_within_kv_block[b_local] = C % BS
        ctx.token_to_block_idx[b_local] = ctx.request_last_kv_block_id[b_global]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _request_q(self, b_local: int, cu_q: torch.Tensor) -> list[torch.Tensor]:
        """This request's observation-window Q rows for every layer."""
        start = int(cu_q[b_local].item())
        end = int(cu_q[b_local + 1].item())
        w_start = max(start, end - self.obs_window)
        out = []
        for i, q in enumerate(self._q_per_layer):
            if q is None:
                raise RuntimeError(
                    f"LiveKVCompactor: no Q captured for attention layer {i} this "
                    "step — prefill ran under a CUDA graph. Launch the server with "
                    "--decode-only-cuda-graphs so prefill forwards run eagerly."
                )
            out.append(q[w_start:end])
        return out

    def _aggregate_snapkv_scores(
        self, keys: torch.Tensor, q_rows: list[torch.Tensor]
    ) -> torch.Tensor:
        """Token-level SnapKV score: window attention summed over layers, KV heads,
        and their grouped query heads. keys (L, S, Hkv, D); q_rows[l] (Tq, Hq, D)."""
        L, S, Hkv, D = keys.shape
        scores = torch.zeros(S, device=keys.device, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)
        for l in range(L):
            q = q_rows[l]                       # (Tq, Hq, D)
            Hq = q.shape[1]
            group = Hq // Hkv
            for g in range(Hkv):
                qg = q[:, g * group:(g + 1) * group, :].reshape(-1, D).float()   # (Tq*grp, D)
                kg = keys[l, :, g, :].float()                                    # (S, D)
                attn = torch.softmax(qg @ kg.T * scale, dim=-1)                  # (·, S)
                scores += attn.sum(dim=0)
        pad = self.pool_kernel // 2
        return F.max_pool1d(
            scores[None, None, :], kernel_size=self.pool_kernel, stride=1, padding=pad
        )[0, 0]
