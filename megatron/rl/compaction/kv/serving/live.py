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

import json
import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from ..compressors import _select_recent_plus_heavy
from .megatron_hook import MegatronInferenceHook

logger = logging.getLogger(__name__)

_LIVE_STRATEGIES = ("snapkv", "streaming_llm", "belief_still", "learned_oracle")


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
        compactor_checkpoint: str | None = None,
        oracle_checkpoint: str | None = None,
        n_compress: int = 64,
        archive: bool = False,
        retrieval_alpha: float = 0.2,
        retrieval_cusum: float = 0.4,
        max_retrievals_per_request: int = 4,
        rope_mode: str | None = None,
        flywheel_dir: str | None = None,
        archive_transfer: str = "pinned",
        budget_final: float | None = None,
        budget_anneal_iters: int | None = None,
    ) -> None:
        if strategy not in _LIVE_STRATEGIES:
            # Route through the factory for the canonical error (h2o explains
            # exactly why it cannot run live).
            from .. import build_kv_compressor
            build_kv_compressor(strategy, inference=True)
            raise ValueError(
                f"strategy {strategy!r} resolves offline but has no live "
                f"deployment; live strategies: {_LIVE_STRATEGIES}"
            )
        if not 0.0 < budget_ratio < 1.0:
            raise ValueError(f"budget_ratio must be in (0, 1), got {budget_ratio}")
        self.strategy = strategy
        self.budget_ratio = budget_ratio
        # budget annealing (RL loop only): budget_ratio moves linearly from
        # its starting value to budget_final over budget_anneal_iters GRPO
        # iterations. Every TP rank runs begin_step in lockstep with the same
        # args, so the schedule stays consistent without any broadcast.
        self._budget_start = budget_ratio
        self.budget_final = budget_final
        self.budget_anneal_iters = budget_anneal_iters
        if (budget_final is None) != (budget_anneal_iters is None):
            raise ValueError(
                "budget_final and budget_anneal_iters must be set together "
                f"(got final={budget_final}, anneal_iters={budget_anneal_iters}).")
        if budget_final is not None:
            if not 0.0 < budget_final < 1.0:
                raise ValueError(f"budget_final must be in (0, 1), got {budget_final}")
            if budget_anneal_iters < 1:
                raise ValueError(
                    f"budget_anneal_iters must be >= 1, got {budget_anneal_iters}")
        self.obs_window = obs_window
        self.pool_kernel = pool_kernel
        self.n_sink = n_sink
        self.min_tokens = min_tokens
        self.compactor_checkpoint = compactor_checkpoint
        self.oracle_checkpoint = oracle_checkpoint
        self.n_compress = n_compress
        self._updater = None   # lazy: belief_still builds/loads on first request
        self._oracle = None    # lazy: learned_oracle builds/loads on first request
        # CPU archive + negative-cache retrieval (archive). Evicted spans are
        # demoted to CPU instead of deleted; a per-decode-step trigger restores
        # them on demand. Needs per-step Q → decode must run EAGER (no CUDA
        # graphs at all); enforced at the first missed capture.
        self.archive_enabled = archive
        # α̂/CUSUM trigger (scale-free, no per-model calibration): fire a span
        # when its centroid attention-mass fraction spikes above the fast-path
        # threshold, or when its CUSUM of (α̂ − own EMA baseline) crosses h —
        # chronically hot spans self-absorb into their baselines, so only a
        # NOVEL, persistent reach for evicted content fires.
        self.retrieval_alpha = retrieval_alpha
        self.retrieval_cusum = retrieval_cusum
        self.cusum_drift = 0.02
        self.ema_decay = 0.9
        self.max_retrievals_per_request = max_retrievals_per_request
        self._archive = None
        self._retrieval_counts: dict[int, int] = {}
        # rid -> span_id -> [ema_baseline, cusum_S]
        self._trigger_state: dict[int, dict[int, list[float]]] = {}
        self._prefetch_stream: torch.cuda.Stream | None = None
        if archive:
            if strategy == "belief_still":
                raise ValueError(
                    "archive mode restores exact evicted spans; belief_still "
                    "replaces the cache with synthetic tokens, so there are no "
                    "evicted spans to archive. Use snapkv or learned_oracle.")
            if strategy == "streaming_llm":
                logger.warning(
                    "[kv-compaction] archive+streaming_llm: under content-blind "
                    "mass eviction the trigger signal is weak (measured); the "
                    "archive is most effective with content-aware strategies "
                    "(snapkv, learned_oracle).")
            if not 0.0 < retrieval_alpha < 1.0:
                raise ValueError(
                    f"retrieval_alpha is an attention-mass fraction in (0, 1); "
                    f"got {retrieval_alpha}")
            if retrieval_cusum <= 0.0:
                raise ValueError(f"retrieval_cusum must be > 0, got {retrieval_cusum}")
        self.rope_mode = rope_mode
        self._inv_freq: torch.Tensor | None = None
        self._rope_interleaved = False
        self._logical_pos: dict[int, int] = {}
        # RoPE handling: 'logical' keeps original token positions (exact,
        # counterfactual semantics; archive splice-back needs no rotation);
        # 'renumber' is StreamingLLM semantics (contiguous cache positions,
        # retained keys delta-rotated, restored spans to the cache tail).
        from megatron.rl.compaction.learned.capture.kv_capture import _unwrap_model
        model = _unwrap_model(engine.controller.inference_wrapped_model.model)
        pos_emb = getattr(model, "position_embedding_type", "none")
        if pos_emb in ("none", None):
            if rope_mode is not None:
                raise ValueError(
                    f"rope_mode={rope_mode!r} given but the model has no positional "
                    "embedding (position_embedding_type none) — remove the flag.")
        elif pos_emb == "rope":
            if rope_mode not in ("logical", "renumber"):
                raise NotImplementedError(
                    "LiveKVCompactor on a RoPE model requires an explicit "
                    "rope_mode: 'logical' (keep original positions — exact, the "
                    "measurement setting) or 'renumber' (contiguous cache "
                    f"positions + key re-rotation, StreamingLLM semantics); got "
                    f"{rope_mode!r}.")
            if strategy == "belief_still":
                raise NotImplementedError(
                    "belief_still under RoPE is unsupported: synthetic KV has no "
                    "position convention yet (the compactor must be trained to "
                    "emit keys pre-rotated for assigned slots).")
            if rope_mode == "renumber":
                rotary = getattr(model, "rotary_pos_emb", None)
                if rotary is None or not hasattr(rotary, "inv_freq"):
                    raise RuntimeError(
                        "rope_mode='renumber': model has no rotary_pos_emb.inv_freq "
                        "to re-rotate keys with.")
                self._inv_freq = rotary.inv_freq
                self._rope_interleaved = bool(
                    getattr(rotary, "rotary_interleaved", False))
        else:
            raise NotImplementedError(
                f"LiveKVCompactor: position_embedding_type={pos_emb!r} is not "
                "supported (only 'none' and 'rope').")

        self._ctx = engine.context
        self._hook = MegatronInferenceHook(self._ctx)
        self._capturing = False
        self._prefill_rids: list[tuple[int, int, int]] = []
        self._q_per_layer: list[torch.Tensor] = []
        if strategy == "snapkv" or archive:
            self._register_q_hooks(engine)
        self.compacted_requests = 0
        self.tokens_evicted = 0

    def stats(self) -> dict:
        """Cumulative counters, identical on every TP rank (compaction runs in
        lockstep), so the LAST [kv-compaction-stats] log line is exact — no
        divide-by-TP guesswork for eval scrapers."""
        s = {"compactions": self.compacted_requests,
             "tokens_evicted": self.tokens_evicted,
             "budget_ratio": round(self.budget_ratio, 4)}
        if self._archive is not None:
            s["retrievals"] = self._archive.retrievals
            s["prefetch_hits"] = self._archive.prefetch_hits
        return s

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

    def schedule_ratio(self, iteration: int) -> float:
        """Linear anneal from the starting budget to budget_final."""
        frac = min(1.0, max(0.0, iteration / self.budget_anneal_iters))
        return self._budget_start + (self.budget_final - self._budget_start) * frac

    def _apply_budget_schedule(self) -> None:
        from megatron.training import get_args
        iteration = getattr(get_args(), "curr_iteration", None)
        if iteration is None:
            return
        new_ratio = self.schedule_ratio(iteration)
        if abs(new_ratio - self.budget_ratio) > 1e-9:
            logger.info("[kv-compaction] budget anneal: ratio %.3f -> %.3f "
                        "(iteration %d)", self.budget_ratio, new_ratio, iteration)
            self.budget_ratio = new_ratio

    def begin_step(self, is_decode_only: bool) -> None:
        """Snapshot which active requests will prefill this step, and arm Q capture.

        Runs pre-forward (after ``schedule_waiting_requests``): the prefill flags
        and per-request query lengths are definitive here, whereas the flags are
        cleared by ``update_requests`` inside the forward call itself.
        """
        self._prefill_rids: list[tuple[int, int, int]] = []
        self._capturing = self.archive_enabled
        if self.budget_final is not None:
            self._apply_budget_schedule()
        if self.archive_enabled and self._q_per_layer:
            self._q_per_layer = [None] * len(self._q_per_layer)  # type: ignore[list-item]
        if os.environ.get("KV_COMPACTION_DEBUG"):
            self._debug_dump()
        if self.rope_mode == "logical" and self._logical_pos:
            self._patch_logical_positions()
        if is_decode_only:
            return
        ctx = self._ctx
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return
        active = slice(ctx.paused_request_count, ctx.total_request_count)
        flags = ctx.request_in_prefill_status_tensor[active]
        prefill_locals = torch.nonzero(flags == 1).flatten().tolist()
        if not prefill_locals:
            return
        # Record (request id, packed-Q row range). update_requests — which runs
        # INSIDE the step, before compact_prefilled_requests — reorders the
        # active slice (finished requests swap left, block-full requests pause,
        # paused ones resume), so pre-forward b_local indices are stale by
        # compaction time; request ids are the only stable handle.
        qlens = ctx.request_query_lengths[active].to(torch.long)
        cu_q = torch.cat([qlens.new_zeros(1), qlens.cumsum(0)])
        self._prefill_rids = [
            (int(ctx.request_ids[ctx.paused_request_count + b].item()),
             int(cu_q[b].item()), int(cu_q[b + 1].item()))
            for b in prefill_locals
        ]
        if self.strategy == "snapkv" and not self._capturing:
            self._capturing = True
            self._q_per_layer = [None] * len(self._q_per_layer)  # type: ignore[list-item]

    def _patch_logical_positions(self) -> None:
        """rope_mode='logical': give this step's decode tokens their ORIGINAL
        sequence positions before the forward runs.

        The engine derives ``token_to_pos_ids`` (which rotary indexes directly)
        from ``request_kv_length_offsets`` — the compacted cache length. For a
        pruned request that under-rotates every future query relative to its
        history. The stored keys keep their original rotations, so patching the
        query position to the request's logical (uncompacted) position restores
        exact relative geometry. Cache write slots are unaffected: they come
        from ``token_to_local_position_within_kv_block``/``token_to_block_idx``,
        which the previous step's bookkeeping computed before this patch.
        """
        ctx = self._ctx
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0:
            return
        active = slice(ctx.paused_request_count, ctx.total_request_count)
        qlens = ctx.request_query_lengths[active].to(torch.long)
        starts = torch.cat([qlens.new_zeros(1), qlens.cumsum(0)])
        in_prefill = ctx.request_in_prefill_status_tensor[active]
        # GC against ALL requests (paused included): a compacted request the
        # engine pauses must keep its logical-position record for resume.
        all_ids = {int(r) for r in
                   ctx.request_ids[:ctx.total_request_count].tolist()}
        live_ids = set()
        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            rid = int(ctx.request_ids[b_global].item())
            live_ids.add(rid)
            pos = self._logical_pos.get(rid)
            if pos is None or bool(in_prefill[b_local] == 1):
                continue
            ctx.token_to_pos_ids[int(starts[b_local].item())] = pos
            self._logical_pos[rid] = pos + 1
        for rid in [r for r in self._logical_pos if r not in all_ids]:
            del self._logical_pos[rid]

    def retrieve_for_decoding_requests(self) -> None:
        """Negative-cache retrieval: restore archived spans the decode queries ask for.

        Runs post-bookkeep on decode-only steps. Uses this step's captured Q
        (one token per active request, active-slice order under eager decode).
        """
        ctx = self._ctx
        n_active = ctx.total_request_count - ctx.paused_request_count
        if n_active <= 0 or self._archive.empty:
            return
        if any(q is None for q in self._q_per_layer):
            raise RuntimeError(
                "archive mode: no Q captured this decode step — decode ran under a "
                "CUDA graph. Archive retrieval needs fully eager decoding: launch "
                "with --cuda-graph-impl none.")
        live_ids = set()
        for b_local in range(n_active):
            b_global = ctx.paused_request_count + b_local
            rid = int(ctx.request_ids[b_global].item())
            live_ids.add(rid)
            if not self._archive.has(rid):
                continue
            if self._retrieval_counts.get(rid, 0) >= self.max_retrievals_per_request:
                continue
            k, v = self._hook.get_kv_for_request(b_local)      # (L, C, H, D)
            q_step = [q[b_local] for q in self._q_per_layer]   # per layer (Hq, D)
            scored = self._archive.span_alphas(rid, q_step, k)
            if scored is None:
                continue
            alphas, span_ids = scored
            state = self._trigger_state.setdefault(rid, {})
            fire_idx, fire_alpha, best_S, best_idx = -1, 0.0, 0.0, -1
            for i, (a, sid) in enumerate(zip(alphas.tolist(), span_ids)):
                b, S = state.get(sid, (a, 0.0))     # EMA seeds at first sight
                S = max(0.0, S + a - b - self.cusum_drift)
                b = self.ema_decay * b + (1.0 - self.ema_decay) * a
                state[sid] = [b, S]
                if S > best_S:
                    best_S, best_idx = S, i
                fired = a >= self.retrieval_alpha or S >= self.retrieval_cusum
                if fired and a > fire_alpha:
                    fire_idx, fire_alpha = i, a
            if os.environ.get("KV_COMPACTION_DEBUG"):
                logger.info(
                    "[kv-retrieval] request id=%d: alpha max %.3f (fast %.2f) "
                    "cusum max %.3f (h %.2f) spans=%d",
                    rid, float(alphas.max()), self.retrieval_alpha,
                    best_S, self.retrieval_cusum, len(span_ids))
            if fire_idx < 0:
                # Speculative staging: the leading CUSUM candidate is warming
                # up toward the firing threshold — start its PCIe copy early.
                if best_idx >= 0 and (best_S >= self.retrieval_cusum / 2
                                      or float(alphas[best_idx]) >= self.retrieval_alpha / 2):
                    self._archive.prefetch(rid, best_idx, self._prefetch_stream)
                continue
            span_idx = fire_idx
            state.pop(span_ids[span_idx], None)
            ak, av, apos = self._archive.take(rid, span_idx)  # staged GPU or pinned CPU
            was_staged = ak.is_cuda
            ak, av = ak.cuda(), av.cuda()
            if self.rope_mode == "renumber":
                from .rope import delta_rotate_keys
                start = int(ctx.request_kv_length_offsets[b_global].item())
                old_pos = torch.tensor(apos, device=ak.device, dtype=torch.long)
                new_pos = torch.arange(start, start + ak.shape[1], device=ak.device)
                ak = delta_rotate_keys(
                    ak, old_pos, new_pos, self._inv_freq, self._rope_interleaved)
            self._hook.append_kv_to_request(b_local, ak, av)
            self._repoint_pending_token(
                b_local, b_global, int(ctx.request_kv_length_offsets[b_global]))
            ctx.request_output_lengths[b_global] += ak.shape[1]
            self._retrieval_counts[rid] = self._retrieval_counts.get(rid, 0) + 1
            logger.info(
                "[kv-retrieval] request id=%d: restored %d tokens (alpha %.3f, "
                "%d spans left, %s)", rid, ak.shape[1], fire_alpha,
                len(self._archive._spans.get(rid, [])),
                "prefetched" if was_staged else "sync copy")
        self._archive.drop_all_except(live_ids)
        for rid in [r for r in self._retrieval_counts if r not in live_ids]:
            del self._retrieval_counts[rid]
        for rid in [r for r in self._trigger_state if r not in live_ids]:
            del self._trigger_state[rid]

    def compact_prefilled_requests(self, is_decode_only: bool) -> None:
        """Prune the paged KV of every request whose prefill just completed.

        Runs after the step is fully bookkept. ``update_requests`` has already
        precomputed the pending first decode token's placement (position = old
        prompt length) in the ``token_to_*`` tensors, so after pruning we repoint
        those entries at the compacted cache. A chunked-prefill request is
        skipped — its prompt is still streaming in.
        """
        if is_decode_only:
            if self.archive_enabled:
                self.retrieve_for_decoding_requests()
            return
        if not self._prefill_rids:
            return
        self._capturing = self.archive_enabled
        ctx = self._ctx
        chunked_global = ctx.get_index_of_chunked_prefill_request(safe=True)

        active_ids = ctx.request_ids[ctx.paused_request_count:ctx.total_request_count]
        for rid, q0, q1 in self._prefill_rids:
            match = torch.nonzero(active_ids == rid).flatten()
            if match.numel() == 0:
                # Finished or paused during this step's bookkeeping — nothing
                # to compact (a paused request resumes with its full cache).
                continue
            b_local = int(match[0].item())
            b_global = ctx.paused_request_count + b_local
            if b_global == chunked_global:
                continue
            # Per-request arm flag (split-group): kv_compact=False exempts
            # this request — the control arm of a compact-vs-full comparison.
            if not bool(ctx.request_metadata["kv_compact"][b_global].item()):
                continue
            k, v = self._hook.get_kv_for_request(b_local)   # (L, S, H, D)
            S = k.shape[1]
            if S < self.min_tokens:
                continue
            if self.strategy == "belief_still":
                if self.n_compress >= S:
                    continue
                self._compact_with_updater(b_local, b_global, k, v)
                continue

            budget = max(1, int(S * self.budget_ratio))
            if budget >= S:
                continue

            if self.strategy == "snapkv":
                q_rows = self._request_q(q0, q1)            # per layer (Tq, Hq, D)
                scores = self._aggregate_snapkv_scores(k, q_rows)
                positions = _select_recent_plus_heavy(
                    scores, S, budget, n_recent=min(self.obs_window, budget)
                )
            elif self.strategy == "learned_oracle":
                # Query-free: the scorer predicts each key's future attention
                # mass from content + position alone — no Q capture, so this
                # strategy has no eager-prefill requirement.
                if self._oracle is None:
                    self._init_oracle(n_layers=k.shape[0], d_key=k.shape[2] * k.shape[3])
                scores = self._oracle.score_tokens(k)
                positions = _select_recent_plus_heavy(
                    scores, S, budget, n_recent=min(self.obs_window, budget)
                )
            else:  # streaming_llm: sinks + recent window, no scores.
                n_sink = min(self.n_sink, budget)
                sinks = list(range(n_sink))
                recent_start = max(n_sink, S - (budget - n_sink))
                positions = sorted(set(sinks + list(range(recent_start, S))))

            if self._archive is not None:
                rid = int(ctx.request_ids[b_global].item())
                self._archive.store_evicted(rid, k, v, positions)
            rotated_keys = None
            if self.rope_mode == "renumber":
                from .rope import delta_rotate_keys
                old_pos = torch.tensor(positions, device=k.device, dtype=torch.long)
                new_pos = torch.arange(len(positions), device=k.device)
                rotated_keys = delta_rotate_keys(
                    k[:, positions], old_pos, new_pos,
                    self._inv_freq, self._rope_interleaved)
            self._hook.apply_mask_for_request(b_local, positions)
            if rotated_keys is not None:
                self._hook.overwrite_keys_for_request(b_local, rotated_keys)
            if self.rope_mode == "logical":
                # The pending decode token's original position is S (prompt
                # length); begin_step patches it in before the next forward.
                self._logical_pos[int(ctx.request_ids[b_global].item())] = S
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
            logger.info("[kv-compaction-stats] %s", json.dumps(self.stats()))
        self._prefill_rids = []

    def _compact_with_updater(self, b_local: int, b_global: int,
                              k: torch.Tensor, v: torch.Tensor) -> None:
        """belief_still: replace the request's prompt KV with the learned
        compactor's C synthetic tokens (Perceiver/gated-updater initial_compress),
        injected via apply_belief_memory_for_request. k/v: (L, S, H, D) TP-local."""
        ctx = self._ctx
        L, S, H, D = k.shape
        if self._updater is None:
            self._init_updater(n_attn_layers=L, d_kv=H * D)
        keys_pl = [k[li].reshape(1, S, H * D).to(torch.bfloat16) for li in range(L)]
        vals_pl = [v[li].reshape(1, S, H * D).to(torch.bfloat16) for li in range(L)]
        with torch.no_grad():
            memory = self._updater.initial_compress(keys_pl, vals_pl)
        C = memory.keys.shape[2]
        self._hook.apply_belief_memory_for_request(b_local, memory)
        self._repoint_pending_token(b_local, b_global, C)
        ctx.request_output_lengths[b_global] -= S - C
        self.compacted_requests += 1
        self.tokens_evicted += S - C
        logger.info(
            "[kv-compaction] request b_local=%d: %d -> %d synthetic tokens (belief_still)",
            b_local, S, C,
        )
        logger.info("[kv-compaction-stats] %s", json.dumps(self.stats()))

    def _init_oracle(self, n_layers: int, d_key: int) -> None:
        """Build (or load) the learned heavy-hitter scorer, replicated per rank.

        With a checkpoint: every rank torch.loads the same file (the scorer is
        trained offline, deployed read-only). Without one: RANDOM INIT — the
        selection is garbage; plumbing smoke only, logged loudly.
        """
        from ..selectors.oracle import LearnedOracleScorer, OracleScorerConfig, load_oracle_scorer
        from megatron.rl.compaction.learned.training.parallel import (
            build_compactor_pg_collection,
        )
        # Singleton TP groups: the scorer is replicated per rank — without this
        # the TE linears shard over the WORLD TP group (and the 1-dim output
        # layer cannot shard at all).
        pgc = build_compactor_pg_collection()
        if self.oracle_checkpoint:
            self._oracle = load_oracle_scorer(
                self.oracle_checkpoint, params_dtype=torch.bfloat16,
                pg_collection=pgc)
            if (self._oracle.cfg.d_key != d_key
                    or self._oracle.cfg.n_layers != n_layers):
                raise ValueError(
                    f"oracle checkpoint was trained for d_key="
                    f"{self._oracle.cfg.d_key}, n_layers={self._oracle.cfg.n_layers} "
                    f"but this model's TP-local KV is d_key={d_key}, "
                    f"n_layers={n_layers} — retrain on captures from this "
                    "model/TP configuration.")
            logger.info("[kv-compaction] learned_oracle checkpoint loaded: %s",
                        self.oracle_checkpoint)
        else:
            logger.warning(
                "[kv-compaction] learned_oracle WITHOUT a checkpoint: RANDOM-INIT "
                "scorer — selection is garbage, plumbing smoke only.")
            self._oracle = LearnedOracleScorer(
                OracleScorerConfig(d_key=d_key, n_layers=n_layers),
                params_dtype=torch.bfloat16, pg_collection=pgc,
            ).cuda().to(torch.bfloat16).eval()

    def _init_updater(self, n_attn_layers: int, d_kv: int) -> None:
        """Build (or load) the learned compactor, replicated on every rank.

        With a checkpoint: collective dist_checkpointing load (all TP ranks call
        this together — compaction runs in lockstep on every rank). Without one:
        RANDOM INIT, useful only for plumbing smoke tests — generation quality
        will be garbage, and we log a loud warning.
        """
        from megatron.rl.compaction.learned import (
            BeliefUpdater, GatedRecurrentUpdater, GatedUpdaterConfig, PerceiverCompactor,
        )
        from megatron.rl.compaction.learned.training.parallel import (
            build_compactor_pg_collection,
        )
        pgc = build_compactor_pg_collection()
        if self.compactor_checkpoint:
            from megatron.rl.compaction.learned import load_checkpoint
            model, _meta = load_checkpoint(
                self.compactor_checkpoint, map_location="cuda",
                params_dtype=torch.bfloat16, pg_collection=pgc,
            )
            model = model.to(torch.bfloat16).eval()
            self._updater = (BeliefUpdater(model)
                             if isinstance(model, PerceiverCompactor) else model)
            logger.info("[kv-compaction] belief_still checkpoint loaded: %s",
                        self.compactor_checkpoint)
        else:
            logger.warning(
                "[kv-compaction] belief_still WITHOUT a checkpoint: RANDOM-INIT "
                "compactor — plumbing smoke only, generation will be degraded.")
            cfg = GatedUpdaterConfig(n_compress=self.n_compress, n_heads=8,
                                     d_kv=d_kv, n_attn_layers=n_attn_layers)
            self._updater = GatedRecurrentUpdater(
                cfg, params_dtype=torch.bfloat16, pg_collection=pgc,
            ).cuda().to(torch.bfloat16).eval()

    def _debug_dump(self) -> None:
        ctx = self._ctx
        if ctx.total_request_count - ctx.paused_request_count <= 0:
            return  # no active request to dump
        b = ctx.paused_request_count  # first active (or first paused if none active)
        logger.info(
            "[kv-dbg] step=%s paused=%d total=%d | kvlen=%d qlen=%d blocks=%d "
            "lastblk=%d off=%d outlen=%d | t2pos=%d t2blk=%d t2loc=%d",
            ctx.step_count, ctx.paused_request_count, ctx.total_request_count,
            int(ctx.request_kv_length_offsets[b]), int(ctx.request_query_lengths[b]),
            int(ctx.request_kv_block_counts[b]), int(ctx.request_last_kv_block_id[b]),
            int(ctx.request_last_kv_block_offset[b]), int(ctx.request_output_lengths[b]),
            int(ctx.token_to_pos_ids[0]), int(ctx.token_to_block_idx[0]),
            int(ctx.token_to_local_position_within_kv_block[0]),
        )

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

    def _request_q(self, start: int, end: int) -> list[torch.Tensor]:
        """This request's observation-window Q rows for every layer.

        (start, end) are the packed-Q row bounds recorded at begin_step — the
        Q tensor layout is fixed at forward time, so the rows stay valid even
        though the bookkeeping reorders request slots afterwards."""
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
        Tq = q_rows[0].shape[0]
        # CAUSAL mask for the observation window (official SnapKV): window
        # query i (the (S - Tq + i)-th token) must not attend to later keys.
        col = torch.arange(S, device=keys.device)
        row = torch.arange(Tq, device=keys.device)
        future = col[None, :] > (S - Tq + row)[:, None]         # (Tq, S)
        for l in range(L):
            q = q_rows[l]                       # (Tq, Hq, D)
            Tq_l, Hq, _ = q.shape
            group = Hq // Hkv
            # One batched matmul over all KV groups instead of Hkv small GEMMs.
            qb = (q.reshape(Tq_l, Hkv, group, D).permute(1, 0, 2, 3)
                   .reshape(Hkv, Tq_l * group, D).float())      # (Hkv, Tq*grp, D)
            kb = keys[l].permute(1, 0, 2).float()               # (Hkv, S, D)
            logits = qb @ kb.transpose(1, 2) * scale            # (Hkv, Tq*grp, S)
            mask = future.repeat_interleave(group, dim=0)       # (Tq*grp, S)
            attn = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)
            scores += attn.sum(dim=(0, 1))
        # Pool over PREFIX scores only (official drops the window columns
        # before pooling): pooling across the boundary leaks the window keys'
        # large scores into the last pool_kernel//2 prefix positions.
        n_win = min(Tq, S)
        pad = self.pool_kernel // 2
        prefix = scores[: S - n_win]
        if prefix.numel():
            pooled_prefix = F.max_pool1d(
                prefix[None, None, :], kernel_size=self.pool_kernel, stride=1, padding=pad
            )[0, 0]
            return torch.cat([pooled_prefix, scores[S - n_win:]])
        return scores
