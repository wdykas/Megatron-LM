# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable STILL student forward pass.

Runs the Megatron GPT model on response tokens where each attention layer
attends to compact_kv (from the compactor) instead of the full KV cache.

Gradient flows:
    loss → student_outputs → attention(Q, compact_k, compact_v) → compact_kv → compactor

Model weights are temporarily frozen so backward only touches the compactor.

This implements the STILL paper's teacher-student training objective:
    minimize CE(model(response | compact_kv), response_token_ids)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import List, Tuple

import torch

from .kv_capture import _unwrap_model


def _attention_layers(model):
    """(layer_index, layer) for each REAL attention layer.

    Same predicate as kv_capture: hybrid models stub non-attention layers with
    self_attention=IdentityOp, which still HAS the attribute — only layers
    whose self_attention owns a core_attention are attention layers, and this
    ordering matches the captured KV list layer-for-layer.
    """
    gpt = _unwrap_model(model)
    return [
        (i, layer)
        for i, layer in enumerate(gpt.decoder.layers)
        if hasattr(layer, "self_attention")
        and hasattr(layer.self_attention, "core_attention")
    ]


@contextmanager
def _inject_compact_kv(model, compact_kv_list: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Context manager: replace K, V in each attention layer with compact_kv.

    compact_kv_list[i] = (compact_k, compact_v) for attention layer i.
        compact_k shape: (B, C, d_kv)  — from BeliefMemory.keys[i]
        compact_v shape: (B, C, d_kv)

    The hook fires on DotProductAttention.forward (the core_attention module),
    whose first four positional args are (query, key, value, attention_mask).
    We replace key and value with the compact versions and clear the mask so
    all query positions can attend to all C slots without restriction.

    Gradient flows through compact_k and compact_v because they are leaf
    tensors with requires_grad=True.  Query comes from the frozen model so it
    carries no gradient.
    """
    attn_layers = _attention_layers(model)
    if len(attn_layers) != len(compact_kv_list):
        raise ValueError(
            f"Attention layer count mismatch: model has {len(attn_layers)} "
            f"attention layers, compact_kv_list has {len(compact_kv_list)}"
        )

    hooks = []
    for hook_idx, (_, layer) in enumerate(attn_layers):
        ck, cv = compact_kv_list[hook_idx]  # (B, C, d_kv)

        def _make_hook(ck_cap, cv_cap):
            def _pre_hook(module, args):
                if len(args) < 3:
                    return args
                query = args[0]
                orig_key = args[1]
                C = ck_cap.shape[1]

                if orig_key.dim() == 3:
                    # THD packed format: (T_kv, n_kv_groups, d_head). Only
                    # B=1 is meaningful for a packed injected forward.
                    if ck_cap.shape[0] != 1:
                        raise ValueError(
                            f"THD injection requires B=1 compact KV, got "
                            f"B={ck_cap.shape[0]}")
                    n_kv_groups, d_head = orig_key.shape[1], orig_key.shape[2]
                    ck_r = ck_cap.reshape(C, n_kv_groups, d_head).clone()
                    cv_r = cv_cap.reshape(C, n_kv_groups, d_head).clone()
                else:
                    # SBHD: (S_k, B, n_kv_groups, d_head).
                    B = orig_key.shape[1]
                    n_kv_groups, d_head = orig_key.shape[2], orig_key.shape[3]
                    # clone(), not contiguous(): BeliefMemory hands per-layer
                    # VIEWS at storage offset l*C*d, and .contiguous() is a
                    # NO-OP when the non-contiguity hides in size-1 dims — the
                    # offset survives and TE's three-chunk layout check
                    # (which requires zero storage offsets) rejects every
                    # layer past the first. clone() gives fresh storage at
                    # offset 0.
                    ck_r = (ck_cap.reshape(B, C, n_kv_groups, d_head)
                            .permute(1, 0, 2, 3).clone())
                    cv_r = (cv_cap.reshape(B, C, n_kv_groups, d_head)
                            .permute(1, 0, 2, 3).clone())

                # attention_mask=None: all query positions attend to all C slots.
                return (query.contiguous(), ck_r, cv_r, None) + tuple(args[4:])
            return _pre_hook

        h = layer.self_attention.core_attention.register_forward_pre_hook(
            _make_hook(ck, cv)
        )
        hooks.append(h)

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def _capture_final_hidden(gpt, sink: list):
    """Capture the decoder's final hidden state (post final norm) into sink.

    ``gpt.decoder`` (TransformerBlock) returns the normed hidden states that
    feed the output layer — the target space of the NextLat future-latent
    loss. Output is (S, B, d_model); the caller transposes to batch-first.
    """
    def _hook(module, args, output):
        sink.append(output[0] if isinstance(output, tuple) else output)

    h = gpt.decoder.register_forward_hook(_hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def _allow_nonflash_attention():
    """Permit fused/unfused attention for the injected forward.

    Serving configs force the flash backend (--attention-backend flash sets
    NVTE_FUSED_ATTN=0 and NVTE_UNFUSED_ATTN=0), and flash cannot run the
    injected student forward's shape (S_q queries over C replaced slots,
    no_mask) — leaving NO available backend. Re-enable the other backends and
    force TE to re-select for this forward, then restore the serving
    configuration (and force re-selection again) on exit.
    """
    import transformer_engine.pytorch.attention.dot_product_attention as _dpa

    saved = {k: os.environ.get(k) for k in ("NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")}
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    _dpa._attention_backends["backend_selection_requires_update"] = True
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _dpa._attention_backends["backend_selection_requires_update"] = True


def _forward_outputs(gpt, response_token_ids: torch.Tensor,
                     gather_logits: bool = False,
                     packed_thd_kv_len: int | None = None):
    """One forward returning (logits, final hidden), both batch-first.

    ``gather_logits`` all-gathers the vocab-parallel output across TP — REQUIRED
    for any softmax/KL over the vocab axis under TP>1 (a vocab shard softmax is
    silently wrong). Default False preserves the trainer's behaviour (TP-1
    training, CE handled elsewhere).
    """
    from megatron.rl.compaction.learned.training.data import StudentOutput

    # Probe tokens are stored CPU-side (trajectories are CPU-resident by
    # design); the forward runs wherever the model lives.
    response_token_ids = response_token_ids.to(next(gpt.parameters()).device)
    forward_kwargs = dict(
        position_ids=None,
        attention_mask=None,
        runtime_gather_output=gather_logits or None,
    )
    saved_flash_decode = None
    if packed_thd_kv_len is not None:
        # The RL training model runs THD/packed attention — mirror the
        # capture-forward recipe (kv_capture.py): THD PackedSeqParams +
        # flash_decode disabled. cu_seqlens_kv reflects the INJECTED length,
        # identical for every attention layer (one compact memory size C).
        from megatron.core.packed_seq_params import PackedSeqParams
        S = response_token_ids.shape[1]
        device = response_token_ids.device
        forward_kwargs["position_ids"] = torch.arange(
            S, device=device).unsqueeze(0)
        forward_kwargs["packed_seq_params"] = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=torch.tensor([0, S], dtype=torch.int32, device=device),
            cu_seqlens_kv=torch.tensor([0, packed_thd_kv_len], dtype=torch.int32,
                                       device=device),
            max_seqlen_q=S,
            max_seqlen_kv=packed_thd_kv_len,
            total_tokens=S,
        )
        saved_flash_decode = gpt.config.flash_decode
        gpt.config.flash_decode = False
    hidden_sink: list = []
    try:
        with _capture_final_hidden(gpt, hidden_sink), _allow_nonflash_attention():
            try:
                output = gpt(
                    input_ids=response_token_ids,
                    **forward_kwargs,
                )
            except RuntimeError as e:
                if "qkv memory layout" in str(e):
                    raise RuntimeError(
                        "injected student forward hit TE qkv-layout rejection: "
                        f"tokens={tuple(response_token_ids.shape)} "
                        f"packed_thd_kv_len={packed_thd_kv_len} "
                        f"flash_decode={getattr(gpt.config, 'flash_decode', None)} "
                        f"sequence_parallel={getattr(gpt.config, 'sequence_parallel', None)}. "
                        "Known-good path: the inference-engine model (SBHD, the"
                        " eviction-RL/ladder recipe). The TRAINING model's packed/THD attention "
                        "still rejects injected KV — track shapes at the "
                        "core_attention pre-hook to close this.") from e
                raise
    finally:
        if saved_flash_decode is not None:
            gpt.config.flash_decode = saved_flash_decode
    # GPTModel returns (logits, ...) or just logits depending on version.
    logits = output[0] if isinstance(output, (tuple, list)) else output
    if len(hidden_sink) != 1:
        raise RuntimeError(
            f"final-hidden capture fired {len(hidden_sink)} times (expected 1).")
    hidden = hidden_sink[0].transpose(0, 1)          # (S, B, d) -> (B, S, d)
    return StudentOutput(logits=logits, hidden=hidden)


def student_outputs(
    model,
    response_token_ids: torch.Tensor,
    compact_kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
    gather_logits: bool = False,
    packed_thd: bool = False,
):
    """Student forward with compact KV as context → StudentOutput.

    Runs model(response_token_ids) with each attention layer's K, V replaced
    by the corresponding compact_kv.  Gradients flow through compact_kv to the
    compactor; model weights contribute no gradient (temporarily frozen).

    Args:
        model: Megatron model (list or single module, as passed to forward_step).
        response_token_ids: (B, S_resp) LongTensor — response tokens.
        compact_kv_list: list of (compact_k, compact_v) per attention layer,
            each (B, C, d_kv) with requires_grad=True.

    Returns:
        StudentOutput(logits (B, S_resp, vocab), hidden (B, S_resp, d_model)),
        both differentiable w.r.t. compact_kv.
    """
    gpt = _unwrap_model(model)

    # Temporarily freeze model parameters — only compactor accumulates grad.
    frozen = [p for p in gpt.parameters() if p.requires_grad]
    for p in frozen:
        p.requires_grad_(False)

    kv_len = compact_kv_list[0][0].shape[1] if packed_thd else None
    try:
        with _inject_compact_kv(model, compact_kv_list):
            return _forward_outputs(gpt, response_token_ids, gather_logits,
                                    packed_thd_kv_len=kv_len)
    finally:
        for p in frozen:
            p.requires_grad_(True)


@torch.no_grad()
def teacher_outputs(model, response_token_ids: torch.Tensor,
                    gather_logits: bool = False):
    """Full-KV teacher forward → StudentOutput (logits + final hidden).

    The capture side of the future-latent loss: run the frozen model with its
    REAL context (no injection) over the probe tokens and record the final
    hidden states the compact cache must reproduce. Store the results on
    ``TrainingProbe.teacher_logits`` / ``teacher_hidden``.
    """
    gpt = _unwrap_model(model)
    return _forward_outputs(gpt, response_token_ids, gather_logits)
