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

from contextlib import contextmanager
from typing import List, Tuple

import torch

from .kv_capture import _unwrap_model


def _attention_layers(model):
    """Return (layer_index, layer) for each TransformerLayer that has self_attention."""
    gpt = _unwrap_model(model)
    return [
        (i, layer)
        for i, layer in enumerate(gpt.decoder.layers)
        if hasattr(layer, "self_attention")
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
                query = args[0]   # (S_q, B, n_heads, d_head)
                orig_key = args[1]  # (S_k, B, n_kv_groups, d_head)

                B = orig_key.shape[1]
                n_kv_groups = orig_key.shape[2]
                d_head = orig_key.shape[3]
                C = ck_cap.shape[1]

                # (B, C, d_kv) → (B, C, n_kv_groups, d_head) → (C, B, n_kv_groups, d_head)
                ck_r = ck_cap.reshape(B, C, n_kv_groups, d_head).permute(1, 0, 2, 3)
                cv_r = cv_cap.reshape(B, C, n_kv_groups, d_head).permute(1, 0, 2, 3)

                # attention_mask=None: all query positions attend to all C slots
                return (query, ck_r, cv_r, None) + tuple(args[4:])
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


def _forward_outputs(gpt, response_token_ids: torch.Tensor):
    """One forward returning (logits, final hidden), both batch-first."""
    from megatron.rl.compaction.learned.training.data import StudentOutput

    hidden_sink: list = []
    with _capture_final_hidden(gpt, hidden_sink):
        output = gpt(
            input_ids=response_token_ids,
            position_ids=None,
            attention_mask=None,
        )
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

    try:
        with _inject_compact_kv(model, compact_kv_list):
            return _forward_outputs(gpt, response_token_ids)
    finally:
        for p in frozen:
            p.requires_grad_(True)


@torch.no_grad()
def teacher_outputs(model, response_token_ids: torch.Tensor):
    """Full-KV teacher forward → StudentOutput (logits + final hidden).

    The capture side of the future-latent loss: run the frozen model with its
    REAL context (no injection) over the probe tokens and record the final
    hidden states the compact cache must reproduce. Store the results on
    ``TrainingProbe.teacher_logits`` / ``teacher_hidden``.
    """
    gpt = _unwrap_model(model)
    return _forward_outputs(gpt, response_token_ids)
