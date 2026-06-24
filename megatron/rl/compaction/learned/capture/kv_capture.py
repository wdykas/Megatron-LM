# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Capture K/V matrices from a Megatron model forward pass via attention hooks.

Unlike MegatronInferenceHook (which reads the paged KV cache), this module
hooks directly into core_attention.forward_pre_hook to capture key/value
tensors as they are computed.  Works when the inference engine is suspended
and a regular (non-CUDA-graph) forward pass is available.

Usage
-----
    keys, vals = capture_kv_from_forward(gpt_model, tokens, position_ids)
    # keys: list of (1, S, H_kv * d_head) per attention layer
    # vals: same shape
"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple

try:
    from megatron.core.packed_seq_params import PackedSeqParams
    _HAVE_PACKED_SEQ_PARAMS = True
except ImportError:
    _HAVE_PACKED_SEQ_PARAMS = False


def _unwrap_model(model):
    """Strip list/DDP wrappers to get the raw GPTModel."""
    m = model[0] if isinstance(model, (list, tuple)) else model
    while hasattr(m, "module"):
        m = m.module
    return m


def _attn_core_modules(model):
    """Return list of core_attention modules for all self-attention layers."""
    gpt = _unwrap_model(model)
    cores = []
    for layer in gpt.decoder.layers:
        if hasattr(layer, "self_attention") and hasattr(layer.self_attention, "core_attention"):
            cores.append(layer.self_attention.core_attention)
    return cores


def capture_kv_from_forward(
    model,
    tokens: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
    """Run a forward pass and capture K/V from all attention layers.

    All TP ranks must call this simultaneously (it issues collective
    communications inside the model forward).  Only the calling rank's
    local K/V partition is captured — when GQA KV heads are replicated
    across TP ranks, rank 0 has the full K/V and no all-gather is needed.

    Parameters
    ----------
    model:
        Megatron model (list-of-modules or single module).
    tokens:
        (1, S) LongTensor on CUDA.
    position_ids:
        (1, S) LongTensor on CUDA, or None (model infers positions).

    Returns
    -------
    (keys_per_layer, vals_per_layer) or None.
        Each list has one tensor per attention layer, shape (1, S, H_kv * d_head).
    """
    gpt = _unwrap_model(model)
    cores = _attn_core_modules(model)
    n_layers = len(cores)

    if n_layers == 0:
        return None

    captured: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
    hooks = []

    for idx, core in enumerate(cores):
        def _make_hook(i):
            def _pre_hook(module, args):
                # args: (query, key, value, attention_mask, ...)
                # key shape: SBHD (S, B, H_kv, d_head) or THD (T, H_kv, d_head)
                if len(args) < 3:
                    return args
                k = args[1]
                v = args[2]
                if k.dim() == 4:
                    S, B, H_kv, D = k.shape
                    captured[i] = (
                        k.permute(1, 0, 2, 3).reshape(B, S, H_kv * D).detach().cpu(),
                        v.permute(1, 0, 2, 3).reshape(B, S, H_kv * D).detach().cpu(),
                    )
                elif k.dim() == 3:
                    # THD format when packed_seq_params is provided
                    T, H_kv, D = k.shape
                    captured[i] = (
                        k.reshape(1, T, H_kv * D).detach().cpu(),
                        v.reshape(1, T, H_kv * D).detach().cpu(),
                    )
                return args
            return _pre_hook
        hooks.append(core.register_forward_pre_hook(_make_hook(idx)))

    # Build PackedSeqParams matching get_logprobs() so the forward takes the same
    # code path as logprob computation (avoids CUDA-graph signature mismatch and
    # flash-attention format errors that cause rank 0 to fail before the first
    # NCCL collective, deadlocking the other TP ranks).
    packed_seq_params = None
    if _HAVE_PACKED_SEQ_PARAMS:
        S = tokens.shape[1]
        cu_seqlens = torch.tensor([0, S], dtype=torch.int32, device=tokens.device)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=S,
            max_seqlen_kv=S,
            total_tokens=S,
        )

    # Match the logprobs forward path: eval mode + flash_decode disabled.
    flash_decode = getattr(gpt.config, 'flash_decode', False)
    gpt.config.flash_decode = False
    was_training = gpt.training
    gpt.eval()

    try:
        with torch.no_grad():
            gpt(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                runtime_gather_output=True,
            )
    except Exception as exc:
        import warnings
        warnings.warn(f"capture_kv_from_forward: forward pass failed: {exc}")
    finally:
        for h in hooks:
            h.remove()
        gpt.config.flash_decode = flash_decode
        if was_training:
            gpt.train()

    good = [c for c in captured if c is not None]
    if len(good) < n_layers:
        import warnings
        warnings.warn(
            f"capture_kv_from_forward: only {len(good)}/{n_layers} layers captured K/V. "
            "The hooks may have fired in a format other than SBHD/THD."
        )
    if not good:
        return None

    keys = [c[0] for c in captured if c is not None]
    vals = [c[1] for c in captured if c is not None]
    return keys, vals
