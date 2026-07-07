# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Delta re-rotation of cached RoPE keys.

The paged cache stores keys already rotated by their original positions
(Megatron applies RoPE before ``core_attention``). RoPE is a block-diagonal
family of 2D rotations, one plane per frequency, and rotations in a plane
compose additively: a key stored at position ``m`` moves to position ``m'`` by
applying the rotation for ``m' - m`` — no un-rotate/re-rotate round trip and no
access to the pre-rotation key needed. Values are untouched (RoPE applies only
to Q/K).

Used by ``LiveKVCompactor`` rope_mode='renumber' (StreamingLLM-style contiguous
cache positions: retained keys re-rotate to 0..C-1 at prune time; restored
archive spans re-rotate to the cache tail). rope_mode='logical' never calls
this — it keeps original rotations and patches query position ids instead.

Frequencies come from the model's own ``RotaryEmbedding.inv_freq`` so base,
rotary_percent (partial rotary: ``2 * len(inv_freq) < D`` leaves the tail dims
unrotated), and scaling always match what the model applied.
"""

from __future__ import annotations

import torch


def delta_rotate_keys(
    keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """Re-rotate cached keys from their old positions to new positions.

    Args:
        keys: (..., T, H, D) keys as stored in the cache (rotated at
            ``old_positions``). Any number of leading dims (layers).
        old_positions / new_positions: (T,) integer position tensors.
        inv_freq: (rot_dim / 2,) inverse frequencies from the model's
            RotaryEmbedding.
        interleaved: RoPE layout — False = NeoX halves (Megatron default),
            True = adjacent even/odd pairs (``--rotary-interleaved``).

    Returns:
        Keys rotated as if they had been written at ``new_positions``; same
        shape and dtype as ``keys``.
    """
    if old_positions.shape != new_positions.shape or old_positions.dim() != 1:
        raise ValueError(
            f"positions must be matching 1-D tensors, got {tuple(old_positions.shape)} "
            f"vs {tuple(new_positions.shape)}")
    T = old_positions.shape[0]
    if keys.shape[-3] != T:
        raise ValueError(
            f"keys token dim {keys.shape[-3]} != positions length {T}")
    rot_dim = 2 * inv_freq.shape[0]
    D = keys.shape[-1]
    if rot_dim > D:
        raise ValueError(f"rot_dim {rot_dim} > head dim {D}")

    delta = (new_positions - old_positions).to(device=keys.device, dtype=torch.float32)
    angles = torch.outer(delta, inv_freq.to(keys.device).float())     # (T, rot/2)
    if interleaved:
        angles = angles.repeat_interleave(2, dim=-1)                  # (T, rot)
    else:
        angles = torch.cat([angles, angles], dim=-1)                  # (T, rot)
    # Broadcast over leading dims and heads: (T, rot) -> (T, 1, rot).
    cos = angles.cos().unsqueeze(-2)
    sin = angles.sin().unsqueeze(-2)

    x = keys[..., :rot_dim].float()
    if interleaved:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        half = torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)
    else:
        x1, x2 = x.chunk(2, dim=-1)
        half = torch.cat((-x2, x1), dim=-1)
    rotated = (x * cos + half * sin).to(keys.dtype)
    if rot_dim == D:
        return rotated
    return torch.cat([rotated, keys[..., rot_dim:]], dim=-1)
