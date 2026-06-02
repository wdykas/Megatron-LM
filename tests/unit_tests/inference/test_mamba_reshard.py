# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Hetero TP/PP reshard of Mamba conv/ssm state (pure, CPU).

Builds a known global Mamba state, shards it to a source (tp,pp) the exact way
mamba_mixer does ([x|B|C] conv bands + head-sharded ssm, layers split by PP),
runs plan_mamba_reshard to a different destination (tp,pp), and asserts every
destination rank ends up byte-identical to a direct shard of the global state.
This validates the band/layer index math against the real sharding model
without a hybrid checkpoint (the residual gap is a real-model functional run).
"""

import pytest
import torch

from megatron.core.inference.disaggregation.mamba_reshard import (
    MambaShardLayout,
    apply_conv_transfer,
    apply_ssm_transfer,
    plan_mamba_reshard,
)

# Global model dims (chosen divisible by the tp values under test).
NHEADS, HEADDIM, DSTATE, NGROUPS, DCONV = 8, 4, 2, 2, 3
M = 4  # global Mamba layers
D_INNER = NHEADS * HEADDIM            # 32
G = NGROUPS * DSTATE                  # 4  (B and C band global size)
CONV_DIM = D_INNER + 2 * G            # 40


def _global_state():
    """Distinct value per (layer, channel, ...) so any mis-slice is caught."""
    conv = torch.arange(M * CONV_DIM * DCONV, dtype=torch.float32).reshape(M, CONV_DIM, DCONV)
    ssm = (
        torch.arange(M * NHEADS * HEADDIM * DSTATE, dtype=torch.float32).reshape(
            M, NHEADS, HEADDIM, DSTATE
        )
        + 10_000.0
    )
    return conv, ssm


def _layouts(tp, pp):
    """One MambaShardLayout per rank for a (tp, pp) instance; rank = p*tp + r.
    PP splits the M layers evenly (contiguous per stage)."""
    per = M // pp
    out = {}
    for p in range(pp):
        for r in range(tp):
            rank = p * tp + r
            out[rank] = MambaShardLayout(
                global_rank=rank, tp_size=tp, tp_rank=r,
                layer_start=p * per, num_layers=per,
                nheads=NHEADS, headdim=HEADDIM, d_state=DSTATE, ngroups=NGROUPS, d_conv=DCONV,
            )
    return out


def _shard(conv_g, ssm_g, lay: MambaShardLayout):
    """Shard the global state to one rank exactly as mamba_mixer does."""
    s, e = lay.layer_range()
    r, tp = lay.tp_rank, lay.tp_size
    di_l = D_INNER // tp
    g_l = (NGROUPS // tp) * DSTATE
    x = conv_g[s:e, 0:D_INNER][:, r * di_l:(r + 1) * di_l]
    b = conv_g[s:e, D_INNER:D_INNER + G][:, r * g_l:(r + 1) * g_l]
    c = conv_g[s:e, D_INNER + G:D_INNER + 2 * G][:, r * g_l:(r + 1) * g_l]
    conv_l = torch.cat([x, b, c], dim=1).contiguous()
    nh_l = NHEADS // tp
    ssm_l = ssm_g[s:e, r * nh_l:(r + 1) * nh_l, :, :].contiguous()
    return conv_l, ssm_l


@pytest.mark.parametrize(
    "src,dst",
    [
        ((2, 1), (1, 1)),   # TP2 -> TP1 (band merge)
        ((1, 1), (2, 1)),   # TP1 -> TP2 (band split)
        ((1, 2), (1, 1)),   # PP2 -> PP1 (layer merge)
        ((1, 1), (1, 2)),   # PP1 -> PP2 (layer split)
        ((2, 2), (1, 1)),   # both axes hetero
        ((2, 1), (2, 1)),   # identity
    ],
)
def test_mamba_reshard_reconstructs_destination(src, dst):
    conv_g, ssm_g = _global_state()
    src_lay, dst_lay = _layouts(*src), _layouts(*dst)

    # Source per-rank tensors (as a prefill instance would hold them).
    src_t = {rk: _shard(conv_g, ssm_g, lay) for rk, lay in src_lay.items()}
    # Destination buffers, zero-filled at each rank's local shape.
    dst_t = {}
    for rk, lay in dst_lay.items():
        dst_t[rk] = (
            torch.zeros(lay.num_layers, lay.conv_dim_local, DCONV),
            torch.zeros(lay.num_layers, lay.nheads_local, HEADDIM, DSTATE),
        )

    plan = plan_mamba_reshard(list(src_lay.values()), list(dst_lay.values()))
    for t in plan:
        if t.is_conv:
            apply_conv_transfer(t, src_t[t.src_rank][0], dst_t[t.dst_rank][0])
        else:
            apply_ssm_transfer(t, src_t[t.src_rank][1], dst_t[t.dst_rank][1])

    # Every destination rank must match a direct shard of the global state.
    for rk, lay in dst_lay.items():
        want_conv, want_ssm = _shard(conv_g, ssm_g, lay)
        assert torch.equal(dst_t[rk][0], want_conv), f"conv mismatch at rank {rk} ({src}->{dst})"
        assert torch.equal(dst_t[rk][1], want_ssm), f"ssm mismatch at rank {rk} ({src}->{dst})"
