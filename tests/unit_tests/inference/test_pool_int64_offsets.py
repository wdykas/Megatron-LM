# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Inference kernels that index a per-layer KV / state pool by a block or slot id must compute that offset in int64.

Each test builds a pool whose last block / slot starts past 2^31 elements and runs the kernel on that block / slot.
The output must be bitwise equal to the same kernel run on a small pool that holds the same data at index 0. An int32
product wraps to a wrong address, so the output differs or the kernel faults (the probe runs in a subprocess).
Footprint: up to ~9 GiB of GPU memory per test.
"""

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _need(nbytes: int) -> None:
    torch.cuda.empty_cache()
    if torch.cuda.mem_get_info()[0] < nbytes + (2 << 30):
        pytest.skip(f"needs {nbytes / 2**30:.1f} GiB free")


def _probe(code: str) -> None:
    """Run in a subprocess: an int32 wrap can fault (illegal address), which would poison this process's context."""
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "PROBE_OK" in r.stdout, (r.stdout[-600:], r.stderr[-1500:])


def test_kv_append_per_layer_slice_past_2pow31():
    _need(2 * 32776 * 256 * 2 * 128 * 2)
    _probe(
        """
        import torch
        from megatron.core.inference.contexts.fused_kv_append_kernel import triton_append_key_value_cache
        nb = 32776  # one layer's K slice: nb x 65536 elements > 2^31; block nb - 3 starts past 2^31
        buf = torch.zeros(2, 1, nb, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
        assert (nb - 3) * buf.stride(2) >= 2**31
        g = torch.Generator(device="cuda").manual_seed(0)
        k = torch.randn(3, 1, 2, 128, device="cuda", generator=g).bfloat16()
        v = torch.randn(3, 1, 2, 128, device="cuda", generator=g).bfloat16()
        # int32 block ids: the engine passes int64 (gpu_view.token_to_block_idx), any other caller may not
        blocks = torch.tensor([nb - 3, nb - 2, 7], dtype=torch.int32, device="cuda")
        pos = torch.tensor([5, 255, 0], dtype=torch.int32, device="cuda")
        triton_append_key_value_cache(0, k, v, buf, 3, blocks, pos, dummy_block_idx=nb - 1)
        torch.cuda.synchronize()
        for i in range(3):
            b, p = int(blocks[i]), int(pos[i])
            assert torch.equal(buf[0, 0, b, p], k[i, 0]) and torch.equal(buf[1, 0, b, p], v[i, 0]), i
        assert int(buf.count_nonzero()) == int(k.count_nonzero()) + int(v.count_nonzero())
        print("PROBE_OK")
        """
    )


def test_selective_state_update_slot_past_2pow31():
    S, H, P, N = 32772, 8, 64, 128  # 65536 fp32 elements per slot: slot S - 2 starts past 2^31
    _need(S * H * P * N * 4)
    _probe(
        f"""
        import torch
        from megatron.core.ssm.ops.mamba2.mamba_ssm import selective_state_update
        S, H, P, N, G = {S}, {H}, {P}, {N}, 2
        g = torch.Generator(device="cuda").manual_seed(1)
        r = lambda *s: torch.randn(*s, device="cuda", generator=g)
        big = torch.zeros(S, H, P, N, device="cuda")
        slot = S - 2
        assert slot * big.stride(0) >= 2**31
        init = r(1, H, P, N)
        big[slot] = init[0]
        small = init.clone()
        x, dt, z = r(1, H, P), r(1, H, P), r(1, H, P)
        A, D, dtb = -r(H, P, N).abs(), r(H, P), r(H, P)
        B, C = r(1, G, N), r(1, G, N)
        idx = torch.tensor([slot], dtype=torch.int32, device="cuda")
        out_big = selective_state_update(big, x, dt, A, B, C, D=D, z=z, dt_bias=dtb, dt_softplus=True,
                                         state_batch_indices=idx)
        out_small = selective_state_update(small, x, dt, A, B, C, D=D, z=z, dt_bias=dtb, dt_softplus=True,
                                           state_batch_indices=torch.zeros(1, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        assert torch.equal(out_big, out_small)
        assert torch.equal(big[slot], small[0])
        assert int(big[: S // 2].count_nonzero()) == 0
        print("PROBE_OK")
        """
    )


def test_causal_conv1d_update_slot_past_2pow31():
    dim, W = 6144, 4
    S = (2**31) // (dim * (W - 1)) + 8
    _need(S * dim * (W - 1) * 2)
    _probe(
        f"""
        import torch
        from megatron.core.ssm.ops.common.causal_conv1d_triton import causal_conv1d_update
        S, dim, W = {S}, {dim}, {W}
        g = torch.Generator(device="cuda").manual_seed(2)
        big = torch.zeros(S, dim, W - 1, device="cuda", dtype=torch.bfloat16)
        slot = S - 3
        assert slot * big.stride(0) >= 2**31
        init = torch.randn(1, dim, W - 1, device="cuda", generator=g).bfloat16()
        big[slot] = init[0]
        small = init.clone()
        x = torch.randn(1, dim, device="cuda", generator=g).bfloat16()
        w = torch.randn(dim, W, device="cuda", generator=g).bfloat16()
        b = torch.randn(dim, device="cuda", generator=g).bfloat16()
        o_big = causal_conv1d_update(x, big, w, b, True, torch.tensor([slot], dtype=torch.int32, device="cuda"))
        o_small = causal_conv1d_update(x, small, w, b, True, torch.zeros(1, dtype=torch.int32, device="cuda"))
        torch.cuda.synchronize()
        assert torch.equal(o_big, o_small)
        assert torch.equal(big[slot], small[0])
        assert int(big[: S // 2].count_nonzero()) == 0
        print("PROBE_OK")
        """
    )


def test_bi_masked_update_rows_past_2pow31():
    R, ROW = 4100, 524288  # one Nano Mamba SSM state per row (64 x 64 x 128 fp32)
    _need(R * ROW * 4)
    _probe(
        f"""
        import torch
        from megatron.core.ssm.ops.mamba2.batch_invariant_decode import _masked_update_rows
        R, ROW = {R}, {ROW}
        states = torch.zeros(R, ROW, device="cuda")
        vals = torch.randn(2, ROW, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
        idx = torch.tensor([R - 3, -1], dtype=torch.int32, device="cuda")
        assert (R - 3) * states.stride(0) >= 2**31
        _masked_update_rows(states, idx, vals)
        torch.cuda.synchronize()
        assert torch.equal(states[R - 3], vals[0])
        assert int(states.count_nonzero()) == int(vals[0].count_nonzero())
        print("PROBE_OK")
        """
    )
