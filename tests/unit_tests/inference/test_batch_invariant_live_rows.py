# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Batch-invariant MoE inference kernels bounded by the live row count.

The permuted expert buffers are sized for the worst case (max_tokens x EP x topk rows) so CUDA
graphs see static shapes. These tests check that

- MXFP8 quantization indexes rows in int64 (rows x K >= 2**31 must not overflow);
- the row-bounded kernels touch only the live rows and produce exactly the bits of the
  whole-buffer kernels on those rows, eagerly and under CUDA-graph replay;
- the batch-invariant fused-MoE decode step (permute -> activation -> MXFP8 quantize ->
  unpermute) costs O(live rows), not O(max_tokens).
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.internal,
]


def _scale_bytes(n_rows: int, K: int) -> int:
    from megatron.core.inference.quantization.mxfp8_quantize import MXFP8_SCALE_ROW_BLOCK

    n_col_blocks = (K // 32 + 3) // 4
    return n_rows // MXFP8_SCALE_ROW_BLOCK * n_col_blocks * 512


def _check_rows(x, n, data, scale):
    from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize

    ref_data, ref_scale = mxfp8_quantize(x)
    assert torch.equal(ref_data[:n].view(torch.uint8), data[:n].view(torch.uint8))
    nb = _scale_bytes(n, x.shape[1])
    assert torch.equal(ref_scale.view(torch.uint8)[:nb], scale.view(torch.uint8)[:nb])


# --- int64 row indexing ------------------------------------------------------------------------

# Runs in a subprocess: on an int32-indexed kernel this is an illegal memory access, which would
# poison the CUDA context of the whole pytest session.
_OVERFLOW_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch
    from megatron.core.inference.quantization import mxfp8_quantize as mq

    K = 7168
    R = 2**31 // K + 256  # rows * K >= 2**31 elements
    assert R * K >= 2**31
    tail = 512  # last rows, starting on a 128-row scale tile
    start = (R - tail) // 128 * 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.empty(R, K, device="cuda", dtype=torch.bfloat16)
    x[:start].fill_(1.0)
    x[start:] = torch.randn(R - start, K, device="cuda", dtype=torch.bfloat16, generator=g) * 4

    ref_data, ref_scale = mq.mxfp8_quantize(x[start:].contiguous())
    n_col_blocks = (K // 32 + 3) // 4
    tile_bytes = n_col_blocks * 512
    full_tiles = (R - start) // 128  # compare whole 128-row scale tiles only

    def check(name, data, scale):
        torch.cuda.synchronize()
        ok_data = torch.equal(data[start:].view(torch.uint8), ref_data.view(torch.uint8))
        s0 = start // 128 * tile_bytes
        ok_scale = torch.equal(
            scale.view(torch.uint8)[s0 : s0 + full_tiles * tile_bytes],
            ref_scale.view(torch.uint8)[: full_tiles * tile_bytes],
        )
        print(f"{name}: data_equal={ok_data} scale_equal={ok_scale}", flush=True)
        return ok_data and ok_scale

    data, scale = mq.mxfp8_quantize(x)
    ok = check("mxfp8_quantize", data, scale)
    del data, scale
    if hasattr(mq, "mxfp8_quantize_rows"):
        n = torch.tensor([R], device="cuda", dtype=torch.int32)
        data, scale = mq.mxfp8_quantize_rows(x, n)
        ok = check("mxfp8_quantize_rows", data, scale) and ok
    sys.exit(0 if ok else 3)
    """
)


def test_mxfp8_quantize_rows_times_k_over_int32():
    """Quantizing a [R, 7168] bf16 tensor with R*K >= 2**31 (~4.3 GB) must neither fault nor
    corrupt the last rows: they must equal quantizing those rows alone, bit for bit."""
    free, _ = torch.cuda.mem_get_info()
    if free < 12 * 2**30:
        pytest.skip("needs ~12 GB of free GPU memory")
    proc = subprocess.run(
        [sys.executable, "-c", _OVERFLOW_SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        env=dict(os.environ),
    )
    assert proc.returncode == 0, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    )


# --- live-row MXFP8 quantization ---------------------------------------------------------------


class TestMxfp8QuantizeRows:
    @pytest.mark.parametrize("K", [1856, 2688, 4096])
    @pytest.mark.parametrize("n", [0, 128, 1280, 8192])
    def test_live_rows_bitwise_equal(self, K, n):
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize_rows

        torch.manual_seed(0)
        x = torch.randn(8192, K, device="cuda", dtype=torch.bfloat16) * 4
        n_t = torch.tensor([n], device="cuda", dtype=torch.int32)
        data, scale = mxfp8_quantize_rows(x, n_t)
        _check_rows(x, n, data, scale)
        # Rows past n are never quantized: their scales stay zero.
        assert not scale.view(torch.uint8)[_scale_bytes(n, K) :].any()

    def test_cuda_graph_replay_with_changing_count(self):
        from megatron.core.inference.quantization.mxfp8_quantize import mxfp8_quantize_rows

        torch.manual_seed(1)
        x = torch.randn(4096, 2688, device="cuda", dtype=torch.bfloat16)
        n_t = torch.tensor([256], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mxfp8_quantize_rows(x, n_t)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            data, scale = mxfp8_quantize_rows(x, n_t)
        for n in (256, 3840, 128):
            n_t.fill_(n)
            graph.replay()
            _check_rows(x, n, data, scale)


# --- live-token batch-invariant unpermute ------------------------------------------------------


class TestUnpermuteInExpertOrderLiveTokens:
    @pytest.mark.parametrize("valid", [1, 40, 1000, 3000])
    def test_matches_dense_reference(self, valid):
        from megatron.core.inference.moe.batch_invariant import unpermute_tokens_in_expert_order

        torch.manual_seed(2)
        hidden, experts, topk, num_tokens = 512, 16, 4, 4096
        rows = valid * topk + 128 * experts
        expert_out = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        inverse = torch.full((num_tokens, experts), -1, device="cuda", dtype=torch.int32)
        pos = torch.randperm(rows, device="cuda")[: valid * topk].view(valid, topk).int()
        eid = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(valid)])
        inverse[torch.arange(valid, device="cuda")[:, None], eid] = pos
        out = torch.full((num_tokens, hidden), 7.0, device="cuda", dtype=torch.float32)
        valid_t = torch.tensor([valid], device="cuda", dtype=torch.int32)
        unpermute_tokens_in_expert_order(expert_out, inverse, valid_t, out)
        # Reference: fp32 sum in increasing local-expert order (this kernel's accumulator).
        ref = torch.zeros(valid, hidden, device="cuda", dtype=torch.float32)
        for lid in range(experts):
            p = inverse[:valid, lid].long()
            m = p >= 0
            ref[m] += expert_out[p[m]].float()
        assert torch.equal(out[:valid], ref)
        # Tokens past the live count are not written.
        assert torch.all(out[valid:] == 7.0)


# --- O(live rows) decode step --------------------------------------------------------------------


def _bi_decode_step_ms(max_tokens, valid, use_mxfp8, monkeypatch):
    """GPU ms of one CUDA-graphed batch-invariant mcore_fused_moe step with `valid` live tokens
    and buffers sized for `max_tokens`. The expert GEMMs are replaced by a fixed buffer so that
    only the permute / activation / quantize / unpermute kernels are timed."""
    from megatron.core.inference.moe import batch_invariant, fused_moe
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
    )

    hidden, experts, topk = 256, 8, 2
    gen = torch.Generator(device="cuda").manual_seed(0)
    stub_out = {}

    def stub_mm(x, w, offs):
        rows = (x.data if isinstance(x, MXFP8Tensor) else x).shape[0]
        n_out = (w.data if isinstance(w, MXFP8Tensor) else w).shape[1]
        if (rows, n_out) not in stub_out:
            stub_out[(rows, n_out)] = torch.randn(
                rows, n_out, device="cuda", dtype=torch.bfloat16, generator=gen
            )
        return stub_out[(rows, n_out)]

    if use_mxfp8:
        if not fused_moe.HAVE_SCALED_GMM:
            pytest.skip("MXFP8 fused MoE needs torch scaled_grouped_mm")
        qs = [
            MXFP8Tensor.from_bf16(
                torch.randn(hidden, hidden, device="cuda", dtype=torch.bfloat16), backend="triton"
            )
            for _ in range(experts)
        ]
        weight = MXFP8Tensor(
            data=torch.stack([q.data for q in qs]).contiguous(),
            scale=torch.stack([q.scale for q in qs]).contiguous(),
            dtype=torch.bfloat16,
            backend="triton",
        )
        monkeypatch.setattr(fused_moe, "_mxfp8_grouped_mm", stub_mm)
    else:
        weight = torch.randn(experts, hidden, hidden, device="cuda", dtype=torch.bfloat16)
        monkeypatch.setattr(batch_invariant, "grouped_mm", stub_mm)
        monkeypatch.setattr(batch_invariant, "grouped_mm_alignment", lambda: 128)

    x = torch.randn(max_tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
    probs = torch.rand(max_tokens, topk, device="cuda", generator=gen)
    routing = torch.randint(0, experts, (max_tokens, topk), device="cuda", generator=gen)
    valid_t = torch.tensor([valid], device="cuda", dtype=torch.int32)

    def step():
        return fused_moe.mcore_fused_moe(
            x,
            probs,
            weight,
            weight,
            fused_moe.ActivationType.SQUARED_RELU,
            experts,
            0,
            valid_t,
            routing,
        )

    with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    reps, per = 15, 10
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(per):
            graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / per)
    del graph
    torch.cuda.empty_cache()
    return sorted(times)[reps // 2]


@pytest.mark.parametrize("use_mxfp8", [False, True], ids=["bf16", "mxfp8"])
def test_bi_decode_step_cost_independent_of_max_tokens(use_mxfp8, monkeypatch):
    """With 16 live tokens the graphed BI MoE step must cost about the same whether the
    buffers are sized for 2**10 or 2**20 tokens (worst-case sizing is for CUDA graphs only)."""
    small = _bi_decode_step_ms(2**10, 16, use_mxfp8, monkeypatch)
    large = _bi_decode_step_ms(2**20, 16, use_mxfp8, monkeypatch)
    print(f"bi decode step ({'mxfp8' if use_mxfp8 else 'bf16'}): {small:.4f} ms @2^10, "
          f"{large:.4f} ms @2^20, ratio {large / small:.2f}")
    assert large < 4 * small, f"{large:.4f} ms at 2**20 vs {small:.4f} ms at 2**10"
