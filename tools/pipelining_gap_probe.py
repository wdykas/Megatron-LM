# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Measure the per-layer decode time on Nano, partition layers along
the canonical M / * / E disagg pattern, and compute the gap between
a fully-serial multi-shard step and a fully-pipelined one.

Method:

1. Load Nano 30B (single-shard collocated). Run K warmup forwards.
2. For each transformer layer, time its forward via CUDA events on
   a synthetic one-token decode input. Record the per-layer cost
   ``t_l`` for l = 0..51.
3. Walk the canonical Nano layer pattern
   ``MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME`` and
   partition consecutive layers of the same kind into "shards"
   (one entry per contiguous run of the same layer kind owned by a
   hypothetical disagg shard).
4. Estimate per-step wall time under three regimes:

   (a) **Serial**: every shard finishes its layers, ships across
       NVSHMEM, the next shard receives and starts. Step time =
       ``sum(shard_compute_times) + (N - 1) * hop_latency``.
   (b) **Async-send only** (today's default): cross-shard sends
       are non-blocking, but the receiver's ``s.synchronize()`` in
       ``receive_hidden`` blocks the host. So the next shard's
       compute can't start until the host returns from the wait.
       Step time = same as serial unless the engine has *other*
       work to do during the wait — which it doesn't in v1.
   (c) **Pipelined**: with host sync removed and a scheduler that
       can dispatch the next batch's work during cross-shard
       waits, compute on shard B for batch ``A_{n-1}`` runs
       concurrently with compute on shard A for batch ``A_n``.
       Effective step time ≈ ``max(shard_compute_times) + N *
       hop_latency / pipeline_depth``.

The "gap" is the ratio of serial step time to pipelined step time.
If it's > 1.5 we should build pipelining; if < 1.2 the implicit
async-send already does most of the work and we should focus
elsewhere.

Run as a single-process probe — TP=4 inside the process, no real
multi-shard disagg setup needed (we're measuring the per-layer
cost, not real cross-shard transport).
"""

import os
import sys
import time
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import torch

from hybrid_builders import hybrid_builder
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_model
from megatron.training.arguments import parse_and_validate_args  # noqa
from model_provider import model_provider


# Default NVSHMEM hop latency for a small (~5KB) BF16 activation.
# Source: literature numbers for NVLink put_signal + signal_wait
# round-trip on H100 / B200. Typical small-payload latencies are
# 5-15us depending on link congestion + flag overhead.
DEFAULT_HOP_LATENCY_US = 10.0


def add_probe_args(parser):
    group = parser.add_argument_group(title="pipelining gap probe")
    group.add_argument("--probe-warmup-steps", type=int, default=4)
    group.add_argument("--probe-measure-steps", type=int, default=20)
    group.add_argument(
        "--probe-hop-latency-us",
        type=float,
        default=DEFAULT_HOP_LATENCY_US,
        help="Assumed cross-shard hop latency in microseconds. "
        "Reported numbers are linear in this — adjust if you have a "
        "measured value.",
    )
    group.add_argument(
        "--probe-output-json", type=str, default=None,
    )
    return parser


def _unwrap(model):
    inner = model
    while not hasattr(inner, "decoder"):
        if hasattr(inner, "module"):
            inner = inner.module
        else:
            raise RuntimeError(f"could not find .decoder on {type(inner)}")
    return inner


def _partition_layers_by_kind(layer_type_list, layer_times_ms):
    """Walk the layer pattern, group consecutive same-kind layers
    into hops. Returns a list of dicts with kind, layer_indices,
    compute_ms — one per contiguous same-kind run."""
    if not layer_type_list:
        return []
    hops = []
    cur_kind = layer_type_list[0]
    cur_layers = [0]
    cur_time = layer_times_ms[0]
    for i in range(1, len(layer_type_list)):
        k = layer_type_list[i]
        if k == cur_kind:
            cur_layers.append(i)
            cur_time += layer_times_ms[i]
        else:
            hops.append({"kind": cur_kind, "layers": cur_layers, "compute_ms": cur_time})
            cur_kind = k
            cur_layers = [i]
            cur_time = layer_times_ms[i]
    hops.append({"kind": cur_kind, "layers": cur_layers, "compute_ms": cur_time})
    return hops


def _partition_by_layer_kind_3shard(layer_type_list, layer_times_ms):
    """Realistic disagg grouping: each layer kind to its own shard.
    Produces one shard per distinct kind, with that shard's
    compute = sum of all its layers. The route walks one hop per
    contiguous same-kind run (so the hop count is the same as
    _partition_layers_by_kind), but the shards represent the
    physical compute distribution.
    """
    shards = {}
    for i, k in enumerate(layer_type_list):
        if k not in shards:
            shards[k] = {"kind": k, "layers": [], "compute_ms": 0.0}
        shards[k]["layers"].append(i)
        shards[k]["compute_ms"] += layer_times_ms[i]
    return list(shards.values())


def _compute_gap(hops, hop_latency_us, pipeline_depth=None):
    """Given the hop list (one entry per contiguous same-kind run)
    + transport hop latency, compute the three step-time regimes.

    pipeline_depth: how many concurrent batches the pipelined model
    can have in flight. None means use the number of hops (max
    depth), which gives the most optimistic pipelining bound.
    """
    N = len(hops)
    if N == 0:
        return {}

    total_compute_ms = sum(h["compute_ms"] for h in hops)
    max_shard_ms = max(h["compute_ms"] for h in hops)
    hop_latency_ms = hop_latency_us / 1000.0

    # Serial regime: every shard runs sequentially.
    serial_step_ms = total_compute_ms + (N - 1) * hop_latency_ms

    # Async-send-only regime (today): receiver's host-sync means
    # the next shard's compute can't start until activation lands.
    # Equivalent to serial in v1 (no scheduler to dispatch other
    # work during wait).
    async_send_step_ms = serial_step_ms

    # Pipelined regime: with multiple batches in flight, the
    # bottleneck is the slowest shard's compute. Per-batch latency
    # is still serial, but per-step *throughput* is bound by the
    # slowest shard. The interesting metric is wall time for a long
    # decode run — steady-state per-token latency.
    #
    # In steady state with full pipeline depth, every shard is
    # always busy, so per-token throughput = 1 / max_shard_ms.
    # Per-token wall time = max_shard_ms + hop_latency_ms.
    pipelined_step_ms = max_shard_ms + hop_latency_ms

    return {
        "n_hops": N,
        "total_compute_ms": total_compute_ms,
        "max_shard_compute_ms": max_shard_ms,
        "min_shard_compute_ms": min(h["compute_ms"] for h in hops),
        "hop_latency_ms": hop_latency_ms,
        "serial_step_ms": serial_step_ms,
        "async_send_step_ms": async_send_step_ms,
        "pipelined_step_ms": pipelined_step_ms,
        "serial_to_pipelined_gap": serial_step_ms / pipelined_step_ms,
        "shard_imbalance": max_shard_ms / (total_compute_ms / N) if N > 0 else 1.0,
    }


@torch.no_grad()
def _time_layers_via_hooks(
    model, decoder, input_ids, position_ids, attention_mask, n_warmup=4, n_measure=10
):
    """Install forward-pre and forward hooks on every layer in the
    decoder; run a real model forward; collect per-layer wall time.

    This is more reliable than calling individual layers in isolation
    because:
      - the model's forward sets up all the context layers expect
      - layers run in their natural sequence (no first-call init dominating)
      - we get real decode-time numbers, not synthetic ones

    Runs n_warmup forwards first to flush any lazy CUDA init, then
    n_measure forwards with hooks to record per-layer times. Returns
    layer_times_ms as the mean across measure runs.
    """
    n_layers = len(decoder.layers)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_layers)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_layers)]

    sums = [0.0] * n_layers
    counts = [0] * n_layers

    def pre_hook_factory(idx):
        def hook(_module, _inputs):
            starts[idx].record()
        return hook

    def post_hook_factory(idx):
        def hook(_module, _inputs, _output):
            ends[idx].record()
        return hook

    handles = []
    for i, layer in enumerate(decoder.layers):
        handles.append(layer.register_forward_pre_hook(pre_hook_factory(i)))
        handles.append(layer.register_forward_hook(post_hook_factory(i)))

    try:
        # Warmup forwards — flush lazy init, kernel compilation, etc.
        for _ in range(n_warmup):
            _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            torch.cuda.synchronize()

        # Measure forwards.
        for _ in range(n_measure):
            _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            torch.cuda.synchronize()
            for i in range(n_layers):
                try:
                    ms = starts[i].elapsed_time(ends[i])
                    sums[i] += ms
                    counts[i] += 1
                except Exception:
                    pass
    finally:
        for h in handles:
            h.remove()

    return [
        sums[i] / counts[i] if counts[i] > 0 else 0.0 for i in range(n_layers)
    ]


def main():
    parse_and_validate_args(
        extra_args_provider=add_probe_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
        },
    )
    initialize_megatron()
    args = get_args()

    print_rank_0("[probe] loading Nano 30B...")
    model = get_model(partial(model_provider, hybrid_builder), wrap_with_ddp=False)
    _ = load_checkpoint(model, None, None, strict=False)
    model = model[0]
    model.eval()
    inner = _unwrap(model)
    print_rank_0("[probe] loaded.")

    decoder = inner.decoder
    layer_type_list = decoder.layer_type_list
    n_layers = len(layer_type_list)
    pattern = "".join(layer_type_list)
    print_rank_0(f"[probe] layer pattern: {pattern}")
    print_rank_0(f"[probe] n_layers: {n_layers}")

    device = torch.cuda.current_device()
    # Construct a minimal "real" decode input: a short prompt that
    # produces sensible shape through the model's normal forward.
    # We use seq_len=8 (typical small prompt) to avoid edge cases
    # at seq_len=1.
    seq_len = 8
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = torch.tril(
        torch.ones((1, 1, seq_len, seq_len), device=device)
    ).bool()

    print_rank_0(
        f"[probe] timing each of {n_layers} layers via forward hooks "
        f"({args.probe_warmup_steps} warmup + {args.probe_measure_steps} measure)..."
    )
    layer_times_ms = _time_layers_via_hooks(
        model, decoder, input_ids, position_ids, attention_mask,
        n_warmup=args.probe_warmup_steps,
        n_measure=args.probe_measure_steps,
    )

    valid_layers = [i for i, t in enumerate(layer_times_ms) if t > 0]
    if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        print_rank_0(
            f"[probe] timed {len(valid_layers)}/{n_layers} layers successfully"
        )
        for i in range(0, n_layers, 8):
            chunk = layer_times_ms[i : i + 8]
            kinds = "".join(layer_type_list[i : i + 8])
            print_rank_0(
                f"[probe]   layers {i:2d}-{i + len(chunk) - 1:2d} ({kinds}): "
                + ", ".join(f"{t:.3f}" for t in chunk)
            )

    # Two partition models:
    #   (a) Contiguous-run hops: route walks the pattern, one hop
    #       per same-kind run. This is what the route planner emits.
    #   (b) 3-shard by kind: one shard per distinct kind (M, *, E),
    #       each running ALL its layers. This is the realistic
    #       disagg layout where we'd actually deploy.
    contiguous_hops = _partition_layers_by_kind(list(layer_type_list), layer_times_ms)
    kind_shards = _partition_by_layer_kind_3shard(list(layer_type_list), layer_times_ms)

    print_rank_0(f"\n[probe] {len(contiguous_hops)} contiguous-run hops on the route")
    print_rank_0(f"[probe] {len(kind_shards)} distinct layer kinds → {len(kind_shards)} shards")
    print_rank_0("\n[probe] Per-kind shard compute:")
    for s in sorted(kind_shards, key=lambda x: -x["compute_ms"]):
        print_rank_0(
            f"[probe]   shard kind={s['kind']!r}: "
            f"{len(s['layers'])} layers, "
            f"compute={s['compute_ms']:.2f} ms"
        )

    # Compute the gap under BOTH models.
    gap_contig = _compute_gap(contiguous_hops, args.probe_hop_latency_us)
    gap_kind = _compute_gap(kind_shards, args.probe_hop_latency_us)

    print_rank_0("\n" + "=" * 64)
    print_rank_0(
        f"[probe] assumed hop latency: {args.probe_hop_latency_us:.1f} us"
    )
    print_rank_0(
        f"[probe] total per-step compute (sum of layers): {gap_contig['total_compute_ms']:.2f} ms"
    )
    print_rank_0("-" * 64)
    print_rank_0("[probe] CONTIGUOUS-RUN HOP MODEL (one hop per same-kind run)")
    print_rank_0(f"[probe]   hops:          {gap_contig['n_hops']}")
    print_rank_0(f"[probe]   serial:        {gap_contig['serial_step_ms']:.2f} ms")
    print_rank_0(f"[probe]   pipelined:     {gap_contig['pipelined_step_ms']:.2f} ms")
    print_rank_0(
        f"[probe]   gap:           {gap_contig['serial_to_pipelined_gap']:.2f}x"
    )
    print_rank_0(f"[probe]   max/mean:      {gap_contig['shard_imbalance']:.2f}")
    print_rank_0("-" * 64)
    print_rank_0("[probe] KIND-BASED SHARD MODEL (one shard per layer kind)")
    print_rank_0(f"[probe]   shards:        {gap_kind['n_hops']}")
    print_rank_0(f"[probe]   serial:        {gap_kind['serial_step_ms']:.2f} ms")
    print_rank_0(f"[probe]   pipelined:     {gap_kind['pipelined_step_ms']:.2f} ms")
    print_rank_0(
        f"[probe]   gap:           {gap_kind['serial_to_pipelined_gap']:.2f}x"
    )
    print_rank_0(f"[probe]   max/mean:      {gap_kind['shard_imbalance']:.2f}")
    print_rank_0("=" * 64)

    # Use kind-based model for the headline verdict — that's the
    # realistic disagg layout.
    if gap_kind["serial_to_pipelined_gap"] > 1.5:
        print_rank_0(
            "[probe] VERDICT: pipelining gap is meaningful — implement it"
        )
    elif gap_kind["serial_to_pipelined_gap"] > 1.2:
        print_rank_0(
            "[probe] VERDICT: modest gap — implement if scheduler change is cheap"
        )
    else:
        print_rank_0(
            "[probe] VERDICT: gap small — existing async-send captures most of the win"
        )

    if (
        args.probe_output_json is not None
        and mpu.get_data_parallel_rank() == 0
        and mpu.get_tensor_model_parallel_rank() == 0
    ):
        import json
        with open(args.probe_output_json, "w") as f:
            json.dump(
                {
                    "layer_pattern": pattern,
                    "n_layers": n_layers,
                    "layer_times_ms": layer_times_ms,
                    "contiguous_hops": contiguous_hops,
                    "kind_shards": kind_shards,
                    "hop_latency_us": args.probe_hop_latency_us,
                    "gap_contiguous": gap_contig,
                    "gap_kind_based": gap_kind,
                },
                f,
                indent=2,
            )
        print_rank_0(f"[probe] wrote {args.probe_output_json}")


if __name__ == "__main__":
    main()
