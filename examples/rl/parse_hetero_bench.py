#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Parse iteration timings from `bench_hetero_inference.sh` runs.

Reads each experiment's ``test.log`` and reports a side-by-side table
of per-iteration tail metrics across the heterogeneous-inference
shard configurations submitted by the bench script.

The signal under partial rollouts + L+1 batches in flight: any slow
rollout in any of the L+1 batches stalls the next training step, so
the per-iteration tail (P95 / P99 / max) directly tracks the rollout
tail. The rollout-collection slice of each iteration is the pure
inference cost — broken out separately so it's clear how much of any
speedup came from the inference layer vs everything else.

Usage:
  python examples/rl/parse_hetero_bench.py <exp_name> [<exp_name> ...]

Or pass the run_paths.txt the orchestrator wrote:
  python examples/rl/parse_hetero_bench.py $(cat bench_hetero_runs/run_paths.txt)
"""
import argparse
import os
import re
import sys
from statistics import median


# Default location the bench script tells run_job.sh to write to.
# Matches BENCH_OUT_DIR/runs in bench_hetero_inference.sh. Override
# via ``--runs-base`` if your bench used a different BENCH_OUT_DIR
# or the runs landed in the cluster default
# (/lustre/fsw/.../megatron_rl/runs).
DEFAULT_RUNS_BASE = (
    "/lustre/fsw/portfolios/llmservice/users/wdykas/code/rl-hetero/bench_hetero_runs/runs"
)

ITER_RE = re.compile(
    r"iteration\s+(\d+)\D.*elapsed time per iteration \(ms\):\s*([\d.]+)"
)
# Section timings the trainer logs as "name ............: (min, max)".
SECTION_RE = re.compile(r"\s+(rl/\S+|forward-backward|optimizer)\s*\.+:\s*\(([\d.]+),\s*([\d.]+)\)")


def parse_log(log_path):
    """Return per-iteration elapsed times (ms) + per-section max-across-ranks lists."""
    iter_times = []
    sections: dict[str, list[float]] = {}
    if not os.path.exists(log_path):
        return iter_times, sections
    with open(log_path) as f:
        for line in f:
            m = ITER_RE.search(line)
            if m:
                iter_times.append(float(m.group(2)))
                continue
            m = SECTION_RE.search(line)
            if m:
                # Use max-across-ranks as the iteration's section cost.
                sections.setdefault(m.group(1), []).append(float(m.group(3)))
    return iter_times, sections


def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(int(len(s) * p), len(s) - 1)]


def fmt(name, body, label, ms_factor=1000.0):
    if len(body) < 2:
        return f"{name:36} {label:20} <too few iters>"
    return (
        f"{name:36} {label:20} "
        f"n={len(body):3d} "
        f"med={median(body) / ms_factor:7.2f}s "
        f"p95={percentile(body, 0.95) / ms_factor:7.2f}s "
        f"p99={percentile(body, 0.99) / ms_factor:7.2f}s "
        f"max={max(body) / ms_factor:7.2f}s "
        f"tail={max(body) / max(median(body), 1e-9):4.2f}x"
    )


def summarize(name, log_path):
    iter_times, sections = parse_log(log_path)
    if not iter_times:
        print(f"{name}: no iteration times found in {log_path}")
        return
    # Drop iter 1 (cold start: coord boot, cuda graphs, weight refit etc.).
    iters = iter_times[1:]
    rollout = sections.get("rl/rollout-collection", [])[1:]
    print(fmt(name, iters, "all iter time"))
    if rollout:
        print(fmt(name, rollout, "  rl/rollout-collect"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_names", nargs="+")
    parser.add_argument(
        "--runs-base",
        default=DEFAULT_RUNS_BASE,
        help=f"Base path runs live under (default: {DEFAULT_RUNS_BASE})",
    )
    args = parser.parse_args()

    print(
        f"{'config':36} {'metric':20} {'count':>5} {'median':>10} {'p95':>10} "
        f"{'p99':>10} {'max':>10} {'tail':>5}"
    )
    print("-" * 120)
    for name in args.exp_names:
        log_path = os.path.join(args.runs_base, name, "logs", "test.log")
        summarize(name, log_path)


if __name__ == "__main__":
    main()
