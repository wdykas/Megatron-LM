# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregated prefill->decode smoke for ``launch_inference_server.py``.

Same high-level-API server as ``gpt_inference_server_smoke``, but launched with
``--inference-shards "tp=1,role=prefill+tp=1,role=decode"`` so the 2-rank job is
split into one prefill shard (rank 0) and one decode shard (rank 1). The HTTP
frontend runs on the primary rank and submits to the shared coordinator, which
2-hop routes each request (prefill -> KV handoff -> decode). We tail stdout for
the readiness banner, send one OpenAI-compatible ``/v1/completions`` request,
and assert a 200 with non-empty ``choices[0].text``.

No golden values: a pass/fail HTTP smoke of the disaggregated control plane
(REGISTER_ROLE, 2-hop routing, KV handoff over NCCL, request round-trip).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

READINESS_MARKER = "Running on http"
READINESS_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 60
SHUTDOWN_TIMEOUT_S = 60
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000

# One prefill shard + one decode shard, TP=1 each -> 2 ranks.
INFERENCE_SHARDS = "tp=1,role=prefill+tp=1,role=decode"


def build_server_cmd(
    checkpoint_dir: str, tokenizer_model: str, server_log_dir: str = None
) -> list[str]:
    """torchrun command for ``launch_inference_server.py`` (Mistral 0.5B, 2 ranks
    split prefill/decode). Mirrors gpt_inference_server_smoke's model args; the
    only difference is ``--nproc-per-node=2`` + ``--inference-shards``."""
    log_args = ["--log-dir", server_log_dir, "--tee", "3"] if server_log_dir else []
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *log_args,
        "--nproc-per-node=2",
        "-m",
        "examples.inference.launch_inference_server",
        "--inference-shards",
        INFERENCE_SHARDS,
        "--tiktoken-pattern",
        "v2",
        "--use-mcore-models",
        "--tokenizer-type",
        "TikTokenizer",
        "--tokenizer-model",
        tokenizer_model,
        "--auto-detect-ckpt-format",
        "--max-tokens-to-oom",
        "3600000",
        "--inference-max-seq-length",
        "4096",
        "--attention-backend",
        "flash",
        "--use-checkpoint-args",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-use-tokenizer-model-from-checkpoint-args",
        "--load",
        checkpoint_dir,
        "--distributed-backend",
        "nccl",
        "--transformer-impl",
        "inference_optimized",
        "--tensor-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        "1",
        "--deterministic-mode",
        "--ckpt-format",
        "torch_dist",
        "--bf16",
        "--num-layers",
        "24",
        "--hidden-size",
        "1152",
        "--num-attention-heads",
        "16",
        "--max-position-embeddings",
        "1024",
        "--seq-length",
        "1024",
        "--inference-dynamic-batching-buffer-size-gb",
        "20",
        "--dist-ckpt-strictness",
        "log_unexpected",
        "--inference-ckpt-non-strict",
        "--port",
        str(SERVER_PORT),
        "--host",
        SERVER_HOST,
    ]


def cleaned_env() -> dict:
    """Strip torchrun-specific env vars so the spawned server starts a fresh
    distributed setup instead of inheriting a stale one."""
    env = os.environ.copy()
    for v in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ):
        env.pop(v, None)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env["NCCL_ALGO"] = "Ring"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


def post_completion() -> dict:
    body = json.dumps(
        {"model": "EMPTY", "prompt": "Hello, world!", "max_tokens": 10, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
        if resp.status != 200:
            raise AssertionError(f"server returned status {resp.status}")
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument(
        "--server-log-dir",
        default=None,
        help="torchrun --log-dir for the spawned server; CI passes the JET assets "
        "dir so per-rank logs land where the harness expects them.",
    )
    args = parser.parse_args()

    cmd = build_server_cmd(args.checkpoint_dir, args.tokenizer_model, args.server_log_dir)
    print(f"[disagg-smoke] spawning server: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=cleaned_env(),
    )

    ready = threading.Event()

    def watch():
        for line in proc.stdout:
            print(f"[server] {line}", end="", flush=True)
            if READINESS_MARKER in line:
                ready.set()

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    rc = 1
    try:
        if not ready.wait(READINESS_TIMEOUT_S):
            print(
                f"[disagg-smoke] FAIL: readiness banner not seen in {READINESS_TIMEOUT_S}s",
                flush=True,
            )
            return rc

        # Allow a beat after the readiness banner for the frontend replicas to
        # be reachable.
        time.sleep(2)

        print("[disagg-smoke] sending /v1/completions request", flush=True)
        body = post_completion()
        choices = body.get("choices") or []
        if not choices:
            print(f"[disagg-smoke] FAIL: no choices in response: {body}", flush=True)
            return rc
        text = choices[0].get("text", "")
        if not text:
            print(f"[disagg-smoke] FAIL: empty completion text: {body}", flush=True)
            return rc

        print(f"[disagg-smoke] PASS: completion={text!r}", flush=True)
        rc = 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print("[disagg-smoke] server didn't exit on SIGTERM; SIGKILL", flush=True)
                proc.kill()
                proc.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
