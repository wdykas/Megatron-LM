# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import argparse
import asyncio

import torch

from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    start_text_gen_server,
    stop_text_gen_server,
)
from megatron.core.utils import trace_async_exceptions
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron


def add_text_generation_server_args(parser: argparse.ArgumentParser):
    """Adds the required command line arguments for running the text generation server."""
    parser = add_modelopt_args(parser)
    parser = add_inference_args(parser)
    parser.add_argument("--port", type=int, default=5000, help="Port for Flask server to run on")
    parser.add_argument(
        "--parsers", type=str, nargs="+", default=[], help="Parsers to use for parsing the response"
    )
    parser.add_argument(
        "--kv-compaction-strategy", type=str, default=None,
        help="Live post-prefill KV compaction strategy (snapkv, streaming_llm, "
             "belief_still, learned_oracle). snapkv requires "
             "--decode-only-cuda-graphs so prefill forwards run eagerly; "
             "learned_oracle is query-free and has no such requirement.",
    )
    parser.add_argument(
        "--kv-compaction-budget-ratio", type=float, default=0.5,
        help="Fraction of prompt tokens to keep when --kv-compaction-strategy is set.",
    )
    parser.add_argument(
        "--kv-compaction-obs-window", type=int, default=32,
        help="SnapKV observation window (last W prompt queries score the rest).",
    )
    parser.add_argument(
        "--kv-compaction-min-tokens", type=int, default=128,
        help="Skip live compaction for prompts shorter than this.",
    )
    parser.add_argument(
        "--kv-compaction-max-retrievals", type=int, default=4,
        help="Archive retrievals allowed per request (cap on trigger fires).",
    )
    parser.add_argument(
        "--kv-compaction-score-weighting", choices=["none", "value_norm"],
        default="none",
        help="Weight snapkv selection scores by per-token value norms "
             "(VATP-style attention x ||v|| output contribution).",
    )
    parser.add_argument(
        "--kv-compaction-compactor-checkpoint", type=str, default=None,
        help="Trained compactor checkpoint for --kv-compaction-strategy belief_still.",
    )
    parser.add_argument(
        "--kv-compaction-oracle-checkpoint", type=str, default=None,
        help="Trained heavy-hitter scorer (save_oracle_scorer output) for "
             "--kv-compaction-strategy learned_oracle.",
    )
    parser.add_argument(
        "--kv-compaction-n-compress", type=int, default=64,
        help="Synthetic memory slots for belief_still.",
    )
    parser.add_argument(
        "--kv-compaction-belief-keep-recent", type=int, default=64,
        help="belief_still: keep this many raw recent prompt tokens after the "
             "memory (the question region — the training format).",
    )
    parser.add_argument(
        "--kv-compaction-archive", action="store_true", default=False,
        help="Demote evicted KV spans to a CPU archive and restore them on demand "
             "via the negative-cache trigger. Needs fully eager decoding "
             "(--cuda-graph-impl none).",
    )
    parser.add_argument(
        "--kv-compaction-retrieval-alpha", type=float, default=0.2,
        help="Archive trigger fast path: fire a span whose centroid "
             "attention-mass fraction reaches this (scale-free, in (0,1)).")
    parser.add_argument(
        "--kv-compaction-retrieval-cusum", type=float, default=0.4,
        help="Archive trigger CUSUM threshold h: fire when a span's "
             "accumulated (alpha - own EMA baseline - drift) crosses h — "
             "novel persistent reaches fire, chronically hot spans don't.")
    parser.add_argument(
        "--kv-compaction-flywheel-dir", type=str, default=None,
        help="Log restored/unused span labels per finished request (the "
             "retrieval flywheel's self-labeling training data).",
    )
    parser.add_argument(
        "--kv-compaction-archive-transfer", type=str, default="pinned",
        choices=["pinned", "nixl"],
        help="Span-store backend: pinned host memory (on-node default) or "
             "NIXL for a remote/disaggregated archive tier.",
    )
    parser.add_argument(
        "--kv-compaction-rope-mode", type=str, default=None,
        choices=["logical", "renumber"],
        help="Required for RoPE models. 'logical': keep original positions of "
             "record (stored rotations stay exact; decode queries get their "
             "original sequence positions) — the exact/measurement setting. "
             "'renumber': contiguous cache positions with key re-rotation "
             "(StreamingLLM semantics, positions bounded by cache size).",
    )
    return parser


@trace_async_exceptions
async def run_text_generation_server(
    engine: DynamicInferenceEngine, coordinator_port: int, server_port: int
):
    """
    Runs the text generation server from rank 0 and initializes the
    DynamicInferenceEngine on all ranks.

    Args:
        engine (DynamicInferenceEngine): The dynamic inference engine.
        coordinator_port (int): The network port for the dynamic inference DP coordinator.
        server_port (int): The network for port the frontend text generation server.
    """

    rank = torch.distributed.get_rank()

    coordinator_addr = await engine.start_listening_to_data_parallel_coordinator(
        inference_coordinator_port=coordinator_port, launch_inference_coordinator=True
    )

    try:
        if rank == 0:
            start_text_gen_server(
                coordinator_addr=coordinator_addr,
                tokenizer=engine.controller.tokenizer,
                parsers=args.parsers,
                rank=rank,
                server_port=server_port,
                verbose=args.inference_text_gen_server_logging,
            )

        # Await the engine loop directly since the server is running in a separate process
        await engine.engine_loop_task

    finally:
        # Guarantee that the separate process is terminated when the engine loop stops or is interrupted
        if rank == 0:
            stop_text_gen_server()


if __name__ == "__main__":
    with torch.inference_mode():
        initialize_megatron(
            extra_args_provider=add_text_generation_server_args,
            args_defaults={'no_load_rng': True, 'no_load_optim': True},
        )

        # Enable return_log_probs to allow prompt logprobs computation for echo=True requests
        # This sets materialize_only_last_token_logits=False in the inference context,
        # which is required for lm-eval compatibility (loglikelihood evaluation tasks)
        args = get_args()
        args.return_log_probs = True

        # Archive mode must install its attention wrapper BEFORE the decode
        # CUDA graphs are captured (the trigger's Q copy_ has to be baked
        # into the graphs), and a delete+re-capture doubles graph memory
        # (the first capture's pools are not reclaimed -> OOM). Defer the
        # engine's init-time capture and run it exactly once, after install.
        defer_capture = (args.kv_compaction_strategy is not None
                         and args.kv_compaction_archive)
        if defer_capture:
            from megatron.core.inference.engines.dynamic_engine import (
                DynamicInferenceEngine,
            )
            _orig_create = DynamicInferenceEngine.create_cuda_graphs
            DynamicInferenceEngine.create_cuda_graphs = (
                lambda self, reset_context=True: None)
            try:
                engine = get_dynamic_inference_engine()
            finally:
                DynamicInferenceEngine.create_cuda_graphs = _orig_create
        else:
            engine = get_dynamic_inference_engine()

        if args.kv_compaction_strategy is not None:
            from megatron.rl.compaction.kv.serving.live import LiveKVCompactor
            if not args.decode_only_cuda_graphs and args.kv_compaction_strategy == "snapkv":
                raise ValueError(
                    "--kv-compaction-strategy snapkv needs eager prefill forwards for "
                    "observation-window Q capture: relaunch with --decode-only-cuda-graphs."
                )
            engine.kv_compactor = LiveKVCompactor(
                engine,
                strategy=args.kv_compaction_strategy,
                budget_ratio=args.kv_compaction_budget_ratio,
                obs_window=args.kv_compaction_obs_window,
                min_tokens=args.kv_compaction_min_tokens,
                compactor_checkpoint=args.kv_compaction_compactor_checkpoint,
                oracle_checkpoint=args.kv_compaction_oracle_checkpoint,
                n_compress=args.kv_compaction_n_compress,
                belief_keep_recent=args.kv_compaction_belief_keep_recent,
                archive=args.kv_compaction_archive,
                retrieval_alpha=args.kv_compaction_retrieval_alpha,
                retrieval_cusum=args.kv_compaction_retrieval_cusum,
                rope_mode=args.kv_compaction_rope_mode,
                flywheel_dir=args.kv_compaction_flywheel_dir,
                archive_transfer=args.kv_compaction_archive_transfer,
                score_weighting=args.kv_compaction_score_weighting,
                max_retrievals_per_request=args.kv_compaction_max_retrievals,
            )
            print(f"[kv-compaction] live compaction enabled: "
                  f"{args.kv_compaction_strategy} @ {args.kv_compaction_budget_ratio}")
            if defer_capture:
                print("[kv-compaction] capturing decode CUDA graphs with the "
                      "archive Q hook baked in ...", flush=True)
                engine.create_cuda_graphs()

        try:
            asyncio.run(
                run_text_generation_server(engine, args.inference_coordinator_port, args.port)
            )
        except KeyboardInterrupt:
            # Catching at the top level ensures clean stdout without spamming the traceback
            print("Server process interrupted by user.")
        finally:
            # Clean up PyTorch distributed groups properly
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
