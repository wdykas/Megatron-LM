# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Disaggregated prefill->decode dynamic inference driver.

Mirrors ``gpt_dynamic_inference.py`` but splits the data-parallel
replicas into roles: replica 0 = prefill, replica 1 = decode. A
request's prompt is prefilled on the prefill replica; its KV cache is
handed to the decode replica via :class:`DisaggCoordinator`
(layout handshake -> route -> reshard -> transport -> import); the
decode replica then admits the request -- the imported KV registers in
the prefix cache, so ``add_request(prompt)`` matches the whole prompt
and generation continues without re-prefilling. Output is written in
the same JSON format ``gpt_dynamic_inference.py`` produces, so the
disaggregated run can be validated against the colocated golden values
(disagg should be token-identical to colocated under matched layouts).

Run with data-parallel size 2 (``--num-tokens-to-generate`` etc. as in
the colocated test). Matched prefill/decode TP/PP is supported here;
heterogeneous TP/PP also needs the inference process-group construction
(separate MR) and raises until then.

NOTE: this driver is exercised by the functional test harness with a
real checkpoint; the coordinator/transport/reshard layers it builds on
are unit- and multi-process-tested independently.
"""

import json
import os
import sys

import torch

from megatron.training.arguments import parse_and_validate_args

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from examples.inference.gpt.utils import build_requests, get_curr_time
from megatron.core import parallel_state as ps
from megatron.core.inference.disagg_coordinator import DisaggCoordinator
from megatron.core.inference.kv_shard_layout import KVShardLayout, is_matched
from megatron.core.inference.kv_transport_backend import NcclTransportBackend
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.training import get_args, initialize_megatron


def _add_disagg_args(base_provider):
    def provider(parser):
        parser = base_provider(parser)
        group = parser.add_argument_group(title="Disaggregation")
        group.add_argument(
            "--disagg-decode-replica-id",
            type=str,
            default="decode0",
            help="Replica id assigned to the decode data-parallel replica.",
        )
        return parser

    return provider


def _layout_for_this_rank() -> KVShardLayout:
    """Build this rank's KV shard layout from parallel_state. num_heads
    uses the model's KV-head count (GQA-aware via num_query_groups when
    set, else attention heads)."""
    args = get_args()
    num_heads = getattr(args, "num_query_groups", None) or args.num_attention_heads
    return KVShardLayout(
        num_layers=args.num_layers,
        num_heads=num_heads,
        tp_size=ps.get_tensor_model_parallel_world_size(),
        tp_rank=ps.get_tensor_model_parallel_rank(),
        pp_size=ps.get_pipeline_model_parallel_world_size(),
        pp_rank=ps.get_pipeline_model_parallel_rank(),
        global_rank=torch.distributed.get_rank(),
    )


@torch.inference_mode()
def main():
    args = parse_and_validate_args(
        extra_args_provider=_add_disagg_args(add_inference_args),
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    initialize_megatron()

    dp_size = ps.get_data_parallel_world_size()
    dp_rank = ps.get_data_parallel_rank()
    if dp_size != 2:
        raise RuntimeError(
            f"gpt_disagg_inference expects data-parallel size 2 "
            f"(replica 0=prefill, replica 1=decode); got dp_size={dp_size}."
        )
    role = "prefill" if dp_rank == 0 else "decode"
    replica_id = "prefill" if role == "prefill" else args.disagg_decode_replica_id

    tokenizer = build_tokenizer(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_log_probs=args.return_log_probs,
        num_tokens_to_generate=args.num_tokens_to_generate,
        termination_id=args.termination_id if args.termination_id is not None else tokenizer.eod,
    )

    engine = get_dynamic_inference_engine()
    layout = _layout_for_this_rank()

    backend = NcclTransportBackend()
    backend.init()
    coord = DisaggCoordinator(
        role=role, replica_id=replica_id, my_layout=layout, backend=backend
    )
    coord.handshake()

    # Hetero TP/PP needs the inference process-group construction (separate
    # MR). Until then, require matched prefill/decode layouts.
    for tgt in coord.decode_targets:
        for dl in tgt.layouts:
            if not is_matched(layout, dl):
                raise NotImplementedError(
                    "gpt_disagg_inference currently supports matched prefill/decode "
                    "TP/PP only; heterogeneous layouts require the inference "
                    "process-group construction (separate MR)."
                )

    requests = build_requests(args, tokenizer, sampling_params)

    if role == "prefill":
        _run_prefill(engine, coord, requests)
    else:
        results = _run_decode(engine, coord, requests, sampling_params)
        if torch.distributed.get_rank() == ps.get_data_parallel_src_rank() and args.output_path:
            _write_output(args, results)


def _run_prefill(engine, coord, requests):
    """Prefill each request's prompt, then hand its KV to the decode
    replica. Generation continues on the decode side."""
    for i, req in enumerate(requests):
        engine.add_request(i, req.prompt_text, req.sampling_params)
    # Drive the engine until every request has its prompt KV populated
    # (one decode step suffices for non-chunked prefill); then hand off.
    handed = set()
    while len(handed) < len(requests):
        engine.step_modern()
        for i, req in enumerate(requests):
            if i in handed:
                continue
            # Once the request's KV blocks exist, export + route + send.
            ho = coord.prefill_handoff(engine, i, req.prompt_tokens)
            if ho is not None:
                ho.wait()
            handed.add(i)


def _run_decode(engine, coord, requests, sampling_params):
    """Receive each request's KV, admit it (prefix-cache match skips
    re-prefill), and generate. Returns per-request outputs."""
    n = len(requests)
    for _ in range(n):
        request_id, _imported = coord.decode_intake(engine)
        req = requests[request_id]
        engine.add_request(request_id, req.prompt_text, req.sampling_params)

    finished = {}
    while len(finished) < n:
        result = engine.step_modern()
        for rec in result.get("finished_request_records", []):
            merged = rec.merge()
            finished[merged.request_id] = merged
    return finished


def _write_output(args, finished):
    """Match gpt_dynamic_inference.py's output JSON so the disaggregated
    run validates against the colocated golden values."""
    out = {}
    for rid, req in sorted(finished.items()):
        d = {
            "input_prompt": req.prompt,
            "generated_text": req.generated_text,
            "generated_tokens": (
                req.generated_tokens.tolist()
                if torch.is_tensor(req.generated_tokens)
                else req.generated_tokens
            ),
        }
        if getattr(req.sampling_params, "return_log_probs", False):
            d["logprobs"] = getattr(req, "generated_log_probs", None)
        out[str(rid)] = d
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
