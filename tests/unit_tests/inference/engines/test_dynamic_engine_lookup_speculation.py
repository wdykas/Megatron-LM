# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end tests for lookup ("n-gram") speculative decoding in the dynamic engine.

Each test runs the same requests through a plain engine (no speculation) and a lookup engine
(`speculative_method="lookup"`) that share one model, and checks that the outputs agree token
for token, while the lookup engine commits more than one token per step.
"""

import types
from typing import Dict, List, Optional
from unittest import mock

import msgpack
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.inference.engines.test_dynamic_engine import reset_rounder, set_rounder
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

VOCAB = 128
SEED = 123

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    ),
]


def _build_model(max_sequence_length: int) -> GPTModel:
    clear_nvte_env_vars()
    torch.manual_seed(SEED)
    model_parallel_cuda_manual_seed(
        seed=SEED, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    config = TransformerConfig(
        params_dtype=torch.bfloat16,
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        cuda_graph_impl="none",
        inference_rng_tracker=True,
        pipeline_dtype=torch.bfloat16,
        add_bias_linear=True,
        inference_sampling_seed=SEED,
        flash_attention_version=2,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCAB,
        max_sequence_length=max_sequence_length,
        parallel_output=True,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
        position_embedding_type="rope",
    ).cuda()
    for param in model.parameters():
        param.data = param.data.to(torch.bfloat16)
    model.eval()
    return model


def _build_engine(
    model: GPTModel,
    *,
    num_speculative_tokens: int,
    max_sequence_length: int,
    buffer_size_gb: float = 0.05,
    paused_buffer_size_gb: Optional[float] = None,
    block_size_tokens: int = 256,
    materialize_only_last_token_logits: bool = True,
) -> DynamicInferenceEngine:
    context = DynamicInferenceContext(
        model_config=model.config,
        inference_config=InferenceConfig(
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            paused_buffer_size_gb=(
                0.2 * buffer_size_gb if paused_buffer_size_gb is None else paused_buffer_size_gb
            ),
            block_size_tokens=block_size_tokens,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            unified_memory_level=0,
            num_speculative_tokens=num_speculative_tokens,
            speculative_method="lookup",
            lookup_ngram_max=4,
            lookup_ngram_min=1,
            sampling_backend="torch",
            async_sched_mode=AsyncScheduleMode.LEGACY,
        ),
    )
    wrapper = GPTInferenceWrapper(model, context)
    wrapper.model_is_pipeline_parallel = False
    tokenizer = types.SimpleNamespace(
        vocab_size=VOCAB,
        bos=None,
        detokenize=lambda tokens, **_: " ".join(str(t) for t in tokens),
        tokenize=lambda text: [int(t) for t in text.split()],
    )
    controller = TextGenerationController(inference_wrapped_model=wrapper, tokenizer=tokenizer)
    delete_cuda_graphs()
    return DynamicInferenceEngine(controller, context)


def _install_table_logits(engine: DynamicInferenceEngine, table: torch.Tensor):
    """Replace the model's logits with `table[input_token]` (a pure function of the input).

    Such a model gives bitwise identical logits however a step batches its tokens, so plain
    and speculative runs must agree exactly, including under sampling.
    """
    model = engine.controller.inference_wrapped_model.model
    context = engine.context
    original = model.forward

    def forward(*args, **kwargs):
        logits = original(*args, **kwargs)
        tokens = kwargs.get("tokens", kwargs.get("input_ids", args[0] if args else None))
        if logits.shape[1] != tokens.shape[1]:
            tokens = tokens[:, context.active_logit_idxs[: logits.shape[1]].long()]
        logits.copy_(table[tokens].to(logits.dtype))
        return logits

    model.forward = forward


def _cycle_table(modulus: int, bonus: float = 6.0) -> torch.Tensor:
    """Logits whose argmax after token t is (t + 1) % modulus, with a non-trivial tail."""
    generator = torch.Generator().manual_seed(7)
    table = torch.randn(VOCAB, VOCAB, generator=generator)
    table[torch.arange(VOCAB), (torch.arange(VOCAB) + 1) % modulus] += bonus
    return table.cuda()


def _run(
    engine: DynamicInferenceEngine,
    requests: List[DynamicInferenceRequest],
    stream: bool = False,
    max_steps: int = 2000,
):
    """Drive `step_modern` until every request finishes. Returns outputs and stream deltas."""
    for request in requests:
        engine._add_request(request)
    finished: Dict[int, DynamicInferenceRequest] = {}
    streamed: Dict[int, List[List[int]]] = {}
    if stream:
        engine.socket_for_receiving_requests = mock.Mock()
    for _ in range(max_steps):
        if not engine.has_unfinished_requests():
            break
        result = engine.step_modern()
        for request in result["finished_requests"]:
            finished[request.request_id] = request
        if stream:
            socket = engine.socket_for_receiving_requests
            engine._try_send_streaming_partials()
            for call in socket.send_multipart.call_args_list:
                for frame in call.args[0][1:]:
                    partial = msgpack.unpackb(frame, raw=False)
                    streamed.setdefault(partial["request_id"], []).append(partial)
            socket.send_multipart.reset_mock()
    assert not engine.has_unfinished_requests()
    return finished, streamed


def _make_requests(prompts, num_tokens, references=None, **sampling_kwargs):
    requests = []
    for request_id, prompt in enumerate(prompts):
        params = SamplingParams(
            num_tokens_to_generate=(
                num_tokens if isinstance(num_tokens, int) else num_tokens[request_id]
            ),
            lookup_references=None if references is None else references[request_id],
            **sampling_kwargs,
        )
        requests.append(
            DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=torch.tensor(prompt, dtype=torch.int64, device="cuda"),
                sampling_params=params,
            )
        )
    return requests


def _random_prompts(num, low, high, seed=0):
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(low, high + 1, (num,), generator=generator).tolist()
    return [torch.randint(0, VOCAB, (n,), generator=generator).tolist() for n in lengths]


def _tokens_per_step(requests: Dict[int, DynamicInferenceRequest], request_ids) -> float:
    lengths = [n for rid in request_ids for n in requests[rid].acceptance_step_lengths]
    return sum(lengths) / len(lengths)


class TestDynamicEngineLookupSpeculation:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        reset_rounder()
        Utils.destroy_model_parallel()

    def setup_method(self, method):
        set_rounder(4)

    def _plain_and_lookup(self, model, k, max_sequence_length, **engine_kwargs):
        # Each engine gets its own copy of the same weights: a model module is bound to one
        # inference context (and the logit overrides below wrap one engine's context).
        twin = _build_model(max_sequence_length)
        twin.load_state_dict(model.state_dict())
        plain = _build_engine(
            model,
            num_speculative_tokens=0,
            max_sequence_length=max_sequence_length,
            **engine_kwargs,
        )
        lookup = _build_engine(
            twin, num_speculative_tokens=k, max_sequence_length=max_sequence_length, **engine_kwargs
        )
        return plain, lookup

    @torch.inference_mode()
    @pytest.mark.parametrize("sampling", ["greedy", "seeded"])
    def test_real_model_matches_plain_decoding(self, sampling):
        """Tiny random GPT: lookup output == plain output; references give >1 token/step."""
        k, num_tokens, max_len = 4, 48, 128
        model = _build_model(max_len)
        plain, lookup = self._plain_and_lookup(model, k, max_len)
        prompts = _random_prompts(8, 6, 24)
        sampling_kwargs = dict(termination_id=-1)
        if sampling == "greedy":
            sampling_kwargs["top_k"] = 1
        else:
            sampling_kwargs["temperature"] = 1.0

        def build(references=None):
            requests = _make_requests(prompts, num_tokens, references, **sampling_kwargs)
            if sampling == "seeded":
                for request in requests:
                    request.sampling_params.seed = 1000 + request.request_id
            return requests

        expected, _ = _run(plain, build())
        # Half the requests get the plain continuation as a reference (as when re-decoding a
        # rejected draft); the other half rely on their own context only.
        references = [
            [prompts[i] + expected[i].generated_tokens] if i % 2 == 0 else None
            for i in range(len(prompts))
        ]
        actual, _ = _run(lookup, build(references))
        for i in range(len(prompts)):
            assert actual[i].status == Status.COMPLETED
            assert actual[i].generated_tokens == expected[i].generated_tokens, i
        with_refs = _tokens_per_step(actual, range(0, len(prompts), 2))
        assert with_refs > 3.0, with_refs
        proposer = lookup.controller._lookup_proposer
        assert proposer.num_requests == 0  # finished requests release their index

    @torch.inference_mode()
    @pytest.mark.parametrize("materialize_only_last_token_logits", [True, False])
    def test_sampling_logprobs_and_streaming_match_plain(self, materialize_only_last_token_logits):
        """Seeded temperature sampling with log probs and streaming, on an exact table model."""
        k, max_len = 3, 160
        model = _build_model(max_len)
        plain, lookup = self._plain_and_lookup(
            model, k, max_len, materialize_only_last_token_logits=materialize_only_last_token_logits
        )
        table = _cycle_table(modulus=9, bonus=4.0)
        _install_table_logits(plain, table)
        _install_table_logits(lookup, table)
        prompts = _random_prompts(6, 4, 20, seed=1)
        num_tokens = [40, 41, 42, 43, 44, 45]

        def build(references=None):
            requests = _make_requests(
                prompts,
                num_tokens,
                references,
                temperature=0.8,
                termination_id=-1,
                return_log_probs=True,
                skip_prompt_log_probs=materialize_only_last_token_logits,
                streaming=True,
            )
            for request in requests:
                request.sampling_params.seed = 77 + request.request_id
            return requests

        expected, _ = _run(plain, build())
        # Sampled text rarely repeats itself; reference the plain run's output as a stand-in
        # for a previously drafted continuation, so most steps commit several tokens.
        references = [[prompts[i] + expected[i].generated_tokens] for i in range(len(prompts))]
        actual, streamed = _run(lookup, build(references), stream=True)
        for i in range(len(prompts)):
            assert actual[i].generated_tokens == expected[i].generated_tokens, i
            assert len(actual[i].generated_tokens) == num_tokens[i]
            torch.testing.assert_close(
                torch.tensor(actual[i].generated_log_probs),
                torch.tensor(expected[i].generated_log_probs),
                rtol=0,
                atol=1e-5,
            )
            assert len(actual[i].generated_log_probs) == num_tokens[i]
            if not materialize_only_last_token_logits:
                torch.testing.assert_close(
                    torch.tensor(actual[i].prompt_log_probs),
                    torch.tensor(expected[i].prompt_log_probs),
                    rtol=0,
                    atol=1e-5,
                )
            # Streamed deltas are a prefix of the final output, with matching log probs.
            tokens = [t for p in streamed[i] for t in p["new_tokens"]]
            log_probs = [lp for p in streamed[i] for lp in p["new_log_probs"]]
            assert tokens == actual[i].generated_tokens[: len(tokens)]
            assert log_probs == actual[i].generated_log_probs[: len(log_probs)]
            assert max(len(p["new_tokens"]) for p in streamed[i]) > 1
        assert _tokens_per_step(actual, range(len(prompts))) > 2.0

    @torch.inference_mode()
    def test_stop_conditions_with_multi_token_commits(self):
        """Max tokens, EOS, stop token ids and stop words cut multi-token commits exactly."""
        k, max_len = 4, 128
        model = _build_model(max_len)
        plain, lookup = self._plain_and_lookup(model, k, max_len)
        modulus = 11
        table = _cycle_table(modulus=modulus)
        _install_table_logits(plain, table)
        _install_table_logits(lookup, table)
        # Greedy output cycles 0..10, so after one period the own context predicts perfectly.
        prompts = [[0, 1, 2, 3], list(range(modulus)) + [5, 6], [7, 8, 9], [1, 2], [4, 5, 6]]

        def build():
            requests = _make_requests(prompts, 37, top_k=1, termination_id=-1)
            requests[0].sampling_params.num_tokens_to_generate = 23  # not a multiple of k + 1
            requests[1].sampling_params.termination_id = 4  # EOS inside a speculative block
            requests[2].sampling_params.stop_token_ids = [3]
            requests[3].sampling_params.stop_words = ["8 9 10"]
            requests[3].sampling_params.detokenize_stop_sequence = True
            requests[4].sampling_params.stop_words = ["2 3"]
            return requests

        expected, _ = _run(plain, build())
        actual, _ = _run(lookup, build())
        for i in range(len(prompts)):
            assert actual[i].generated_tokens == expected[i].generated_tokens, i
            assert actual[i].finish_reason == expected[i].finish_reason, i
        assert len(actual[0].generated_tokens) == 23
        assert actual[1].generated_tokens[-1] == 4
        assert actual[2].generated_tokens[-1] == 3
        assert expected[3].generated_tokens[-3:] == [8, 9, 10]
        assert actual[1].generated_tokens == [7, 8, 9, 10, 0, 1, 2, 3, 4]
        # Request 1's prompt already holds a full period, so its EOS lands inside a
        # multi-token commit: [7] at prefill, [8, 9, 10, 0], then [1, 2, 3, 4].
        assert actual[1].acceptance_step_lengths == [1, 4, 4]
        # The first period (11 tokens) has nothing to look up; after it, K + 1 tokens per step.
        assert _tokens_per_step(actual, [0]) > 2.0

    @torch.inference_mode()
    def test_eviction_and_resume(self):
        """A tiny KV pool forces pauses and evictions mid-generation; outputs still match."""
        k, max_len = 3, 1024
        model = _build_model(max_len)
        kwargs = dict(buffer_size_gb=0.00064, paused_buffer_size_gb=0.0, block_size_tokens=256)
        plain, lookup = self._plain_and_lookup(model, k, max_len, **kwargs)
        table = _cycle_table(modulus=13)
        _install_table_logits(plain, table)
        _install_table_logits(lookup, table)
        prompts = _random_prompts(3, 256, 256, seed=3)
        expected, _ = _run(plain, _make_requests(prompts, 400, top_k=1, termination_id=-1))
        actual, _ = _run(lookup, _make_requests(prompts, 400, top_k=1, termination_id=-1))
        assert lookup.evicted_request_count > 0
        for i in range(len(prompts)):
            assert actual[i].generated_tokens == expected[i].generated_tokens, i
        assert _tokens_per_step(actual, range(len(prompts))) > 3.0

    def test_config_rejects_async_scheduling_and_bad_ngrams(self):
        with pytest.raises(ValueError, match="LEGACY"):
            InferenceConfig(num_speculative_tokens=2, speculative_method="lookup")
        with pytest.raises(ValueError, match="lookup_ngram_min"):
            InferenceConfig(
                num_speculative_tokens=2,
                speculative_method="lookup",
                async_sched_mode=AsyncScheduleMode.LEGACY,
                lookup_ngram_min=3,
                lookup_ngram_max=2,
            )
        with pytest.raises(ValueError, match="speculative_method"):
            InferenceConfig(speculative_method="eagle")
        with pytest.raises(ValueError, match="lookup_references"):
            SamplingParams(lookup_references=[[1, -2]])
        with pytest.raises(ValueError, match="lookup_references"):
            SamplingParams(lookup_references=[1, 2])
        params = SamplingParams(lookup_references=[(1, 2), [3]])
        assert params.lookup_references == [[1, 2], [3]]
        assert SamplingParams.deserialize(params.serialize()).lookup_references == [[1, 2], [3]]
