# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import random
import types

import pytest
import torch
import torch.nn.functional as F
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig
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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version, is_te_min_version
from tests.unit_tests.test_utilities import Utils

try:
    from flashinfer import mm_mxfp8 as _mm_mxfp8  # noqa: F401
    from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor

    HAVE_MXFP8 = True
except ImportError:
    HAVE_MXFP8 = False


RANDOM_SEED = 42
VOCAB_SIZE = 100

# NeMo-RL threshold: token_mult_prob_error must be below 1.05 for stable RL.
TOKEN_MULT_PROB_ERROR_THRESHOLD = 1.05


def _set_rounder(value):
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


def _compute_token_mult_prob_error(inference_lps, training_lps):
    """Compute the token_mult_prob_error metric (NeMo-RL definition).

    token_mult_prob_error = mean(exp(|inference_logprobs - training_logprobs|))

    A value of 1.0 means the log probs are identical.  Values > 1.0 indicate
    divergence; the RL training loop considers > 1.05 problematic.
    """
    inf_t = torch.tensor(inference_lps, dtype=torch.float64)
    train_t = torch.tensor(training_lps, dtype=torch.float64)
    lp_error = (inf_t - train_t).abs()
    return torch.exp(lp_error).mean().item()


def _skip_if_mxfp8_unsupported():
    """Skip the current test if MXFP8 requirements are not met."""
    if not HAVE_MXFP8:
        pytest.skip("FlashInfer mm_mxfp8 / MXFP8Tensor not available")
    fp8_available, reason = check_fp8_support()
    if not fp8_available:
        pytest.skip(f"FP8 not supported: {reason}")
    if not is_te_min_version("2.3.0"):
        pytest.skip("MXFP8 requires TransformerEngine >= 2.3.0")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("MXFP8 requires Blackwell architecture (SM >= 10.0)")


def _quantize_model_to_mxfp8(model):
    """Quantize inference-optimized linear-layer weights from bf16 to FlashInfer MXFP8."""
    from megatron.core.tensor_parallel.inference_layers import (
        InferenceLayerNormColumnParallelLinear,
        InferenceRowParallelLinear,
    )

    inference_linear_types = (
        InferenceLayerNormColumnParallelLinear,
        InferenceRowParallelLinear,
    )

    for module in model.modules():
        if not isinstance(module, inference_linear_types):
            continue
        param = module._parameters.get("weight")
        if param is not None and param.ndim == 2:
            fi_tensor = MXFP8Tensor.from_bf16(param.data)
            del module._parameters["weight"]
            module.weight = fi_tensor


class TestInferenceVsTrainingLogProbs:
    """Verify that MXFP8 inference log probs match training forward pass log probs."""

    def teardown_method(self, method):
        _set_rounder(64)
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_model_and_engine(max_sequence_length):
        """Build an MXFP8 inference-optimized GPT model + dynamic engine."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
        _set_rounder(4)

        random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        model_parallel_cuda_manual_seed(
            seed=RANDOM_SEED,
            inference_rng_tracker=True,
            use_cudagraphable_rng=False,
            force_reset_rng=True,
        )

        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=4,
            hidden_size=128,
            num_attention_heads=4,
            use_cpu_initialization=True,
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=False,
            inference_sampling_seed=RANDOM_SEED,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            fp8="hybrid",
            fp8_recipe="mxfp8",
        )

        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_inference_spec(),
            vocab_size=VOCAB_SIZE,
            max_sequence_length=max_sequence_length,
            parallel_output=True,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()

        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)
        model.eval()

        from megatron.core.utils import unwrap_model

        _quantize_model_to_mxfp8(unwrap_model(model))

        inference_context = DynamicInferenceContext(
            model_config=transformer_config,
            inference_config=InferenceConfig(
                max_sequence_length=max_sequence_length,
                buffer_size_gb=0.1,
                paused_buffer_size_gb=0.02,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                use_flashinfer_fused_rope=None,
                unified_memory_level=0,
            ),
        )

        inference_wrapped_model = GPTInferenceWrapper(model, inference_context)
        inference_wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage()
            and parallel_state.is_pipeline_last_stage()
        )

        controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE,
                detokenize=lambda tokens, **kwargs: "text",
            ),
        )

        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

        engine = DynamicInferenceEngine(controller, inference_context)
        return model, engine

    @staticmethod
    @torch.inference_mode()
    def _run_inference(engine, prompt_tokens, sampling_params):
        """Run dynamic inference to completion and return the finished request."""
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=prompt_tokens.clone(),
            sampling_params=sampling_params,
        )
        engine._add_request(request)

        all_finished_records = []
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            all_finished_records.extend(result["finished_request_records"])

        assert len(all_finished_records) == 1, (
            f"Expected 1 finished request, got {len(all_finished_records)}"
        )
        return all_finished_records[0].merge()

    @staticmethod
    def _training_forward_log_probs(model, full_sequence):
        """Run a training-style forward pass and compute per-token log probs."""
        tokens = full_sequence.unsqueeze(0).cuda()
        seq_len = tokens.shape[1]
        position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        with torch.inference_mode():
            logits = model(tokens, position_ids, None)

        log_probs_all = F.log_softmax(logits.float().squeeze(0), dim=-1)
        target_tokens = full_sequence[1:].to(log_probs_all.device)
        per_token_log_probs = log_probs_all[:-1].gather(
            1, target_tokens.unsqueeze(1)
        ).squeeze(1)

        return per_token_log_probs.cpu().tolist()

    # =========================================================================
    # Parametrized test
    # =========================================================================

    @pytest.mark.internal
    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"),
        reason="need latest flash attn for dynamic batching",
    )
    @pytest.mark.parametrize(
        "prompt_length",
        [8, 16, 32, 64, 128],
        ids=["prompt_8", "prompt_16", "prompt_32", "prompt_64", "prompt_128"],
    )
    @pytest.mark.parametrize(
        "num_tokens_to_generate",
        [16, 32, 64, 128],
        ids=["gen_16", "gen_32", "gen_64", "gen_128"],
    )
    @pytest.mark.parametrize(
        "sampling_config",
        [
            {"temperature": 1.0, "top_k": 1, "top_p": 0.0},
            {"temperature": 0.7, "top_k": 0, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 10, "top_p": 0.0},
            {"temperature": 1.0, "top_k": 0, "top_p": 0.9},
            {"temperature": 1.5, "top_k": 0, "top_p": 0.0},
            {"temperature": 0.3, "top_k": 0, "top_p": 0.0},
        ],
        ids=["greedy", "temp_0.7", "top_k_10", "top_p_0.9", "temp_1.5", "temp_0.3"],
    )
    @pytest.mark.parametrize(
        "num_concurrent_requests", [1, 3], ids=["single_req", "batched_3"]
    )
    @torch.inference_mode()
    def test_inference_vs_training_log_probs(
        self,
        prompt_length,
        num_tokens_to_generate,
        sampling_config,
        num_concurrent_requests,
    ):
        _skip_if_mxfp8_unsupported()

        max_sequence_length = prompt_length + num_tokens_to_generate
        model, engine = self._build_model_and_engine(
            max_sequence_length=max_sequence_length,
        )

        # Build requests — each gets a unique but deterministic prompt.
        request_data = []
        for i in range(num_concurrent_requests):
            torch.manual_seed(RANDOM_SEED + i)
            prompt_tokens = torch.randint(
                0,
                VOCAB_SIZE - 1,
                (prompt_length,),
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            sampling_params = SamplingParams(
                temperature=sampling_config["temperature"],
                top_k=sampling_config["top_k"],
                top_p=sampling_config["top_p"],
                return_log_probs=True,
                skip_prompt_log_probs=False,
                num_tokens_to_generate=num_tokens_to_generate,
                termination_id=-1,
            )
            req = DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=prompt_tokens.clone(),
                sampling_params=sampling_params,
            )
            request_data.append((req, prompt_tokens))

        for req, _ in request_data:
            engine._add_request(req)

        all_finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                finished = record.merge()
                all_finished[finished.request_id] = finished

        assert len(all_finished) == num_concurrent_requests

        # Compare each request using the NeMo-RL token_mult_prob_error metric.
        for req, prompt_tokens in request_data:
            finished = all_finished[req.request_id]
            assert finished.status == Status.COMPLETED

            inf_prompt_lps = finished.prompt_log_probs or []
            inf_gen_lps = finished.generated_log_probs or []
            generated_tokens = finished.generated_tokens

            assert len(generated_tokens) == num_tokens_to_generate, (
                f"Request {req.request_id}: expected {num_tokens_to_generate} "
                f"tokens, got {len(generated_tokens)}"
            )

            full_sequence = torch.cat([
                prompt_tokens.cpu(),
                torch.tensor(generated_tokens, dtype=torch.int64),
            ])
            training_lps = self._training_forward_log_probs(model, full_sequence)
            assert len(training_lps) == len(full_sequence) - 1

            # Check prompt log probs.
            expected_prompt_count = prompt_length - 1
            assert len(inf_prompt_lps) == expected_prompt_count, (
                f"Request {req.request_id}: expected {expected_prompt_count} "
                f"prompt log probs, got {len(inf_prompt_lps)}"
            )

            prompt_error = _compute_token_mult_prob_error(
                inf_prompt_lps, training_lps[: prompt_length - 1]
            )

            # Check generated log probs.
            assert len(inf_gen_lps) == num_tokens_to_generate, (
                f"Request {req.request_id}: expected {num_tokens_to_generate} "
                f"generated log probs, got {len(inf_gen_lps)}"
            )

            gen_error = _compute_token_mult_prob_error(
                inf_gen_lps, training_lps[prompt_length - 1:]
            )

            # Combined over all tokens (prompt + generated), same as NeMo-RL.
            all_inf_lps = inf_prompt_lps + inf_gen_lps
            all_train_lps = training_lps
            total_error = _compute_token_mult_prob_error(all_inf_lps, all_train_lps)

            assert total_error < TOKEN_MULT_PROB_ERROR_THRESHOLD, (
                f"Request {req.request_id}: token_mult_prob_error {total_error:.6f} "
                f">= {TOKEN_MULT_PROB_ERROR_THRESHOLD} "
                f"(prompt_error={prompt_error:.6f}, gen_error={gen_error:.6f}, "
                f"prompt_len={prompt_length}, gen_len={num_tokens_to_generate})"
            )
