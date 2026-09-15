# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Configurable MIMO training -> standard LLaVA refit correctness smoke.

Run with torch.distributed.run; language, vision and inference use disjoint
rank ranges whose TP sizes are configurable. Extra ranks remain idle. Synthetic images and random tiny
weights avoid checkpoint dependencies. The explicit activation/gradient
exchange tests refit independently of a training framework's scheduler.
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.resharding.refit import (
    clear_all_caches,
    clear_plan_cache,
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.resharding.utils import named_refit_tensors
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

LOG = logging.getLogger(__name__)


@dataclass
class SmokeConfig:
    """Small model and mesh configuration for the distributed correctness test."""

    output: Path
    language_tp: int = 2
    vision_tp: int = 1
    inference_tp: int = 1
    steps: int = 2
    layers: int = 1
    query_groups: int = 4
    batch_size: int = 1
    dtype: str = 'bf16'
    backend: str = 'nccl'
    projector: str = 'mlp'
    tied: bool = False
    untie_target: bool = False
    bias_free: bool = False
    wrapped: bool = False
    rebuild: bool = False
    execution_batch_bytes: int | None = None

    @property
    def torch_dtype(self):
        return torch.bfloat16 if self.dtype == 'bf16' else torch.float16

    @property
    def language_ranks(self):
        return range(self.language_tp)

    @property
    def vision_ranks(self):
        return range(self.language_tp, self.language_tp + self.vision_tp)

    @property
    def inference_ranks(self):
        start = self.language_tp + self.vision_tp
        return range(start, start + self.inference_tp)


def make_grid(tp, offset):
    grid = HyperCommGrid(
        [tp, 1, 1, 1], ['tp', 'pp', 'dp', 'cp'], rank_offset=offset, backend='nccl'
    )
    for axis in ('tp', 'pp', 'dp', 'cp'):
        grid.create_pg([axis])
    if not grid.is_current_rank_in_grid():
        return grid, None
    tp_group, singleton = grid.get_pg(['tp']), grid.get_pg(['dp'])
    return grid, ProcessGroupCollection(
        tp=tp_group,
        pp=grid.get_pg(['pp']),
        dp=singleton,
        cp=singleton,
        dp_cp=singleton,
        expt_tp=tp_group,
        ep=singleton,
        expt_dp=singleton,
        embd=singleton,
        pos_embd=singleton,
        mp=tp_group,
        tp_cp=tp_group,
    )


def config(args, tp, language=False):
    return TransformerConfig(
        num_layers=args.layers,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=args.query_groups if language else 4,
        add_bias_linear=not args.bias_free,
        tensor_model_parallel_size=tp,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=args.torch_dtype,
        bf16=args.dtype == 'bf16',
        fp16=args.dtype == 'fp16',
    )


def build_models(args, rank, grids, language_pg, image_pg, inference_pg):
    if rank >= args.inference_ranks.stop:
        return None, {}
    layer = get_gpt_layer_with_transformer_engine_spec()
    # Affine projectors request gather_output, which TEColumnParallelLinear
    # does not implement. The native column-parallel layer supports it.
    projection = MLPSubmodules(
        linear_fc1=ColumnParallelLinear if args.projector == 'affine' else TEColumnParallelLinear,
        linear_fc2=TERowParallelLinear,
    )
    vision_tp = args.inference_tp if rank in args.inference_ranks else args.vision_tp
    vision_config = config(args, vision_tp)
    vision_config.vision_model_type = 'clip'
    if rank in args.inference_ranks:
        model = (
            LLaVAModel(
                language_transformer_config=config(args, args.inference_tp, language=True),
                language_transformer_layer_spec=layer,
                language_vocab_size=256,
                language_max_sequence_length=64,
                vision_transformer_config=vision_config,
                vision_transformer_layer_spec=layer,
                drop_vision_class_token=False,
                vision_projection_config=config(args, args.inference_tp),
                vision_projection_type=args.projector,
                share_embeddings_and_output_weights=args.tied and not args.untie_target,
                vision_projection_layer_spec=projection,
                language_position_embedding_type='rope',
                img_h=32,
                img_w=32,
                patch_dim=16,
                image_token_index=255,
                parallel_output=False,
                pg_collection=inference_pg,
            )
            .cuda()
            .to(args.torch_dtype)
        )
        components = {
            'language': model.language_model,
            'vision': model.vision_model,
            'projector': model.vision_projection,
        }
    else:
        language = ModuleSpec(
            module=GPTModel,
            params={
                'config': config(args, args.language_tp, language=True),
                'share_embeddings_and_output_weights': args.tied,
                'transformer_layer_spec': layer,
                'vocab_size': 256,
                'max_sequence_length': 64,
                'parallel_output': False,
                'position_embedding_type': 'rope',
                'pg_collection': language_pg,
            },
        )
        vision = ModuleSpec(
            module=CLIPViTModel,
            params={
                'transformer_config': vision_config,
                'transformer_layer_spec': layer,
                'img_h': 32,
                'img_w': 32,
                'patch_dim': 16,
                'pg_collection': image_pg,
            },
        )
        projector = ModuleSpec(
            module=MultimodalProjector,
            params={
                'config': config(args, args.vision_tp),
                'submodules': projection,
                'projector_type': args.projector,
                'input_size': 128,
                'pg_collection': image_pg,
            },
        )
        images = ModuleSpec(
            module=VisionModalitySubmodules,
            params={'pg_collection': image_pg},
            submodules={'encoders': {'clip_encoder': vision}, 'input_projections': [projector]},
        )
        model = (
            MimoModel(
                MimoModelConfig(
                    language_model_spec=language,
                    modality_submodules_spec={'images': images},
                    special_token_ids={'images': 255},
                    module_to_grid_map=grids,
                )
            )
            .cuda()
            .to(args.torch_dtype)
        )
        if rank in args.language_ranks:
            components = {'language': model.language_model}
        else:
            tower = model.modality_submodules['images']
            components = {
                'vision': tower.encoders['clip_encoder'],
                'projector': tower.input_projections[0],
            }
    # Persistent state must travel alongside the trainable weights.
    if 'vision' in components:
        components['vision'].register_buffer('refit_counter', torch.zeros(1, device='cuda'))
    return model, components


def training_forward(args, model, rank, tokens, positions, image, train):
    if rank not in args.language_ranks and rank not in args.vision_ranks:
        return None, None
    model.train(train)
    if rank in args.vision_ranks:
        outputs, _ = model(
            input_ids=tokens,
            position_ids=positions,
            modality_inputs={'images': {'clip_encoder': {'x': image}}},
        )
        features = outputs['images']
        if rank == args.vision_ranks.start:
            for peer in args.language_ranks:
                dist.send(features.detach().contiguous(), dst=peer)
        if train:
            gradient = torch.empty_like(features)
            dist.recv(gradient, src=0)
            features.backward(gradient)
        return None, None
    features = torch.empty((args.batch_size * 5, 128), dtype=args.torch_dtype, device='cuda')
    dist.recv(features, src=args.vision_ranks.start)
    features.requires_grad_(train)
    model.set_input_tensor({'images': features})
    logits, _ = model(input_ids=tokens, position_ids=positions)
    loss = None
    if train:
        labels = (tokens + 1) % 254
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
        loss.backward()
        if rank == 0:
            for peer in args.vision_ranks:
                dist.send(features.grad.contiguous(), dst=peer)
    return logits, loss


def verify_weights(args, model, components, rank):
    """Independently reconstruct source shards and compare every destination shard."""
    local = {}
    if rank < args.inference_ranks.start:
        paths = {
            'language': 'language_model',
            'vision': 'vision_model',
            'projector': 'vision_projection',
        }
        for component, module in components.items():
            for name, tensor in named_refit_tensors(module):
                local[f'{paths[component]}.{name}'] = (
                    tensor.detach().cpu(),
                    bool(getattr(tensor, 'tensor_model_parallel', False)),
                    getattr(tensor, 'partition_dim', 0),
                    getattr(tensor, 'partition_stride', 1),
                )
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if rank not in args.inference_ranks:
        return 0
    target = dict(named_refit_tensors(model))
    source_names = set().union(*(set(part) for part in gathered))
    allowed_aliases = (
        {'language_model.output_layer.weight'} if args.tied and args.untie_target else set()
    )
    assert set(target) - allowed_aliases == source_names
    for name, tensor in target.items():
        source_name = (
            'language_model.embedding.word_embeddings.weight' if name in allowed_aliases else name
        )
        parts = [part[source_name] for part in gathered if source_name in part]
        first, sharded, dim, stride = parts[0]
        if sharded and len(parts) > 1:
            chunks = [part[0].chunk(stride, dim=dim) for part in parts]
            expected = torch.cat(
                [torch.cat([c[i] for c in chunks], dim=dim) for i in range(stride)], dim=dim
            )
        else:
            expected = first
            for replica in parts[1:]:
                torch.testing.assert_close(replica[0], first, rtol=0, atol=0)
        if getattr(tensor, 'tensor_model_parallel', False) and args.inference_tp > 1:
            dim = tensor.partition_dim
            stride = tensor.partition_stride
            tp_rank = rank - args.inference_ranks.start
            expected = torch.cat(
                [
                    part.chunk(args.inference_tp, dim=dim)[tp_rank]
                    for part in expected.chunk(stride, dim=dim)
                ],
                dim=dim,
            )
        torch.testing.assert_close(
            tensor.cpu(), expected, rtol=0, atol=0, msg=lambda m: f'{name}: {m}'
        )
    return len(target)


def generate(args, model, inference_pg, text, image):
    """Run standard LLaVA prefill and a decode step that reuses the KV cache."""
    context = StaticInferenceContext(args.batch_size, 64)
    context.config.pg_collection = inference_pg
    wrapper = VLMInferenceWrapper(model, context)
    inputs = wrapper.prep_inference_input(
        text, 5, image, torch.ones(args.batch_size, device='cuda', dtype=torch.int32), 64
    )
    wrapper.inference_context.config.pg_collection = inference_pg
    wrapper.prep_model_for_inference()
    prefill = wrapper.run_one_forward_step(inputs)
    assert torch.isfinite(prefill).all()
    first = prefill[:, -1].argmax(-1)
    # A random tiny model may emit the reserved image token; mask it during generation.
    if (first == 255).any():
        prefill[:, -1, 255] = -torch.inf
        first = prefill[:, -1].argmax(-1)
    inputs['tokens'] = first[:, None].to(torch.int32)
    inputs['position_ids'] = torch.full((args.batch_size, 1), 12, device='cuda', dtype=torch.int32)
    decode = wrapper.run_one_forward_step(inputs)
    assert torch.isfinite(decode).all()
    return [first.tolist(), decode[:, -1].argmax(-1).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for name, default in [
        ('language-tp', 2),
        ('vision-tp', 1),
        ('inference-tp', 1),
        ('steps', 2),
        ('layers', 1),
        ('query-groups', 4),
        ('batch-size', 1),
    ]:
        parser.add_argument(f'--{name}', type=int, default=default)
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--backend', choices=['nccl', 'gloo'], default='nccl')
    parser.add_argument('--projector', choices=['mlp', 'affine'], default='mlp')
    parser.add_argument('--execution-batch-bytes', type=int)
    for name in ('tied', 'untie-target', 'bias-free', 'wrapped', 'rebuild'):
        parser.add_argument(f'--{name}', action='store_true')
    args = SmokeConfig(**vars(parser.parse_args()))
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    logging.basicConfig(level=logging.INFO, format=f'rank={rank} %(message)s', force=True)
    dist.init_process_group('cpu:gloo,cuda:nccl', timeout=timedelta(seconds=120))
    assert (
        args.inference_ranks.stop <= dist.get_world_size()
    ), 'Not enough ranks for the requested meshes'
    parallel_state.initialize_model_parallel(1, 1)
    language_grid, language_pg = make_grid(args.language_tp, 0)
    image_grid, image_pg = make_grid(args.vision_tp, args.vision_ranks.start)
    _, inference_pg = make_grid(args.inference_tp, args.inference_ranks.start)
    pg = language_pg or image_pg or inference_pg
    tp_rank = pg.tp.rank() if pg is not None else 0
    torch.manual_seed(100 if rank < args.inference_ranks.start else 200)
    model_parallel_cuda_manual_seed(100, tp_rank=tp_rank, ep_rank=0, etp_rank=0)
    model, components = build_models(
        args,
        rank,
        {'language': language_grid, 'images': image_grid},
        language_pg,
        image_pg,
        inference_pg,
    )
    source = model if rank < args.inference_ranks.start else None
    target = model if rank in args.inference_ranks else None
    inference_model = Float16Module(model.config, model) if target is not None else None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02) if source is not None else None
    if args.wrapped:
        source = [Float16Module(source.config, source)] if source is not None else None
        target = [inference_model] if target is not None else None
    text = torch.tensor([[11, 12, 255, 13, 14, 15, 16, 17, 18, 19, 20, 21]], device='cuda').repeat(
        args.batch_size, 1
    )
    tokens = torch.cat(
        [text[:, :2], torch.full((args.batch_size, 5), 255, device='cuda'), text[:, 3:]], dim=1
    )
    positions = torch.arange(tokens.size(1), device='cuda').unsqueeze(0).expand(args.batch_size, -1)
    image = (
        torch.linspace(-1, 1, 3 * 32 * 32, device='cuda', dtype=args.torch_dtype)
        .reshape(1, 3, 32, 32)
        .repeat(args.batch_size, 1, 1, 1)
    )
    refit_options = {'group': dist.group.WORLD, 'execution_batch_bytes': args.execution_batch_bytes}
    prepare_swap_model_weights(source, target, **refit_options)
    report = {
        'config': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'rounds': [],
    }
    previous = None
    for step in range(args.steps + 1):
        if step:
            if optimizer is not None:
                optimizer.zero_grad()
            _, loss = training_forward(args, model, rank, tokens, positions, image, train=True)
            if optimizer is not None:
                for component, module in components.items():
                    norm = sum(
                        p.grad.float().square().sum().item()
                        for p in module.parameters()
                        if p.grad is not None
                    )
                    assert norm > 0, f'{component} received no gradient'
                optimizer.step()
                LOG.info('optimizer step %s; loss=%s', step, loss)
            if rank in args.vision_ranks:
                components['vision'].refit_counter.add_(1)
            if args.rebuild:
                clear_plan_cache()
                prepare_swap_model_weights(source, target, **refit_options)
        swap_model_weights(source, target, refit_method=args.backend, **refit_options)
        verified = verify_weights(args, model, components, rank)
        with torch.no_grad():
            expected, _ = training_forward(args, model, rank, tokens, positions, image, train=False)
            if rank != 0:
                expected = torch.empty(
                    (args.batch_size, 16, 256), dtype=args.torch_dtype, device='cuda'
                )
            dist.broadcast(expected, src=0)
            if rank in args.inference_ranks:
                model.eval()
                actual = model(
                    image,
                    text,
                    torch.arange(12, device='cuda').unsqueeze(0).expand(args.batch_size, -1),
                    None,
                    runtime_gather_output=True,
                )
                if isinstance(actual, tuple):
                    actual = actual[0]
                error = (actual.float() - expected.float()).abs().max().item()
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.015)
                generated = generate(args, inference_model, inference_pg, text, image)
                if previous is not None:
                    assert not torch.equal(
                        previous, actual
                    ), 'Training/refit did not change inference logits'
                previous = actual.clone()
                report['rounds'].append(
                    {
                        'step': step,
                        'exact_tensors_verified': verified,
                        'max_logit_error': error,
                        'generated_tokens': generated,
                    }
                )
                LOG.info('PASS round %s: %s', step, report['rounds'][-1])
        dist.barrier()
    if rank == args.inference_ranks.start:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    clear_all_caches()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
