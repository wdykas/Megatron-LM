# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Composite refit discovers ownership without changing the model's storage tree."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from megatron.core.resharding.model_metadata import _named_owned_tensors
from megatron.core.resharding.planner import _extract_module_metadata, _find_source_metadata
from megatron.core.resharding.refit import (
    _get_parallel_config,
    _harmonize_buffer_dtypes,
    _unwrap_model_cores,
)
from megatron.core.resharding.utils import ReshardPlan, TransferOp, get_refit_tensor_dict
from megatron.core.transformer.module import Float16Module


def groups(ranks=(0,)):
    return SimpleNamespace(tp=ranks, pp=ranks, dp=ranks, ep=ranks, expt_tp=ranks)


def linear(ranks=(0,)):
    module = torch.nn.Linear(2, 2, bias=False)
    module.pg_collection = groups(ranks)
    return module


@pytest.fixture(autouse=True)
def group_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda group: list(group))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)


def metadata(module, offset=0):
    return _extract_module_metadata(module, 0, None, offset, {})


def mimo(**modalities):
    from megatron.core.models.mimo import MimoModel

    # Preserve real type dispatch without constructing unrelated CUDA layers.
    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    model.language_model = linear()
    model.modality_submodules = torch.nn.ModuleDict(modalities)
    return model


def test_generic_composite_uses_child_groups_and_rank_offsets():
    model = torch.nn.ModuleDict({'language': linear((0, 1)), 'vision': linear((2,))})
    model['language'].weight.tensor_model_parallel = True
    model['language'].weight.partition_dim = 0
    entries = {entry.name: entry for entry in metadata(model, offset=3)}
    assert entries['language.weight'].tensor_parallel_group_ranks == [3, 4]
    assert entries['language.weight'].is_tp
    assert entries['vision.weight'].tensor_parallel_group_ranks == [5]
    assert all(entry.name == entry.resolved_name for entry in entries.values())
    assert metadata(torch.nn.Module()) == []
    assert _unwrap_model_cores([model], None) == (model, None, None)


def test_projector_inherits_nearest_groups_through_containers():
    projector = torch.nn.Linear(2, 2, bias=False)
    projector.tp_group = (2,)
    tower = torch.nn.ModuleDict({'projections': torch.nn.ModuleList([projector])})
    tower.pg_collection = groups((2,))
    model = torch.nn.ModuleDict({'images': tower})
    entry = metadata(model)[0]
    assert entry.data_parallel_group_ranks == [2]
    projector.tp_group = (0, 1)
    with pytest.raises(ValueError, match='TP group disagrees'):
        metadata(model)


def test_expert_layers_check_expert_tp_ownership():
    module = linear()
    module.config = SimpleNamespace(num_moe_experts=8)
    module.pg_collection.expt_tp = (1, 2)
    module.tp_group = (1, 2)  # Grouped expert layers do not always have is_expert.
    assert len(list(_named_owned_tensors(module))) == 1
    module.tp_group = (3, 4)
    with pytest.raises(ValueError, match='TP group disagrees'):
        list(_named_owned_tensors(module))


def test_nearest_config_controls_expert_count():
    tower = torch.nn.ModuleDict({'expert': linear(), 'dense': linear()})
    tower.config = SimpleNamespace(num_moe_experts=8)
    tower['dense'].config = SimpleNamespace(num_moe_experts=None)
    entries = {name: experts for name, _, _, _, experts in _named_owned_tensors(tower)}
    assert entries == {'expert.weight': 8, 'dense.weight': None}
    tower['expert'].pg_collection.ep = None
    with pytest.raises(ValueError, match='MoE requires explicit'):
        list(_named_owned_tensors(tower))


@pytest.mark.parametrize('child_groups', [None, SimpleNamespace(tp=(0,))])
def test_missing_ownership_fails_instead_of_using_global_groups(child_groups):
    child = torch.nn.Linear(2, 2)
    child.pg_collection = child_groups
    with pytest.raises(ValueError, match='requires explicit tp, pp and dp'):
        metadata(torch.nn.ModuleDict({'child': child}))


def test_partial_child_groups_do_not_merge_with_parent_groups():
    model = torch.nn.ModuleDict({'child': linear()})
    model.pg_collection = groups()
    model['child'].pg_collection.dp = None
    with pytest.raises(ValueError, match='requires explicit tp, pp and dp'):
        list(_named_owned_tensors(model))


def test_wrappers_and_pipeline_indices_only_change_matching_names():
    layer = linear()
    layer.layer_number = 7
    wrapper = Float16Module.__new__(Float16Module)
    torch.nn.Module.__init__(wrapper)
    wrapper.module = torch.nn.ModuleDict({'layers': torch.nn.ModuleList([layer])})
    model = torch.nn.ModuleDict({'language': wrapper})
    entry = metadata(model)[0]
    assert entry.name == 'language.module.layers.0.weight'
    assert entry.resolved_name == 'language.layers.6.weight'
    assert dict(model.named_parameters())[entry.name] is layer.weight


def test_tied_parameters_are_deduplicated_but_conflicting_ownership_fails():
    first, second = linear(), linear()
    second.weight = first.weight
    model = torch.nn.ModuleDict({'first': first, 'second': second})
    assert [entry.name for entry in metadata(model)] == ['first.weight']
    second.pg_collection = groups((1,))
    with pytest.raises(ValueError, match='shared tensor has conflicting ownership'):
        metadata(model)


def test_persistent_buffer_paths_are_retained_and_scratch_is_excluded():
    first, second = linear(), linear()
    state, scratch = torch.ones(1), torch.zeros(1)
    for module in (first, second):
        module.register_buffer('state', state)
        module.register_buffer('scratch', scratch, persistent=False)
    names = {entry.name for entry in metadata(torch.nn.ModuleDict({'a': first, 'b': second}))}
    assert names == {'a.weight', 'a.state', 'b.weight', 'b.state'}


def test_shared_module_buffers_use_first_path_but_validate_every_owner():
    shared = torch.nn.Module()
    shared.register_buffer('state', torch.ones(1))
    first, second = torch.nn.ModuleDict({'shared': shared}), torch.nn.ModuleDict({'shared': shared})
    first.pg_collection, second.pg_collection = groups(), groups()
    model = torch.nn.ModuleDict({'first': first, 'second': second})
    assert [entry.name for entry in metadata(model)] == ['first.shared.state']
    second.pg_collection = groups((1,))
    with pytest.raises(ValueError, match='shared tensor has conflicting ownership'):
        metadata(model)


def test_mimo_names_map_to_llava_and_reject_collisions():
    tower = torch.nn.ModuleDict(
        {
            'encoders': torch.nn.ModuleDict({'clip': linear()}),
            'input_projections': torch.nn.ModuleList([linear()]),
        }
    )
    model = mimo(images=tower)
    assert {entry.resolved_name for entry in metadata(model)} == {
        'language_model.weight',
        'vision_model.weight',
        'vision_projection.weight',
    }
    tower['encoders']['second'] = linear()
    with pytest.raises(ValueError, match='Duplicate refit name'):
        metadata(model)


def test_unmapped_mimo_state_raises():
    with pytest.raises(ValueError, match='No refit name mapping'):
        metadata(mimo(audio=linear()))


def test_tied_embedding_alias_stays_inside_its_component():
    embedding = object()
    roster = {'language.embedding.word_embeddings.weight': [embedding]}
    assert _find_source_metadata(roster, 'language.output_layer.weight') == [embedding]
    assert _find_source_metadata(roster, 'other.output_layer.weight') is None


@pytest.mark.parametrize('format', ['fp8', 'fp4'])
def test_quantized_composite_config_is_rejected(format):
    module = linear()
    module.config = SimpleNamespace(**{format: True})
    with pytest.raises(ValueError, match='Quantized composite refit'):
        metadata(torch.nn.ModuleDict({'child': module}))


def test_composite_cache_key_distinguishes_models_without_retaining_them():
    first, second = torch.nn.Module(), torch.nn.Module()
    key = _get_parallel_config(first)
    assert key == _get_parallel_config(first)
    assert key != _get_parallel_config(second)
    cached_plan = {key: object()}
    reference = weakref.ref(first)
    del first
    gc.collect()
    assert reference() is None
    assert key in cached_plan


@pytest.mark.parametrize('conflicting_shard', [False, True])
def test_buffer_dtypes_match_transfer_ids_across_different_storage_names(
    monkeypatch, conflicting_shard
):
    source, target = torch.nn.ModuleDict({'source_tower': torch.nn.Module()}), torch.nn.ModuleDict(
        {'inference': torch.nn.Module()}
    )
    source['source_tower'].register_buffer('state', torch.ones(2, dtype=torch.float32))
    target['inference'].register_buffer('state', torch.zeros(2, dtype=torch.bfloat16))
    old_tensors = get_refit_tensor_dict(target)
    plan = ReshardPlan(
        [TransferOp('source_tower.state', 1, True, (slice(None),), (slice(None),), task_id=7)],
        [
            TransferOp('inference.state', 0, False, (slice(None),), (slice(None),), task_id=task)
            for task in (7, 9)
        ],
    )
    calls = []

    def gather(gathered, local, group=None):
        calls.append(local)
        assert local == {7: torch.float32}
        gathered[:] = [local, {9: torch.float16 if conflicting_shard else torch.float32}]

    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 2)
    monkeypatch.setattr(torch.distributed, 'all_gather_object', gather)
    if conflicting_shard:
        with pytest.raises(ValueError, match='Source shards disagree on buffer dtype'):
            _harmonize_buffer_dtypes(plan, source, target)
        return
    _harmonize_buffer_dtypes(plan, source, target)
    assert plan.buffer_dtypes == {'inference.state': torch.float32}
    assert target['inference'].state.dtype == torch.float32
    assert get_refit_tensor_dict(target) is not old_tensors
    assert get_refit_tensor_dict(target)['inference.state'] is target['inference'].state
    _harmonize_buffer_dtypes(plan, source, target)
    assert len(calls) == 1
