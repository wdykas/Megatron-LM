# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Component refit metadata must reflect tensor ownership, not the parent mesh."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.resharding.model_adapters import _as_refit_model, _refit_components
from megatron.core.resharding.model_view import _RefitComponent, _RefitModelView
from megatron.core.resharding.planner import (
    _extract_module_metadata,
    _find_source_metadata,
    build_plan_from_rosters,
    index_metadata_rosters,
)
from megatron.core.resharding.refit import (
    _get_parallel_config,
    _unwrap_model_cores,
    clear_plan_cache,
)
from megatron.core.resharding.utils import get_refit_tensor_dict


def groups(ranks=(0,)):
    return SimpleNamespace(tp=ranks, pp=ranks, dp=ranks, ep=ranks, expt_tp=ranks)


@pytest.fixture(autouse=True)
def group_ranks(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda group: list(group))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)


def test_view_preserves_original_tensors_and_buffer_ownership():
    tower = torch.nn.Linear(3, 4)
    tower.register_buffer('state', torch.ones(4))
    tower.register_buffer('scratch', torch.zeros(4), persistent=False)
    original_names = tuple(tower.state_dict())
    view = _RefitModelView({'vision': _RefitComponent(tower, groups())})
    tensors = get_refit_tensor_dict(view)
    assert tensors['vision.weight'] is tower.weight
    assert tensors['vision.state'] is tower.state
    assert 'vision.scratch' not in tensors
    assert tuple(tower.state_dict()) == original_names
    assert _unwrap_model_cores(view, None) == (view, None, None)


def test_components_use_their_own_groups_and_cross_world_offset():
    vision = torch.nn.Linear(3, 4)
    language = torch.nn.Linear(4, 8)
    language.weight.tensor_model_parallel = True
    language.weight.partition_dim = 0
    view = _RefitModelView(
        {
            'vision': _RefitComponent(vision, groups((2,))),
            'language': _RefitComponent(language, groups((0, 1))),
        }
    )
    metadata = {m.name: m for m in _extract_module_metadata(view, 5, None, 3, {})}
    assert metadata['vision.weight'].tensor_parallel_group_ranks == [5]
    assert metadata['language.weight'].tensor_parallel_group_ranks == [3, 4]
    assert metadata['language.weight'].is_tp
    assert all(m.owner_rank == 5 for m in metadata.values())


def test_different_model_paths_match_and_transfer_all_components():
    # The native source and destination paths need not agree.
    source = torch.nn.ModuleDict({'modality_submodules': torch.nn.Linear(3, 4)})
    target = torch.nn.ModuleDict({'vision_model': torch.nn.Linear(3, 4)})
    src = _RefitModelView({'vision': _RefitComponent(source['modality_submodules'], groups((0,)))})
    dst = _RefitModelView({'vision': _RefitComponent(target['vision_model'], groups((1,)))})
    src_meta = _extract_module_metadata(src, 0, None, 0, {})
    dst_meta = _extract_module_metadata(dst, 1, None, 0, {})
    dst_roster, src_roster = index_metadata_rosters([(src_meta, []), ([], dst_meta)])
    send = build_plan_from_rosters(dst_roster, src_roster, 0)
    receive = build_plan_from_rosters(dst_roster, src_roster, 1)
    assert len(send.send_ops) == len(receive.recv_ops) == 2
    assert [op.task_id for op in send.send_ops] == [op.task_id for op in receive.recv_ops]
    assert {op.param_name for op in send.send_ops} == {'vision.weight', 'vision.bias'}


def test_absent_components_and_missing_destination_weights():
    assert _extract_module_metadata(_RefitModelView({}), 0, None, 0, {}) == []
    dst = _RefitModelView({'vision': _RefitComponent(torch.nn.Linear(2, 2), groups((1,)))})
    dst_meta = _extract_module_metadata(dst, 1, None, 0, {})
    dst_roster, src_roster = index_metadata_rosters([([], []), ([], dst_meta)])
    with pytest.raises(RuntimeError, match='not found in source'):
        build_plan_from_rosters(dst_roster, src_roster, 1)


def test_cache_distinguishes_equal_size_but_different_mesh_members():
    module = torch.nn.Linear(2, 2)
    first = _RefitModelView({'vision': _RefitComponent(module, groups((0, 1)))})
    second = _RefitModelView({'vision': _RefitComponent(module, groups((2, 3)))})
    renamed = _RefitModelView({'language': _RefitComponent(module, groups((0, 1)))})
    assert _get_parallel_config(first) != _get_parallel_config(second)
    assert _get_parallel_config(first) != _get_parallel_config(renamed)
    assert _get_parallel_config(first) == _get_parallel_config(first)


def test_tied_embedding_alias_stays_inside_its_component():
    embedding = object()
    unrelated = object()
    roster = {
        'language.embedding.word_embeddings.weight': [embedding],
        'embedding.word_embeddings.weight': [unrelated],
    }
    assert _find_source_metadata(roster, 'language.output_layer.weight') == [embedding]
    assert _find_source_metadata(roster, 'other.output_layer.weight') is None


def test_global_pipeline_layer_indices_are_preserved_under_component_prefix():
    layer = torch.nn.Linear(2, 2)
    layer.layer_number = 7
    module = torch.nn.ModuleDict({'layers': torch.nn.ModuleList([layer])})
    view = _RefitModelView({'language': _RefitComponent(module, groups())})
    metadata = _extract_module_metadata(view, 0, None, 0, {})
    assert metadata[0].name == 'language.layers.0.weight'
    assert metadata[0].resolved_name == 'language.layers.6.weight'


def test_rejects_shared_tensors_across_components():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match='share a tensor'):
        _RefitModelView(
            {'a': _RefitComponent(module, groups()), 'b': _RefitComponent(module, groups())}
        )


def test_shared_nonpersistent_scratch_is_not_part_of_refit():
    first, second = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    scratch = torch.zeros(1)
    for module in (first, second):
        module.register_buffer('scratch', scratch, persistent=False)
    view = _RefitModelView(
        {'a': _RefitComponent(first, groups()), 'b': _RefitComponent(second, groups())}
    )
    assert len(get_refit_tensor_dict(view)) == 4


def test_cache_distinguishes_gtp_rematerialization_groups():
    module = torch.nn.Linear(2, 2)
    first, second = groups(), groups()
    first.gtp_remat, second.gtp_remat = (0, 1), (2, 3)
    a = _RefitModelView({'language': _RefitComponent(module, first)})
    b = _RefitModelView({'language': _RefitComponent(module, second)})
    assert _get_parallel_config(a) != _get_parallel_config(b)


def test_rejects_quantization_and_missing_process_groups():
    module = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match='explicit tp, pp and dp'):
        _RefitModelView({'vision': _RefitComponent(module, SimpleNamespace(tp=None))})
    module.config = SimpleNamespace(fp8='e4m3')
    with pytest.raises(ValueError, match='quantized refit views'):
        _RefitModelView({'vision': _RefitComponent(module, groups())})


def test_component_expert_count_uses_config_and_rejects_conflicts():
    module = torch.nn.Linear(2, 2)
    module.config = SimpleNamespace(num_moe_experts=8)
    view = _RefitModelView({'language': _RefitComponent(module, groups())})
    assert view.local_components()[0][1].num_experts == 8
    with pytest.raises(ValueError, match='disagrees with its config'):
        _RefitModelView({'language': _RefitComponent(module, groups(), num_experts=4)})
    incomplete = groups()
    incomplete.ep = None
    with pytest.raises(ValueError, match='MoE requires explicit'):
        _RefitModelView({'language': _RefitComponent(module, incomplete)})


@pytest.mark.parametrize('name', ['', 'nested.name'])
def test_rejects_ambiguous_component_names(name):
    with pytest.raises(ValueError, match='Component name'):
        _RefitModelView({name: _RefitComponent(torch.nn.Linear(2, 2), groups())})


def language_only_mimo_rank():
    from megatron.core.models.mimo import MimoModel

    # Construct the local structure without initializing unrelated CUDA layers.
    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    model.mimo_config = SimpleNamespace(modality_submodules_spec={'images': object()})
    model.language_model = torch.nn.Linear(2, 2)
    model.language_model.pg_collection = groups()
    model.modality_submodules = torch.nn.ModuleDict()
    return model


def test_automatic_adapter_reuses_view_and_cache_clear_rebuilds_it():
    model = language_only_mimo_rank()
    source, target, _ = _unwrap_model_cores([model], None)
    assert target is None
    assert source is _as_refit_model(model)
    assert get_refit_tensor_dict(source)['language_model.weight'] is model.language_model.weight
    clear_plan_cache()
    assert _as_refit_model(model) is not source
    assert _as_refit_model(model.language_model) is model.language_model


def test_automatic_adapter_rejects_unmapped_modalities_and_uncovered_state():
    model = language_only_mimo_rank()
    model.mimo_config.modality_submodules_spec['audio'] = object()
    with pytest.raises(ValueError, match="one 'images' modality"):
        _as_refit_model(model)
    del model.mimo_config.modality_submodules_spec['audio']
    model.register_buffer('unmapped_state', torch.ones(1))
    with pytest.raises(ValueError, match='does not cover all'):
        _as_refit_model(model)


def vision_only_mimo_rank():
    from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
    from megatron.core.models.vision.clip_vit_model import CLIPViTModel
    from megatron.core.models.vision.multimodal_projector import MultimodalProjector

    model = language_only_mimo_rank()
    model.language_model = None
    vision = CLIPViTModel.__new__(CLIPViTModel)
    torch.nn.Module.__init__(vision)
    vision.register_parameter('weight', torch.nn.Parameter(torch.ones(2, 2)))
    vision.pg_collection = groups((2,))
    projector = MultimodalProjector.__new__(MultimodalProjector)
    torch.nn.Module.__init__(projector)
    projector.register_parameter('weight', torch.nn.Parameter(torch.ones(2, 2)))
    projector.tp_group = (2,)
    tower = VisionModalitySubmodules(
        encoders={'clip': vision}, input_projections=[projector], pg_collection=groups((2,))
    )
    model.modality_submodules['images'] = tower
    return model, tower


def test_automatic_adapter_maps_vision_and_uses_projector_modality_groups():
    model, tower = vision_only_mimo_rank()
    view = _as_refit_model(model)
    tensors = get_refit_tensor_dict(view)
    assert tensors['vision_model.weight'] is tower.encoders['clip'].weight
    assert tensors['vision_projection.weight'] is tower.input_projections[0].weight
    metadata = _extract_module_metadata(view, 2, None, 0, {})
    assert all(m.data_parallel_group_ranks == [2] for m in metadata)


def test_automatic_adapter_rejects_projector_mesh_mismatch():
    model, tower = vision_only_mimo_rank()
    tower.input_projections[0].tp_group = (0, 1)
    with pytest.raises(ValueError, match='projector TP group must match'):
        _as_refit_model(model)


@pytest.mark.parametrize(
    'field', ['encoders', 'decoders', 'input_projections', 'output_projections']
)
def test_automatic_adapter_rejects_unsupported_vision_structure(field):
    model, tower = vision_only_mimo_rank()
    modules = getattr(tower, field)
    if isinstance(modules, torch.nn.ModuleDict):
        modules['extra'] = torch.nn.Linear(2, 2)
    else:
        modules.append(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match='one vision encoder and one input projector'):
        _as_refit_model(model)


def test_cached_adapter_does_not_keep_original_model_alive():
    import gc
    import weakref

    model = language_only_mimo_rank()
    _as_refit_model(model)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


def test_cached_layout_does_not_keep_component_alive_or_match_replacement():
    import gc
    import weakref

    module = torch.nn.Linear(2, 2)
    view = _RefitModelView({'language': _RefitComponent(module, groups())})
    reference = weakref.ref(module)
    layout = _get_parallel_config(view)
    cached = {layout: object()}
    assert _get_parallel_config(view) in cached

    del module, view
    gc.collect()
    assert reference() is None
    assert layout in cached

    replacement = _RefitModelView({'language': _RefitComponent(torch.nn.Linear(2, 2), groups())})
    assert _get_parallel_config(replacement) not in cached


def test_registered_adapter_uses_generic_validation_and_cache():
    class Composite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)

    @_refit_components.register(Composite)
    def components(model):
        return {'vision_model': _RefitComponent(model.encoder, groups())}

    model = Composite()
    view = _as_refit_model(model)
    assert _as_refit_model(model) is view
    assert get_refit_tensor_dict(view)['vision_model.weight'] is model.encoder.weight
    clear_plan_cache()
    assert _as_refit_model(model) is not view

    incomplete = Composite()
    incomplete.register_buffer('unmapped', torch.ones(1))
    with pytest.raises(ValueError, match='Composite refit adapter does not cover all'):
        _as_refit_model(incomplete)


def test_registered_adapter_cannot_retain_its_cache_key():
    class RootComponent(torch.nn.Module):
        pass

    @_refit_components.register(RootComponent)
    def components(model):
        return {'language_model': _RefitComponent(model, groups())}

    with pytest.raises(ValueError, match='child modules, not the original model'):
        _as_refit_model(RootComponent())
