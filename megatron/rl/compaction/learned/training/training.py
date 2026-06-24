# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compactor training core — the single learned-compactor training path.

`train_compactor_trajectory` (used by both the online RL loop and the offline
trainer), the trainer classes, the Megatron mixed-precision optimizer adapter
`_CompactorOptimizer`, and the probe-term helpers.  Extracted from rl_utils.py so
the `learned/` package is self-contained; rl_utils keeps only the RL-loop
integration glue (maybe_train_compactor / _build_compactor_trajectories /
_init_* / _attach_*) which depends on the GRPO runtime state.
"""
from __future__ import annotations

import torch

@torch.enable_grad()
def train_compactor_trajectory(
    updater,            # BeliefUpdater or GatedRecurrentUpdater
    optimizer,          # torch.optim.Optimizer
    trajectory,         # Trajectory
    student_fn,         # StudentFn
    cfg,                # CompactorTrainerConfig
    feature_extractor=None,  # optional ChunkFeatureExtractor (z_t features for the gate)
    value_head=None,         # optional ValueHead (belief-state value prediction logging)
) -> dict:
    """THE single STILL training path — shared by online RL and offline training.

    Processes one Trajectory chunk-by-chunk, maintaining the belief memory M_t,
    accumulating every enabled loss term, with truncated BPTT.  Each objective is
    gated by its cfg weight AND by data availability, so the same function covers
    every pipeline:

      * online RL (vd_frozen / vd_joint): kv_reconstruction (+advantage weighting),
        predictive, consistency, dynamics, future_kv_reconstruction, merged_chunk,
        future_accuracy_weight.
      * online true-STILL (teacher-KL): still_ce via student_fn.
      * offline (BeliefCompactorTrainer): all of the above PLUS the probe-distillation
        terms (teacher_kl / future_kl / weighted_kl / retrieval / future_horizon_kl)
        when probes carry teacher_logits, plus optional feature_extractor (z_t) and
        value_head — passed by BeliefCompactorTrainer.train_trajectory, which delegates
        here.

    Advantage weighting is applied when cfg.vd_cfg is set and probe.advantage (or
    the trajectory return) is available.

    Decorated with @torch.enable_grad() because maybe_train_compactor is invoked from
    get_grpo_data_iterator under a `with torch.no_grad()` block (rollout collection).
    Without this, the compactor forward builds no autograd graph, chunk_loss never
    requires grad, and optimizer.step() is a silent no-op (weights never change).
    """
    import inspect
    # All loss/weight helpers used in the per-chunk loop, hoisted here so they are
    # imported once per trajectory (not per chunk/probe) and visible in one place.
    from megatron.rl.compaction.learned.training.losses import (
        CompactorLosses, predictive_coding_loss, consistency_loss,
        kv_reconstruction_loss, future_kv_reconstruction_loss,
        dynamics_prediction_loss, future_horizon_kl_loss,
    )
    from megatron.rl.compaction.learned.training.value_directed import _probe_weight
    from megatron.rl.compaction.learned.training.data import TrainingProbe

    # Detect whether the updater accepts a `features` kwarg (gate conditioning).
    def _accepts_features(fn):
        try:
            return "features" in inspect.signature(fn).parameters
        except (ValueError, TypeError):
            return False
    _init_has_features = feature_extractor is not None and _accepts_features(updater.initial_compress)
    _fwd_has_features = feature_extractor is not None and _accepts_features(getattr(updater, "update", updater.forward))

    optimizer.zero_grad()
    memory = None
    all_probe_terms = []

    # Per-chunk loss accumulators (not tied to probes).  Collected in one dict that
    # is the single source of truth for both the empty-result check and the final
    # result assembly — keyed by the result-dict name it maps to.
    pred_loss_vals: list[float] = []
    consist_loss_vals: list[float] = []
    value_preds: list[float] = []
    kv_recon_vals: list[float] = []
    adv_weight_vals: list[float] = []
    still_ce_vals: list[float] = []
    future_kv_recon_vals: list = []
    dynamics_vals: list = []
    future_horizon_kl_vals: list = []
    merged_consist_vals: list = []
    _per_chunk_accumulators = {
        "kv_reconstruction": kv_recon_vals,
        "predictive": pred_loss_vals,
        "consistency": consist_loss_vals,
        "advantage_weight": adv_weight_vals,
        "still_ce": still_ce_vals,
        "future_kv_reconstruction": future_kv_recon_vals,
        "dynamics": dynamics_vals,
        "future_horizon_kl": future_horizon_kl_vals,
        "merged_consistency": merged_consist_vals,
        "value_mean": value_preds,
    }

    _dev = next(updater.parameters()).device
    _dtype = next(updater.parameters()).dtype
    chunk_loss = torch.zeros([], device=_dev)
    n_since_detach = 0
    _use_update = hasattr(updater, 'update')
    prev_chunk_keys = None
    prev_chunk_values = None

    for chunk_idx, (chunk_keys, chunk_values) in enumerate(trajectory.chunks):
        chunk_keys = [k.to(_dev, dtype=_dtype) for k in chunk_keys]
        chunk_values = [v.to(_dev, dtype=_dtype) for v in chunk_values]

        # Optional chunk features z_t for gate conditioning (offline path).
        z_t = None
        if feature_extractor is not None:
            current_slots = 0 if memory is None else feature_extractor.memory_budget
            feat_vec = feature_extractor.batch_extract(chunk_keys, chunk_idx, current_slots)
            _B = chunk_keys[0].shape[0]
            z_t = feat_vec.unsqueeze(0).expand(_B, -1).to(chunk_keys[0].device)

        memory_before_update = None
        preds = None
        if memory is None:
            # chunk 0: bootstrap from zero prior; get predictions to train the
            # prediction head toward the data mean even on the first step.
            if _use_update:
                _kw = {"features": z_t} if (_init_has_features and z_t is not None) else {}
                memory, preds = updater.initial_compress(
                    chunk_keys, chunk_values, return_preds=True, **_kw)
            else:
                _kw = {"features": z_t} if (_init_has_features and z_t is not None) else {}
                memory = updater.initial_compress(chunk_keys, chunk_values, **_kw)

            if preds is not None and cfg.loss_weights.predictive > 0.0:
                pred_l = predictive_coding_loss(preds, chunk_keys)
                chunk_loss = chunk_loss + cfg.loss_weights.predictive * pred_l
                pred_loss_vals.append(pred_l.item())
        else:
            # chunk > 0: save reference belief before update for consistency loss
            memory_before_update = memory
            memory_prev = memory.detach() if cfg.loss_weights.consistency > 0.0 else None

            if _use_update:
                _kw = {"features": z_t} if (_fwd_has_features and z_t is not None) else {}
                memory, gates, preds = updater.update(memory, chunk_keys, chunk_values, **_kw)
            else:
                gates = None
                _kw = {"features": z_t} if (_fwd_has_features and z_t is not None) else {}
                memory = updater(memory, chunk_keys, chunk_values, **_kw)

            # Predictive loss: each slot predicts R_t from M_{t-1}
            if preds is not None and cfg.loss_weights.predictive > 0.0:
                pred_l = predictive_coding_loss(preds, chunk_keys)
                chunk_loss = chunk_loss + cfg.loss_weights.predictive * pred_l
                pred_loss_vals.append(pred_l.item())

            # Gate-weighted consistency: only penalise protected slots (g≈0) for changing
            if memory_prev is not None and cfg.loss_weights.consistency > 0.0:
                consist_l = consistency_loss(memory_prev, memory, gates=gates)
                chunk_loss = chunk_loss + cfg.loss_weights.consistency * consist_l
                consist_loss_vals.append(consist_l.item())

        # Optional belief-state value prediction (logging only; offline path).
        if value_head is not None:
            with torch.no_grad():
                v_pred = value_head(memory, z_t)
            value_preds.append(float(v_pred.mean().item()))

        compact_keys_list = [memory.keys[l] for l in range(memory.n_layers)]
        compact_vals_list = [memory.values[l] for l in range(memory.n_layers)]
        compact_kv = [(compact_keys_list[l], compact_vals_list[l]) for l in range(memory.n_layers)]

        # NextLat dynamics-prediction loss (predict M_{t+1} via the dynamics head)
        dyn_pred = None
        if (cfg.loss_weights.dynamics > 0.0
                and hasattr(updater, 'predict_next_memory')
                and memory_before_update is not None):
            dyn_pred = updater.predict_next_memory(memory_before_update)
        if dyn_pred is not None:
            dyn_l = dynamics_prediction_loss(
                dyn_pred[0], dyn_pred[1],
                [memory.keys[l] for l in range(memory.n_layers)],
                [memory.values[l] for l in range(memory.n_layers)],
            )
            chunk_loss = chunk_loss + cfg.loss_weights.dynamics * dyn_l
            dynamics_vals.append(dyn_l.item())

        # NextLat future-KV-reconstruction loss (old memory answers future-chunk queries)
        if (cfg.loss_weights.future_kv_reconstruction > 0.0
                and chunk_idx > 0
                and memory_before_update is not None):
            fkv_l = future_kv_reconstruction_loss(
                [memory_before_update.keys[l] for l in range(memory_before_update.n_layers)],
                [memory_before_update.values[l] for l in range(memory_before_update.n_layers)],
                chunk_keys, chunk_values,
            )
            chunk_loss = chunk_loss + cfg.loss_weights.future_kv_reconstruction * fkv_l
            future_kv_recon_vals.append(fkv_l.item())

        # KV reconstruction loss — applies even without a live student model.
        # Advantage weighting: use teacher_logprob_return (STILL mode) or
        # rollout_return (VD mode) to scale the gradient per trajectory.
        if cfg.loss_weights.kv_reconstruction > 0.0:
            kv_recon_l = kv_reconstruction_loss(
                compact_keys_list, compact_vals_list, chunk_keys, chunk_values,
            )
            kv_recon_w = 1.0
            if cfg.vd_cfg is not None:
                # STILL mode: weight by teacher confidence (log-prob of full-KV model).
                # VD mode:    weight by task reward (rollout_return).
                _vd_return = (
                    trajectory.teacher_logprob_return
                    if (getattr(cfg, "use_teacher_logprob_weight", False)
                        and trajectory.teacher_logprob_return is not None)
                    else trajectory.rollout_return
                )
                if _vd_return is not None:
                    _vd_probe = TrainingProbe(
                        query_tokens=torch.zeros(1, dtype=torch.long),
                        advantage=float(_vd_return),
                    )
                    kv_recon_w = _probe_weight(_vd_probe, cfg.vd_cfg, _vd_return)
                    adv_weight_vals.append(kv_recon_w)
            # Future-accuracy weighting of kv_reconstruction
            if getattr(cfg, 'use_future_accuracy_weight', False) and future_kv_recon_vals and kv_recon_vals:
                _last_future = future_kv_recon_vals[-1]
                _last_current = kv_recon_vals[-1]
                kv_recon_w = kv_recon_w * float(min(4.0, max(0.5, _last_future / (_last_current + 1e-6))))
            chunk_loss = chunk_loss + cfg.loss_weights.kv_reconstruction * kv_recon_l * kv_recon_w
            kv_recon_vals.append(kv_recon_l.item())

        if student_fn is not None:
            for probe in trajectory.probes_at(chunk_idx):
                if getattr(cfg, "use_teacher_kl", False):
                    # True STILL paper objective: CE(model(response | compact_kv), response_tokens).
                    # student_fn returns (B, S, vocab) logits; shift to get next-token prediction.
                    _logits = student_fn(probe.query_tokens, compact_kv)  # (B, S, V)
                    _B, _S, _V = _logits.shape
                    if _S > 1:
                        _labels = probe.query_tokens[:, 1:].to(_logits.device).reshape(-1)
                        _ce = torch.nn.functional.cross_entropy(
                            _logits[:, :-1].reshape(-1, _V), _labels, ignore_index=-100
                        )
                    else:
                        _ce = torch.zeros([], device=_logits.device)
                    _ce_w = 1.0
                    if probe.advantage is not None and cfg.vd_cfg is not None:
                        _ce_w = _probe_weight(probe, cfg.vd_cfg, trajectory.rollout_return)
                        adv_weight_vals.append(_ce_w)
                    chunk_loss = chunk_loss + _ce * _ce_w
                    still_ce_vals.append(_ce.item())
                    all_probe_terms.append(None)
                else:
                    _student_logits = student_fn(probe.query_tokens, compact_kv)
                    terms = _compute_probe_terms(cfg, probe, _student_logits)
                    probe_total = terms.total
                    if probe.advantage is not None and cfg.vd_cfg is not None:
                        w = _probe_weight(probe, cfg.vd_cfg, trajectory.rollout_return)
                        probe_total = probe_total * w
                        adv_weight_vals.append(w)
                    chunk_loss = chunk_loss + probe_total
                    all_probe_terms.append(terms)
                    # Future-horizon KL (position-weighted distillation)
                    if (cfg.loss_weights.future_horizon_kl > 0.0
                            and probe.teacher_logits is not None
                            and getattr(cfg, 'future_horizon_gamma', 1.0) < 1.0):
                        fh_l = future_horizon_kl_loss(
                            probe.teacher_logits, _student_logits,
                            temperature=cfg.temperature,
                            gamma=cfg.future_horizon_gamma,
                        )
                        chunk_loss = chunk_loss + cfg.loss_weights.future_horizon_kl * fh_l
                        future_horizon_kl_vals.append(fh_l.item())

        # Merged-chunk consistency (path-independence of compression)
        if (getattr(cfg, 'merged_chunk_prob', 0.0) > 0.0
                and chunk_idx > 0
                and prev_chunk_keys is not None
                and _use_update):
            import random as _rand
            if _rand.random() < cfg.merged_chunk_prob:
                merged_keys = [torch.cat([pk, ck], dim=1) for pk, ck in zip(prev_chunk_keys, chunk_keys)]
                merged_vals = [torch.cat([pv, cv], dim=1) for pv, cv in zip(prev_chunk_values, chunk_values)]
                with torch.no_grad():
                    mem_merged = updater.initial_compress(merged_keys, merged_vals)
                merged_ck = [mem_merged.keys[l] for l in range(mem_merged.n_layers)]
                merged_cv = [mem_merged.values[l] for l in range(mem_merged.n_layers)]
                mc_l = future_kv_reconstruction_loss(
                    compact_keys_list, compact_vals_list,
                    merged_ck, merged_cv,
                )
                chunk_loss = chunk_loss + cfg.loss_weights.consistency * mc_l
                merged_consist_vals.append(mc_l.item())

        prev_chunk_keys = [k.detach() for k in chunk_keys]
        prev_chunk_values = [v.detach() for v in chunk_values]

        n_since_detach += 1
        if n_since_detach >= cfg.truncated_bptt_steps:
            if chunk_loss.requires_grad:
                chunk_loss.backward()
                if cfg.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(list(updater.parameters()), cfg.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            chunk_loss = torch.zeros([], device=_dev)
            memory = memory.detach()
            n_since_detach = 0

    # Final partial BPTT window: flush any chunks accumulated since the last
    # detach/step.  With chunk counts < truncated_bptt_steps (the common case),
    # this is the ONLY backward/step for the trajectory.
    if n_since_detach > 0 and chunk_loss.requires_grad:
        chunk_loss.backward()
        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(updater.parameters()), cfg.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    if not all_probe_terms and not any(_per_chunk_accumulators.values()):
        return {}

    # Probe-distillation averages (offline path): teacher_kl / future_kl / weighted_kl
    # / retrieval / path_consistency / task / total.  all_probe_terms may contain None
    # placeholders from the teacher-KL (still_ce) branch, so filter them out.
    _real_probe_terms = [t for t in all_probe_terms if t is not None]
    result = (_average_compactor_terms(_real_probe_terms).as_dict(include_total=True)
              if _real_probe_terms else {})

    # Per-chunk term averages — overlaid on top (these win over probe-derived
    # duplicates such as consistency, matching the prior offline behaviour).
    for _name, _vals in _per_chunk_accumulators.items():
        if _vals:
            result[_name] = sum(_vals) / len(_vals)

    return result



class _CompactorOptimizer:
    """Thin adapter around Megatron's Float16OptimizerWithFloat16Params.

    The compactor trains in BF16 with FP32 master weights exactly like the main
    Megatron model (and like the trainable projector in a frozen-LLM VLM such as
    LLaVA): the inner Adam steps the FP32 masters, gradients are copied BF16→FP32
    before the step and the masters copied FP32→BF16 after.  We reuse Megatron's
    own optimizer rather than re-deriving the mixed-precision bookkeeping.

    Why construct Float16OptimizerWithFloat16Params directly instead of via the
    get_megatron_optimizer() factory (the VLM flow)?  The factory requires
    DDP-wrapped model chunks — it reads model_chunks[0].ddp_config and builds grad
    buffers + a DP-sharded distributed optimizer (optimizer/__init__.py:713).  The
    VLM works because its whole model is DistributedDataParallel-wrapped.  The
    compactor is a standalone nn.Module, replicated across TP (not DP-sharded), and
    trained with truncated BPTT (multiple backward() per trajectory) — DDP grad
    buckets would be wrong/heavy here.  For the bf16 case the factory ultimately
    instantiates THIS SAME class, so constructing it directly is the correct
    Megatron-native optimizer for a non-DDP replicated module.  Used by both the
    online RL loop and the offline trainer (scripts/train_still.py).

    Standalone — NOT a param group inside the LLM optimizer.  In joint training
    the LLM master params still hold their previous-iteration gradient when the
    compactor steps (the LLM zero_grad runs later, in train_step), so sharing the
    optimizer would re-step and corrupt the LLM.  A dedicated optimizer keeps the
    two independent.  (A VLM achieves the analogous separation by freezing the LLM
    via requires_grad=False so the shared optimizer skips it; our LLM runs
    inference-only with no optimizer at all, so the compactor gets its own.)

    The compactor is replicated across TP ranks (not sharded), so Megatron's
    grad-norm / zero-count collectives are disabled (clip_grad=0,
    log_num_zeros_in_grad=False) to avoid cross-rank all-reduces; gradient
    clipping is applied to the BF16 grads in the training loop before step().
    No loss scaling (bf16 has fp32 exponent range).

    Checkpointing uses Megatron's state_dict()/load_state_dict(), which serialise
    the FP32 masters (`fp32_from_fp16_params`, the authoritative weights) plus the
    inner Adam moments — so resume is full precision, not reconstructed from BF16.
    """

    def __init__(self, compactor, lr):
        from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
        from megatron.core.optimizer.optimizer_config import OptimizerConfig

        # Param group carries the identifier keys Megatron's optimizer uses to
        # match groups across save/load (wd_mult, lr_mult, is_decoupled_lr,
        # is_expert_parallel); a bare AdamW group omits them and load_state_dict's
        # _filter_and_reorder_param_groups raises KeyError 'pre_wd_mult'.
        base = torch.optim.AdamW(
            [{
                'params': list(compactor.parameters()),
                'wd_mult': 1.0,
                'lr_mult': 1.0,
                'is_decoupled_lr': False,
                'is_expert_parallel': False,
            }],
            lr=lr, betas=(0.9, 0.999), eps=1e-8)
        cfg = OptimizerConfig(
            bf16=True,
            fp16=False,
            clip_grad=0.0,                # clipped manually in the loop; avoids TP all-reduce
            log_num_zeros_in_grad=False,  # avoids the count-zeros all-reduce
            params_dtype=torch.float32,
            barrier_with_L1_time=False,
            timers=None,
        )

        def _init_state_fn(opt, config=None):
            # Pre-create Adam moments so optimizer state exists before a load.
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state.get(p, {})) == 0:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

        # grad_scaler=None: bf16 with no loss scale (Megatron's documented case).
        self._mopt = Float16OptimizerWithFloat16Params(
            optimizer=base, config=cfg, grad_scaler=None, init_state_fn=_init_state_fn)
        self._compactor = compactor

    # --- duck-type interface used by train_compactor_trajectory / maybe_train_compactor ---

    @property
    def state(self):
        return self._mopt.optimizer.state  # inner Adam state, keyed by FP32 masters

    def zero_grad(self):
        self._mopt.zero_grad()

    def step(self):
        # Float16 optimizer: copy BF16 grads → FP32 master grads, step inner Adam,
        # copy FP32 masters → BF16 params.  Returns (success, grad_norm, n_zeros);
        # the caller does not need it.
        self._mopt.step()

    # --- checkpoint support (plain torch.save; FP32 masters included) ---

    def state_dict(self):
        return self._mopt.state_dict()

    def load_state_dict(self, sd):
        self._mopt.load_state_dict(sd)
        # Masters are authoritative; push the restored FP32 values into BF16 params.
        self._mopt._copy_main_params_to_model_params()



class SinglePassCompactorTrainer:
    """Single-pass Still training (section 16 stage 1 of design doc).

    The PerceiverCompactor is trained to compress a full KV cache into C
    synthetic tokens that preserve the frozen model's output distribution.

    Usage:
        trainer = SinglePassCompactorTrainer(compactor, optimizer, CompactorTrainerConfig())
        terms = trainer.train_step(keys, values, probes, student_fn)
        print(terms.as_dict(include_total=True))  # {"total": ..., "teacher_kl": ..., ...}
    """

    def __init__(self, compactor, optimizer, config) -> None:
        self.compactor = compactor
        self.optimizer = optimizer
        self.config = config

    def train_step(self, keys_per_layer, values_per_layer, probes, student_fn):
        """One training step: compress → evaluate probes → backward → step.

        When ``student_fn`` is None, logit-based losses are skipped and
        ``cfg.loss_weights.kv_reconstruction`` must be > 0 to produce gradient.
        """
        from megatron.rl.compaction.learned.training.losses import CompactorLosses

        cfg = self.config
        self.optimizer.zero_grad()

        ck_list, cv_list = self.compactor.compress_all_layers(keys_per_layer, values_per_layer)
        compact_kv = [(ck_list[l], cv_list[l]) for l in range(len(ck_list))]

        probe_terms = []
        if student_fn is not None and probes:
            for probe in probes:
                student_logits = student_fn(probe.query_tokens, compact_kv)
                terms = _compute_probe_terms(cfg, probe, student_logits)
                probe_terms.append(terms)
        else:
            # No live model: use KV reconstruction loss.
            terms = CompactorLosses.compute(
                weights=cfg.loss_weights,
                compact_keys=ck_list,
                compact_values=cv_list,
                full_keys=keys_per_layer,
                full_values=values_per_layer,
            )
            probe_terms.append(terms)

        avg = _average_compactor_terms(probe_terms)
        if not avg.total.requires_grad:
            return avg
        avg.total.backward()

        if cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.compactor.parameters(), cfg.clip_grad_norm)

        self.optimizer.step()
        return avg

    def eval_step(self, keys_per_layer, values_per_layer, probes, student_fn):
        """Validation step — no gradient, no optimizer."""
        from megatron.rl.compaction.learned.training.losses import CompactorLosses

        cfg = self.config
        with torch.no_grad():
            ck_list, cv_list = self.compactor.compress_all_layers(keys_per_layer, values_per_layer)
            compact_kv = [(ck_list[l], cv_list[l]) for l in range(len(ck_list))]
            probe_terms = []
            for probe in probes:
                student_logits = student_fn(probe.query_tokens, compact_kv)
                terms = _compute_probe_terms(cfg, probe, student_logits)
                probe_terms.append(terms)
        return _average_compactor_terms(probe_terms)


class BeliefCompactorTrainer:
    """Recurrent Belief-Still training (design doc section 14).

    Processes a Trajectory chunk by chunk, maintaining a fixed-size BeliefMemory
    M_t that is updated by the updater at every chunk.  Loss is accumulated from
    all probes across all chunks and backpropagated every
    config.truncated_bptt_steps chunks.

    Usage:
        trainer = BeliefCompactorTrainer(updater, optimizer, CompactorTrainerConfig(),
                                     scheduler=CurriculumScheduler.default_4stage())
        log = trainer.train_trajectory(trajectory, student_fn)
    """

    def __init__(
        self,
        updater,
        optimizer,
        config,
        feature_extractor=None,
        value_head=None,
        scheduler=None,
    ) -> None:
        self.updater = updater
        self.optimizer = optimizer
        self.config = config
        self.feature_extractor = feature_extractor
        self.value_head = value_head
        self.scheduler = scheduler

    @torch.enable_grad()
    def train_trajectory(self, trajectory, student_fn) -> dict:
        """Train on one full trajectory — delegates to the unified
        ``train_compactor_trajectory`` (the single STILL training path), passing this
        trainer's feature_extractor and value_head.  Offline-only objectives
        (probe distillation: teacher_kl / future_kl / weighted_kl / retrieval /
        future_horizon_kl) activate automatically because offline probes carry
        teacher_logits and student_fn is provided; the online-introduced NextLat
        objectives (dynamics / future_kv_reconstruction / merged_chunk / future
        accuracy weighting) are likewise available here for free.

        After the trajectory, advances the optional curriculum scheduler (which the
        online path does not use).
        """
        import dataclasses as _dc

        result = train_compactor_trajectory(
            updater=self.updater,
            optimizer=self.optimizer,
            trajectory=trajectory,
            student_fn=student_fn,
            cfg=self.config,
            feature_extractor=self.feature_extractor,
            value_head=self.value_head,
        )

        # Curriculum: advance once per trajectory and swap in the new loss weights.
        if self.scheduler is not None:
            new_weights = self.scheduler.step()
            self.config = _dc.replace(self.config, loss_weights=new_weights)

        return result


def _compute_probe_terms(cfg, probe, student_logits):
    """Run CompactorLosses.compute for one probe (the distillation loss terms).

    Centralises the is_exact_retrieval → target_ids/retrieval_ids routing so the
    probe call sites (SinglePassCompactorTrainer.train_step/eval_step and the unified
    train_compactor_trajectory) cannot drift apart.
    """
    from megatron.rl.compaction.learned.training.losses import CompactorLosses
    return CompactorLosses.compute(
        weights=cfg.loss_weights,
        full_logits=probe.teacher_logits,
        compact_logits=student_logits,
        temperature=cfg.temperature,
        weighted_kl_rho=cfg.weighted_kl_rho,
        target_ids=probe.answer_tokens if not probe.is_exact_retrieval else None,
        retrieval_ids=probe.answer_tokens if probe.is_exact_retrieval else None,
    )


def _average_compactor_terms(terms_list):
    """Average a list of CompactorLossTerms into one. Helper for SinglePassCompactorTrainer."""
    from megatron.rl.compaction.learned.training.losses import CompactorLossTerms

    n = len(terms_list)
    assert n > 0, "No loss terms to average"

    def _avg(attr):
        vals = [getattr(t, attr) for t in terms_list if getattr(t, attr) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    return CompactorLossTerms(
        teacher_kl=_avg("teacher_kl"),
        future_kl=_avg("future_kl"),
        consistency=_avg("consistency"),
        task=_avg("task"),
        retrieval=_avg("retrieval"),
        weighted_kl=_avg("weighted_kl"),
        path_consistency=_avg("path_consistency"),
        predictive=_avg("predictive"),
        kv_reconstruction=_avg("kv_reconstruction"),
        future_kv_reconstruction=_avg("future_kv_reconstruction"),
        dynamics=_avg("dynamics"),
        future_horizon_kl=_avg("future_horizon_kl"),
        total=sum(t.total for t in terms_list) / n,
    )


def still_train_step(
    compactor,          # PerceiverCompactor
    optimizer,          # torch.optim.Optimizer for the compactor
    keys_per_layer,     # list[Tensor] — (B, T, d) per layer
    values_per_layer,   # list[Tensor]
    probes,             # list[TrainingProbe]
    student_fn,         # StudentFn callable, or None for reconstruction-only training
    cfg,                # CompactorTrainerConfig
) -> dict:
    """Single-chunk training step for a single-pass PerceiverCompactor (stateless).

    When ``student_fn`` is None the logit-based losses (teacher_kl, etc.) are
    skipped and ``cfg.loss_weights.kv_reconstruction`` must be > 0 to produce any
    gradient. The predictive / consistency losses are trajectory-level and are
    never computed here regardless.
    """
    from megatron.rl.compaction.learned.training.losses import CompactorLosses
    import warnings

    if cfg.loss_weights.consistency > 0.0 or cfg.loss_weights.predictive > 0.0:
        warnings.warn(
            "still_train_step received loss_weights with consistency or predictive > 0, "
            "but these losses require a trajectory and are not computed here. "
            "Use train_compactor_trajectory with GatedRecurrentUpdater instead.",
            stacklevel=2,
        )

    optimizer.zero_grad()
    ck_list, cv_list = compactor.compress_all_layers(keys_per_layer, values_per_layer)
    compact_kv = [(ck_list[l], cv_list[l]) for l in range(len(ck_list))]

    terms_list = []

    if student_fn is not None and probes:
        for probe in probes:
            student_logits = student_fn(probe.query_tokens, compact_kv)
            terms = _compute_probe_terms(cfg, probe, student_logits)
            terms_list.append(terms)
    else:
        # No live model — compute kv_reconstruction loss directly from KV tensors.
        terms = CompactorLosses.compute(
            weights=cfg.loss_weights,
            compact_keys=ck_list,
            compact_values=cv_list,
            full_keys=keys_per_layer,
            full_values=values_per_layer,
        )
        terms_list.append(terms)

    if not terms_list:
        return {}

    n = len(terms_list)
    avg_total = sum(t.total for t in terms_list) / n
    if not avg_total.requires_grad:
        return {}
    avg_total.backward()

    if cfg.clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(compactor.parameters(), cfg.clip_grad_norm)
    optimizer.step()

    def _avg(attr):
        vals = [getattr(t, attr) for t in terms_list if getattr(t, attr) is not None]
        return float(sum(v.item() if hasattr(v, 'item') else v for v in vals) / len(vals)) if vals else None

    return {k: v for k, v in (
        (k, _avg(k)) for k in ("teacher_kl", "kv_reconstruction", "future_kl", "consistency", "task")
    ) if v is not None}
