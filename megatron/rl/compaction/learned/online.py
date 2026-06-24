# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Online compactor glue for the Megatron RL loop.

These functions are the bridge between the RL rollout loop (megatron/rl/rl_utils.py)
and the compactor training core (compaction/learned/training/training.py). They
live in the compaction package so the logic is owned here; rl_utils only calls
them at the loop sites, passing its ``runtime_state`` (an ``RLRuntimeState``).

``runtime_state`` is duck-typed — these functions read/write the compactor_*
fields on it:
    compactor, compactor_ddp, compactor_optimizer, compactor_cfg,
    compactor_trajectories, compactor_raw_sequences, compactor_step_offset,
    compactor_student_model, _compactor_lr, _compactor_ckpt_path,
    _compactor_pg_collection
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from megatron.core.utils import log_single_rank
from megatron.rl.compaction.learned.training.training import train_compactor_trajectory
from megatron.rl.compaction.learned.training.parallel import (
    build_compactor_pg_collection,
    wrap_compactor_for_training,
)

logger = logging.getLogger(__name__)


def init_compactor_from_kv(runtime_state: Any, args, n_attn_layers: int, d_kv: int) -> None:
    """Initialize the online compactor from known K/V dimensions.

    Called lazily once dims are known — either from a live kv_hook or from the
    first forward-pass K/V capture. No-op if already initialized.
    """
    if runtime_state.compactor is not None:
        return

    from megatron.rl.compaction.learned import GatedUpdaterConfig, GatedRecurrentUpdater
    from megatron.rl.compaction.learned.training.data import CompactorTrainerConfig
    from megatron.rl.compaction.learned.training.value_directed import ValueDirectedConfig
    from megatron.rl.compaction.learned.training.losses import CompactorLossWeights

    n_compress = getattr(args, "rl_compaction_n_compress", 64)
    n_heads = 8
    updater_cfg = GatedUpdaterConfig(
        n_compress=n_compress,
        n_heads=n_heads,
        d_kv=d_kv,
        n_attn_layers=n_attn_layers,
        use_dynamics_head=getattr(args, "rl_compaction_compactor_use_dynamics_head", False),
    )
    _dtype = (torch.bfloat16 if getattr(args, 'bf16', False)
              else torch.float16 if getattr(args, 'fp16', False)
              else torch.float32)

    # The compactor is replicated on every rank, data-parallel across the world.
    # build_compactor_pg_collection() gives it a singleton tensor-parallel group
    # (so the Megatron sub-modules build replicated, not sharded across the real TP
    # group) and a world data-parallel group (used by DDP at optimizer-attach time).
    pg_collection = build_compactor_pg_collection()
    runtime_state._compactor_pg_collection = pg_collection
    compactor = GatedRecurrentUpdater(
        updater_cfg, params_dtype=_dtype, pg_collection=pg_collection,
    ).cuda().to(_dtype)

    # Load compactor weights from checkpoint if provided (replicated across all ranks).
    _ckpt_path = getattr(args, "rl_compaction_compactor_checkpoint", None)
    _loaded_step = None
    _loaded_ckpt_payload = None
    if _ckpt_path:
        from megatron.rl.compaction.learned import load_checkpoint as _load_ckpt
        try:
            # dist_checkpointing builds TE layers (CUDA-only) and load() is collective —
            # all ranks reach here together via build_compactor_trajectories.
            compactor, _meta = _load_ckpt(
                _ckpt_path, map_location="cuda", params_dtype=_dtype, pg_collection=pg_collection,
            )
            compactor = compactor.to(_dtype).train()
            _loaded_step = getattr(_meta, "step", None)
            _loaded_ckpt_payload = _ckpt_path  # stash path for optimizer state restore
            log_single_rank(logger, logging.INFO,
                            f"[STILL online] Loaded compactor checkpoint: {_ckpt_path} "
                            f"(step={_loaded_step})")
        except Exception as _e:
            log_single_rank(logger, logging.WARNING,
                            f"[STILL online] Failed to load compactor checkpoint {_ckpt_path}: {_e}")

    # Optimizer is created lazily in attach_compactor_optimizer once the Megatron
    # optimizer is available (first maybe_train_compactor call).
    runtime_state._compactor_lr = getattr(args, "rl_compaction_compactor_lr", 3e-4)
    runtime_state._compactor_ckpt_path = _loaded_ckpt_payload

    _use_teacher_kl = getattr(args, "rl_compaction_compactor_teacher_kl", False)
    trainer_cfg = CompactorTrainerConfig(
        loss_weights=CompactorLossWeights(
            kv_reconstruction=0.0 if _use_teacher_kl else 1.0,
            dynamics=getattr(args, "rl_compaction_compactor_dynamics", 0.0),
            future_kv_reconstruction=getattr(args, "rl_compaction_compactor_future_kv_reconstruction", 0.0),
            future_horizon_kl=getattr(args, "rl_compaction_compactor_future_horizon_kl", 0.0),
        ),
        vd_cfg=ValueDirectedConfig(
            advantage_clip=getattr(args, "rl_compaction_compactor_advantage_clip", 5.0),
            min_weight=getattr(args, "rl_compaction_compactor_advantage_min_weight", 0.1),
        ),
        use_teacher_logprob_weight=getattr(args, "rl_compaction_compactor_use_teacher_logprob", False),
        use_teacher_kl=_use_teacher_kl,
        future_horizon_gamma=getattr(args, "rl_compaction_compactor_future_horizon_gamma", 1.0),
        use_future_accuracy_weight=getattr(args, "rl_compaction_compactor_use_future_accuracy_weight", False),
        merged_chunk_prob=getattr(args, "rl_compaction_compactor_merged_chunk_prob", 0.0),
    )
    # Warn on loss weights that are silently inert online (term enabled but its
    # precondition is off, so it contributes no gradient).
    _w = trainer_cfg.loss_weights
    if _w.dynamics > 0.0 and not getattr(args, "rl_compaction_compactor_use_dynamics_head", False):
        log_single_rank(logger, logging.WARNING,
                        "[STILL online] dynamics loss > 0 but --use-dynamics-head is off; it is skipped.")
    if _w.future_horizon_kl > 0.0 and (trainer_cfg.future_horizon_gamma >= 1.0 or _use_teacher_kl):
        log_single_rank(logger, logging.WARNING,
                        "[STILL online] future-horizon-KL > 0 is inert online (needs gamma < 1.0 and "
                        "per-probe teacher_logits, which only the offline pipeline captures).")

    runtime_state.compactor = compactor
    runtime_state.compactor_optimizer = None  # set by attach_compactor_optimizer
    runtime_state.compactor_cfg = trainer_cfg
    runtime_state.compactor_step_offset = _loaded_step if _loaded_step is not None else 0
    _ckpt_suffix = f", resumed from step={_loaded_step}" if _loaded_step else ""
    log_single_rank(logger, logging.INFO,
                    f"[STILL online] Compactor initialized via K/V capture: "
                    f"n_layers={n_attn_layers} d_kv={d_kv} n_compress={n_compress}{_ckpt_suffix}")


def attach_compactor_optimizer(runtime_state: Any, megatron_opt=None) -> None:
    """DDP-wrap the compactor and build its Megatron optimizer.

    Uses the standard Megatron machinery (``wrap_compactor_for_training`` ->
    ``DistributedDataParallel`` + ``get_megatron_optimizer``) over the world
    data-parallel group, so gradients are all-reduced across all ranks and the
    replicas stay bit-identical. The compactor has its OWN optimizer, separate from
    the LLM's — in joint training the LLM master params still hold the previous
    iteration's gradient when the compactor steps, so sharing would corrupt the LLM.

    ``megatron_opt`` (the LLM optimizer) is accepted for call-site stability but
    intentionally unused. Called once, collectively on ALL ranks, on the first
    maybe_train_compactor invocation.
    """
    del megatron_opt  # intentionally unused — see docstring
    if runtime_state.compactor_optimizer is not None:
        return
    if runtime_state.compactor is None:
        return

    ddp_model, optimizer = wrap_compactor_for_training(
        runtime_state.compactor,
        runtime_state._compactor_lr,
        pg_collection=getattr(runtime_state, "_compactor_pg_collection", None),
    )

    # Restore FP32 masters + Adam moments from checkpoint if available.
    if runtime_state._compactor_ckpt_path is not None:
        from megatron.rl.compaction.learned.training.checkpoint import load_optimizer_state
        try:
            load_optimizer_state(
                runtime_state._compactor_ckpt_path, optimizer,
                model_sharded_state_dict=runtime_state.compactor.sharded_state_dict(),
            )
            log_single_rank(logger, logging.INFO,
                            "[STILL online] Restored optimizer state "
                            "(FP32 masters + Adam moments) from checkpoint.")
        except KeyError:
            # Weights-only / old-format checkpoint.
            log_single_rank(logger, logging.WARNING,
                            "[STILL online] checkpoint has no optimizer state; "
                            "FP32 masters + Adam moments start fresh from the loaded weights.")

    runtime_state.compactor_ddp = ddp_model
    runtime_state.compactor_optimizer = optimizer
    log_single_rank(logger, logging.INFO,
                    "[STILL online] Compactor optimizer attached "
                    "(DDP over the world DP group + Megatron optimizer).")


def build_compactor_trajectories(runtime_state: Any, model, args) -> None:
    """Build compactor training trajectories via a collective forward-pass KV capture.

    Option C — called on ALL ranks. ``compactor_raw_sequences`` is identical on
    every rank (populated from the broadcast ``rollouts``), so all ranks replay the
    same sequences through one collective forward and EACH rank captures its own
    LOCAL tensor-parallel KV partition (a different subset of KV heads per rank —
    with GQA+TP, e.g. d_kv = 1*kv_channels for tp=4/num_query_groups=2; no
    all-gather). Each rank builds its own Trajectory from its local slice and trains
    on it; the DDP wrapping the compactor all-reduces gradients across the world so
    the replicas stay bit-identical. The compactor is sized from the actual captured
    local d_kv (consistent across ranks by symmetric GQA+TP replication), so it
    always matches the data exactly.

    No-op when online training is disabled. Collective: every rank loops the same
    number of sequences and runs the forward together.
    """
    from megatron.training import get_args
    args_obj = args if args is not None else get_args()
    if not getattr(args_obj, "rl_compaction_compactor_train", False):
        return

    # Training model for the teacher-KL student forward (set on all ranks).
    runtime_state.compactor_student_model = model

    raw_seqs = runtime_state.compactor_raw_sequences
    runtime_state.compactor_raw_sequences = []

    from megatron.rl.compaction.learned.capture.kv_capture import capture_kv_from_forward
    from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe

    # Lockstep count across ranks (raw_seqs is identical by construction, but guard
    # against any drift so the collective forwards below stay matched).
    n_seqs_t = torch.tensor([len(raw_seqs)], dtype=torch.long, device="cuda")
    torch.distributed.all_reduce(n_seqs_t, op=torch.distributed.ReduceOp.MIN)
    n_seqs = int(n_seqs_t.item())

    log_single_rank(logger, logging.INFO,
                    f"[STILL online] build_compactor_trajectories: n_seqs={n_seqs}")
    if n_seqs == 0:
        return

    chunk_size = getattr(args_obj, "rl_compaction_chunk_size", 256)
    # Sequence parallel requires S divisible by TP size.
    tp_size = max(1, getattr(args_obj, "tensor_model_parallel_size", 1))

    for i in range(n_seqs):
        seq_ids, reward = raw_seqs[i]
        seq_len = len(seq_ids)
        if seq_len == 0:
            continue

        # Pad to the model's full seq_length so CUDA graphs (captured at that length)
        # replay correctly, and to a TP boundary for sequence-parallel scatter/gather.
        model_seq_len = getattr(args_obj, "seq_length", 8192)
        full_len = max(model_seq_len, seq_len)
        pad_len = (tp_size - full_len % tp_size) % tp_size
        padded_len = full_len + pad_len

        token_t = torch.zeros(padded_len, dtype=torch.long, device="cuda")
        token_t[:seq_len].copy_(torch.tensor(seq_ids, dtype=torch.long))
        tokens = token_t.unsqueeze(0)
        pos_ids = torch.arange(padded_len, dtype=torch.long, device="cuda").unsqueeze(0)

        # Free any cached GPU memory so the forward pass has headroom.
        torch.cuda.empty_cache()

        # All ranks run the SAME collective forward and each captures its own local KV.
        kv_result = None
        _fwd_ok = 1
        try:
            kv_result = capture_kv_from_forward(model, tokens, pos_ids)
            torch.cuda.synchronize()
        except Exception as exc:
            log_single_rank(logger, logging.WARNING,
                            f"[STILL online] forward pass failed for seq {i}: {exc}")
            _fwd_ok = 0

        # Collective agreement: if the forward failed on ANY rank, ALL ranks skip
        # this sequence together to keep collectives matched (avoids NCCL deadlock).
        _ok_t = torch.tensor([_fwd_ok], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(_ok_t, op=torch.distributed.ReduceOp.MIN)
        if _ok_t.item() == 0 or kv_result is None:
            continue

        keys, vals = kv_result
        n_attn_layers = len(keys)
        d_kv = keys[0].shape[-1]  # local partition size, identical across ranks

        # Initialize compactor from the actual captured local d_kv (collective: the
        # checkpoint load inside is collective and all ranks reach it together).
        init_compactor_from_kv(runtime_state, args_obj, n_attn_layers, d_kv)

        # Trim padding back to the original sequence length before chunking.
        keys = [k[:, :seq_len, :] for k in keys]
        vals = [v[:, :seq_len, :] for v in vals]

        chunks = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunks.append((
                [k[:, chunk_start:chunk_end, :] for k in keys],
                [v[:, chunk_start:chunk_end, :] for v in vals],
            ))
        if not chunks:
            continue

        # Probe at the last chunk so kv_recon has a query target.
        last_idx = len(chunks) - 1
        probe = TrainingProbe(
            query_tokens=token_t[:seq_len].unsqueeze(0).cpu(),
            teacher_logits=None,
            answer_tokens=None,
            advantage=reward if reward != 0.0 else None,
        )
        runtime_state.compactor_trajectories.append(Trajectory(
            chunks=chunks,
            probes_by_chunk={last_idx: [probe]},
            rollout_return=reward,
        ))


def maybe_train_compactor(runtime_state: Any, args=None, optimizer=None) -> None:
    """Train the online compactor on trajectories accumulated this iteration.

    No-op when online training is disabled or the compactor is not yet
    initialized. Clears runtime_state.compactor_trajectories after training.

    optimizer: the Megatron MegatronOptimizer for the LLM (passed through to the
               attach step, which deliberately builds a standalone optimizer).
    """
    # Always drain the trajectory buffer to avoid unbounded GPU memory growth
    # even when the compactor is not yet ready.
    trajectories = runtime_state.compactor_trajectories
    runtime_state.compactor_trajectories = []

    if runtime_state.compactor is None:
        return

    # Attach the compactor optimizer on first call (deferred because the compactor
    # dims are only known after the first KV capture). This is COLLECTIVE — it wraps
    # the compactor in world-DP DDP — so it must run on all ranks; it does, because
    # the compactor is created collectively in build_compactor_trajectories.
    if runtime_state.compactor_optimizer is None:
        attach_compactor_optimizer(runtime_state, optimizer)

    if runtime_state.compactor_optimizer is None or runtime_state.compactor_cfg is None:
        return

    if not trajectories:
        return

    # Option C: every rank trains on its OWN local KV slice. The optimizer's step()
    # runs finalize_model_grads, which all-reduces gradients across the whole world
    # (DP+TP collapsed), so replicas stay bit-identical without any bespoke sync.
    # All ranks have the same number of trajectories (built in lockstep) and each
    # trajectory drives the same number of optimizer steps (same chunk count / BPTT
    # schedule), so the per-step finalize collectives stay matched across ranks.

    # Build student_fn for true STILL (teacher-KL) mode.
    # student_fn: (query_tokens, compact_kv) → logits (B, S, vocab)
    # The CE loss with correct next-token shift is applied inside train_compactor_trajectory.
    _student_fn = None
    if getattr(runtime_state.compactor_cfg, "use_teacher_kl", False):
        _still_model = getattr(runtime_state, "compactor_student_model", None)
        if _still_model is not None:
            from megatron.rl.compaction.learned.capture.student_forward import student_logits as _sf
            _m = _still_model  # capture for closure
            _student_fn = lambda q, kv: _sf(_m, q, kv)
        else:
            log_single_rank(logger, logging.WARNING,
                            "[STILL online] teacher-KL mode active but compactor_student_model is None; "
                            "skipping student forward this iteration.")

    _opt_state_before = len(runtime_state.compactor_optimizer.state)
    for trajectory in trajectories:
        try:
            log = train_compactor_trajectory(
                model=runtime_state.compactor_ddp,
                optimizer=runtime_state.compactor_optimizer,
                trajectory=trajectory,
                student_fn=_student_fn,
                cfg=runtime_state.compactor_cfg,
            )
            if log:
                log_single_rank(logger, logging.INFO, f"[STILL online] {log}")
        except Exception as exc:
            import traceback as _tb
            log_single_rank(logger, logging.WARNING,
                            f"[STILL online] training step failed: {exc}\n{_tb.format_exc()}")
    _opt_state_after = len(runtime_state.compactor_optimizer.state)
    log_single_rank(logger, logging.INFO,
                    f"[STILL online] opt state: {_opt_state_before} -> {_opt_state_after}, "
                    f"n_trajs={len(trajectories)}")

    # Advance the persistent global training-step counter. This is monotonic across
    # job restarts (restored from the loaded checkpoint's step and only increments),
    # unlike args.curr_iteration which is the LLM loop counter and resets to 0 every
    # process — using that could re-emit and overwrite an existing checkpoint with
    # different weights after a restart.
    runtime_state.compactor_step_offset += 1
    global_step = runtime_state.compactor_step_offset

    # Periodic checkpoint via dist_checkpointing — COLLECTIVE (all ranks call it).
    # global_step is identical across ranks, so the gate fires on all ranks together.
    # Each checkpoint is a DIRECTORY (sharded model + common optimizer/step state).
    if args is not None:
        ckpt_dir = getattr(args, "rl_compaction_compactor_checkpoint_dir", None)
        ckpt_every = getattr(args, "rl_compaction_compactor_checkpoint_every", 100)
        if ckpt_dir and global_step % ckpt_every == 0:
            import os
            from megatron.rl.compaction.learned import save_checkpoint
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:07d}")
            save_checkpoint(
                runtime_state.compactor,
                ckpt_path,
                step=global_step,
                optimizer=runtime_state.compactor_optimizer,
            )
            log_single_rank(logger, logging.INFO, f"[STILL online] checkpoint saved: {ckpt_path}")
