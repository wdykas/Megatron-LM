# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Online compactor glue for the Megatron RL loop.

These functions are the bridge between the RL rollout loop (megatron/rl/rl_utils.py)
and the compactor training core (compaction/learned/training/training.py). They
live in the compaction package so the logic is owned here; rl_utils only calls
them at the loop sites, passing its ``runtime_state`` (an ``RLRuntimeState``).

``runtime_state`` is duck-typed — these functions read/write the compactor_*
fields on it:
    compactor, compactor_optimizer, compactor_cfg, compactor_trajectories,
    compactor_raw_sequences, compactor_step_offset, compactor_student_model,
    compactor_scheduler, _compactor_lr, _compactor_ckpt_path
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from megatron.core.utils import log_single_rank
from megatron.rl.compaction.learned.training.training import (
    train_compactor_trajectory,
    _CompactorOptimizer,
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
    updater_cfg = GatedUpdaterConfig(
        n_compress=n_compress,
        n_heads=8,
        d_kv=d_kv,
        n_attn_layers=n_attn_layers,
        use_dynamics_head=getattr(args, "rl_compaction_compactor_use_dynamics_head", False),
    )
    compactor = GatedRecurrentUpdater(updater_cfg).cuda()
    _dtype = (torch.bfloat16 if getattr(args, 'bf16', False)
              else torch.float16 if getattr(args, 'fp16', False)
              else torch.float32)
    compactor = compactor.to(_dtype)

    # Load compactor weights from checkpoint if provided (replicated across TP ranks).
    _ckpt_path = getattr(args, "rl_compaction_compactor_checkpoint", None)
    _loaded_step = None
    _loaded_ckpt_payload = None
    if _ckpt_path:
        from megatron.rl.compaction.learned import load_checkpoint as _load_ckpt
        try:
            compactor, _meta = _load_ckpt(_ckpt_path, map_location="cpu")
            compactor = compactor.cuda().to(_dtype).train()
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
    # Guard against silent no-op misconfigurations: a positive loss weight whose
    # enabling flag/condition is off means the term is computed-and-logged but
    # contributes nothing (or is skipped entirely).
    if trainer_cfg.loss_weights.dynamics > 0.0 and not getattr(args, "rl_compaction_compactor_use_dynamics_head", False):
        log_single_rank(logger, logging.WARNING,
                        "[STILL online] --rl-compaction-compactor-dynamics > 0 but "
                        "--rl-compaction-compactor-use-dynamics-head is NOT set; the dynamics "
                        "loss will be silently skipped (no dynamics head to predict M_{t+1}).")
    if trainer_cfg.loss_weights.future_horizon_kl > 0.0 and trainer_cfg.future_horizon_gamma >= 1.0:
        log_single_rank(logger, logging.WARNING,
                        "[STILL online] --rl-compaction-compactor-future-horizon-kl > 0 but "
                        "--rl-compaction-compactor-future-horizon-gamma >= 1.0; the future-horizon-KL "
                        "term is gated off (gamma must be < 1.0 to upweight later positions).")
    # The probe-distillation losses (teacher_kl / future_kl / future_horizon_kl) are
    # computed only in the use_teacher_kl=False student-forward branch and require
    # per-probe teacher_logits. Online rollouts do NOT capture teacher_logits, and
    # teacher-KL mode uses the CE branch instead — so these terms are OFFLINE-only
    # today and silently inert online. Warn rather than mislead.
    if trainer_cfg.loss_weights.future_horizon_kl > 0.0 and _use_teacher_kl:
        log_single_rank(logger, logging.WARNING,
                        "[STILL online] --rl-compaction-compactor-future-horizon-kl > 0 with "
                        "--rl-compaction-compactor-teacher-kl: teacher-KL mode uses the CE branch, so "
                        "future-horizon-KL is NOT computed online (it needs per-probe teacher "
                        "logits, captured only in the offline pipeline). This term is inert here.")

    runtime_state.compactor = compactor
    runtime_state.compactor_optimizer = None  # set by attach_compactor_optimizer
    runtime_state.compactor_cfg = trainer_cfg
    runtime_state.compactor_step_offset = _loaded_step if _loaded_step is not None else 0
    _ckpt_suffix = f", resumed from step={_loaded_step}" if _loaded_step else ""
    log_single_rank(logger, logging.INFO,
                    f"[STILL online] Compactor initialized via K/V capture: "
                    f"n_layers={n_attn_layers} d_kv={d_kv} n_compress={n_compress}{_ckpt_suffix}")


def attach_compactor_optimizer(runtime_state: Any, megatron_opt=None) -> None:
    """Build the compactor's standalone Megatron mixed-precision optimizer.

    Always gives the compactor its own Float16OptimizerWithFloat16Params (BF16
    params + FP32 masters) rather than a param group inside the LLM optimizer —
    see _CompactorOptimizer for why sharing corrupts the LLM in joint training.

    ``megatron_opt`` is accepted for call-site stability but intentionally unused:
    frozen-LLM pipelines pass None (Megatron skips the optimizer under
    --skip-train --no-load-optim) and joint pipelines pass the real optimizer,
    which we deliberately do not touch.

    Called once on the first maybe_train_compactor invocation. No-op if
    compactor_optimizer is already set or the compactor is not yet built.
    """
    del megatron_opt  # intentionally unused — see docstring
    if runtime_state.compactor_optimizer is not None:
        return
    if runtime_state.compactor is None:
        return

    compactor_opt = _CompactorOptimizer(runtime_state.compactor, runtime_state._compactor_lr)

    # Restore FP32 masters + Adam moments from checkpoint if available.
    if runtime_state._compactor_ckpt_path is not None:
        from megatron.rl.compaction.learned.training.checkpoint import load_optimizer_state
        try:
            load_optimizer_state(
                runtime_state._compactor_ckpt_path, compactor_opt,
                map_location=next(runtime_state.compactor.parameters()).device,
            )
            log_single_rank(logger, logging.INFO,
                            "[STILL online] Restored optimizer state "
                            "(FP32 masters + Adam moments) from checkpoint.")
        except KeyError:
            # Weights-only / old-format checkpoint.
            log_single_rank(logger, logging.WARNING,
                            "[STILL online] checkpoint has no optimizer state; "
                            "FP32 masters + Adam moments start fresh from the loaded weights.")

    runtime_state.compactor_optimizer = compactor_opt
    log_single_rank(logger, logging.INFO,
                    "[STILL online] Compactor optimizer attached "
                    "(Megatron Float16OptimizerWithFloat16Params, standalone).")


def build_compactor_trajectories(runtime_state: Any, model, args) -> None:
    """Build trajectories via model forward-pass KV capture.

    Called on ALL TP ranks simultaneously (the forward pass is collective). Raw
    sequences saved during rollout collection are consumed; K/V is captured on
    rank 0 (GQA KV heads are replicated across TP ranks when tp_size >
    num_kv_groups, so rank 0 has the full K/V). Trajectories are appended to
    runtime_state.compactor_trajectories (rank 0 only).

    No-op when online training is disabled, sequences list is empty, or the
    HookTrajectoryCollector already produced non-empty trajectories.
    """
    from megatron.training import get_args
    args_obj = args if args is not None else get_args()
    if not getattr(args_obj, "rl_compaction_compactor_train", False):
        return

    # If paged-cache capture already populated trajectories, skip replay.
    # Broadcast rank-0's status so ALL TP ranks take the same branch.
    _has_trajs_t = torch.tensor(
        [1 if runtime_state.compactor_trajectories else 0],
        dtype=torch.long, device="cuda",
    )
    torch.distributed.broadcast(_has_trajs_t, src=0)
    if _has_trajs_t.item():
        runtime_state.compactor_raw_sequences = []
        # Ensure the compactor is initialized so maybe_train_compactor can train on
        # these paged-cache trajectories. Init from the first trajectory's KV shape.
        if torch.distributed.get_rank() == 0 and runtime_state.compactor is None and runtime_state.compactor_trajectories:
            try:
                _t0 = runtime_state.compactor_trajectories[0]
                if _t0.chunks:
                    _ck, _ = _t0.chunks[0]
                    _n_layers = len(_ck)
                    _d_kv = _ck[0].shape[-1]
                    if _n_layers > 0 and _d_kv > 0:
                        init_compactor_from_kv(runtime_state, args_obj, _n_layers, _d_kv)
            except Exception as _e:
                log_single_rank(logger, logging.WARNING,
                                f"[STILL online] compactor init from paged-cache traj failed: {_e}")
        return

    raw_seqs = runtime_state.compactor_raw_sequences
    runtime_state.compactor_raw_sequences = []

    rank = torch.distributed.get_rank()
    log_single_rank(logger, logging.INFO,
                    f"[STILL online] build_compactor_trajectories entered: "
                    f"n_raw_seqs={len(raw_seqs)} rank={rank}")

    n_seqs = len(raw_seqs) if rank == 0 else 0
    n_seqs_t = torch.tensor([n_seqs], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(n_seqs_t, src=0)
    n_seqs = int(n_seqs_t.item())

    log_single_rank(logger, logging.INFO,
                    f"[STILL online] n_seqs after broadcast: {n_seqs}")

    if n_seqs == 0:
        return

    from megatron.rl.compaction.learned.capture.kv_capture import capture_kv_from_forward, _unwrap_model
    from megatron.rl.compaction.learned.training.data import Trajectory, TrainingProbe

    # Initialize compactor from model structure NOW (before any forward) so it is
    # always available for maybe_train_compactor even if KV capture later fails/hangs.
    if rank == 0 and runtime_state.compactor is None:
        try:
            _gpt_tmp = _unwrap_model(model)
            # Use the SAME predicate as kv_capture._attn_core_modules (which defines
            # the captured KV layer count) — in a Mamba+attention hybrid, counting
            # bare `self_attention` can disagree with the layers that actually expose
            # core_attention, producing a compactor sized for the wrong layer count.
            _n_attn = sum(
                1 for _l in _gpt_tmp.decoder.layers
                if hasattr(_l, "self_attention") and hasattr(_l.self_attention, "core_attention")
            )
            _d_kv = (
                getattr(args_obj, "num_query_groups", 1)
                * getattr(args_obj, "kv_channels", 128)
            )
            if _n_attn > 0 and _d_kv > 0:
                init_compactor_from_kv(runtime_state, args_obj, _n_attn, _d_kv)
        except Exception as _e:
            log_single_rank(logger, logging.WARNING,
                            f"[STILL online] early compactor init failed: {_e}")

    chunk_size = getattr(args_obj, "rl_compaction_chunk_size", 256)

    # Sequence parallel requires S divisible by TP size.
    tp_size = max(1, getattr(args_obj, "tensor_model_parallel_size", 1))

    for i in range(n_seqs):
        # Broadcast sequence length so all ranks can allocate the buffer.
        if rank == 0:
            seq_ids, reward = raw_seqs[i]
        else:
            seq_ids, reward = [], 0.0

        seq_len = len(seq_ids) if rank == 0 else 0
        seq_len_t = torch.tensor([seq_len], dtype=torch.long, device="cuda")
        torch.distributed.broadcast(seq_len_t, src=0)
        seq_len = int(seq_len_t.item())
        if seq_len == 0:
            continue

        # Pad to the model's full seq_length so CUDA graphs (captured at that length)
        # replay correctly. Padding to only TP-boundary would give a different input
        # shape than what the graph was captured with, causing rank 0 to fail before
        # the first NCCL collective and deadlocking the other TP ranks.
        model_seq_len = getattr(args_obj, "seq_length", 8192)
        # Also ensure divisibility by TP for sequence-parallel scatter/gather.
        full_len = max(model_seq_len, seq_len)
        pad_len = (tp_size - full_len % tp_size) % tp_size
        padded_len = full_len + pad_len

        token_t = torch.zeros(padded_len, dtype=torch.long, device="cuda")
        if rank == 0:
            token_t[:seq_len].copy_(torch.tensor(seq_ids, dtype=torch.long))
        torch.distributed.broadcast(token_t, src=0)
        tokens = token_t.unsqueeze(0)  # (1, padded_len=model_seq_len)
        pos_ids = torch.arange(padded_len, dtype=torch.long, device="cuda").unsqueeze(0)

        log_single_rank(logger, logging.INFO,
                        f"[STILL online] forward pass KV capture: seq {i+1}/{n_seqs}, "
                        f"seq_len={seq_len} padded_to={padded_len}")

        # Free any cached GPU memory so the forward pass has headroom.
        torch.cuda.empty_cache()

        # All ranks run forward collectively; rank 0 captures K/V via hooks.
        # Build PackedSeqParams matching get_logprobs() to avoid CUDA-graph mismatch.
        try:
            from megatron.core.packed_seq_params import PackedSeqParams as _PSP
            _S = tokens.shape[1]
            _cu = torch.tensor([0, _S], dtype=torch.int32, device="cuda")
            _packed_seq_params = _PSP(
                qkv_format='thd',
                cu_seqlens_q=_cu,
                cu_seqlens_kv=_cu,
                max_seqlen_q=_S,
                max_seqlen_kv=_S,
                total_tokens=_S,
            )
        except Exception:
            _packed_seq_params = None

        kv_result = None
        _fwd_ok = 1
        try:
            if rank == 0:
                kv_result = capture_kv_from_forward(model, tokens, pos_ids)
            else:
                # Non-rank-0: use identical forward path as rank 0 to ensure all
                # TP ranks hit the same NCCL collectives.
                gpt = _unwrap_model(model)
                _flash_decode = getattr(gpt.config, 'flash_decode', False)
                gpt.config.flash_decode = False
                _was_training = gpt.training
                gpt.eval()
                try:
                    with torch.no_grad():
                        gpt(
                            input_ids=tokens,
                            position_ids=pos_ids,
                            attention_mask=None,
                            packed_seq_params=_packed_seq_params,
                            runtime_gather_output=True,
                        )
                finally:
                    gpt.config.flash_decode = _flash_decode
                    if _was_training:
                        gpt.train()
            torch.cuda.synchronize()
            log_single_rank(logger, logging.INFO,
                            f"[STILL online] forward pass completed for seq {i+1}/{n_seqs}, "
                            f"kv_captured={'yes' if kv_result is not None else 'no'}")
        except Exception as exc:
            log_single_rank(logger, logging.WARNING,
                            f"[STILL online] forward pass failed for seq {i}: {exc}")
            _fwd_ok = 0

        # Collective agreement: if the forward failed on ANY rank, ALL ranks skip
        # this sequence together. Without this, a rank that hit `continue` would
        # race ahead to the next sequence's broadcast while the others are still
        # synced here → mismatched collectives → NCCL deadlock. (This all_reduce is
        # reached by every rank whether the forward succeeded or raised.)
        _ok_t = torch.tensor([_fwd_ok], dtype=torch.long, device="cuda")
        torch.distributed.all_reduce(_ok_t, op=torch.distributed.ReduceOp.MIN)
        if _ok_t.item() == 0:
            continue

        if rank == 0 and kv_result is not None:
            keys, vals = kv_result
            n_attn_layers = len(keys)
            d_kv = keys[0].shape[-1]  # (1, S_pad, d_kv)

            # Initialize compactor on first successful capture.
            init_compactor_from_kv(runtime_state, args_obj, n_attn_layers, d_kv)

            # Trim padding back to original sequence length before chunking.
            keys = [k[:, :seq_len, :] for k in keys]
            vals = [v[:, :seq_len, :] for v in vals]

            # Slice K/V into chunks and build a Trajectory.
            S = seq_len
            chunks = []
            for chunk_start in range(0, S, chunk_size):
                chunk_end = min(chunk_start + chunk_size, S)
                chunk_keys = [k[:, chunk_start:chunk_end, :] for k in keys]
                chunk_vals = [v[:, chunk_start:chunk_end, :] for v in vals]
                chunks.append((chunk_keys, chunk_vals))

            if not chunks:
                continue

            # Add a probe at the last chunk so kv_recon has a query target.
            last_idx = len(chunks) - 1
            resp_t = token_t[:seq_len].unsqueeze(0).cpu()  # (1, seq_len)
            probe = TrainingProbe(
                query_tokens=resp_t,
                teacher_logits=None,
                answer_tokens=None,
                advantage=reward if reward != 0.0 else None,
            )
            traj = Trajectory(
                chunks=chunks,
                probes_by_chunk={last_idx: [probe]},
                rollout_return=reward,
            )
            runtime_state.compactor_trajectories.append(traj)
            log_single_rank(logger, logging.DEBUG,
                            f"[STILL online] Built trajectory from forward pass: "
                            f"n_chunks={len(chunks)} d_kv={d_kv}")


def maybe_init_compactor_online(runtime_state: Any, args, kv_hook) -> None:
    """Lazily create the online compactor from a live KV hook if dims are available.

    Falls back to forward-pass KV capture (build_compactor_trajectories) when the
    hook returns None (paged cache cleared after inference). No-op if online
    training is disabled or the compactor is already initialized.
    """
    if not getattr(args, "rl_compaction_compactor_train", False):
        return
    if runtime_state.compactor is not None:
        return

    kv = kv_hook.get_kv_matrices()
    if kv is None:
        return  # dims not yet available; build_compactor_trajectories will init on first forward

    keys_per_layer, _ = kv
    init_compactor_from_kv(runtime_state, args, len(keys_per_layer), keys_per_layer[0].shape[-1])


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

    # Attach the compactor optimizer on first call (deferred because the Megatron
    # optimizer is not available during init_compactor_from_kv). optimizer may
    # legitimately be None in frozen-LLM pipelines (--skip-train --no-load-optim);
    # the attach builds a standalone optimizer in that case.
    if runtime_state.compactor_optimizer is None:
        attach_compactor_optimizer(runtime_state, optimizer)

    if runtime_state.compactor_optimizer is None or runtime_state.compactor_cfg is None:
        return

    if not trajectories:
        return

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
                updater=runtime_state.compactor,
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
    # process — using that could re-emit and overwrite an existing step_N.pt with
    # different weights after a restart.
    runtime_state.compactor_step_offset += 1
    global_step = runtime_state.compactor_step_offset

    # Periodic checkpoint
    if args is not None:
        ckpt_dir = getattr(args, "rl_compaction_compactor_checkpoint_dir", None)
        ckpt_every = getattr(args, "rl_compaction_compactor_checkpoint_every", 100)
        if ckpt_dir and global_step % ckpt_every == 0:
            import os
            from megatron.rl.compaction.learned import save_checkpoint
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"step_{global_step:07d}.pt")
            scheduler = getattr(runtime_state, "compactor_scheduler", None)
            save_checkpoint(
                runtime_state.compactor,
                ckpt_path,
                step=global_step,
                optimizer=runtime_state.compactor_optimizer,
                scheduler=scheduler,
            )
            log_single_rank(logger, logging.INFO, f"[STILL online] checkpoint saved: {ckpt_path}")
