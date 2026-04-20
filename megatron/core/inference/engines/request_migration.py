# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inter-engine request migration for heterogeneous inference shards.

Moves an in-flight generation request — prompt, generated tokens, sampling
params, logprob history, *and* its KV cache — from one
:class:`DynamicInferenceEngine` to another whose parallelism differs. The
existing :class:`~megatron.core.resharding.copy_services.nccl_copy_service.NCCLCopyService`
is reused for the KV tensor transport. The module supplies:

- :class:`RequestMigrationBundle` — metadata envelope that travels with
  the request (everything except the KV tensors themselves).
- :class:`KVLayout` — descriptor of one shard's KV-tensor geometry
  (TP/PP size, head count, block size, MLA).
- :func:`build_kv_migration_plan` — given both shards' layouts + block
  ids, returns a flat list of :class:`KVMigrationOp` describing the
  tensor slices each rank sends / receives. Handles heterogeneous TP
  by reshaping the head-dim partition. PP=1 on both sides for v0.
- :func:`execute_kv_migration_plan` — drives the plan's ops through
  :class:`NCCLCopyService` over a ``cross_shard_group`` process group.
- :func:`migrate_request_cross_shard` — the one-shot collective
  orchestration: snapshot on src → broadcast bundle → inject on dst →
  broadcast block ids → transport → detach on src.

Engine-side surgery (``snapshot_request`` / ``detach_request`` /
``inject_request``) lives alongside :class:`DynamicInferenceEngine`.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass
class MambaLayout:
    """Per-shard descriptor for Mamba/hybrid model state.

    The state tensors live at
    ``(num_mamba_layers_pp, max_requests, *state_shape)`` on each rank
    — conv state and SSM state are separate tensors. ``num_mamba_layers_pp``
    is the count of Mamba layers this PP-rank is responsible for (zero
    for pure-attention ranks on a hybrid PP split, though v1 matches PP
    across shards anyway). The ``*_shape`` tuples and ``*_dtype_name``
    come from the engine's :class:`MambaInferenceStateConfig` and are
    identical on both sides for matched TP/PP.
    """

    num_mamba_layers_pp: int
    conv_states_shape: Tuple[int, ...]
    ssm_states_shape: Tuple[int, ...]
    conv_states_dtype_name: str  # e.g. "bfloat16" / "float32"
    ssm_states_dtype_name: str


@dataclass
class KVLayout:
    """Per-shard KV-tensor layout descriptor.

    ``num_layers_total`` / ``num_kv_heads_total`` / ``head_dim`` /
    ``block_size_tokens`` are whole-model invariants that must match on
    both sides of a migration. ``tp_size`` / ``pp_size`` describe the
    owning shard's partition of that model. ``num_kv_heads_total`` is
    the count of attention heads that key/value tensors are split over
    (``num_query_groups`` for GQA, ``num_attention_heads`` otherwise).
    TP shards the head dim; PP shards the layer dim.

    Hybrid (Mamba + attention) models populate ``mamba`` with the
    per-request state shapes so the migration transport can also carry
    the conv / SSM state. ``None`` on pure-attention models.
    """

    tp_size: int
    pp_size: int
    num_layers_total: int
    num_kv_heads_total: int
    head_dim: int
    block_size_tokens: int
    is_mla: bool = False
    # MLA compresses K/V into one reduced latent dim; shape is
    # ``(num_layers_pp, total_blocks, block_size, kv_reduced_dim)`` instead
    # of the standard ``(2, num_layers_pp, total_blocks, block_size, heads_per_tp, head_dim)``.
    kv_reduced_dim: Optional[int] = None
    mamba: Optional[MambaLayout] = None

    def assert_divisible(self) -> None:
        """Preconditions: TP divides head count, PP divides layer count."""
        assert self.num_kv_heads_total % self.tp_size == 0, (
            f"num_kv_heads_total={self.num_kv_heads_total} must be divisible by "
            f"tp_size={self.tp_size}"
        )
        assert self.num_layers_total % self.pp_size == 0, (
            f"num_layers_total={self.num_layers_total} must be divisible by "
            f"pp_size={self.pp_size}"
        )
        if self.is_mla:
            assert self.kv_reduced_dim is not None, "MLA layout requires kv_reduced_dim"

    @property
    def heads_per_tp(self) -> int:
        return self.num_kv_heads_total // self.tp_size


@dataclass
class RequestMigrationBundle:
    """Metadata envelope for migrating an in-flight request.

    Contains everything the destination engine needs to resume generation
    *except* the KV cache tensors themselves (those move via
    :func:`build_kv_migration_plan` ops).

    The bundle is designed to be msgpack-serializable: all fields are
    plain Python types. Tensors that belong to the request metadata
    (``generated_log_probs``, ``generated_top_n_logprobs``) are serialized
    as lists; they're CPU-resident in
    :class:`megatron.core.inference.inference_request.DynamicInferenceRequest`
    anyway.
    """

    # --- Request identity -------------------------------------------------
    request_id: int

    # --- Tokens -----------------------------------------------------------
    prompt_tokens: List[int]
    generated_tokens: List[int]

    # --- Sampling / generation params (SamplingParams.serialize() dict) --
    sampling_params: dict

    # --- Generation history ----------------------------------------------
    # `generated_log_probs` is a flat float list (position-ordered).
    generated_log_probs: Optional[List[float]] = None
    # Per-position list of {token_str: logprob} dicts for top-n tracking.
    generated_top_n_logprobs: Optional[List[List[Tuple[int, float]]]] = None
    # kv_cache_epoch = list of (start_position, policy_epoch) pairs marking
    # segments of the KV that were produced by different policy weights.
    kv_cache_epoch: List[Tuple[int, int]] = field(default_factory=list)

    # --- Block bookkeeping (must match what the source engine had) -------
    # Number of KV blocks the request currently occupies.
    num_kv_blocks: int = 0
    # Number of tokens in the final (possibly partial) block.
    last_block_offset: int = 0
    # Global block ids as the source engine saw them (in logical order,
    # i.e. request's chronological token order). The destination will
    # allocate fresh block ids and write incoming tensors into them.
    src_block_ids: List[int] = field(default_factory=list)

    # --- Source / destination KV layout (for plan construction) ----------
    src_layout: Optional[KVLayout] = None
    dst_layout: Optional[KVLayout] = None


@dataclass
class KVMigrationOp:
    """One tensor transfer in the KV migration plan.

    Represents "source global rank ``src_rank`` sends the slice of its
    memory buffer covering ``layer_range`` × ``head_slice`` of blocks
    ``src_block_ids`` to destination global rank ``dst_rank``, which
    writes it into blocks ``dst_block_ids`` at the same head offsets".
    Shape of the transferred tensor:

        (kv=2 or 1 for MLA, layer_span, num_blocks, block_size,
         head_span, head_dim)

    For an MLA layout the ``head_span`` dimension collapses into
    ``kv_reduced_dim`` (and ``kv=1``) — see :class:`KVLayout`.
    """

    src_rank: int
    dst_rank: int
    # Global layer indices covered by this op (contiguous range).
    layer_range: Tuple[int, int]  # [start, end)
    # Head indices (contiguous range across the head dim) this op moves.
    head_range: Tuple[int, int]  # [start, end)
    # Parallel lists: position i in src_block_ids is written to
    # dst_block_ids[i] on the destination side.
    src_block_ids: List[int]
    dst_block_ids: List[int]


def build_kv_migration_plan(
    bundle: RequestMigrationBundle,
    src_global_rank_of: "Callable[[int, int], int]",
    dst_global_rank_of: "Callable[[int, int], int]",
    dst_block_ids: List[int],
) -> List[KVMigrationOp]:
    """Compute the set of tensor transfers that move a request's KV cache
    from the source shard to the destination shard.

    Args:
        bundle: Request envelope. ``src_layout`` and ``dst_layout`` must be
            populated.
        src_global_rank_of: Function ``(tp_rank, pp_rank) -> global_rank``
            for the source shard.
        dst_global_rank_of: Same for the destination shard.
        dst_block_ids: Block ids the destination engine's allocator
            already reserved for this request, in the same chronological
            order as ``bundle.src_block_ids``.

    Returns:
        A list of :class:`KVMigrationOp`. Every source-rank / destination-
        rank pair that needs to exchange data appears exactly once per
        (layer range × head range) block.

    Heterogeneous-TP invariant: the plan emits one op per
    ``(src_tp_rank, dst_tp_rank)`` combination that shares head ownership.
    With ``src_tp=S`` and ``dst_tp=D`` and ``nh = num_kv_heads_total``:

        - Each src rank owns heads ``[src_tp_rank * nh/S, (src_tp_rank+1) * nh/S)``.
        - Each dst rank owns heads ``[dst_tp_rank * nh/D, (dst_tp_rank+1) * nh/D)``.
        - For each (src, dst) pair, the intersection of their head ranges
          is the slice that moves between them. When that intersection
          is empty the pair has no op.

    PP reshape is intentionally not supported in v0 (``assert src_pp ==
    dst_pp == 1``).
    """
    assert bundle.src_layout is not None, "bundle.src_layout missing"
    assert bundle.dst_layout is not None, "bundle.dst_layout missing"
    src = bundle.src_layout
    dst = bundle.dst_layout
    src.assert_divisible()
    dst.assert_divisible()
    assert src.pp_size == 1 and dst.pp_size == 1, (
        "v0 migration plan only supports PP=1 on both sides; got "
        f"src_pp={src.pp_size}, dst_pp={dst.pp_size}"
    )
    assert src.num_kv_heads_total == dst.num_kv_heads_total, (
        "num_kv_heads_total must match across shards — migration cannot "
        f"change model shape (src={src.num_kv_heads_total}, "
        f"dst={dst.num_kv_heads_total})"
    )
    assert src.num_layers_total == dst.num_layers_total, (
        "num_layers_total must match across shards"
    )
    assert src.is_mla == dst.is_mla, "MLA mode must match across shards"
    assert len(dst_block_ids) == len(bundle.src_block_ids), (
        f"dst_block_ids count ({len(dst_block_ids)}) must match "
        f"bundle.src_block_ids count ({len(bundle.src_block_ids)})"
    )

    # With PP=1 both sides hold all layers on every rank.
    layer_range = (0, src.num_layers_total)

    ops: List[KVMigrationOp] = []

    # MLA collapses the head dim; the "head_range" is just the reduced dim.
    # We still emit one op per (src_tp, dst_tp) for symmetry, but both
    # sides always hold the full reduced-dim tensor, so the op is
    # effectively a full tensor copy between the rank-pair.
    if src.is_mla:
        head_range = (0, src.kv_reduced_dim or 0)
        for sr in range(src.tp_size):
            for dr in range(dst.tp_size):
                ops.append(
                    KVMigrationOp(
                        src_rank=src_global_rank_of(sr, 0),
                        dst_rank=dst_global_rank_of(dr, 0),
                        layer_range=layer_range,
                        head_range=head_range,
                        src_block_ids=list(bundle.src_block_ids),
                        dst_block_ids=list(dst_block_ids),
                    )
                )
        return ops

    src_heads = src.heads_per_tp
    dst_heads = dst.heads_per_tp
    for sr in range(src.tp_size):
        src_lo, src_hi = sr * src_heads, (sr + 1) * src_heads
        for dr in range(dst.tp_size):
            dst_lo, dst_hi = dr * dst_heads, (dr + 1) * dst_heads
            overlap_lo = max(src_lo, dst_lo)
            overlap_hi = min(src_hi, dst_hi)
            if overlap_lo >= overlap_hi:
                continue
            ops.append(
                KVMigrationOp(
                    src_rank=src_global_rank_of(sr, 0),
                    dst_rank=dst_global_rank_of(dr, 0),
                    layer_range=layer_range,
                    head_range=(overlap_lo, overlap_hi),
                    src_block_ids=list(bundle.src_block_ids),
                    dst_block_ids=list(dst_block_ids),
                )
            )
    return ops


def serialize_bundle(bundle: RequestMigrationBundle) -> dict:
    """Flatten the bundle to a msgpack-compatible dict."""

    def _serialize_mamba(m: Optional[MambaLayout]) -> Optional[dict]:
        if m is None:
            return None
        return {
            "num_mamba_layers_pp": m.num_mamba_layers_pp,
            "conv_states_shape": list(m.conv_states_shape),
            "ssm_states_shape": list(m.ssm_states_shape),
            "conv_states_dtype_name": m.conv_states_dtype_name,
            "ssm_states_dtype_name": m.ssm_states_dtype_name,
        }

    def _serialize_layout(layout: Optional[KVLayout]) -> Optional[dict]:
        if layout is None:
            return None
        return {
            "tp_size": layout.tp_size,
            "pp_size": layout.pp_size,
            "num_layers_total": layout.num_layers_total,
            "num_kv_heads_total": layout.num_kv_heads_total,
            "head_dim": layout.head_dim,
            "block_size_tokens": layout.block_size_tokens,
            "is_mla": layout.is_mla,
            "kv_reduced_dim": layout.kv_reduced_dim,
            "mamba": _serialize_mamba(layout.mamba),
        }

    return {
        "request_id": bundle.request_id,
        "prompt_tokens": bundle.prompt_tokens,
        "generated_tokens": bundle.generated_tokens,
        "sampling_params": bundle.sampling_params,
        "generated_log_probs": bundle.generated_log_probs,
        "generated_top_n_logprobs": bundle.generated_top_n_logprobs,
        "kv_cache_epoch": [list(p) for p in bundle.kv_cache_epoch],
        "num_kv_blocks": bundle.num_kv_blocks,
        "last_block_offset": bundle.last_block_offset,
        "src_block_ids": bundle.src_block_ids,
        "src_layout": _serialize_layout(bundle.src_layout),
        "dst_layout": _serialize_layout(bundle.dst_layout),
    }


def _gather_kv_slice(
    memory_buffer: torch.Tensor,
    layer_range: Tuple[int, int],
    block_ids: torch.Tensor,
    head_range: Tuple[int, int],
    is_mla: bool,
) -> torch.Tensor:
    """Copy the KV slice described by an op into a fresh contiguous tensor.

    Advanced indexing on ``memory_buffer`` with a block-id tensor
    produces a copy (not a view), which is what we want for the
    send-side staging buffer.
    """
    layer_lo, layer_hi = layer_range
    head_lo, head_hi = head_range
    if is_mla:
        # shape (num_layers_pp, total_blocks, block_size, kv_reduced_dim)
        return memory_buffer[layer_lo:layer_hi, block_ids, :, head_lo:head_hi].contiguous()
    # shape (2, num_layers_pp, total_blocks, block_size, num_heads_per_tp, head_dim)
    return memory_buffer[
        :, layer_lo:layer_hi, block_ids, :, head_lo:head_hi, :
    ].contiguous()


def _scatter_kv_slice(
    memory_buffer: torch.Tensor,
    layer_range: Tuple[int, int],
    block_ids: torch.Tensor,
    head_range: Tuple[int, int],
    is_mla: bool,
    data: torch.Tensor,
) -> None:
    """Write ``data`` back into ``memory_buffer`` at the op's slice.

    Inverse of :func:`_gather_kv_slice`. Advanced-index assignment
    writes in-place on the original buffer.
    """
    layer_lo, layer_hi = layer_range
    head_lo, head_hi = head_range
    if is_mla:
        memory_buffer[layer_lo:layer_hi, block_ids, :, head_lo:head_hi] = data
    else:
        memory_buffer[:, layer_lo:layer_hi, block_ids, :, head_lo:head_hi, :] = data


def execute_kv_migration_plan(
    ops: List[KVMigrationOp],
    memory_buffer: torch.Tensor,
    layout: KVLayout,
    group: "Optional[dist.ProcessGroup]",
    *,
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
) -> None:
    """Execute a KV migration plan collectively across ``group``.

    Every rank in ``group`` must call this function. For each
    :class:`KVMigrationOp` the current rank plays at most two roles:

    - if ``rank == op.src_rank``: gather the op's slice from the local
      ``memory_buffer`` into a staging tensor, ``submit_send`` it;
    - if ``rank == op.dst_rank``: allocate a staging tensor of the op's
      expected shape, ``submit_recv`` into it, then scatter the result
      back into the local ``memory_buffer``.

    Same-rank ops (``src_rank == dst_rank``) are handled by
    :class:`NCCLCopyService` through its ``task_id`` matching — no
    network traffic, just a device-local copy on its dedicated stream.

    **Global vs local head indices.** ``op.head_range`` is in the
    model's *global* head index space (e.g. heads ``[0, num_kv_heads_total)``).
    Each rank's ``memory_buffer`` holds only its local TP slice. The
    ``my_src_head_offset`` / ``my_dst_head_offset`` parameters tell the
    transport where this rank's slice starts in global head space, so
    the op's global range can be translated into the local buffer's
    index range. Callers that aren't in the src (resp. dst) shard can
    leave those offsets at their default ``0``.

    Args:
        ops: Output of :func:`build_kv_migration_plan`.
        memory_buffer: This rank's KV cache buffer. Contract:
            - non-MLA shape:
              ``(2, num_layers_pp, total_blocks, block_size, heads_per_tp, head_dim)``
            - MLA shape:
              ``(num_layers_pp, total_blocks, block_size, kv_reduced_dim)``
        layout: This rank's KV layout. Its ``is_mla`` flag selects
            between the two buffer shapes.
        group: Cross-shard process group returned by
            :func:`megatron.core.inference.shards.build_cross_shard_group`.
            Must contain the union of all ops' src and dst ranks.
            ``None`` is permitted for the single-rank-local case
            (collectives are no-ops then).
        my_src_head_offset: Global head index at which this rank's src
            buffer slice starts. Only consulted when this rank appears
            as ``op.src_rank`` for some op.
        my_dst_head_offset: Global head index at which this rank's dst
            buffer slice starts. Only consulted when this rank appears
            as ``op.dst_rank`` for some op.

    Notes:
        Every rank in ``group`` must enter this function so the
        collective under the hood (``batch_isend_irecv``) stays balanced.
        Ranks that appear in no op still participate as a no-op at the
        collective level.
    """
    from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService

    rank = dist.get_rank()
    service = NCCLCopyService(group=group)

    # Keep recv-side staging tensors alive until the service runs so
    # their buffers don't get freed mid-flight. Each entry is (op, tensor).
    pending_recvs: List[Tuple[KVMigrationOp, torch.Tensor]] = []

    for task_id, op in enumerate(ops):
        is_src = rank == op.src_rank
        is_dst = rank == op.dst_rank
        if not (is_src or is_dst):
            continue

        device = memory_buffer.device
        dtype = memory_buffer.dtype

        if is_src:
            src_blocks = torch.as_tensor(op.src_block_ids, device=device, dtype=torch.long)
            local_head_range = (
                op.head_range[0] - my_src_head_offset,
                op.head_range[1] - my_src_head_offset,
            )
            send_tensor = _gather_kv_slice(
                memory_buffer, op.layer_range, src_blocks, local_head_range, layout.is_mla
            )
            service.submit_send(send_tensor, dest_rank=op.dst_rank, task_id=task_id)

        if is_dst:
            # Compute the shape the incoming tensor must have.
            num_blocks = len(op.dst_block_ids)
            layer_span = op.layer_range[1] - op.layer_range[0]
            head_span = op.head_range[1] - op.head_range[0]
            if layout.is_mla:
                shape = (layer_span, num_blocks, layout.block_size_tokens, head_span)
            else:
                shape = (
                    2,
                    layer_span,
                    num_blocks,
                    layout.block_size_tokens,
                    head_span,
                    layout.head_dim,
                )
            recv_tensor = torch.empty(shape, dtype=dtype, device=device)
            service.submit_recv(recv_tensor, src_rank=op.src_rank, task_id=task_id)
            pending_recvs.append((op, recv_tensor))

    service.run()

    # Scatter every received slice back into the local memory buffer.
    for op, recv_tensor in pending_recvs:
        dst_blocks = torch.as_tensor(
            op.dst_block_ids, device=memory_buffer.device, dtype=torch.long
        )
        local_head_range = (
            op.head_range[0] - my_dst_head_offset,
            op.head_range[1] - my_dst_head_offset,
        )
        _scatter_kv_slice(
            memory_buffer,
            op.layer_range,
            dst_blocks,
            local_head_range,
            layout.is_mla,
            recv_tensor,
        )


def execute_mamba_state_transport(
    *,
    role: str,
    layout: KVLayout,
    src_ranks: List[int],
    dst_ranks: List[int],
    src_mamba_conv_states: Optional[torch.Tensor] = None,
    src_mamba_ssm_states: Optional[torch.Tensor] = None,
    src_state_idx: Optional[int] = None,
    dst_mamba_conv_states: Optional[torch.Tensor] = None,
    dst_mamba_ssm_states: Optional[torch.Tensor] = None,
    dst_state_idx: Optional[int] = None,
    cross_shard_group: "dist.ProcessGroup",
) -> None:
    """Transport Mamba conv + SSM state for one request across shards.

    For v1 we require **matched TP and PP** between the shards so the
    state tensor shapes are identical on both sides — each src rank
    sends its per-layer state slice to the corresponding dst rank
    (``src_ranks[i] → dst_ranks[i]``) with no reshape. Heterogeneous
    TP/PP on Mamba would need a reshape layer analogous to
    :func:`build_kv_migration_plan`; that's deferred.

    Runs collectively over ``cross_shard_group``. Non-participating
    ranks in the group enter the underlying ``NCCLCopyService`` with
    no ops so the ``batch_isend_irecv`` stays balanced.
    """
    if layout.mamba is None:
        return  # pure attention model; nothing to do

    assert len(src_ranks) == len(dst_ranks), (
        f"Mamba state transport (v1) requires matched TP/PP between "
        f"shards — got src_ranks={src_ranks} and dst_ranks={dst_ranks}"
    )

    from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService

    rank = dist.get_rank()
    service = NCCLCopyService(group=cross_shard_group)

    # Each (src_ranks[i], dst_ranks[i]) pair exchanges a conv-state
    # slice and an SSM-state slice. Tag with distinct task_ids so the
    # same-rank local-copy path can match them.
    pending_recvs: List[Tuple[str, torch.Tensor]] = []
    for i, (sr, dr) in enumerate(zip(src_ranks, dst_ranks)):
        is_src = rank == sr
        is_dst = rank == dr
        if not (is_src or is_dst):
            continue
        if is_src:
            assert src_mamba_conv_states is not None
            assert src_mamba_ssm_states is not None
            assert src_state_idx is not None
            # Gather this request's state across all PP-local Mamba layers.
            conv_send = src_mamba_conv_states[:, src_state_idx].contiguous()
            ssm_send = src_mamba_ssm_states[:, src_state_idx].contiguous()
            service.submit_send(conv_send, dest_rank=dr, task_id=2 * i)
            service.submit_send(ssm_send, dest_rank=dr, task_id=2 * i + 1)
        if is_dst:
            assert dst_mamba_conv_states is not None
            assert dst_mamba_ssm_states is not None
            assert dst_state_idx is not None
            conv_recv = torch.empty(
                (layout.mamba.num_mamba_layers_pp,) + tuple(layout.mamba.conv_states_shape),
                dtype=dst_mamba_conv_states.dtype,
                device=dst_mamba_conv_states.device,
            )
            ssm_recv = torch.empty(
                (layout.mamba.num_mamba_layers_pp,) + tuple(layout.mamba.ssm_states_shape),
                dtype=dst_mamba_ssm_states.dtype,
                device=dst_mamba_ssm_states.device,
            )
            service.submit_recv(conv_recv, src_rank=sr, task_id=2 * i)
            service.submit_recv(ssm_recv, src_rank=sr, task_id=2 * i + 1)
            pending_recvs.append(("conv", conv_recv))
            pending_recvs.append(("ssm", ssm_recv))

    service.run()

    # Scatter received state into the dst slot.
    if role == "dst":
        assert dst_mamba_conv_states is not None
        assert dst_mamba_ssm_states is not None
        assert dst_state_idx is not None
        for kind, recv_tensor in pending_recvs:
            if kind == "conv":
                dst_mamba_conv_states[:, dst_state_idx] = recv_tensor
            else:
                dst_mamba_ssm_states[:, dst_state_idx] = recv_tensor


def migrate_request_cross_shard(
    *,
    role: str,
    engine: "Optional[object]",
    request_id_src: Optional[int],
    src_layout: KVLayout,
    dst_layout: KVLayout,
    src_ranks: List[int],
    dst_ranks: List[int],
    cross_shard_group: "dist.ProcessGroup",
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
    request_id_dst: Optional[int] = None,
) -> Optional[int]:
    """Migrate one in-flight request from the src shard to the dst shard.

    Collective across ``cross_shard_group``. Every rank in
    ``src_ranks ∪ dst_ranks`` must enter this function simultaneously.
    Engines must be quiescent (no in-flight step); the caller (e.g.
    ``MegatronLocalMulti.migrate_request``) is responsible for the
    pause/resume bracket.

    Flow (each step is aligned on every rank in the group):

        1. ``src shard rank 0`` calls ``engine.snapshot_request`` to
           build the migration :class:`RequestMigrationBundle`.
        2. Broadcast the bundle through ``cross_shard_group`` so every
           rank — in particular every rank of the dst shard — has it.
        3. ``dst shard rank 0`` calls ``engine.inject_request(bundle)``
           to register the request in DECODE state with fresh KV
           blocks allocated. Broadcasts the new block ids.
        4. Every rank builds the KV plan from the (now-global-visible)
           bundle + dst_block_ids and drives
           :func:`execute_kv_migration_plan`, which moves the per-layer
           KV slices through ``NCCLCopyService`` over
           ``cross_shard_group``.
        5. ``src shard rank 0`` calls ``engine.detach_request`` with
           ``keep_blocks=False`` to free the now-stale source blocks
           and clean up the active-batch slot.

    Args:
        role: ``"src"``, ``"dst"``, or ``"bystander"`` (latter is only
            legal when the caller is in ``cross_shard_group`` but not
            in either shard — unlikely but permitted).
        engine: The rank-local engine; ``None`` for bystanders.
        request_id_src: Source-side request id. Required when
            ``role == "src"``; ignored otherwise (dst learns it from
            the bundle).
        src_layout / dst_layout: KV layout descriptors. Populated by
            the caller from each shard's process-group config.
        src_ranks / dst_ranks: Global ranks making up the two shards'
            rank windows. Used to pick broadcast roots and to feed the
            plan's global-rank-of callable.
        cross_shard_group: The torch process group spanning
            ``src_ranks ∪ dst_ranks``. Callers typically obtain this
            from :func:`megatron.core.inference.shards.build_cross_shard_group`.
        my_src_head_offset / my_dst_head_offset: See
            :func:`execute_kv_migration_plan`.
        request_id_dst: The request id to register on the destination
            engine. Defaults to the source id; supply a different value
            when the two engines share a caller's request-id namespace
            (e.g. same-engine round-trip tests).

    Returns:
        The dst-side request id (useful to the caller for future
        lifecycle ops). ``None`` on bystanders.
    """
    assert role in ("src", "dst", "bystander"), role
    src_root = src_ranks[0]
    dst_root = dst_ranks[0]

    # --- Step 1+2: src builds bundle, all receive it. ------------------
    bundle_box: List[Optional[RequestMigrationBundle]] = [None]
    src_block_ids_box: List[Optional[torch.Tensor]] = [None]
    if role == "src":
        assert engine is not None and request_id_src is not None
        bundle, src_blocks = engine.snapshot_request(request_id_src)
        bundle.src_layout = src_layout
        bundle.dst_layout = dst_layout
        # Populate the dst request id now so the payload reaching the
        # destination is self-contained.
        bundle.request_id = request_id_dst if request_id_dst is not None else request_id_src
        bundle_box[0] = bundle
        src_block_ids_box[0] = src_blocks.clone()

    # broadcast_object_list sends Python objects via pickle/gloo; it
    # works for mixed device layouts and handles None on non-root
    # ranks.
    dist.broadcast_object_list(bundle_box, src=src_root, group=cross_shard_group)
    bundle = bundle_box[0]
    assert bundle is not None

    # --- Step 3: dst injects, broadcasts new block ids. ---------------
    dst_block_ids_list_box: List[Optional[List[int]]] = [None]
    if role == "dst":
        assert engine is not None
        dst_blocks = engine.inject_request(bundle)
        dst_block_ids_list_box[0] = dst_blocks.tolist()

    dist.broadcast_object_list(dst_block_ids_list_box, src=dst_root, group=cross_shard_group)
    dst_block_ids_list = dst_block_ids_list_box[0]
    assert dst_block_ids_list is not None

    # --- Step 4: every participant runs the plan. ---------------------
    def _src_rank_of(tp: int, pp: int, _start=src_root) -> int:
        # v0 has pp=1 so pp is a no-op; src_ranks is a contiguous
        # [src_root, src_root + tp_size) window.
        return _start + tp

    def _dst_rank_of(tp: int, pp: int, _start=dst_root) -> int:
        return _start + tp

    ops = build_kv_migration_plan(
        bundle,
        src_global_rank_of=_src_rank_of,
        dst_global_rank_of=_dst_rank_of,
        dst_block_ids=dst_block_ids_list,
    )

    # Bystanders in the group still enter execute_kv_migration_plan so
    # the underlying ``batch_isend_irecv`` stays balanced — they just
    # have no send/recv ops matching their rank.
    if role == "src":
        memory_buffer = engine.context.memory_buffer
        layout = src_layout
    elif role == "dst":
        memory_buffer = engine.context.memory_buffer
        layout = dst_layout
    else:
        # Bystander: fabricate a zero-sized buffer of the right dtype /
        # device so the helper's shape computations don't blow up; the
        # helper still visits every op but skips ones not touching our
        # rank. Use the dst layout's dtype — either side's would do.
        memory_buffer = torch.empty(0, device=torch.cuda.current_device())
        layout = dst_layout

    execute_kv_migration_plan(
        ops,
        memory_buffer,
        layout,
        cross_shard_group,
        my_src_head_offset=my_src_head_offset,
        my_dst_head_offset=my_dst_head_offset,
    )

    # --- Step 4b: Mamba/hybrid models — move the per-request conv +
    #   SSM state alongside attention KV. v1 requires matched TP/PP
    #   between src and dst (both layouts must agree on the same mamba
    #   shape). Each rank reads its own state_idx from its engine.
    if src_layout.mamba is not None:
        assert dst_layout.mamba is not None, (
            "src layout has Mamba state but dst layout does not — "
            "shards must agree on model shape for migration"
        )
        src_state_idx = (
            engine.get_mamba_state_idx_for(request_id_src) if role == "src" else None
        )
        dst_state_idx = (
            engine.get_mamba_state_idx_for(bundle.request_id) if role == "dst" else None
        )
        execute_mamba_state_transport(
            role=role,
            layout=layout,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            src_mamba_conv_states=(
                engine.context.mamba_conv_states if role == "src" else None
            ),
            src_mamba_ssm_states=(
                engine.context.mamba_ssm_states if role == "src" else None
            ),
            src_state_idx=src_state_idx,
            dst_mamba_conv_states=(
                engine.context.mamba_conv_states if role == "dst" else None
            ),
            dst_mamba_ssm_states=(
                engine.context.mamba_ssm_states if role == "dst" else None
            ),
            dst_state_idx=dst_state_idx,
            cross_shard_group=cross_shard_group,
        )

    # --- Step 5: src cleans up its slot + releases source blocks. -----
    if role == "src":
        engine.detach_request(request_id_src, keep_blocks=False)

    if role == "dst":
        return bundle.request_id
    return None


def deserialize_bundle(obj: dict) -> RequestMigrationBundle:
    """Inverse of :func:`serialize_bundle`."""

    def _deserialize_mamba(d: Optional[dict]) -> Optional[MambaLayout]:
        if d is None:
            return None
        return MambaLayout(
            num_mamba_layers_pp=d["num_mamba_layers_pp"],
            conv_states_shape=tuple(d["conv_states_shape"]),
            ssm_states_shape=tuple(d["ssm_states_shape"]),
            conv_states_dtype_name=d["conv_states_dtype_name"],
            ssm_states_dtype_name=d["ssm_states_dtype_name"],
        )

    def _deserialize_layout(d: Optional[dict]) -> Optional[KVLayout]:
        if d is None:
            return None
        mamba_d = d.pop("mamba", None)
        layout = KVLayout(**d)
        layout.mamba = _deserialize_mamba(mamba_d)
        return layout

    return RequestMigrationBundle(
        request_id=obj["request_id"],
        prompt_tokens=list(obj["prompt_tokens"]),
        generated_tokens=list(obj["generated_tokens"]),
        sampling_params=dict(obj["sampling_params"]),
        generated_log_probs=(
            list(obj["generated_log_probs"])
            if obj.get("generated_log_probs") is not None
            else None
        ),
        generated_top_n_logprobs=obj.get("generated_top_n_logprobs"),
        kv_cache_epoch=[tuple(p) for p in obj.get("kv_cache_epoch", [])],
        num_kv_blocks=obj["num_kv_blocks"],
        last_block_offset=obj["last_block_offset"],
        src_block_ids=list(obj["src_block_ids"]),
        src_layout=_deserialize_layout(obj.get("src_layout")),
        dst_layout=_deserialize_layout(obj.get("dst_layout")),
    )
