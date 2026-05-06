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
  *and* heterogeneous PP by taking the (layer × head) intersection of
  each (src rank, dst rank) pair's ownership.
- :func:`execute_kv_migration_plan` — drives the plan's ops through
  :class:`NCCLCopyService` over a ``cross_shard_group`` process group.
- :func:`migrate_requests_cross_shard_batch` — collective orchestration
  for N requests per call: snapshot on src → broadcast bundles → inject
  on dst → broadcast block ids → one fused transport → detach on src.
- :func:`migrate_request_cross_shard` — thin wrapper around the batch
  primitive for the single-request case.

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
    # policy_epoch = list of (token_index, generation_epoch) pairs
    # tracking which policy weights generated each prefix of the
    # output. The HTTP InferenceResponse pydantic schema requires a
    # list, so we ship it explicitly through the bundle (defaults to
    # an empty list when the source request had no policy epochs).
    policy_epoch: List[Tuple[int, int]] = field(default_factory=list)

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

    # --- MoE router replay history ---------------------------------------
    # Serialized form of ``DynamicInferenceRequest.routing_indices``,
    # which has shape ``(total_tokens, num_layers, topk)`` and carries
    # per-token per-layer expert routing decisions accumulated during
    # generation. Empty / (0,0,0) means "no routing history". Flattened
    # for wire efficiency; inject_request reshapes back via shape.
    routing_indices_shape: Optional[Tuple[int, int, int]] = None
    routing_indices_dtype_name: Optional[str] = None
    routing_indices_flat: List[int] = field(default_factory=list)


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
        A list of :class:`KVMigrationOp`. Every (src_pp × dst_pp × src_tp
        × dst_tp) combination with non-empty layer and head overlap
        produces exactly one op.

    Heterogeneous-TP invariant: the plan emits one op per
    ``(src_tp_rank, dst_tp_rank)`` combination that shares head ownership.
    With ``src_tp=S`` and ``dst_tp=D`` and ``nh = num_kv_heads_total``:

        - Each src rank owns heads ``[src_tp_rank * nh/S, (src_tp_rank+1) * nh/S)``.
        - Each dst rank owns heads ``[dst_tp_rank * nh/D, (dst_tp_rank+1) * nh/D)``.
        - For each (src, dst) pair, the intersection of their head ranges
          is the slice that moves between them. When that intersection
          is empty the pair has no op.

    Pipeline-parallel reshape: the same intersection logic applies to
    the layer dimension. Each pp_rank owns layers
    ``[pp_rank * L/pp_size, (pp_rank+1) * L/pp_size)``; ops are emitted
    for every (src_pp, dst_pp, src_tp, dst_tp) quadruple with non-empty
    layer AND head overlap. Matched PP (src_pp_size == dst_pp_size) is
    the common case and collapses to one op per pp stage; reshape
    across PP works too.
    """
    assert bundle.src_layout is not None, "bundle.src_layout missing"
    assert bundle.dst_layout is not None, "bundle.dst_layout missing"
    src = bundle.src_layout
    dst = bundle.dst_layout
    src.assert_divisible()
    dst.assert_divisible()
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

    src_layers_per_pp = src.num_layers_total // src.pp_size
    dst_layers_per_pp = dst.num_layers_total // dst.pp_size
    src_heads = src.heads_per_tp
    dst_heads = dst.heads_per_tp
    mla_head_range = (0, src.kv_reduced_dim or 0)
    src_block_ids_copy = list(bundle.src_block_ids)
    dst_block_ids_copy = list(dst_block_ids)

    ops: List[KVMigrationOp] = []
    for src_pp in range(src.pp_size):
        src_layer_lo = src_pp * src_layers_per_pp
        src_layer_hi = src_layer_lo + src_layers_per_pp
        for dst_pp in range(dst.pp_size):
            dst_layer_lo = dst_pp * dst_layers_per_pp
            dst_layer_hi = dst_layer_lo + dst_layers_per_pp
            layer_lo = max(src_layer_lo, dst_layer_lo)
            layer_hi = min(src_layer_hi, dst_layer_hi)
            if layer_lo >= layer_hi:
                continue
            for sr in range(src.tp_size):
                src_head_lo = sr * src_heads
                src_head_hi = src_head_lo + src_heads
                for dr in range(dst.tp_size):
                    if src.is_mla:
                        # MLA collapses head dim — every rank-pair in
                        # the current layer overlap carries the full
                        # reduced-dim tensor.
                        head_range = mla_head_range
                    else:
                        dst_head_lo = dr * dst_heads
                        dst_head_hi = dst_head_lo + dst_heads
                        head_lo = max(src_head_lo, dst_head_lo)
                        head_hi = min(src_head_hi, dst_head_hi)
                        if head_lo >= head_hi:
                            continue
                        head_range = (head_lo, head_hi)
                    ops.append(
                        KVMigrationOp(
                            src_rank=src_global_rank_of(sr, src_pp),
                            dst_rank=dst_global_rank_of(dr, dst_pp),
                            layer_range=(layer_lo, layer_hi),
                            head_range=head_range,
                            src_block_ids=src_block_ids_copy,
                            dst_block_ids=dst_block_ids_copy,
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
        "policy_epoch": [list(p) for p in bundle.policy_epoch],
        "num_kv_blocks": bundle.num_kv_blocks,
        "last_block_offset": bundle.last_block_offset,
        "src_block_ids": bundle.src_block_ids,
        "src_layout": _serialize_layout(bundle.src_layout),
        "dst_layout": _serialize_layout(bundle.dst_layout),
        "routing_indices_shape": (
            list(bundle.routing_indices_shape)
            if bundle.routing_indices_shape is not None
            else None
        ),
        "routing_indices_dtype_name": bundle.routing_indices_dtype_name,
        "routing_indices_flat": bundle.routing_indices_flat,
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


def _execute_kv_migration_plan_nvshmem(
    ops: List[KVMigrationOp],
    memory_buffer: torch.Tensor,
    layout: KVLayout,
    *,
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
    my_src_pp_layer_offset: int = 0,
    my_dst_pp_layer_offset: int = 0,
) -> None:
    """NVSHMEM-direct KV transport. One-sided ``put`` between every
    src/dst rank pair through a symmetric staging buffer, with
    ``quiet`` as the only synchronization. No ``barrier_all``.

    Walks ``ops`` in deterministic order on every rank so the same
    ``StagingArena`` offsets are assigned on src and dst — that's the
    invariant that lets ``nvshmem.core.put`` land each op's bytes at
    the matching offset on the destination's symmetric staging buffer.
    """
    from megatron.core.inference import nvshmem_migration as _nv

    rank = dist.get_rank()
    arena = _nv.StagingArena()
    stream = _nv.migration_stream()

    # First pass on every rank: assign one staging *slot index* per op
    # (regardless of participation), so src and dst agree on slot
    # assignments. Each slot is an independent ``bytetensor`` so
    # NVSHMEM tracks it as a single tensor handle.
    op_slot: List[int] = []
    op_sizes: List[int] = []
    op_shapes: List[tuple] = []
    elem_size = memory_buffer.element_size()
    dtype = memory_buffer.dtype
    for op in ops:
        num_blocks = len(op.src_block_ids)
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
        nelems = 1
        for d in shape:
            nelems *= d
        nbytes = nelems * elem_size
        op_slot.append(arena.take(nbytes))
        op_sizes.append(nbytes)
        op_shapes.append(shape)

    # Each op gets a flag slot in addition to its staging slot — both
    # src and dst look up by op index, so no coordination needed.
    # The flag is what synchronizes src's put with dst's scatter:
    # ``put_signal`` atomically delivers the data + sets the flag,
    # and dst's ``signal_wait`` is a stream-ordered GPU wait. No
    # ``quiet``, no ``barrier_all``.
    op_flag: List[int] = []
    for _ in ops:
        op_flag.append(_nv.acquire_flag_slot())

    sent_op_indices: List[int] = []
    with torch.cuda.stream(stream):
        # Src: gather → slot → put_signal (atomic put + flag set).
        for i, op in enumerate(ops):
            if rank != op.src_rank:
                continue
            slot_idx = op_slot[i]
            flag_slot = op_flag[i]
            nbytes = op_sizes[i]
            shape = op_shapes[i]
            src_blocks = torch.as_tensor(
                op.src_block_ids, device=memory_buffer.device, dtype=torch.long
            )
            local_head_range = (
                op.head_range[0] - my_src_head_offset,
                op.head_range[1] - my_src_head_offset,
            )
            local_layer_range = (
                op.layer_range[0] - my_src_pp_layer_offset,
                op.layer_range[1] - my_src_pp_layer_offset,
            )
            send_tensor = _gather_kv_slice(
                memory_buffer,
                local_layer_range,
                src_blocks,
                local_head_range,
                layout.is_mla,
            )
            slot = _nv.staging_slot(slot_idx)
            slot[:nbytes].view(dtype).reshape(shape).copy_(
                send_tensor, non_blocking=True
            )
            _nv.put_slot_with_signal(
                slot_idx, flag_slot, op.dst_rank, nbytes=nbytes, stream=stream
            )
            sent_op_indices.append(i)

        # Dst: stream-side wait on signal, then scatter slot →
        # memory_buffer. Dependency on src's data arrival is
        # enforced by NVSHMEM's put_signal atomicity.
        for i, op in enumerate(ops):
            if rank != op.dst_rank:
                continue
            slot_idx = op_slot[i]
            flag_slot = op_flag[i]
            nbytes = op_sizes[i]
            shape = op_shapes[i]
            _nv.wait_slot_signal(flag_slot, expected_value=1, stream=stream)
            dst_blocks = torch.as_tensor(
                op.dst_block_ids, device=memory_buffer.device, dtype=torch.long
            )
            local_head_range = (
                op.head_range[0] - my_dst_head_offset,
                op.head_range[1] - my_dst_head_offset,
            )
            local_layer_range = (
                op.layer_range[0] - my_dst_pp_layer_offset,
                op.layer_range[1] - my_dst_pp_layer_offset,
            )
            slot = _nv.staging_slot(slot_idx)
            recv_view = slot[:nbytes].view(dtype).reshape(shape)
            _scatter_kv_slice(
                memory_buffer,
                local_layer_range,
                dst_blocks,
                local_head_range,
                layout.is_mla,
                recv_view,
            )

    # After the migration stream completes, reset every flag we used
    # so the slot can be recycled on the next migration. (Resets run
    # on the host after stream sync so they don't race with the
    # signal_wait.)
    torch.cuda.current_stream().wait_stream(stream)
    for slot in op_flag:
        _nv.reset_flag(slot)


def execute_kv_migration_plan(
    ops: List[KVMigrationOp],
    memory_buffer: torch.Tensor,
    layout: KVLayout,
    group: "Optional[dist.ProcessGroup]",
    *,
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
    my_src_pp_layer_offset: int = 0,
    my_dst_pp_layer_offset: int = 0,
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
    # NVSHMEM-direct path: when the inference interface has initialized
    # the NVSHMEM migration module (cross-shard topology), bypass the
    # collective NCCL service entirely and use one-sided ``put`` over
    # a symmetric staging buffer with a single ``quiet`` for completion.
    # No ``barrier_all``; no engine pause beyond the kernel-launch cost
    # of the gather/put/scatter sequence on the migration stream.
    from megatron.core.inference import nvshmem_migration as _nv

    if _nv.is_initialized():
        _execute_kv_migration_plan_nvshmem(
            ops,
            memory_buffer,
            layout,
            my_src_head_offset=my_src_head_offset,
            my_dst_head_offset=my_dst_head_offset,
            my_src_pp_layer_offset=my_src_pp_layer_offset,
            my_dst_pp_layer_offset=my_dst_pp_layer_offset,
        )
        return

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
            local_layer_range = (
                op.layer_range[0] - my_src_pp_layer_offset,
                op.layer_range[1] - my_src_pp_layer_offset,
            )
            send_tensor = _gather_kv_slice(
                memory_buffer, local_layer_range, src_blocks, local_head_range, layout.is_mla
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
        local_layer_range = (
            op.layer_range[0] - my_dst_pp_layer_offset,
            op.layer_range[1] - my_dst_pp_layer_offset,
        )
        _scatter_kv_slice(
            memory_buffer,
            local_layer_range,
            dst_blocks,
            local_head_range,
            layout.is_mla,
            recv_tensor,
        )


def _execute_mamba_state_transport_nvshmem(
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
) -> None:
    """NVSHMEM-direct Mamba conv + SSM state transport.

    Same shape contract as :func:`execute_mamba_state_transport` (v1
    requires matched TP/PP between shards; per-pair 1:1 send) but
    routed through the symmetric staging buffer with one-sided
    ``put`` + ``quiet``. The Mamba state slot tensors are themselves
    in the symmetric heap; we still stage because conv and ssm slices
    are at distinct offsets within those tensors and we want a single
    contiguous put per (kind, pair).
    """
    from megatron.core.inference import nvshmem_migration as _nv

    rank = dist.get_rank()
    arena = _nv.StagingArena()
    stream = _nv.migration_stream()

    # Pre-walk every (i, conv|ssm) pair on every rank to assign
    # consistent staging slot indices. The mamba layout is matched
    # between shards so the per-rank-pair shapes are identical.
    conv_layers = layout.mamba.num_mamba_layers_pp
    ssm_layers = layout.mamba.num_mamba_layers_pp
    conv_shape = (conv_layers,) + tuple(layout.mamba.conv_states_shape)
    ssm_shape = (ssm_layers,) + tuple(layout.mamba.ssm_states_shape)

    if dst_mamba_conv_states is not None:
        conv_dtype = dst_mamba_conv_states.dtype
        ssm_dtype = dst_mamba_ssm_states.dtype
    elif src_mamba_conv_states is not None:
        conv_dtype = src_mamba_conv_states.dtype
        ssm_dtype = src_mamba_ssm_states.dtype
    else:
        # Bystander: still need the offsets to stay consistent across
        # ranks. Pick a sensible default (matches engine config).
        conv_dtype = torch.bfloat16
        ssm_dtype = torch.bfloat16

    conv_elem = torch.empty((), dtype=conv_dtype).element_size()
    ssm_elem = torch.empty((), dtype=ssm_dtype).element_size()
    conv_nelems = 1
    for d in conv_shape:
        conv_nelems *= d
    ssm_nelems = 1
    for d in ssm_shape:
        ssm_nelems *= d
    conv_nbytes = conv_nelems * conv_elem
    ssm_nbytes = ssm_nelems * ssm_elem

    # Per-pair slots + flags. Flag pattern matches the KV path: src
    # ``put_signal`` atomically delivers the data + sets a flag; dst
    # ``signal_wait`` is a stream-ordered GPU wait. No barrier_all.
    pair_state: List[Tuple[int, int, int, int]] = []
    # (conv_slot, ssm_slot, conv_flag, ssm_flag) per pair
    for _i in range(len(src_ranks)):
        conv_slot = arena.take(conv_nbytes)
        ssm_slot = arena.take(ssm_nbytes)
        conv_flag = _nv.acquire_flag_slot()
        ssm_flag = _nv.acquire_flag_slot()
        pair_state.append((conv_slot, ssm_slot, conv_flag, ssm_flag))

    flags_used: List[int] = []
    with torch.cuda.stream(stream):
        # Src: gather → slot → put_signal.
        for i, (sr, dr) in enumerate(zip(src_ranks, dst_ranks)):
            if rank != sr:
                continue
            assert src_mamba_conv_states is not None
            assert src_mamba_ssm_states is not None
            assert src_state_idx is not None
            conv_slot, ssm_slot, conv_flag, ssm_flag = pair_state[i]
            conv_send = src_mamba_conv_states[:, src_state_idx].contiguous()
            ssm_send = src_mamba_ssm_states[:, src_state_idx].contiguous()
            _nv.staging_slot(conv_slot)[:conv_nbytes].view(conv_dtype).reshape(
                conv_shape
            ).copy_(conv_send, non_blocking=True)
            _nv.staging_slot(ssm_slot)[:ssm_nbytes].view(ssm_dtype).reshape(
                ssm_shape
            ).copy_(ssm_send, non_blocking=True)
            _nv.put_slot_with_signal(
                conv_slot, conv_flag, dr, nbytes=conv_nbytes, stream=stream
            )
            _nv.put_slot_with_signal(
                ssm_slot, ssm_flag, dr, nbytes=ssm_nbytes, stream=stream
            )
            flags_used.extend([conv_flag, ssm_flag])

        # Dst: signal_wait → scatter slot → state.
        for i, (sr, dr) in enumerate(zip(src_ranks, dst_ranks)):
            if rank != dr:
                continue
            assert dst_mamba_conv_states is not None
            assert dst_mamba_ssm_states is not None
            assert dst_state_idx is not None
            conv_slot, ssm_slot, conv_flag, ssm_flag = pair_state[i]
            _nv.wait_slot_signal(conv_flag, expected_value=1, stream=stream)
            _nv.wait_slot_signal(ssm_flag, expected_value=1, stream=stream)
            conv_recv = _nv.staging_slot(conv_slot)[:conv_nbytes].view(
                conv_dtype
            ).reshape(conv_shape)
            ssm_recv = _nv.staging_slot(ssm_slot)[:ssm_nbytes].view(
                ssm_dtype
            ).reshape(ssm_shape)
            dst_mamba_conv_states[:, dst_state_idx] = conv_recv
            dst_mamba_ssm_states[:, dst_state_idx] = ssm_recv
            flags_used.extend([conv_flag, ssm_flag])

    torch.cuda.current_stream().wait_stream(stream)
    for slot in flags_used:
        _nv.reset_flag(slot)


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

    # NVSHMEM-direct path: single ``put`` per (rank, state) pair into
    # symmetric staging, ``quiet``, dst scatters. No ``barrier_all``.
    from megatron.core.inference import nvshmem_migration as _nv

    if _nv.is_initialized():
        _execute_mamba_state_transport_nvshmem(
            role=role,
            layout=layout,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            src_mamba_conv_states=src_mamba_conv_states,
            src_mamba_ssm_states=src_mamba_ssm_states,
            src_state_idx=src_state_idx,
            dst_mamba_conv_states=dst_mamba_conv_states,
            dst_mamba_ssm_states=dst_mamba_ssm_states,
            dst_state_idx=dst_state_idx,
        )
        return

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
    # Thin wrapper around the batch primitive so all of the actual
    # collective / plan / transport logic lives in exactly one place.
    # A 1-element batch is the natural single-request shape.
    migrated_ids = migrate_requests_cross_shard_batch(
        role=role,
        engine=engine,
        request_ids_src=[request_id_src] if request_id_src is not None else None,
        src_layout=src_layout,
        dst_layout=dst_layout,
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
        cross_shard_group=cross_shard_group,
        my_src_head_offset=my_src_head_offset,
        my_dst_head_offset=my_dst_head_offset,
        request_ids_dst=[request_id_dst] if request_id_dst is not None else None,
    )
    if migrated_ids:
        return migrated_ids[0]
    return None


def migrate_requests_cross_shard_batch(
    *,
    role: str,
    engine: "Optional[object]",
    request_ids_src: Optional[List[int]],
    src_layout: KVLayout,
    dst_layout: KVLayout,
    src_ranks: List[int],
    dst_ranks: List[int],
    cross_shard_group: "dist.ProcessGroup",
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
    request_ids_dst: Optional[List[int]] = None,
) -> Optional[List[int]]:
    """Migrate a batch of in-flight requests with a single KV collective.

    Same contract as :func:`migrate_request_cross_shard` but ``N`` requests
    per invocation. The whole batch goes through:

      - one ``snapshot_request`` loop on src,
      - one object-list broadcast (N bundles),
      - one ``inject_request`` loop on dst,
      - one object-list broadcast (N dst-block lists),
      - **one** :func:`execute_kv_migration_plan` call whose op list is
        the concatenation of per-request plans,
      - one ``execute_mamba_state_transport`` per request (hybrid only),
      - one ``detach_request`` loop on src.

    Amortizes the NCCL collective + broadcast setup overhead across the
    batch, which is the main win when the auto-disagg scheduler has
    multiple requests ready to migrate in the same tick.

    Args:
        request_ids_src: Source-side request ids. Required when
            ``role == "src"``; ignored otherwise (dst gets them from
            the bundles).
        request_ids_dst: Optional per-request destination ids. If
            ``None``, each request keeps its src id on the dst side.
        Other args mirror :func:`migrate_request_cross_shard`.

    Returns:
        List of dst-side request ids (same length as the input batch)
        on ``role == "dst"``; ``None`` otherwise. Empty batches return
        an empty list / ``None``.
    """
    assert role in ("src", "dst", "bystander"), role
    src_root = src_ranks[0]
    dst_root = dst_ranks[0]

    # --- Step 1+2: src builds N bundles, broadcasts the list. ---------
    bundles_box: List[Optional[List[RequestMigrationBundle]]] = [None]
    if role == "src":
        assert engine is not None and request_ids_src is not None
        if request_ids_dst is not None:
            assert len(request_ids_dst) == len(request_ids_src), (
                f"request_ids_dst length ({len(request_ids_dst)}) must match "
                f"request_ids_src length ({len(request_ids_src)})"
            )
        bundles: List[RequestMigrationBundle] = []
        for i, req_id in enumerate(request_ids_src):
            bundle, _src_blocks = engine.snapshot_request(req_id)
            bundle.src_layout = src_layout
            bundle.dst_layout = dst_layout
            bundle.request_id = (
                request_ids_dst[i] if request_ids_dst is not None else req_id
            )
            bundles.append(bundle)
        bundles_box[0] = bundles

    dist.broadcast_object_list(bundles_box, src=src_root, group=cross_shard_group)
    bundles = bundles_box[0]
    assert bundles is not None
    if not bundles:
        # Empty batch is a no-op; skip the remaining collectives rather
        # than firing them with zero ops (NCCL tolerates zero ops but
        # the extra barriers aren't worth it).
        return [] if role == "dst" else None

    # --- Step 3: dst injects, collects N dst-block lists. -------------
    dst_blocks_box: List[Optional[List[List[int]]]] = [None]
    if role == "dst":
        assert engine is not None
        all_dst_blocks: List[List[int]] = []
        for bundle in bundles:
            dst_blocks = engine.inject_request(bundle)
            all_dst_blocks.append(dst_blocks.tolist())
        dst_blocks_box[0] = all_dst_blocks

    dist.broadcast_object_list(dst_blocks_box, src=dst_root, group=cross_shard_group)
    all_dst_blocks = dst_blocks_box[0]
    assert all_dst_blocks is not None
    assert len(all_dst_blocks) == len(bundles)

    # --- Step 4: concatenate per-request plans, run one collective. ---
    # TP-major rank layout within each shard: (tp, pp) lives at offset
    # ``pp * tp_size + tp``.
    def _src_rank_of(tp: int, pp: int, _start=src_root, _tp=src_layout.tp_size) -> int:
        return _start + pp * _tp + tp

    def _dst_rank_of(tp: int, pp: int, _start=dst_root, _tp=dst_layout.tp_size) -> int:
        return _start + pp * _tp + tp

    all_ops: List[KVMigrationOp] = []
    for bundle, dst_block_ids in zip(bundles, all_dst_blocks):
        all_ops.extend(
            build_kv_migration_plan(
                bundle,
                src_global_rank_of=_src_rank_of,
                dst_global_rank_of=_dst_rank_of,
                dst_block_ids=dst_block_ids,
            )
        )

    if role == "src":
        memory_buffer = engine.context.memory_buffer
        layout = src_layout
    elif role == "dst":
        memory_buffer = engine.context.memory_buffer
        layout = dst_layout
    else:
        memory_buffer = torch.empty(0, device=torch.cuda.current_device())
        layout = dst_layout

    my_rank = dist.get_rank()
    my_src_pp_stage = (
        (my_rank - src_root) // src_layout.tp_size if role == "src" else 0
    )
    my_dst_pp_stage = (
        (my_rank - dst_root) // dst_layout.tp_size if role == "dst" else 0
    )
    src_layers_per_pp = src_layout.num_layers_total // src_layout.pp_size
    dst_layers_per_pp = dst_layout.num_layers_total // dst_layout.pp_size
    my_src_pp_layer_offset = my_src_pp_stage * src_layers_per_pp
    my_dst_pp_layer_offset = my_dst_pp_stage * dst_layers_per_pp

    execute_kv_migration_plan(
        all_ops,
        memory_buffer,
        layout,
        cross_shard_group,
        my_src_head_offset=my_src_head_offset,
        my_dst_head_offset=my_dst_head_offset,
        my_src_pp_layer_offset=my_src_pp_layer_offset,
        my_dst_pp_layer_offset=my_dst_pp_layer_offset,
    )

    # --- Step 4b: Mamba/hybrid state per request. ---------------------
    # No batched equivalent today — each request owns a distinct mamba
    # slot, and transport reads the slot tensor directly. Called in a
    # loop; the per-request send/recv is already small compared to the
    # attention KV.
    if src_layout.mamba is not None:
        assert dst_layout.mamba is not None, (
            "src layout has Mamba state but dst layout does not"
        )
        for i, bundle in enumerate(bundles):
            src_req_id = request_ids_src[i] if role == "src" else None
            dst_req_id = bundle.request_id
            src_state_idx = (
                engine.get_mamba_state_idx_for(src_req_id) if role == "src" else None
            )
            dst_state_idx = (
                engine.get_mamba_state_idx_for(dst_req_id) if role == "dst" else None
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

    # --- Step 5: src detaches each slot; returns new ids to dst. -----
    if role == "src":
        for req_id in request_ids_src:
            engine.detach_request(req_id, keep_blocks=False)
        return None
    if role == "dst":
        return [b.request_id for b in bundles]
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

    routing_shape = obj.get("routing_indices_shape")
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
        policy_epoch=[tuple(p) for p in obj.get("policy_epoch", [])],
        num_kv_blocks=obj["num_kv_blocks"],
        last_block_offset=obj["last_block_offset"],
        src_block_ids=list(obj["src_block_ids"]),
        src_layout=_deserialize_layout(obj.get("src_layout")),
        dst_layout=_deserialize_layout(obj.get("dst_layout")),
        routing_indices_shape=(
            tuple(routing_shape) if routing_shape is not None else None
        ),
        routing_indices_dtype_name=obj.get("routing_indices_dtype_name"),
        routing_indices_flat=list(obj.get("routing_indices_flat", [])),
    )
