# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inter-engine request migration for heterogeneous inference shards.

Moves an in-flight generation request — prompt, generated tokens, sampling
params, logprob history, *and* its KV cache — from one
:class:`DynamicInferenceEngine` to another whose parallelism differs. KV
tensor transport rides on the symmetric NVSHMEM staging buffer set up
by :mod:`megatron.core.inference.nvshmem_migration`; one-sided
``put_signal`` delivers the data and arms a per-op flag, and the dst
side ``signal_wait``s on the same flag from the migration stream — no
NCCL collective and no ``barrier_all``. The module supplies:

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
  one-sided NVSHMEM ``put_signal`` over a symmetric staging buffer.
- :func:`execute_mamba_state_transport` — same NVSHMEM-direct path for
  per-request Mamba conv + SSM state on hybrid models.

Engine-side surgery (``snapshot_request`` / ``detach_request`` /
``inject_request``) lives alongside :class:`DynamicInferenceEngine`.
The production migration handler in :mod:`megatron.rl.inference.multi_shard`
inlines the NVSHMEM gather/put/scatter sequence directly for the
non-blocking signal path.
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
    dst_block_ids: Optional[List[int]] = None,
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
            order as ``bundle.src_block_ids``. **Required on dst-side
            ranks** (used by the scatter step). On src-side ranks may
            be ``None`` — the gather only consults ``bundle.src_block_ids``,
            and the field is filled with a same-length placeholder so
            the dst-only logic that reads ``op.dst_block_ids`` in
            symmetric code paths still has a value of the right shape.

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
    if dst_block_ids is None:
        # Src-side build: the gather doesn't read this field, but every
        # op carries one for shape symmetry with dst-side ops.
        dst_block_ids = list(bundle.src_block_ids)
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
    """Flatten the bundle to a msgpack-compatible dict.

    ``src_layout`` / ``dst_layout`` are intentionally *not* on the wire
    — they're invariant per (src_shard, dst_shard) pair and the
    receiving handler restamps them from its local ``_migration_meta``,
    saving N× redundant serialization per batch.
    """
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


def execute_kv_migration_plan(
    ops: List[KVMigrationOp],
    memory_buffer: torch.Tensor,
    layout: KVLayout,
    *,
    my_src_head_offset: int = 0,
    my_dst_head_offset: int = 0,
    my_src_pp_layer_offset: int = 0,
    my_dst_pp_layer_offset: int = 0,
) -> None:
    """Execute a KV migration plan over the symmetric NVSHMEM heap.

    For each :class:`KVMigrationOp` the current rank plays at most two
    roles:

    - if ``rank == op.src_rank``: gather the op's slice from the local
      ``memory_buffer`` into a symmetric staging slot, then
      ``put_signal`` it to the matching slot + flag on ``op.dst_rank``;
    - if ``rank == op.dst_rank``: ``signal_wait`` for the per-op flag
      to arrive (stream-ordered GPU wait, not a host barrier), then
      scatter the slot back into the local ``memory_buffer``.

    Walks ``ops`` in deterministic order on every rank so the same
    ``StagingArena`` offsets are assigned on src and dst — that's the
    invariant that lets each op's bytes land at the matching offset on
    the destination's symmetric staging buffer.

    **Global vs local head/layer indices.** ``op.head_range`` and
    ``op.layer_range`` are in the model's *global* index space. Each
    rank's ``memory_buffer`` holds only its local TP/PP slice. The
    ``my_src_*_offset`` / ``my_dst_*_offset`` parameters tell the
    transport where this rank's slice starts in global space, so the
    op's global range can be translated into the local buffer's index
    range. Callers that aren't in the src (resp. dst) shard can leave
    those offsets at their default ``0``.

    Args:
        ops: Output of :func:`build_kv_migration_plan`.
        memory_buffer: This rank's KV cache buffer. Contract:
            - non-MLA shape:
              ``(2, num_layers_pp, total_blocks, block_size, heads_per_tp, head_dim)``
            - MLA shape:
              ``(num_layers_pp, total_blocks, block_size, kv_reduced_dim)``
        layout: This rank's KV layout. Its ``is_mla`` flag selects
            between the two buffer shapes.
        my_src_head_offset: Global head index at which this rank's src
            buffer slice starts. Only consulted when this rank appears
            as ``op.src_rank`` for some op.
        my_dst_head_offset: Global head index at which this rank's dst
            buffer slice starts. Only consulted when this rank appears
            as ``op.dst_rank`` for some op.
        my_src_pp_layer_offset / my_dst_pp_layer_offset: Same idea for
            the PP layer dimension.

    Requires :mod:`megatron.core.inference.nvshmem_migration` to be
    initialized; raises a :class:`RuntimeError` otherwise.
    """
    from megatron.core.inference import nvshmem_migration as _nv

    if not _nv.is_initialized():
        raise RuntimeError(
            "execute_kv_migration_plan requires NVSHMEM migration to be "
            "initialized. Call nvshmem_migration.initialize(...) before "
            "invoking the cross-shard transport."
        )

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
) -> None:
    """Transport Mamba conv + SSM state for one request across shards.

    For v1 we require **matched TP and PP** between the shards so the
    state tensor shapes are identical on both sides — each src rank
    sends its per-layer state slice to the corresponding dst rank
    (``src_ranks[i] → dst_ranks[i]``) with no reshape. Heterogeneous
    TP/PP on Mamba would need a reshape layer analogous to
    :func:`build_kv_migration_plan`; that's deferred.

    Routes per (rank, state) pair through the symmetric NVSHMEM
    staging buffer: src ``put_signal``s a contiguous conv (or ssm)
    slice and arms its flag, dst ``signal_wait``s and scatters back
    into its state slot. The Mamba state tensors themselves live in
    the symmetric heap, but conv and ssm slices are at distinct
    offsets within them, so we still stage to keep one contiguous
    transfer per (kind, pair). No NCCL collective and no
    ``barrier_all``.

    ``role`` is informational and used for assertion clarity. A pure
    attention model (``layout.mamba is None``) is a no-op.

    Requires :mod:`megatron.core.inference.nvshmem_migration` to be
    initialized; raises a :class:`RuntimeError` otherwise.
    """
    if layout.mamba is None:
        return  # pure attention model; nothing to do

    assert len(src_ranks) == len(dst_ranks), (
        f"Mamba state transport (v1) requires matched TP/PP between "
        f"shards — got src_ranks={src_ranks} and dst_ranks={dst_ranks}"
    )

    from megatron.core.inference import nvshmem_migration as _nv

    if not _nv.is_initialized():
        raise RuntimeError(
            "execute_mamba_state_transport requires NVSHMEM migration to "
            "be initialized. Call nvshmem_migration.initialize(...) before "
            "invoking the cross-shard transport."
        )

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


def deserialize_bundle(obj: dict) -> RequestMigrationBundle:
    """Inverse of :func:`serialize_bundle`. Layouts are restamped by
    the caller from its local ``_migration_meta``."""
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
        routing_indices_shape=(
            tuple(routing_shape) if routing_shape is not None else None
        ),
        routing_indices_dtype_name=obj.get("routing_indices_dtype_name"),
        routing_indices_flat=list(obj.get("routing_indices_flat", [])),
    )
