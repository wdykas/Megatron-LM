# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inter-engine request migration for heterogeneous inference shards.

Moves an in-flight generation request — prompt, generated tokens, sampling
params, logprob history, *and* its KV cache — from one
:class:`DynamicInferenceEngine` to another whose parallelism differs. The
module supplies the data structures and plan-building / gather-scatter
helpers; the production migration handler in
:mod:`megatron.rl.inference.multi_shard` (``_on_migrate_batch_signal``)
drives them directly with one-sided NVSHMEM ``put_signal`` /
``signal_wait`` on the migration stream — no NCCL collective and no
``barrier_all``.

- :class:`RequestMigrationBundle` — metadata envelope that travels with
  the request (everything except the KV tensors themselves).
- :class:`KVLayout` — descriptor of one shard's KV-tensor geometry
  (TP/PP size, head count, block size, MLA).
- :func:`build_kv_migration_plan` — given both shards' layouts + block
  ids, returns a flat list of :class:`KVMigrationOp` describing the
  tensor slices each rank sends / receives. Handles heterogeneous TP
  *and* heterogeneous PP by taking the (layer × head) intersection of
  each (src rank, dst rank) pair's ownership.
- :class:`MambaLayout` / :class:`MambaMigrationOp` /
  :func:`build_mamba_migration_plan` — analogous machinery for hybrid
  models. Mamba per-request state has two TP-sharded buffers (conv on
  ``d_inner``, ssm on ``nheads``); the plan emits one op per
  ``(src_pp × dst_pp × src_tp × dst_tp)`` quadruple with non-empty
  layer overlap, each carrying the conv- and ssm-dim intersections to
  transfer.
- :func:`_gather_kv_slice` / :func:`_scatter_kv_slice` — local-buffer
  side of each op; the handler calls them directly around the NVSHMEM
  put/wait.

Engine-side surgery (``snapshot_request`` / ``detach_request`` /
``inject_request``) lives alongside :class:`DynamicInferenceEngine`.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist


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
    plain Python types. ``generated_log_probs`` is serialized as a list;
    it's CPU-resident in
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


# ---- Mamba state migration -------------------------------------------------
#
# Hybrid (Mamba) models carry per-request conv + ssm state alongside the
# attention KV cache. The two state tensors share the same TP axis but
# shard on different dimensions:
#
#   conv_state: (num_layers_pp, max_requests, d_inner_local_tp, d_conv)
#   ssm_state:  (num_layers_pp, max_requests, nheads_local_tp, headdim, d_state)
#
# A single TP rank owns ``d_inner_local_tp`` rows of the conv state and
# ``nheads_local_tp`` rows of the ssm state, both partitioned at the
# same TP boundaries (``d_inner_total = d_inner_local_tp * tp_size``,
# same for nheads). A migration op moves the (layer × d_inner × nheads)
# intersection between one src rank and one dst rank — same shape of
# (pp, tp) overlap as KV migration, just on different state dims.


@dataclass
class MambaLayout:
    """Per-shard Mamba-state layout descriptor.

    All fields except ``tp_size`` / ``pp_size`` are whole-model
    invariants and must match between src and dst layouts. Mirrors
    :class:`KVLayout` for the SSM/conv state buffers a hybrid model
    keeps per-request.
    """

    tp_size: int
    pp_size: int
    num_layers_total: int
    # ``d_inner_total`` shards across TP for the conv state.
    d_inner_total: int
    # ``nheads_total`` shards across TP for the SSM state.
    nheads_total: int
    d_conv: int
    headdim: int
    d_state: int
    # Full model layer-type pattern (length = total transformer
    # layer count). Required for hetero-PP migration because each
    # PP rank only writes to mamba state buffer entries for the
    # mamba layers in its transformer-layer window — with
    # heterogeneous PP, src and dst own different mamba subsets and
    # the plan must intersect them. Stored as a tuple of single-char
    # symbols ('M', 'E', '*', etc. — see
    # ``megatron.core.ssm.mamba_hybrid_layer_allocation.Symbols``).
    layer_type_list: Tuple[str, ...] = ()

    def assert_divisible(self) -> None:
        """TP divides both d_inner and nheads."""
        assert self.d_inner_total % self.tp_size == 0, (
            f"d_inner_total={self.d_inner_total} must be divisible by "
            f"tp_size={self.tp_size}"
        )
        assert self.nheads_total % self.tp_size == 0, (
            f"nheads_total={self.nheads_total} must be divisible by "
            f"tp_size={self.tp_size}"
        )

    @property
    def d_inner_per_tp(self) -> int:
        return self.d_inner_total // self.tp_size

    @property
    def nheads_per_tp(self) -> int:
        return self.nheads_total // self.tp_size

    def mamba_indices_for_pp_rank(self, pp_rank: int) -> List[int]:
        """Mamba-buffer indices that ``pp_rank`` of this shard owns.

        Each PP rank owns a contiguous range of transformer layers
        (uniform partition: ``[pp * L/pp_size, (pp+1) * L/pp_size)``);
        among those, the mamba-typed ones populate the per-request
        mamba state buffer at indices given by their position among
        all mamba layers in the model (the mamba_layer_map). Returns
        the buffer indices this PP rank's engine actually writes,
        sorted ascending.

        Empty ``layer_type_list`` falls back to a uniform
        ``num_layers_total / pp_size`` slice for backward compat
        with callers that don't propagate the layer pattern.
        """
        if not self.layer_type_list:
            # Fallback: assume uniform partition of mamba layers
            # across PP. Correct only when matched PP and mamba
            # layers are uniformly distributed — same assumption
            # the v1 (matched-PP-only) transport made.
            per_pp = self.num_layers_total // self.pp_size
            return list(range(pp_rank * per_pp, (pp_rank + 1) * per_pp))
        n_total = len(self.layer_type_list)
        per_pp = n_total // self.pp_size
        lo = pp_rank * per_pp
        hi = lo + per_pp
        # Walk all mamba layers in document order; emit those whose
        # global transformer index falls in this PP rank's window.
        # Index in the buffer = position among mamba layers globally.
        out: List[int] = []
        mamba_count = 0
        for g in range(n_total):
            if self.layer_type_list[g] == "M":
                if lo <= g < hi:
                    out.append(mamba_count)
                mamba_count += 1
        return out


@dataclass
class MambaMigrationOp:
    """One conv+ssm tensor transfer in the Mamba migration plan.

    Represents "source rank ``src_rank`` sends its slice of the
    mamba state at buffer indices ``mamba_layer_indices`` — conv rows
    ``conv_d_inner_range`` and ssm rows ``ssm_nheads_range`` — to dst
    rank ``dst_rank`` for request ``request_id``". A single op drives
    both conv and ssm transfers because the two share the same TP
    boundary; one (src, dst) rank pair always overlaps on both
    dimensions or neither.

    ``mamba_layer_indices`` is the list of mamba-buffer indices that
    BOTH the src PP rank and the dst PP rank own (the intersection
    of their per-PP-rank mamba ownership). Under matched PP and
    uniform mamba layer distribution, this collapses to a contiguous
    slice; with heterogeneous PP it can be sparse.
    """

    src_rank: int
    dst_rank: int
    request_id: int
    # Which one this is among the per-request mamba ops; used to
    # derive disjoint flag-slot indices from the shared per-request
    # key budget. See ``MAX_OPS_PER_REQ`` in
    # ``megatron.rl.inference.multi_shard``.
    op_index_in_request: int
    mamba_layer_indices: Tuple[int, ...]  # buffer indices along layer dim
    conv_d_inner_range: Tuple[int, int]  # conv d_inner indices, [start, end)
    ssm_nheads_range: Tuple[int, int]  # ssm nheads indices, [start, end)


def build_mamba_migration_plan(
    bundle: RequestMigrationBundle,
    src_layout: MambaLayout,
    dst_layout: MambaLayout,
    src_global_rank_of: "Callable[[int, int], int]",
    dst_global_rank_of: "Callable[[int, int], int]",
) -> List[MambaMigrationOp]:
    """Compute the conv + ssm transfers that move a request's mamba
    state from the source shard to the destination shard.

    Heterogeneous TP **and** PP supported via the same
    intersection-of-overlapping-ranges pattern as
    :func:`build_kv_migration_plan`. The per-request mamba state
    buffer indexes by *global* mamba layer (each PP rank writes only
    to entries for the mamba layers it actually owns); under hetero
    PP src and dst own different subsets, so for each
    ``(src_pp, dst_pp)`` pair we intersect their mamba ownership and
    emit ops carrying the resulting buffer indices via
    ``MambaMigrationOp.mamba_layer_indices``. With matched PP and
    uniform mamba distribution the intersection collapses to a
    contiguous slice.

    For each ``(src_pp × dst_pp × src_tp × dst_tp)`` quadruple, emit
    one op iff:

    - the mamba-buffer-index intersection between ``src_pp`` and
      ``dst_pp`` is non-empty, **and**
    - both the conv ``d_inner`` ranges and the ssm ``nheads`` ranges
      overlap on TP. The two TP-shard dims partition at the same
      boundaries (same TP rank count), so an empty d_inner overlap
      implies an empty nheads overlap and vice versa.

    Args:
        bundle: Request envelope. Only ``request_id`` is read.
        src_layout / dst_layout: shard-level mamba layouts. Whole-model
            invariants must match; PP sizes may differ. Both sides must
            populate ``layer_type_list`` for hetero-PP correctness;
            empty ``layer_type_list`` falls back to uniform mamba
            ownership per PP rank (legacy matched-PP behaviour).
        src_global_rank_of / dst_global_rank_of: ``(tp_rank, pp_rank)
            -> global_rank`` for each shard.

    Returns:
        Flat list of :class:`MambaMigrationOp`. Each op's
        ``op_index_in_request`` is its 0-based position in the
        returned list.
    """
    src_layout.assert_divisible()
    dst_layout.assert_divisible()
    assert src_layout.num_layers_total == dst_layout.num_layers_total, (
        "num_layers_total must match across shards"
    )
    assert src_layout.d_inner_total == dst_layout.d_inner_total, (
        f"d_inner_total must match: src={src_layout.d_inner_total}, "
        f"dst={dst_layout.d_inner_total}"
    )
    assert src_layout.nheads_total == dst_layout.nheads_total, (
        f"nheads_total must match: src={src_layout.nheads_total}, "
        f"dst={dst_layout.nheads_total}"
    )
    assert src_layout.d_conv == dst_layout.d_conv, "d_conv mismatch"
    assert src_layout.headdim == dst_layout.headdim, "headdim mismatch"
    assert src_layout.d_state == dst_layout.d_state, "d_state mismatch"
    assert src_layout.layer_type_list == dst_layout.layer_type_list, (
        "layer_type_list must match across shards (model invariant)"
    )

    src_d_inner = src_layout.d_inner_per_tp
    dst_d_inner = dst_layout.d_inner_per_tp
    src_nheads = src_layout.nheads_per_tp
    dst_nheads = dst_layout.nheads_per_tp

    # Pre-compute per-PP-rank mamba buffer ownership on each side.
    src_pp_indices = [
        src_layout.mamba_indices_for_pp_rank(p) for p in range(src_layout.pp_size)
    ]
    dst_pp_indices = [
        dst_layout.mamba_indices_for_pp_rank(p) for p in range(dst_layout.pp_size)
    ]

    ops: List[MambaMigrationOp] = []
    op_idx = 0
    for src_pp in range(src_layout.pp_size):
        src_set = set(src_pp_indices[src_pp])
        for dst_pp in range(dst_layout.pp_size):
            dst_set = set(dst_pp_indices[dst_pp])
            overlap = sorted(src_set & dst_set)
            if not overlap:
                continue
            overlap_t = tuple(overlap)
            for sr in range(src_layout.tp_size):
                src_d_lo = sr * src_d_inner
                src_d_hi = src_d_lo + src_d_inner
                src_h_lo = sr * src_nheads
                src_h_hi = src_h_lo + src_nheads
                for dr in range(dst_layout.tp_size):
                    dst_d_lo = dr * dst_d_inner
                    dst_d_hi = dst_d_lo + dst_d_inner
                    dst_h_lo = dr * dst_nheads
                    dst_h_hi = dst_h_lo + dst_nheads
                    d_lo = max(src_d_lo, dst_d_lo)
                    d_hi = min(src_d_hi, dst_d_hi)
                    h_lo = max(src_h_lo, dst_h_lo)
                    h_hi = min(src_h_hi, dst_h_hi)
                    if d_lo >= d_hi:
                        continue
                    ops.append(
                        MambaMigrationOp(
                            src_rank=src_global_rank_of(sr, src_pp),
                            dst_rank=dst_global_rank_of(dr, dst_pp),
                            request_id=bundle.request_id,
                            op_index_in_request=op_idx,
                            mamba_layer_indices=overlap_t,
                            conv_d_inner_range=(d_lo, d_hi),
                            ssm_nheads_range=(h_lo, h_hi),
                        )
                    )
                    op_idx += 1
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
