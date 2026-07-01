# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregation runtime for a DynamicInferenceEngine.

All disaggregated prefill<->decode state and the 2-hop KV hand-off live here.
A ``DynamicInferenceEngine`` holds one ``DisaggEngineRuntime`` (or ``None`` for
a normal aggregated engine); every disagg branch in the engine is a guarded
delegation into this class, so the non-disaggregated path stays clean.

The runtime branches only on ``backend.is_pull`` (one-sided NIXL vs two-sided
NCCL); everything else is transport-agnostic. See the package docstring for the
control-plane protocol.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout
from megatron.core.inference.disaggregation.kv_transfer_pull import (
    post_pull_request_kv,
    pull_request_meta,
    pull_static_meta,
)
from megatron.core.inference.disaggregation.kv_transfer_push import (
    post_recv_request_kv_resharded,
    send_request_kv_resharded,
)
from megatron.core.inference.disaggregation.mamba_reshard import MambaShardLayout
from megatron.core.inference.disaggregation.transfer_backends.base import (
    PullRegion,
    construct_kv_transport_backend,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_pg_src_rank, nvtx_range_pop, nvtx_range_push

try:
    import msgpack
except ImportError:
    msgpack = None


class DisaggEngineRuntime:
    """Disaggregation state + the 2-hop KV hand-off for one engine.

    Constructed by :meth:`DynamicInferenceEngine.set_disaggregation_config`
    before ``start_listening_to_data_parallel_coordinator``.

    Prefill KV staging lives on the context (``context.disagg_staged_kv``,
    populated by the controller before the slot is freed); :meth:`send_kv`
    drains it. This object holds the in-flight transfer state:

    - ``pending_{sends,recvs}``: in-flight transfers, reaped one step later in
      :meth:`complete_pending` (collective across MP ranks since coordinator
      messages are TP-broadcast), so the transfer overlaps the engine step.
    - ``max_inflight``: depth window bounding concurrent hand-offs (and staged-KV
      memory) so prefill can't outrun decode.
    - ``pending_acks``: (decode, one-sided) request_ids whose READ has drained,
      queued for a KV_READ_DONE ack so the coordinator releases a credit + the
      prefill releases the request's pinned KV blocks.
    - ``pull_static_metas``: (prefill, one-sided) per-MP-rank request-invariant
      pull metadata, gathered once (lazily on the first publish); None until
      gathered, a list on the MP coordinator, [] on other ranks.
    """

    def __init__(
        self, engine, *, role, instance_layouts, identity, world_group,
        spawn_coordinator, disagg_router="round_robin", kv_transport_backend="nccl",
    ):
        """Args:
            engine: the owning ``DynamicInferenceEngine``.
            role: ``"prefill"`` or ``"decode"``.
            instance_layouts: KV-shard layout dicts for every rank of this
                instance (so the coordinator can build reshard plans).
            identity: unique ZMQ identity for this instance's MP-coordinator
                (must differ across shards/instances).
            world_group: process group spanning all disagg ranks (used to
                broadcast the coordinator address across shards).
            spawn_coordinator: whether THIS rank spawns the single coordinator.
            disagg_router: name of the routing policy the coordinator resolves
                (registered via ``register_disagg_router``; default round-robin).
            kv_transport_backend: KV transport, ``"nccl"`` (two-sided push) or
                ``"nixl"`` (one-sided pull). Explicit -- chosen by the caller, not
                an env var or auto-detection.
        """
        assert role in ("prefill", "decode")
        self.engine = engine
        self.role = role
        self.instance_layouts = instance_layouts
        self.identity = identity
        self.world_group = world_group
        self.spawn_coordinator = spawn_coordinator
        self.router_name = disagg_router
        self.kv_transport_backend = kv_transport_backend

        # In-flight transfer state.
        self.backend = None  # lazily-created KV transport backend
        self.pending_sends = {}  # request_id -> PrefillHandoff
        self.pending_recvs = {}  # request_id -> (recv, prompt, sampling_params)
        self.pending_acks = []
        self.pull_static_metas = None
        # Backpressure window: max KV transfers posted-but-not-yet-reaped at once,
        # bounding concurrent transfers, staged-KV memory, and how far prefill can
        # run ahead of decode. TODO(peter): test what values are good for this.
        self.max_inflight = 8
        # (prefill, one-sided, hybrid) depth of the reset-safe Mamba hold-ring
        # published end-states are copied into; capped to the live slot count.
        # TODO(peter): test what values are good for this.
        self.mamba_hold_slots = 64

        # Precompute the immutable per-rank/instance layouts once: they're fixed
        # for the engine's life, so building them here (instead of rescanning +
        # rebuilding on every hand-off) keeps the send/recv path cheap.
        rank = dist.get_rank()
        self.instance_kv_layouts = self.layouts(instance_layouts)
        self.instance_mamba_layouts = self.mamba_layouts(instance_layouts)
        self.my_layout = next(
            (l for l in self.instance_kv_layouts if l.global_rank == rank), None
        )
        assert (
            self.my_layout is not None
        ), f"rank {rank} not found in its disagg instance layouts"
        self.my_mamba_layout = next(
            (m for m in self.instance_mamba_layouts if m.global_rank == rank), None
        )

        # The prefill controller stages each finished request's KV into the
        # context (while the slot is still valid) -- the engine's finish loop
        # runs after the context has freed the slot. send_kv drains it.
        ctx = self.context
        ctx.disagg_stage_prefill_kv = role == "prefill"
        ctx.disagg_staged_kv = {}
        # request_id -> prompt block count, recorded at SUBMIT. Caps the exported
        # KV to the prompt-covering blocks so it matches the decode side's
        # header-free block_count = ceil(prompt_len/block_size); the prefill
        # otherwise allocates one extra block for its (discarded) generated token
        # when prompt_len is block-aligned.
        ctx.disagg_prompt_block_count = {}
        # One-sided (pull) backends must register their KV buffers and set
        # disagg_pull_mode *before* the first prefill completes, so the
        # controller's staging hook captures block references rather than copying
        # a staging tensor. Construct eagerly here for pull backends; push (NCCL)
        # backends keep their lazy first-use init unchanged.
        ctx.disagg_pull_mode = False
        backend = construct_kv_transport_backend(self.kv_transport_backend)
        if backend.is_pull:
            backend.init()
            self.register_pull_regions(backend)  # sets disagg_pull_mode
            self.backend = backend

    # --- engine proxies (read at call time; set on the engine in
    # start_listening after this runtime is constructed) -----------------
    @property
    def context(self):
        return self.engine.context

    @property
    def pg_collection(self):
        return self.engine.pg_collection

    @property
    def is_mp_coordinator(self):
        return self.engine.is_mp_coordinator

    @property
    def use_coordinator(self):
        return self.engine.use_coordinator

    @property
    def socket_for_receiving_requests(self):
        return self.engine.socket_for_receiving_requests

    # --- layout helpers --------------------------------------------------
    def layouts(self, dicts):
        # Strip the optional hybrid ``mamba`` sub-dict; it's a separate layout.
        return [KVShardLayout(**{k: v for k, v in d.items() if k != "mamba"}) for d in dicts]

    def mamba_layouts(self, dicts):
        """MambaShardLayout list for the hybrid dicts that carry a ``mamba``
        sub-dict (empty for non-hybrid models)."""
        return [MambaShardLayout(**d["mamba"]) for d in dicts if d.get("mamba")]

    # --- backend + one-sided registration --------------------------------
    def get_backend(self):
        """Lazily construct the KV transport backend named by the disaggregation
        config (``kv_transport_backend``: ``"nccl"`` two-sided push, or ``"nixl"``
        one-sided pull). The runtime branches on ``backend.is_pull`` -- everything
        else is transport-agnostic.

        One-sided backends register this rank's paged KV (+ Mamba) buffers once
        here, so a peer can READ entries by block/slot index without any
        per-request registration."""
        if self.backend is None:
            backend = construct_kv_transport_backend(self.kv_transport_backend)
            backend.init()
            if backend.is_pull:
                self.register_pull_regions(backend)
            self.backend = backend
        return self.backend

    def register_pull_regions(self, backend):
        """Register this rank's KV buffers with a one-sided backend, once, so a
        peer can READ any entry by index (no per-request registration). Each
        region is ``(tensor, index_axis)`` -- the axis that enumerates entries:
        KV blocks on axis 2 of ``(2, L, blocks, BS, H, HD)``, Mamba slots on axis
        1 of ``(layers, slots, *state)``. Also sets ``context.disagg_pull_mode``,
        which switches the prefill staging hook to capture block references
        instead of copying.

        Three region kinds:
        - KV blocks: registered by reference on both sides; the decode READs the
          prefill's blocks in place, kept alive by prefix-cache retention + the
          hand-off pin. This is the large payload -- never copied.
        - Mamba end-state (hybrid): the prefill can't expose its live slot (LIFO-
          recycled and reset mid-rollout, so a READ would race reuse), so it
          publishes into a disagg-owned hold-ring and registers that; the decode
          reads into its live slot. See the prefill branch below.
        - Mamba snapshots (hybrid): registered by reference on both sides -- the
          snapshot pool isn't reset mid-rollout and the KV pin already keeps a
          published request's snapshots alive, so no ring is needed.
        """
        ctx = self.context
        regions = {"kv": PullRegion(ctx.memory_buffer, 2)}
        if getattr(ctx, "is_hybrid_model", False):
            conv = getattr(ctx, "mamba_conv_states", None)
            ssm = getattr(ctx, "mamba_ssm_states", None)
            if conv is not None and ssm is not None:
                if self.role == "prefill":
                    # Register a hold-ring (not the live slots): the engine copies
                    # each published end-state into a ring slot, reused only after
                    # n_ring more publishes -- long past the decode's read lag, so
                    # no pinning needed. Capped at the live slot count.
                    n_ring = min(int(conv.shape[1]), self.mamba_hold_slots)
                    ctx.disagg_mamba_hold_conv = torch.empty(
                        (conv.shape[0], n_ring, *conv.shape[2:]),
                        dtype=conv.dtype, device=conv.device,
                    )
                    ctx.disagg_mamba_hold_ssm = torch.empty(
                        (ssm.shape[0], n_ring, *ssm.shape[2:]),
                        dtype=ssm.dtype, device=ssm.device,
                    )
                    ctx.disagg_mamba_hold_n = n_ring
                    ctx.disagg_mamba_hold_count = 0
                    regions["mamba_conv"] = PullRegion(ctx.disagg_mamba_hold_conv, 1)
                    regions["mamba_ssm"] = PullRegion(ctx.disagg_mamba_hold_ssm, 1)
                else:
                    # Decode: READ straight into the live slots.
                    regions["mamba_conv"] = PullRegion(conv, 1)
                    regions["mamba_ssm"] = PullRegion(ssm, 1)
            # Mamba snapshots (block-boundary states): by reference on both sides,
            # protected by the KV pin (see the snapshot note in the docstring).
            sa = getattr(ctx, "mamba_slot_allocator", None)
            if sa is not None and getattr(sa, "conv_states", None) is not None:
                regions["snap_conv"] = PullRegion(sa.conv_states, 1)
                regions["snap_ssm"] = PullRegion(sa.ssm_states, 1)
        backend.register_regions(regions)
        ctx.disagg_pull_mode = True

    def gather_pull_static_metas(self):
        """Gather every MP rank's request-invariant pull metadata to the MP
        coordinator **once** (region meta + buffer geometry never change). The
        coordinator caches the per-rank list and thereafter synthesizes each
        request's hand-off locally, so PREFILL_DONE carries no per-request gather.
        Collective across the MP group -- all ranks must call it in lockstep."""
        ctx = self.context
        _, num_layers, total_blocks, block_size, heads, hidden = ctx.memory_buffer.shape
        kv_dims = {
            "num_layers": int(num_layers), "total_blocks": int(total_blocks),
            "block_size": int(block_size), "heads": int(heads), "hidden": int(hidden),
            "elem": int(ctx.memory_buffer.element_size()),
        }
        mamba_dims = ctx._disagg_mamba_hold_dims() if hasattr(ctx, "_disagg_mamba_hold_dims") else None
        static = pull_static_meta(
            self.get_backend(), self.my_layout, kv_dims, mamba_dims
        )
        mp_group = self.pg_collection.mp
        gathered = (
            [None] * torch.distributed.get_world_size(mp_group)
            if self.is_mp_coordinator else None
        )
        torch.distributed.gather_object(
            static, gathered, dst=get_pg_src_rank(mp_group), group=mp_group
        )
        self.pull_static_metas = (
            [m for m in gathered if m is not None] if self.is_mp_coordinator else []
        )

    def publish_kv(self, request_id):
        """(prefill, one-sided backends) Build the request's hand-off **without a
        per-request gather**. Each rank's request-invariant pull metadata (region
        meta + geometry) is gathered once, lazily, on the first publish
        (:meth:`gather_pull_static_metas`); thereafter the MP coordinator
        synthesizes the per-rank list locally by merging those cached statics with
        this request's block references. The block references are replicated
        across MP ranks (every rank schedules identically), so this rank's copy is
        authoritative for all. Returns the list on the coordinator rank, ``None``
        elsewhere -- attached to PREFILL_DONE and relayed, opaque, to RECV_KV.

        No copy and no per-request registration: the decode READs the blocks from
        the registered ``memory_buffer``; they stay valid in the prefix cache
        (pinned at staging) until the read-done ack releases them. The lazy gather
        is the only collective, and it runs at most once."""
        if self.pull_static_metas is None:
            self.gather_pull_static_metas()  # one-time collective (all MP ranks)
        ref = self.context.disagg_staged_kv.pop(request_id, None)
        if not self.is_mp_coordinator:
            return None
        if ref is None:
            logging.warning(
                "disagg prefill: PREFILL_DONE publish for request %s has no staged "
                "KV ref; skipping", request_id,
            )
            return None
        request_meta = pull_request_meta(ref)
        return [{**static, **request_meta} for static in self.pull_static_metas]

    # --- send / receive --------------------------------------------------
    def send_kv(self, request_id, dst_layout_dicts):
        """(prefill) post the staged KV for ``request_id`` to the decode
        instance (resharded to its layout). Non-blocking: the send is reaped a
        step later in :meth:`complete_pending` so the transfer overlaps the
        engine step."""
        # Backpressure: block on the oldest in-flight send once the window is
        # full, so prefill doesn't run arbitrarily far ahead of decode. The
        # pending set is identical across MP ranks (TP-broadcast messages), so
        # the drain decision is collective.
        while len(self.pending_sends) >= self.max_inflight:
            oldest = next(iter(self.pending_sends))
            self.pending_sends.pop(oldest).wait()

        staged = self.context.disagg_staged_kv.pop(request_id, None)
        if staged is None:
            # No staged KV for this request: the slot was already freed and
            # re-exporting would fail. This can only happen on a stale/duplicate
            # SEND_KV (the request was never staged or already shipped). Skip
            # rather than crash the engine loop.
            logging.warning(
                "disagg prefill: SEND_KV for request %s has no staged KV; skipping",
                request_id,
            )
            return
        handoff = send_request_kv_resharded(
            self.engine, request_id, self.my_layout,
            self.instance_kv_layouts,
            self.layouts(dst_layout_dicts),
            backend=self.get_backend(), payload=staged,
            my_mamba_layout=self.my_mamba_layout,
            src_mamba_layouts=self.instance_mamba_layouts,
            dst_mamba_layouts=self.mamba_layouts(dst_layout_dicts),
        )
        if handoff is not None:
            self.pending_sends[request_id] = handoff

    def recv_kv(self, request_id, src_layout_dicts, prompt, sampling_params, handoff=None):
        """(decode) post the receive of ``request_id``'s KV. Non-blocking: the
        receive is reaped a step later in :meth:`complete_pending`, which imports
        the KV (registers the prefix-cache blocks) and admits the request --
        add_request prefix-hits and continues generation.

        Push backends (NCCL) post a matched receive against the prefill's send.
        One-sided backends (NIXL) instead allocate destination blocks and issue a
        one-sided READ of the prefill's blocks into them; ``handoff`` carries the
        per-rank region meta + source block ids, relayed opaque by the
        coordinator. Both paths yield an object with a ``finish(engine)`` that
        commits, so the rest is symmetric."""
        backend = self.get_backend()
        # Backpressure: complete + admit the oldest in-flight receive once the
        # window is full (collective: identical pending set across MP ranks).
        while len(self.pending_recvs) >= self.max_inflight:
            oldest = next(iter(self.pending_recvs))
            rv, p, sp = self.pending_recvs.pop(oldest)
            rv.finish(self.engine)
            if backend.is_pull:
                self.pending_acks.append(oldest)
            self.engine.add_request(oldest, p, sampling_params=SamplingParams.deserialize(sp))

        if backend.is_pull:
            recv = post_pull_request_kv(
                self.engine, backend, handoff, self.my_layout,
                src_layouts=self.layouts(src_layout_dicts),
                dst_layouts=self.instance_kv_layouts,
                src_mamba_layouts=self.mamba_layouts(src_layout_dicts),
                dst_mamba_layouts=self.instance_mamba_layouts,
                my_mamba_layout=self.my_mamba_layout,
            )
        else:
            recv = post_recv_request_kv_resharded(
                self.engine, self.my_layout,
                self.layouts(src_layout_dicts),
                self.instance_kv_layouts,
                prompt, backend=backend,
                my_mamba_layout=self.my_mamba_layout,
                src_mamba_layouts=self.mamba_layouts(src_layout_dicts),
                dst_mamba_layouts=self.instance_mamba_layouts,
            )
        if recv is None:
            # No KV received: for pull it means the decode KV cache was full, so
            # we admit to re-prefill. The prefill still pinned its blocks, so ack
            # to release the credit + pin (else its flow-control credit leaks).
            if backend.is_pull:
                self.pending_acks.append(request_id)
            sp = SamplingParams.deserialize(sampling_params)
            self.engine.add_request(request_id, prompt, sampling_params=sp)
            return
        self.pending_recvs[request_id] = (recv, prompt, sampling_params)

    def ready_recvs(self, is_pull):
        """Pending receive ids (insertion order) ready to admit THIS step.

        Push (NCCL): all pending -- the matched-collective recvs are waited in
        :meth:`complete_pending`'s ``finish``. Pull (one-sided): the subset whose
        READ has drained on *every* MP rank -- each rank polls its handles
        non-blockingly, then the per-rank done flags are AND-reduced over the MP
        group (``MIN`` all-reduce) so admission stays collective. Requests not yet
        done everywhere stay pending and are rechecked next step."""
        pending = list(self.pending_recvs)
        if not is_pull or not pending:
            return pending
        local = [self.pending_recvs[rid][0].poll() for rid in pending]
        flags = torch.tensor(
            [1 if d else 0 for d in local],
            dtype=torch.int32, device=self.context.memory_buffer.device,
        )
        # MIN over the MP group == logical AND: admit only where all ranks agree.
        torch.distributed.all_reduce(
            flags, op=torch.distributed.ReduceOp.MIN, group=self.pg_collection.mp
        )
        return [rid for rid, f in zip(pending, flags.tolist()) if f]

    def complete_pending(self):
        """Reap KV transfers posted on a previous step.

        Collective across the MP group: the pending sets were populated from
        TP-broadcast coordinator messages, so every rank holds the same requests
        in the same (insertion) order. Prefill: wait each send and release its
        staged KV. Decode: admit each receive that has landed, import its KV
        (registers the prefix-cache blocks), and continue generation.

        For one-sided pulls the receive completion is non-blocking: rather than
        waiting on a possibly-slow READ, :meth:`ready_recvs` polls each and admits
        only those done on *every* MP rank (deferring the rest to a later step),
        so a lagging transfer never stalls the loop. Admission stays in lockstep
        across ranks because that "done on all ranks" set is AND-reduced over the
        MP group -- if ranks admitted different requests the next forward step
        would diverge."""
        is_pull = self.get_backend().is_pull
        for request_id in list(self.pending_sends):
            self.pending_sends.pop(request_id).wait()
        for request_id in self.ready_recvs(is_pull):
            recv, prompt, sampling_params = self.pending_recvs.pop(request_id)
            imported = recv.finish(self.engine)
            if imported is None:
                # KV import failed (e.g. decode KV cache full): the request is
                # still admitted but without the handed-off blocks, so it
                # re-prefills from the prompt -- correct but slower. Surface it.
                logging.warning(
                    "disagg decode: KV import failed for request %s; "
                    "re-prefilling from prompt instead of using handed-off KV",
                    request_id,
                )
            if is_pull:
                # One-sided READ drained -> release the prefill's pin + a credit.
                self.pending_acks.append(request_id)
            sp = SamplingParams.deserialize(sampling_params)
            self.engine.add_request(request_id, prompt, sampling_params=sp)

        # Flush read-done acks: the MP coordinator tells the coordinator each
        # pulled request has drained. Queue is identical across MP ranks
        # (collective finish), so clearing it is consistent.
        if self.pending_acks:
            if self.use_coordinator and self.is_mp_coordinator:
                for rid in self.pending_acks:
                    self.socket_for_receiving_requests.send(
                        msgpack.packb([Headers.KV_READ_DONE.value, rid], use_bin_type=True)
                    )
            self.pending_acks.clear()

    # --- engine seams ----------------------------------------------------
    def registration_message(self):
        """REGISTER_ROLE msgpack: role + this instance's KV layouts + the is_pull
        flag, so the coordinator can 2-hop route + plan reshards, and apply
        credit-based flow control (bounding outstanding hand-offs to the
        prefill's hold-ring/pin window) only for pull instances."""
        is_pull = bool(getattr(self.context, "disagg_pull_mode", False))
        return msgpack.packb(
            [Headers.REGISTER_ROLE.value, self.role, self.instance_layouts, is_pull],
            use_bin_type=True,
        )

    def prepare_prefill_request(self, request_id, prompt, sampling_params):
        """Prefill-only SUBMIT: run prefill (which populates the prompt KV) and
        stop right after, so the request leaves this engine with its prompt-block
        KV intact for the hand-off. The few generated tokens are discarded; decode
        regenerates from the prompt (prefix-cache hit on the imported KV)."""
        sampling_params.num_tokens_to_generate = 1
        bs = int(self.context.block_size_tokens)
        self.context.disagg_prompt_block_count[request_id] = (len(prompt) + bs - 1) // bs

    def send_prefill_done(self, records_to_send):
        """(prefill) Instead of replying to the client, tell the coordinator each
        request finished prefill (KV staged); it names the decode target via
        RECV_KV (and SEND_KV for push backends).

        Pull backends require EVERY MP rank to publish its KV shard here and
        contribute per-rank READ descriptors -- so the publish runs on all ranks
        (records_to_send is identical across the MP group), while only the MP
        coordinator emits the ZMQ control message and attaches the gathered
        handoff. Push backends publish nothing here; the KV ships later on
        SEND_KV."""
        nvtx_range_push("coordinator_communication")
        backend = self.get_backend()
        for r in records_to_send:
            rid = r.requests[-1].request_id
            handoff = self.publish_kv(rid) if backend.is_pull else None
            if self.is_mp_coordinator:
                parts = [Headers.PREFILL_DONE.value, rid]
                if handoff is not None:
                    parts.append(handoff)
                self.socket_for_receiving_requests.send(
                    msgpack.packb(parts, use_bin_type=True)
                )
        nvtx_range_pop("coordinator_communication")

    def release_pinned(self, request_id):
        """(prefill, one-sided) the decode finished its READ -- release the
        request's pinned KV blocks (Mamba used the reset-safe ring, no pin)."""
        self.context.disagg_release_pinned(request_id)
