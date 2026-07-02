# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disaggregation runtime for a DynamicInferenceEngine.

All disaggregated prefill<->decode state and the 2-hop KV hand-off live here.
A DynamicInferenceEngine holds one DisaggEngineRuntime (or None for a normal
aggregated engine); every disagg branch in the engine delegates into this
class. The runtime branches only on self.is_pull (one-sided NIXL vs two-sided
NCCL); everything else is transport-agnostic. See the package docstring for
the control-plane protocol.
"""

from __future__ import annotations

import functools
import logging

import msgpack
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.kv_reshard import KVShardLayout
from megatron.core.inference.disaggregation.kv_transfer_pull import (
    _ctx_kv_dims,
    post_pull_request_kv,
    pull_request_meta,
    pull_static_meta,
)
from megatron.core.inference.disaggregation.kv_transfer_push import (
    post_recv_request_kv_resharded,
    send_request_kv_resharded,
)
from megatron.core.inference.disaggregation.mamba_layout import MambaShardLayout
from megatron.core.inference.disaggregation.transfer_backends.base import (
    PullRegion,
    construct_kv_transport_backend,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.utils import get_pg_src_rank, nvtx_range_pop, nvtx_range_push


def pull_only(method):
    """Assert at call time that the method only runs on a pull (one-sided)
    backend."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        assert self.is_pull, (
            f"{method.__name__} is pull-only (one-sided/NIXL), but this engine "
            f"uses a push backend ({self.kv_transport_backend})"
        )
        return method(self, *args, **kwargs)

    return wrapper


def push_only(method):
    """Assert at call time that the method only runs on a push (two-sided)
    backend."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        assert not self.is_pull, (
            f"{method.__name__} is push-only (two-sided/NCCL), but this engine "
            f"uses a pull backend ({self.kv_transport_backend})"
        )
        return method(self, *args, **kwargs)

    return wrapper


def kv_layouts(dicts):
    """Build the KVShardLayout list from wire dicts, stripping the optional
    hybrid `mamba` sub-dict (a separate layout)."""
    return [KVShardLayout(**{k: v for k, v in d.items() if k != "mamba"}) for d in dicts]


def mamba_layouts(dicts):
    """Build the MambaShardLayout list from the wire dicts that carry a
    `mamba` sub-dict (empty for non-hybrid models)."""
    return [MambaShardLayout(**d["mamba"]) for d in dicts if d.get("mamba")]


class DisaggEngineRuntime:
    """Disaggregation state and the 2-hop KV hand-off for one engine.

    Constructed by DynamicInferenceEngine.set_disaggregation_config before
    start_listening_to_data_parallel_coordinator.

    Prefill KV staging lives on the context (context.disagg_staged_kv,
    populated by the controller before the slot is freed); send_kv drains it.
    This object holds the in-flight transfer state:

    - pending_{sends,recvs}: in-flight transfers, reaped one step later in
      complete_pending (collective across MP ranks since coordinator messages
      are TP-broadcast), so the transfer overlaps the engine step.
    - max_inflight: depth window bounding concurrent hand-offs and staged-KV
      memory, so prefill cannot outrun decode.
    - pending_acks: (decode, pull) request_ids whose read has drained, queued
      for a KV_READ_DONE ack so the coordinator releases an outstanding slot
      and the prefill releases the request's pinned KV blocks.
    - pull_static_metas: (prefill, pull) per-MP-rank request-invariant pull
      metadata, gathered once, lazily on the first publish; None until
      gathered, a list on the MP coordinator, [] on other ranks.
    """

    def __init__(
        self, engine, *, role, instance_layouts, identity,
        spawn_coordinator, disagg_router="round_robin", kv_transport_backend="nccl",
    ):
        """Args:
            engine: the owning DynamicInferenceEngine.
            role: "prefill" or "decode".
            instance_layouts: KV-shard layout dicts for every rank of this
                instance (so the coordinator can build reshard plans).
            identity: unique ZMQ identity for this instance's MP-coordinator
                (must differ across shards/instances).
            spawn_coordinator: whether this rank spawns the single coordinator.
            disagg_router: name of the routing policy the coordinator resolves
                (registered via register_disagg_router; default round-robin).
            kv_transport_backend: KV transport, "nccl" (two-sided push) or
                "nixl" (one-sided pull).
        """
        assert role in ("prefill", "decode")
        self.engine = engine
        self.role = role
        self.instance_layouts = instance_layouts
        self.identity = identity
        self.spawn_coordinator = spawn_coordinator
        self.router_name = disagg_router
        self.kv_transport_backend = kv_transport_backend

        # In-flight transfer state.
        self.backend = None  # lazily-created KV transport backend
        self.pending_sends = {}  # request_id -> PrefillHandoff
        self.pending_recvs = {}  # request_id -> (recv, prompt, sampling_params)
        self.pending_acks = []
        self.pull_static_metas = None
        # Backpressure window: max KV transfers posted but not yet reaped,
        # bounding concurrent transfers, staged-KV memory, and how far prefill
        # can run ahead of decode. TODO: tune.
        self.max_inflight = 8

        # Per-rank layouts are fixed for the engine's life; build them once.
        rank = dist.get_rank()
        self.instance_kv_layouts = kv_layouts(instance_layouts)
        self.instance_mamba_layouts = mamba_layouts(instance_layouts)
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
        # context while the slot is still valid; the engine's finish loop runs
        # after the context has freed the slot. send_kv drains it.
        self.context.disagg_stage_prefill_kv = role == "prefill"
        # Pull backends must register their KV buffers and set disagg_pull_mode
        # before the first prefill completes, so the controller's staging hook
        # captures block references rather than copying; construct them eagerly.
        # Push backends keep their lazy first-use init.
        backend = construct_kv_transport_backend(self.kv_transport_backend)
        self.is_pull = backend.is_pull
        if self.is_pull:
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
    def socket_for_receiving_requests(self):
        return self.engine.socket_for_receiving_requests

    # --- backend + one-sided registration --------------------------------
    def get_backend(self):
        """Lazily construct the configured KV transport backend. Only push
        backends reach the lazy path; pull backends are built and registered
        eagerly in __init__, so self.backend is already set for them."""
        if self.backend is None:
            backend = construct_kv_transport_backend(self.kv_transport_backend)
            backend.init()
            self.backend = backend
        return self.backend

    @pull_only
    def register_pull_regions(self, backend):
        """Register this rank's KV buffers with a one-sided backend, once, so
        a peer can read any entry by index with no per-request registration.
        Each region is (tensor, index_axis), the axis that enumerates entries:
        KV blocks on axis 2 of (2, L, blocks, BS, H, HD), Mamba snapshot slots
        on axis 1 of (layers, slots, *state). Also sets
        context.disagg_pull_mode, which switches the prefill staging hook to
        capture block references instead of copying.

        The decode reads the prefill's KV blocks in place, kept alive by
        prefix-cache retention plus the hand-off pin. Hybrid models also
        register the block-boundary snapshot pools; the pools are not reset
        mid-rollout and the KV pin keeps a published request's snapshots
        alive.
        """
        ctx = self.context
        regions = {"kv": PullRegion(ctx.memory_buffer, 2)}
        if ctx.is_hybrid_model:
            # The snapshot allocator is optional (present only with the Mamba
            # prefix cache); without it a hybrid hand-off degrades to re-prefill.
            sa = ctx.mamba_slot_allocator
            if sa is not None:
                regions["snap_conv"] = PullRegion(sa.conv_states, 1)
                regions["snap_ssm"] = PullRegion(sa.ssm_states, 1)
        backend.register_regions(regions)
        ctx.disagg_pull_mode = True

    @pull_only
    def gather_pull_static_metas(self):
        """Gather every MP rank's request-invariant pull metadata to the MP
        coordinator, once (region meta and buffer geometry never change). The
        coordinator caches the per-rank list and synthesizes each request's
        hand-off locally, so PREFILL_DONE carries no per-request gather.
        Collective across the MP group; all ranks must call it in lockstep."""
        static = pull_static_meta(
            self.get_backend(), self.my_layout, _ctx_kv_dims(self.context)
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

    @pull_only
    def publish_kv(self, request_id):
        """Build a finished request's read descriptors for PREFILL_DONE
        (prefill side).

        Merges the once-gathered per-rank static metadata with this request's
        block references; block ids are replicated across MP ranks, so there
        is no per-request gather. Returns the per-rank list on the MP
        coordinator, None elsewhere; the coordinator relays it to the decode
        in RECV_KV. The decode reads the blocks in place, kept alive by the
        staging pin."""
        if self.pull_static_metas is None:
            self.gather_pull_static_metas()  # one-time collective (all MP ranks)
        ref = self.context.disagg_staged_kv.pop(request_id, None)
        if not self.is_mp_coordinator:
            return None
        # A finished prefill request always staged its KV before
        # send_prefill_done; a missing ref here would emit PREFILL_DONE with
        # no handoff.
        if ref is None:
            raise RuntimeError(
                f"disagg prefill: PREFILL_DONE publish for request {request_id} "
                "has no staged KV ref"
            )
        request_meta = pull_request_meta(ref)
        return [{**static, **request_meta} for static in self.pull_static_metas]

    @push_only
    def publish_push_handoff(self, request_id):
        """Build the PREFILL_DONE hand-off for a push transport: only the
        Mamba snapshot hashes (the decode rebuilds everything else from the
        prompt). Reads the staged export without consuming it; SEND_KV drains
        it later. Returns None when there is nothing to carry."""
        if not self.is_mp_coordinator:
            return None
        # Same invariant as publish_kv: a finished prefill request always staged
        # its KV before send_prefill_done.
        staged = self.context.disagg_staged_kv[request_id]
        snapshots = staged["mamba_snapshots"]
        if snapshots is None:
            return None
        return {"snapshot_hashes": list(snapshots["block_hashes"])}

    # --- send / receive --------------------------------------------------
    @push_only
    def send_kv(self, request_id, dst_layout_dicts):
        """Post the staged KV for `request_id` to the decode instance,
        resharded to its layout. Non-blocking: the send is reaped a step later
        in complete_pending, so the transfer overlaps the engine step."""
        # Backpressure: block on the oldest in-flight send once the window is
        # full, so prefill doesn't run arbitrarily far ahead of decode. The
        # pending set is identical across MP ranks (TP-broadcast messages), so
        # the drain decision is collective.
        while len(self.pending_sends) >= self.max_inflight:
            oldest = next(iter(self.pending_sends))
            self.pending_sends.pop(oldest).wait()

        # The decode posted its matched receives when RECV_KV arrived; a
        # missing staged export here would leave it blocked in an unmatched
        # recv.
        staged = self.context.disagg_staged_kv.pop(request_id, None)
        if staged is None:
            raise RuntimeError(
                f"disagg prefill: SEND_KV for request {request_id} has no staged KV"
            )
        self.pending_sends[request_id] = send_request_kv_resharded(
            self.my_layout,
            self.instance_kv_layouts,
            kv_layouts(dst_layout_dicts),
            backend=self.get_backend(), payload=staged,
            my_mamba_layout=self.my_mamba_layout,
            dst_mamba_layouts=mamba_layouts(dst_layout_dicts),
        )

    def recv_kv(self, request_id, src_layout_dicts, prompt, sampling_params, handoff):
        """Post the receive of `request_id`'s KV (decode side). Non-blocking:
        the receive is reaped a step later in complete_pending, which imports
        the KV and admits the request; add_request then prefix-hits the
        imported blocks and continues generation.

        Push backends post a matched receive against the prefill's send;
        `handoff` carries only the snapshot hashes. Pull backends allocate
        destination blocks and issue a one-sided read of the prefill's blocks
        into them; `handoff` carries the per-rank region meta and source block
        ids, relayed by the coordinator. Both paths yield an object with a
        finish(engine) that commits, so the rest is symmetric."""
        # Backpressure: when the window is full, complete and admit the oldest
        # in-flight receive (collective: identical pending set across MP
        # ranks). finish() blocks if that transfer has not landed yet.
        while len(self.pending_recvs) >= self.max_inflight:
            self._admit_recv(next(iter(self.pending_recvs)))

        if self.is_pull:
            recv = post_pull_request_kv(
                self.engine, self.get_backend(), handoff, self.my_layout,
                src_layouts=kv_layouts(src_layout_dicts),
                dst_layouts=self.instance_kv_layouts,
            )
        else:
            recv = post_recv_request_kv_resharded(
                self.engine, self.my_layout,
                kv_layouts(src_layout_dicts),
                self.instance_kv_layouts,
                prompt, backend=self.get_backend(),
                handoff=handoff,
                my_mamba_layout=self.my_mamba_layout,
                src_mamba_layouts=mamba_layouts(src_layout_dicts),
            )
        if recv is None:
            # No KV received (pull: the decode KV cache was full); admit to
            # re-prefill. The prefill still pinned its blocks, so ack to
            # release the outstanding slot and pin.
            if self.is_pull:
                self.pending_acks.append(request_id)
            sp = SamplingParams.deserialize(sampling_params)
            self.engine.add_request(request_id, prompt, sampling_params=sp)
            return
        self.pending_recvs[request_id] = (recv, prompt, sampling_params)

    def ready_recvs(self):
        """Return the pending receive ids (insertion order) ready to admit
        this step.

        Push: all pending; the matched recvs are waited in finish(). Pull: the
        subset whose read has drained on every MP rank. Each rank polls its
        handles without blocking, then the per-rank done flags are AND-reduced
        over the MP group so admission stays collective; requests not yet done
        everywhere stay pending and are rechecked next step."""
        pending = list(self.pending_recvs)
        if not self.is_pull or not pending:
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

    def _admit_recv(self, request_id):
        """Finish one landed receive and admit its request: import the KV,
        queue the read-done ack (pull), and add_request to continue
        generation. A failed import (None) still admits the request; it just
        re-prefills from the prompt."""
        recv, prompt, sampling_params = self.pending_recvs.pop(request_id)
        imported = recv.finish(self.engine)
        if imported is None:
            logging.warning(
                "disagg decode: KV import failed for request %s; "
                "re-prefilling from prompt instead of using handed-off KV",
                request_id,
            )
        else:
            # Mark for the admission-time wasted-import check (the request
            # should prefix-hit the blocks this import just registered).
            self.context.disagg_imported_request_ids.add(request_id)
        if self.is_pull:
            # Read drained: release the prefill's pin and outstanding slot.
            self.pending_acks.append(request_id)
        self.engine.add_request(
            request_id, prompt, sampling_params=SamplingParams.deserialize(sampling_params)
        )

    def complete_pending(self):
        """Reap KV transfers posted on a previous step.

        Collective across the MP group: the pending sets were populated from
        TP-broadcast coordinator messages, so every rank holds the same requests
        in the same (insertion) order. Prefill: wait each send and release its
        staged KV. Decode: admit each receive that has landed, import its KV
        (registers the prefix-cache blocks), and continue generation.

        For pulls, receive completion is non-blocking: ready_recvs polls each
        transfer and admits only those done on every MP rank, deferring the
        rest to a later step, so a lagging transfer never stalls the loop.
        The done set is AND-reduced over the MP group so all ranks admit the
        same requests."""
        for request_id in list(self.pending_sends):
            self.pending_sends.pop(request_id).wait()
        for request_id in self.ready_recvs():
            self._admit_recv(request_id)

        # Flush read-done acks. The queue is identical across MP ranks
        # (collective finish), so clearing it is consistent.
        if self.pending_acks:
            if self.is_mp_coordinator:
                for rid in self.pending_acks:
                    self.socket_for_receiving_requests.send(
                        msgpack.packb([Headers.KV_READ_DONE.value, rid], use_bin_type=True)
                    )
            self.pending_acks.clear()

    # --- engine seams ----------------------------------------------------
    def registration_message(self):
        """REGISTER_ROLE msgpack: role, this instance's KV layouts, and the
        is_pull flag, so the coordinator can route, plan reshards, and apply
        outstanding-handoff flow control to pull instances."""
        return msgpack.packb(
            [Headers.REGISTER_ROLE.value, self.role, self.instance_layouts, self.is_pull],
            use_bin_type=True,
        )

    def prepare_prefill_request(self, request_id, prompt, sampling_params):
        """Prepare a prefill-only SUBMIT: run prefill (which populates the
        prompt KV) and stop right after, so the request leaves this engine
        with its prompt-block KV intact for the hand-off. The generated token
        is discarded; decode regenerates from the prompt."""
        if isinstance(prompt, str):
            # The prompt block count below needs the tokenized length; text
            # prompts would be counted in characters and corrupt the hand-off.
            raise NotImplementedError(
                "disaggregated inference requires tokenized prompts (List[int])"
            )
        sampling_params.num_tokens_to_generate = 1
        # num_tokens_total also caps generation and is mutually exclusive with
        # num_tokens_to_generate at add time; the prefill's 1-token cap wins.
        sampling_params.num_tokens_total = None
        bs = int(self.context.block_size_tokens)
        self.context.disagg_prompt_block_count[request_id] = (len(prompt) + bs - 1) // bs

    def send_prefill_done(self, records_to_send):
        """Tell the coordinator each request finished prefill (KV staged)
        instead of replying to the client; the coordinator names the decode
        target via RECV_KV (and SEND_KV for push backends).

        Pull backends need every MP rank to publish its KV shard and
        contribute per-rank read descriptors, so the publish runs on all ranks
        (records_to_send is identical across the MP group) while only the MP
        coordinator emits the control message. Push backends attach only the
        snapshot hashes; the KV ships later on SEND_KV."""
        nvtx_range_push("coordinator_communication")
        for r in records_to_send:
            rid = r.requests[-1].request_id
            handoff = self.publish_kv(rid) if self.is_pull else self.publish_push_handoff(rid)
            if self.is_mp_coordinator:
                parts = [Headers.PREFILL_DONE.value, rid]
                if handoff is not None:
                    parts.append(handoff)
                self.socket_for_receiving_requests.send(
                    msgpack.packb(parts, use_bin_type=True)
                )
        nvtx_range_pop("coordinator_communication")

    @pull_only
    def release_pinned(self, request_id):
        """The decode finished its read: release the request's pinned KV
        blocks, which also frees their boundary snapshots for eviction."""
        self.context.disagg_release_pinned(request_id)
