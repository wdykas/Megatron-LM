# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""High-level prefill->decode disaggregation API (offline / SPMD)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

from megatron.core.inference.disaggregation.disagg_coordinator import DisaggCoordinator
from megatron.core.inference.disaggregation.kv_transport_backend import KVTransportBackend
from megatron.core.inference.disaggregation.orchestration import (
    DECODE,
    PREFILL,
    DisaggRequest,
    run_decode_replica,
    run_prefill_replica,
    setup_disagg,
)
from megatron.core.inference.shards_spec import InferenceShardSpec, normalize_shard_specs


class MegatronDisaggLLM:
    """Offline prefill->decode disaggregated inference, shaped like ``MegatronLLM``.

    Every rank constructs this object (SPMD, like the offline example): it
    builds this rank's inference shard + engine and completes the layout
    handshake. :meth:`generate` then runs the role-appropriate loop -- prefill
    ships KV to the decode pool; decode imports KV and generates. Each decode
    instance returns the subset of requests routed to it (sticky-hash), so
    callers gather across instances if they want the full set.

    The KV handoff (transport, reshard, routing) is the tested machinery in
    :mod:`megatron.core.inference.disaggregation`; this class only packages it
    behind a single object so disaggregated inference is as close to regular
    offline inference as the split allows. Building the per-shard engine is the
    one framework-specific step, supplied via ``engine_builder``.

    Args:
        inference_shards: Shard layout -- a spec string
            (``"tp=2,role=prefill+tp=1,role=decode"``) or a list of
            :class:`~megatron.core.inference.shards_spec.InferenceShardSpec`.
            Shards must be tagged ``role=prefill`` / ``role=decode``.
        engine_builder: ``pg_collection -> engine``; builds this rank's engine
            (model + checkpoint) against the shard's process groups. The global
            layer / KV-head counts are read from the engine's model config.
        backend: KV transport backend (default: NCCL/P2P).
        router_name: Decode router policy; must be deterministic (default
            ``"sticky"``).
    """

    def __init__(
        self,
        inference_shards: Union[str, Sequence[InferenceShardSpec]],
        *,
        engine_builder: Callable[[Any], Any],
        backend: Optional[KVTransportBackend] = None,
        router_name: str = "sticky",
        group: Optional[object] = None,
    ) -> None:
        import torch.distributed as dist

        specs = normalize_shard_specs(inference_shards, dist.get_world_size())
        self._setup = setup_disagg(
            specs,
            engine_builder=engine_builder,
            backend=backend,
            router_name=router_name,
            group=group,
        )

    # --- properties (mirror MegatronLLM's accessors) ----------------------

    @property
    def role(self) -> str:
        """``"prefill"`` or ``"decode"`` for this rank's shard."""
        return self._setup.role

    @property
    def replica_id(self) -> str:
        """This shard's replica id (``"prefill"`` / ``"decode_s{i}_dp{k}"``)."""
        return self._setup.replica_id

    @property
    def is_decode(self) -> bool:
        return self._setup.role == DECODE

    @property
    def engine(self):
        """The underlying ``DynamicInferenceEngine`` for this rank's shard."""
        return self._setup.engine

    @property
    def coordinator(self) -> DisaggCoordinator:
        """The disaggregation coordinator (layouts, router, handoff)."""
        return self._setup.coordinator

    @property
    def num_decode_instances(self) -> int:
        return self._setup.num_decode_instances

    # --- run ---------------------------------------------------------------

    def generate(self, requests: Sequence[Any]) -> List[Any]:
        """Run disaggregated inference over ``requests``.

        ``requests`` is the same sequence the regular path builds (e.g. from
        ``build_requests``): each item exposes ``prompt_text``,
        ``prompt_tokens`` and ``sampling_params``. It must be identical on
        every rank (SPMD); the request id is its position in the sequence, so
        prefill and decode agree. Returns the finished
        ``DynamicInferenceRequest`` records for THIS decode instance in input
        order; an empty list on prefill ranks (they ship KV, produce no
        output). With a single decode instance this is the whole batch; with
        several, each instance returns its routed subset.
        """
        disagg = [
            DisaggRequest(
                request_id=i,
                prompt_text=r.prompt_text,
                prompt_tokens=r.prompt_tokens,
                sampling_params=r.sampling_params,
            )
            for i, r in enumerate(requests)
        ]
        if not disagg:
            return []
        if self._setup.role == PREFILL:
            run_prefill_replica(self._setup.coordinator, self._setup.engine, disagg)
            return []
        finished = run_decode_replica(self._setup.coordinator, self._setup.engine, disagg)
        return [finished[i] for i in sorted(finished)]
