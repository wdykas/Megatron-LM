# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""High-level prefill->decode disaggregation API (coordinator-native)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

from megatron.core.inference.apis.llm import MegatronLLM
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.shards_spec import InferenceShardSpec, normalize_shard_specs


class MegatronDisaggLLM(MegatronLLM):
    """Coordinator-native prefill->decode disaggregated inference.

    This is :class:`~megatron.core.inference.apis.llm.MegatronLLM` in
    coordinator mode, with the engine additionally marked as a prefill or
    decode shard. Construction and usage are therefore identical to regular
    coordinator-mode inference -- :meth:`generate` (inherited unchanged) submits
    prompts through one :class:`InferenceClient` on the primary rank, and the
    shared coordinator 2-hop routes them: SUBMIT -> prefill engine -> KV handoff
    -> decode engine -> reply. The KV transport / reshard / routing machinery
    lives in :mod:`megatron.core.inference.disaggregation`.

    Like ``MegatronLLM``, the model is built **outside** and passed in. The one
    extra step versus regular inference is that the caller builds the per-shard
    process groups first (so each rank's model is built against its shard's
    parallelism) and passes that shard's ``pg_collection`` here::

        specs  = parse_inference_shards_spec(args.inference_shards, world)
        shards = build_inference_pg_collections_for_shards(world, specs)
        my     = next(s for s in shards if s.pg_collection is not None)
        model  = get_model_for_inference(pg_collection=my.pg_collection)

        llm = MegatronDisaggLLM(
            model=model, tokenizer=tokenizer, inference_config=cfg,
            inference_shards=specs, pg_collection=my.pg_collection,
        )
        if llm.is_primary_rank:
            results = llm.generate(prompts, sampling_params)
        llm.shutdown()

    Args:
        model: This rank's model, already built against ``pg_collection``
            (eval mode), exactly as for ``MegatronLLM``.
        tokenizer: Tokenizer for the controller.
        inference_config: Inference config. ``pg_collection`` is overridden with
            the shard's groups so the engine runs on this shard, not global mpu.
        inference_shards: Shard layout -- a spec string
            (``"tp=2,role=prefill+tp=1,role=decode"``) or a list of
            :class:`~megatron.core.inference.shards_spec.InferenceShardSpec`.
            Every shard must be tagged ``role=prefill`` / ``role=decode``.
        pg_collection: This rank's shard ``ProcessGroupCollection`` (the one the
            model was built against).
        coordinator_host / coordinator_port: optional coordinator bind address.
    """

    def __init__(
        self,
        *,
        model,
        tokenizer,
        inference_shards: Union[str, Sequence[InferenceShardSpec]],
        pg_collection,
        inference_config: Optional[InferenceConfig] = None,
        coordinator_host: Optional[str] = None,
        coordinator_port: Optional[int] = None,
    ) -> None:
        import torch.distributed as dist

        # Stash for the _post_build_engine hook (runs inside super().__init__()).
        self._disagg_specs = normalize_shard_specs(inference_shards, dist.get_world_size())
        self._disagg_pg = pg_collection
        self._disagg_setup = None

        if inference_config is None:
            inference_config = InferenceConfig()
        # The engine must run on THIS shard's process groups, not global mpu.
        inference_config.pg_collection = pg_collection

        # Disaggregation is coordinator-native: one client submits, the shared
        # coordinator routes prefill -> decode. Direct mode does not apply.
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
            use_coordinator=True,
            coordinator_host=coordinator_host,
            coordinator_port=coordinator_port,
        )

    def _post_build_engine(self) -> None:
        # Mark the freshly-built engine as a prefill/decode shard before the
        # coordinator runtime (start_listening) brings it up.
        from megatron.core.inference.disaggregation.coordinator_setup import (
            configure_prebuilt_disagg_engine,
        )

        self._disagg_setup = configure_prebuilt_disagg_engine(
            self._engine, self._disagg_pg, self._disagg_specs
        )

    # --- disagg-specific accessors ----------------------------------------

    @property
    def role(self) -> str:
        """``"prefill"`` or ``"decode"`` for this rank's shard."""
        return self._disagg_setup.role

    @property
    def replica_id(self) -> str:
        """This shard's replica id (``"prefill"`` / ``"decode_s{i}_dp{k}"``)."""
        return self._disagg_setup.replica_id

    @property
    def is_decode(self) -> bool:
        return self._disagg_setup.role == "decode"

    @property
    def num_instances(self) -> int:
        """Total prefill + decode instances in this job."""
        return self._disagg_setup.total_instances
