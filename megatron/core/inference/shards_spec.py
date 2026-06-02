# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parser for the ``--inference-shards`` / ``--rl-inference-shards`` CLI string.

Turns the user-supplied shard layout string into a list of per-shard
spec dicts describing the heterogeneous inference layout. Each shard
spec carries the (TP, PP, EP, expt_tp, DP) parallelism for one
contiguous slice of the world; ranks are partitioned across shards
in the order they appear.

Spec syntax (one shard per ``;`` or ``+`` separator):

    "tp=2,pp=1,ep=1,dp=1+tp=1,dp=2"

Per-shard keys (all optional, default ``1``; ``expt_tp`` defaults to
``tp``): ``tp``, ``pp``, ``ep``, ``expt_tp``, ``dp``.

Disaggregation role (optional): ``role=prefill`` / ``role=decode``
tags a shard as a prefill source or a decode target for
prefill->decode KV disaggregation, e.g.

    "tp=2,role=prefill+tp=1,dp=2,role=decode"

Role is metadata the shard layer ignores; the disaggregation
coordinator groups shards by it. A ``dp>1`` decode shard is several
independent decode instances (each takes a routed subset of
requests), not KV replicas.

Layer-kind disaggregation (optional):

    "tp=4,kinds=M+tp=4,kinds=*D-+tp=8,ep=8,kinds=E"

The ``kinds=`` value is a string of single-char layer-kind symbols
from :class:`megatron.core.models.hybrid.hybrid_layer_allocation.Symbols`
(M = mamba, * = attention, D = DS attention, E = MoE block, - = MLP
block). A shard with ``kinds`` set runs only blocks of those kinds;
hidden-state activations flow between shards mid-forward-pass via the
NVSHMEM activation-transport channel. Shards without a ``kinds`` key
run every block in their PP-rank layer range (back-compat with
collocated and hetero-PP layouts).

Examples for a 4-rank world:

    "tp=2,dp=1+tp=1,dp=2"     -> shard 0 owns ranks [0,1], shard 1 owns [2,3]
    "tp=4,dp=1"               -> single shard over all 4 ranks
"""

from typing import List

# Integer-valued shard keys. ``kinds`` is a string and handled separately.
VALID_INT_KEYS = ("tp", "pp", "ep", "expt_tp", "dp")
VALID_ROLES = ("prefill", "decode")
VALID_KEYS = (*VALID_INT_KEYS, "kinds", "role")

# Valid layer-kind symbols. Mirrors
# ``megatron.core.models.hybrid.hybrid_layer_allocation.Symbols.VALID_LAYERS``
# but kept inline so the parser has no heavy import for early CLI
# validation. The unit test asserts these stay in sync with Symbols.
VALID_KIND_SYMBOLS = ("M", "*", "D", "E", "-", "G")


def _parse_kinds(value: str) -> tuple:
    """Parse a ``kinds=`` value into a deduplicated tuple of symbols.

    The value is a string of single-char symbols (e.g., ``"*D-"`` for
    attention + DS-attention + MLP). Empty is rejected — a ``kinds``
    key with no symbols is almost certainly a typo and silently
    swallowing it would route every block away from the shard.
    """
    cleaned = value.strip()
    assert cleaned, "kinds= cannot be empty; omit the key for 'all kinds'."
    seen: list = []
    for ch in cleaned:
        if ch.isspace():
            continue
        if ch not in VALID_KIND_SYMBOLS:
            raise AssertionError(
                f"Unknown kind symbol {ch!r} in kinds={cleaned!r} "
                f"(allowed: {','.join(VALID_KIND_SYMBOLS)})."
            )
        if ch not in seen:
            seen.append(ch)
    return tuple(seen)


def parse_inference_shards_spec(
    spec_str: str, world_size: int
) -> List[dict]:
    """Parse + validate the ``--rl-inference-shards`` string.

    Args:
        spec_str: Raw CLI value, e.g. ``"tp=2,dp=1+tp=1,dp=2"`` or with
            layer-kind disaggregation ``"tp=4,kinds=M+tp=4,kinds=*"``.
        world_size: Total number of ranks. Specs must partition it
            exactly (no idle ranks; see note below).

    Returns:
        List of spec dicts, one per shard. Each carries the integer
        parallelism keys ``tp``, ``pp``, ``ep``, ``expt_tp``, ``dp``,
        and optionally a tuple-valued ``kinds`` key (deduplicated,
        order-preserving) when the shard declared one. Order matches
        the input (left-to-right corresponds to ascending
        ``rank_offset``).

    Raises:
        AssertionError: on syntax errors, unknown keys, expert-grid
            mismatch within a shard, or a rank-count mismatch with
            ``world_size``. Idle ranks are rejected to keep the
            partition explicit — any world-collective consumer must
            be able to enumerate every rank's shard membership from
            the parsed list alone.
    """
    parsed: List[dict] = []
    total_ranks = 0
    # ``+`` is convenient from shell recipes where ``;`` would otherwise
    # be treated as a command terminator. Normalize before splitting.
    shards_raw = spec_str.replace("+", ";")
    for shard_str in shards_raw.split(";"):
        shard_str = shard_str.strip()
        if not shard_str:
            continue
        spec: dict = {}
        for kv in shard_str.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise AssertionError(
                    f"Bad --rl-inference-shards spec entry {kv!r}: expected key=value."
                )
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k not in VALID_KEYS:
                raise AssertionError(
                    f"Unknown key {k!r} in --rl-inference-shards "
                    f"(allowed: {','.join(VALID_KEYS)})."
                )
            if k == "kinds":
                spec[k] = _parse_kinds(v)
            elif k == "role":
                role = v.strip().lower()
                assert role in VALID_ROLES, (
                    f"Unknown role {v!r} in --inference-shards "
                    f"(allowed: {','.join(VALID_ROLES)})."
                )
                spec[k] = role
            else:
                spec[k] = int(v)
        # Defaults: integer keys default to 1; expt_tp defaults to tp.
        # ``kinds`` has no default; absence means "all kinds in my layer
        # range" (collocated / hetero-PP back-compat).
        spec.setdefault("tp", 1)
        spec.setdefault("pp", 1)
        spec.setdefault("ep", 1)
        spec.setdefault("dp", 1)
        spec.setdefault("expt_tp", spec["tp"])
        # Validate expert decomposition tiles cleanly within the shard.
        shard_world = spec["tp"] * spec["pp"] * spec["dp"]
        expert_block = spec["expt_tp"] * spec["ep"] * spec["pp"]
        assert shard_world % expert_block == 0, (
            f"Shard {spec} has tp*pp*dp={shard_world} but "
            f"expt_tp*ep*pp={expert_block} does not divide it; "
            f"choose compatible sizes."
        )
        parsed.append(spec)
        total_ranks += shard_world

    assert parsed, "--rl-inference-shards was empty after parsing."
    assert total_ranks == world_size, (
        f"--rl-inference-shards consumes {total_ranks} ranks but world size is "
        f"{world_size}; specs must partition the full world."
    )
    return parsed


def assert_kinds_partition_layers(
    specs: List[dict], layer_type_list: tuple
) -> None:
    """Validate that every block in ``layer_type_list`` is owned by
    exactly one shard with a matching ``kinds`` declaration.

    Only enforced for layouts where **at least one** shard declares
    ``kinds=``; layouts where every shard runs all kinds are unaffected.
    Mixed layouts (some shards with ``kinds=``, others without) are
    rejected — disagg is all-or-nothing for v1, since the back-compat
    "runs all kinds" semantics would otherwise silently cover blocks
    that a kind-restricted shard expected to migrate to.

    Args:
        specs: Output of :func:`parse_inference_shards_spec`.
        layer_type_list: Per-block kind symbols for the model (length =
            number of blocks). Typically
            ``inference_state_config.layer_type_list`` for hybrid models;
            a tuple of ``("*",) * num_layers`` for pure transformer
            models (every block is an attention block).

    Raises:
        AssertionError: on unowned blocks, double-owned blocks, or a
            mixed layout (some specs with ``kinds=``, others without).
    """
    have_kinds = [s for s in specs if "kinds" in s]
    if not have_kinds:
        # No disagg; nothing to validate.
        return
    assert len(have_kinds) == len(specs), (
        "Mixed layouts (some shards with kinds=, others without) are "
        "not supported in v1: every shard must declare kinds= or none "
        "of them should. Add kinds= to the remaining shards or remove "
        "it everywhere."
    )
    # Each block goes to exactly one shard.
    coverage: dict = {}  # layer_idx -> shard_index_in_specs
    for shard_idx, spec in enumerate(specs):
        kinds = set(spec["kinds"])
        for layer_idx, layer_kind in enumerate(layer_type_list):
            if layer_kind in kinds:
                if layer_idx in coverage:
                    raise AssertionError(
                        f"Layer {layer_idx} (kind {layer_kind!r}) is "
                        f"claimed by both shard {coverage[layer_idx]} "
                        f"and shard {shard_idx}; kinds= sets must "
                        f"partition the layer-type pattern."
                    )
                coverage[layer_idx] = shard_idx
    missing = [
        (i, layer_type_list[i])
        for i in range(len(layer_type_list))
        if i not in coverage
    ]
    if missing:
        raise AssertionError(
            f"{len(missing)} block(s) have no owning shard: "
            f"first few = {missing[:5]}. Extend a shard's kinds= to "
            f"include the missing kind(s)."
        )


def compute_layer_indices_for_kinds(
    kinds: tuple, layer_type_list: tuple
) -> tuple:
    """Return the global layer indices a shard with these ``kinds`` owns.

    Stable, order-preserving. Used by
    :func:`megatron.core.inference.shards.build_inference_pg_collections_for_shards`
    to populate :attr:`InferenceShard.layer_indices`.
    """
    kinds_set = set(kinds)
    return tuple(
        i for i, k in enumerate(layer_type_list) if k in kinds_set
    )
