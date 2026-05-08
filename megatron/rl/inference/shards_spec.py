# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parser for the ``--rl-inference-shards`` CLI string.

Used by :mod:`megatron.training.arguments` to turn the user-supplied
shard layout string into a list of per-shard spec dicts that
:func:`megatron.core.inference.shards.build_inference_pg_collections_for_shards`
can consume.

Spec syntax (one shard per `;` or `+` separator):

    "tp=2,pp=1,ep=1,dp=1+tp=1,dp=2"

Per-shard keys (all optional, default ``1``; ``expt_tp`` defaults to
``tp``): ``tp``, ``pp``, ``ep``, ``expt_tp``, ``dp``.

Examples for a 4-rank world:

    "tp=2,dp=1+tp=1,dp=2"     -> shard 0 owns ranks [0,1], shard 1 owns [2,3]
    "tp=4,dp=1"               -> single shard over all 4 ranks
"""

from typing import List

VALID_KEYS = ("tp", "pp", "ep", "expt_tp", "dp")


def parse_inference_shards_spec(
    spec_str: str, world_size: int
) -> List[dict]:
    """Parse + validate the ``--rl-inference-shards`` string.

    Args:
        spec_str: Raw CLI value, e.g. ``"tp=2,dp=1+tp=1,dp=2"``.
        world_size: Total number of ranks. Specs must partition it
            exactly (no idle ranks; see note below).

    Returns:
        List of spec dicts, one per shard, each with keys ``tp``,
        ``pp``, ``ep``, ``expt_tp``, ``dp``. Order matches the input
        (left-to-right corresponds to ascending ``rank_offset``).

    Raises:
        AssertionError: on syntax errors, unknown keys, expert-grid
            mismatch within a shard, or a rank-count mismatch with
            ``world_size``. Idle ranks would silently skip the
            world-collective refit loop
            (:func:`swap_model_weights_across_shards` runs one swap
            per shard on every rank) and deadlock the surviving ranks,
            so we require equality. Lifting that constraint is a
            follow-up that needs the refit gate in ``training/rl_utils``
            to route ``None``-targets through the collective.
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
            spec[k] = int(v)
        # Defaults: everything else is 1; expt_tp defaults to tp.
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
