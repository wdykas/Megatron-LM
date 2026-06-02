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

Examples for a 4-rank world:

    "tp=2,dp=1+tp=1,dp=2"     -> shard 0 owns ranks [0,1], shard 1 owns [2,3]
    "tp=4,dp=1"               -> single shard over all 4 ranks
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

VALID_INT_KEYS = ("tp", "pp", "ep", "expt_tp", "dp")
VALID_ROLES = ("prefill", "decode")
VALID_KEYS = (*VALID_INT_KEYS, "role")


@dataclass(frozen=True)
class InferenceShardSpec:
    """One inference shard's parallelism -- the programmatic form of a shard
    spec string. ``expt_tp`` defaults to ``tp``; ``role`` is ``"prefill"`` /
    ``"decode"`` for disaggregation, else ``None``."""

    tp: int
    pp: int = 1
    ep: int = 1
    dp: int = 1
    expt_tp: Optional[int] = None
    role: Optional[str] = None

    def __post_init__(self):
        if self.role is not None and self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES} or None; got {self.role!r}")

    def to_dict(self) -> dict:
        d = {"tp": self.tp, "pp": self.pp, "ep": self.ep, "dp": self.dp,
             "expt_tp": self.expt_tp if self.expt_tp is not None else self.tp}
        if self.role is not None:
            d["role"] = self.role
        return d


def normalize_shard_specs(
    shards: Union[str, Sequence["InferenceShardSpec"], Sequence[dict]], world_size: int
) -> List[dict]:
    """Coerce the public shard-spec input (a spec string, a list of
    :class:`InferenceShardSpec`, or a list of raw dicts) into the validated
    dict list the shard builders consume."""
    if isinstance(shards, str):
        return parse_inference_shards_spec(shards, world_size)
    out: List[dict] = []
    for s in shards:
        out.append(s.to_dict() if isinstance(s, InferenceShardSpec) else dict(s))
    return _finalize_and_validate(out, world_size)


def spec_declares_disaggregation(spec_str: str) -> bool:
    """Whether a shard spec tags any shard with a ``role=`` (prefill/decode).

    A role tag is what marks the layout as a prefill->decode handoff rather
    than plain multi-shard / data-parallel inference. Cheap and world_size-
    free, so it can be checked at arg-validation time; full parsing +
    validation is :func:`parse_inference_shards_spec`.
    """
    if not spec_str:
        return False
    return any(
        kv.strip().startswith("role=")
        for shard in spec_str.replace("+", ";").split(";")
        for kv in shard.split(",")
    )


def parse_inference_shards_spec(spec_str: str, world_size: int) -> List[dict]:
    """Parse + validate the ``--inference-shards`` string.

    Args:
        spec_str: Raw CLI value, e.g. ``"tp=2,dp=1+tp=1,dp=2"`` or with
            disaggregation roles ``"tp=2,role=prefill+tp=1,role=decode"``.
        world_size: Total number of ranks. Specs must partition it
            exactly (no idle ranks; see note below).

    Returns:
        List of spec dicts, one per shard. Each carries the integer
        parallelism keys ``tp``, ``pp``, ``ep``, ``expt_tp``, ``dp``,
        and optionally a ``role`` string. Order matches the input
        (left-to-right corresponds to ascending ``rank_offset``).

    Raises:
        AssertionError: on syntax errors, unknown keys, expert-grid
            mismatch within a shard, or a rank-count mismatch with
            ``world_size``. Idle ranks are rejected to keep the
            partition explicit — any world-collective consumer must
            be able to enumerate every rank's shard membership from
            the parsed list alone.
    """
    parsed: List[dict] = []
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
                    f"Bad --inference-shards spec entry {kv!r}: expected key=value."
                )
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k not in VALID_KEYS:
                raise AssertionError(
                    f"Unknown key {k!r} in --inference-shards "
                    f"(allowed: {','.join(VALID_KEYS)})."
                )
            if k == "role":
                role = v.lower()
                assert role in VALID_ROLES, (
                    f"Unknown role {v!r} in --inference-shards "
                    f"(allowed: {','.join(VALID_ROLES)})."
                )
                spec[k] = role
            else:
                spec[k] = int(v)
        parsed.append(spec)

    return _finalize_and_validate(parsed, world_size)


def _finalize_and_validate(specs: List[dict], world_size: int) -> List[dict]:
    """Apply per-shard defaults and assert the shards partition the world.

    Shared by the string parser and the object path (:func:`normalize_shard_specs`).
    Idle ranks are rejected so any world-collective consumer can enumerate
    every rank's shard membership from the list alone.
    """
    assert specs, "--inference-shards was empty."
    total_ranks = 0
    for spec in specs:
        # Defaults: integer keys default to 1; expt_tp defaults to tp.
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
        total_ranks += shard_world

    assert total_ranks == world_size, (
        f"--inference-shards consumes {total_ranks} ranks but world size is "
        f"{world_size}; specs must partition the full world."
    )
    return specs
