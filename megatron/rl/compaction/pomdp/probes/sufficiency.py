# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Field-level importance via belief ablation (§1.2 in research directions).

Ablates individual BeliefState fields one at a time and measures the KL
divergence between the full-belief policy and the ablated-belief policy.
A high KL means that field is load-bearing; near-zero KL means the compactor
could safely drop it.

Usage
-----
    probe = FieldAblationProbe()
    results = probe.measure(
        policy_fn=my_policy,
        belief=belief,
        context_builder=builder,
        base_tokens=full_context_tokens,
    )
    for fi in sorted(results, key=lambda r: r.kl_when_ablated, reverse=True):
        print(f"{fi.field}: KL={fi.kl_when_ablated:.4f}")
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from .kl_probe import KLSufficiencyProbe

PolicyFn = Callable[[list[int]], list[float]]


# Fields that should never be ablated — removing them would corrupt the belief.
_PROTECTED_FIELDS = {"belief_id", "run_id", "step_id", "task"}

# BeliefState fields that are meaningful to ablate.
_DEFAULT_FIELDS = [
    "environment_state",
    "open_questions",
    "progress",
    "action_history_digest",
    "uncertainty",
    "raw_trace_refs",
]


@dataclass
class FieldImportance:
    """Importance of a single BeliefState field, measured by ablation KL."""

    field: str

    # KL(π(·|full) || π(·|ablated)) — how much the policy distribution shifts
    # when this field is zeroed out. Higher = more load-bearing.
    kl_when_ablated: float

    full_token_count: int
    ablated_token_count: int


class FieldAblationProbe:
    """Measures per-field importance by zeroing each field and measuring KL.

    Requires a context builder that can render a modified belief into tokens,
    and a policy callable that maps tokens -> log-probs.
    """

    def __init__(self) -> None:
        self._kl_probe = KLSufficiencyProbe()

    def ablate(self, belief: Any, field: str) -> Any:
        """Return a deep copy of ``belief`` with ``field`` set to its zero value.

        Dicts → {}, lists → [], scalars → None.
        """
        ablated = copy.deepcopy(belief)
        if not hasattr(ablated, field):
            return ablated
        current = getattr(ablated, field)
        if isinstance(current, dict):
            setattr(ablated, field, {})
        elif isinstance(current, list):
            setattr(ablated, field, [])
        else:
            setattr(ablated, field, None)
        return ablated

    def measure(
        self,
        policy_fn: PolicyFn,
        belief: Any,
        context_builder: Any,
        base_tokens: list[int],
        fields: list[str] | None = None,
    ) -> list[FieldImportance]:
        """Ablate each field in ``fields`` and record the KL against the full belief.

        Parameters
        ----------
        policy_fn:        tokens -> log-probs over vocabulary.
        belief:           The belief state to probe (not mutated).
        context_builder:  Must have an ``encode(text) -> list[int]`` method.
        base_tokens:      Token sequence produced from the unablated belief.
        fields:           Fields to probe. Defaults to all non-protected fields.
        """
        if fields is None:
            fields = [
                f for f in _DEFAULT_FIELDS
                if hasattr(belief, f) and f not in _PROTECTED_FIELDS
            ]

        results: list[FieldImportance] = []
        for field in fields:
            ablated_belief = self.ablate(belief, field)
            ablated_text = (
                ablated_belief.to_context_str()
                if hasattr(ablated_belief, "to_context_str")
                else str(ablated_belief)
            )
            ablated_tokens = context_builder.encode(ablated_text) or []

            kl_result = self._kl_probe.measure(policy_fn, base_tokens, ablated_tokens)
            results.append(FieldImportance(
                field=field,
                kl_when_ablated=kl_result.kl_full_to_compact,
                full_token_count=len(base_tokens),
                ablated_token_count=len(ablated_tokens),
            ))

        return results
