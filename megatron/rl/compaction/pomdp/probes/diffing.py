# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Belief state diffing (§4.3 in research directions).

Tracks what changed between consecutive belief states. The key safety signal
is ``constraint_violations``: hard constraints that were present in the
previous belief but missing from the next. That is always a compactor bug.

Usage
-----
    differ = BeliefDiffer(hard_constraint_key="task.hard_constraints")
    diff = differ.diff(prev_belief, next_belief)

    if diff.constraint_violations:
        logger.error("Compactor dropped hard constraints: %s", diff.constraint_violations)
    if not differ.is_monotone(diff):
        logger.warning("Belief lost information at step %d", next_belief.step_id)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BeliefFieldDiff:
    """Structured diff between two consecutive belief states."""

    step_from: int
    step_to: int

    # Top-level BeliefState fields that changed.
    added_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    modified_keys: list[str] = field(default_factory=list)

    # Constraint items that were in prev.task["hard_constraints"] but not next.
    # Non-empty means the compactor silently dropped a safety constraint.
    constraint_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_from": self.step_from,
            "step_to": self.step_to,
            "added_keys": self.added_keys,
            "removed_keys": self.removed_keys,
            "modified_keys": self.modified_keys,
            "constraint_violations": self.constraint_violations,
        }


class BeliefDiffer:
    """Computes structured diffs between consecutive BeliefState objects.

    ``hard_constraint_key`` is a dot-separated path into the belief dict that
    holds the list of hard constraints that must never be removed.
    """

    def __init__(self, hard_constraint_key: str = "task.hard_constraints") -> None:
        self._constraint_path = hard_constraint_key.split(".")

    def diff(self, prev: Any, curr: Any) -> BeliefFieldDiff:
        """Diff two beliefs. Works with any object that has ``to_dict()``."""
        prev_d = prev.to_dict() if hasattr(prev, "to_dict") else {}
        curr_d = curr.to_dict() if hasattr(curr, "to_dict") else {}

        prev_keys = set(prev_d.keys())
        curr_keys = set(curr_d.keys())

        added = sorted(curr_keys - prev_keys)
        removed = sorted(prev_keys - curr_keys)
        modified = sorted(
            k for k in prev_keys & curr_keys if prev_d[k] != curr_d[k]
        )

        constraint_violations = self._find_constraint_violations(prev_d, curr_d)

        return BeliefFieldDiff(
            step_from=getattr(prev, "step_id", -1),
            step_to=getattr(curr, "step_id", -1),
            added_keys=added,
            removed_keys=removed,
            modified_keys=modified,
            constraint_violations=constraint_violations,
        )

    def is_monotone(self, diff: BeliefFieldDiff) -> bool:
        """True if the diff contains no safety violations (dropped constraints)."""
        return len(diff.constraint_violations) == 0

    def _find_constraint_violations(
        self, prev_d: dict, curr_d: dict
    ) -> list[str]:
        """Return hard constraints present in prev but missing in curr."""
        prev_constraints = self._get_nested(prev_d, self._constraint_path)
        curr_constraints = self._get_nested(curr_d, self._constraint_path)
        if not isinstance(prev_constraints, list) or not isinstance(curr_constraints, list):
            return []
        curr_set = set(curr_constraints)
        return [c for c in prev_constraints if c not in curr_set]

    @staticmethod
    def _get_nested(d: dict, path: list[str]) -> Any:
        val: Any = d
        for key in path:
            if not isinstance(val, dict):
                return None
            val = val.get(key)
        return val
