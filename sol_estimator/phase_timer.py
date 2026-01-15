# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Wall-clock phase timing for SOL analysis.

Tracks actual elapsed time per phase, including CPU work that GPU hooks miss.
This complements the GPU kernel timing from LayerSOLHooks/FunctionalPatcher.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PhaseTimeRecord:
    """Record of time spent in a phase."""
    phase: str
    wall_time_us: float  # Wall-clock time
    gpu_sync_time_us: float = 0  # Time spent in GPU sync (if any)
    is_complete: bool = True
    is_top_level: bool = False  # True if this phase was entered when stack was empty
    parent_phase: Optional[str] = None  # The parent phase when this was pushed


class PhaseTimer:
    """Tracks wall-clock time per phase.
    
    Unlike GPU kernel timing, this captures ALL time spent in a phase,
    including CPU work, memory operations, and idle time.
    
    Key concept: "top-level" phases are those pushed when the stack is empty.
    Only top-level phases should be summed for total wall-clock time to avoid
    double-counting nested phases.
    
    This class also tracks parent-child relationships based on the actual call
    stack, not by name prefix. This allows accurate aggregation of nested phase
    times.
    """
    
    def __init__(self):
        self._phase_stack: List[Tuple[str, float, bool, Optional[str]]] = []  # (phase_name, start_time, is_top_level, parent_phase)
        self._records: List[PhaseTimeRecord] = []
        self._phase_totals: Dict[str, float] = defaultdict(float)  # phase -> total wall time
        self._top_level_totals: Dict[str, float] = defaultdict(float)  # Only top-level phases
        self._top_level_phases: Set[str] = set()  # Phases that were ever top-level
        # Track parent-child relationships: parent_phase -> set of child phases
        self._phase_children: Dict[str, Set[str]] = defaultdict(set)
        
    def push_phase(self, phase: str):
        """Start timing a phase.
        
        If this phase is pushed when the stack is empty, it's marked as "top-level".
        Only top-level phases are counted toward the total wall-clock time.
        """
        is_top_level = len(self._phase_stack) == 0
        parent_phase = self._phase_stack[-1][0] if self._phase_stack else None
        
        self._phase_stack.append((phase, time.perf_counter(), is_top_level, parent_phase))
        
        if is_top_level:
            self._top_level_phases.add(phase)
        
        # Track parent-child relationship
        if parent_phase is not None:
            self._phase_children[parent_phase].add(phase)
        
    def pop_phase(self) -> Optional[str]:
        """Stop timing the current phase and record elapsed time."""
        if not self._phase_stack:
            return None
            
        phase, start_time, is_top_level, parent_phase = self._phase_stack.pop()
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        
        self._records.append(PhaseTimeRecord(
            phase=phase,
            wall_time_us=elapsed_us,
            is_top_level=is_top_level,
            parent_phase=parent_phase,
        ))
        self._phase_totals[phase] += elapsed_us
        
        if is_top_level:
            self._top_level_totals[phase] += elapsed_us
        
        return phase
    
    def current_phase(self) -> Optional[str]:
        """Get current phase name."""
        if self._phase_stack:
            return self._phase_stack[-1][0]
        return None
    
    def get_all_descendants(self, phase: str) -> Set[str]:
        """Get all descendant phases (children, grandchildren, etc.) of a phase."""
        descendants = set()
        to_visit = list(self._phase_children.get(phase, set()))
        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self._phase_children.get(child, set()))
        return descendants
    
    def clear(self):
        """Clear recorded times (but keep phase stack for ongoing phases)."""
        self._records.clear()
        self._phase_totals.clear()
        self._top_level_totals.clear()
        self._top_level_phases.clear()
        self._phase_children.clear()
        
    def cleanup(self):
        """Full cleanup including phase stack."""
        self._phase_stack.clear()
        self._records.clear()
        self._phase_totals.clear()
        self._top_level_totals.clear()
        self._top_level_phases.clear()
        self._phase_children.clear()
    
    def get_summary(self) -> Dict:
        """Get summary of wall-clock time per phase.
        
        Returns:
            - total_wall_time_us: Sum of ALL phase times (may have overlaps)
            - top_level_wall_time_us: Sum of only top-level phases (no overlaps, accurate total)
            - by_phase: Per-phase breakdown with counts and times
            - top_level_phases: List of phases that were entered at the top level
            - phase_children: Dict mapping each phase to its direct children
            - phase_descendants: Dict mapping each phase to all its descendants
        """
        if not self._records and not self._phase_totals:
            return {}
        
        # Build phase summary
        by_phase = {}
        for phase, total_us in self._phase_totals.items():
            count = sum(1 for r in self._records if r.phase == phase)
            is_top_level = phase in self._top_level_phases
            by_phase[phase] = {
                'count': count,
                'total_wall_time_us': total_us,
                'avg_wall_time_us': total_us / count if count > 0 else 0,
                'is_top_level': is_top_level,
            }
        
        # Calculate totals
        total_wall_time_us = sum(self._phase_totals.values())
        top_level_wall_time_us = sum(self._top_level_totals.values())
        
        # Build descendants map for each phase
        phase_descendants = {}
        for phase in self._phase_totals.keys():
            phase_descendants[phase] = list(self.get_all_descendants(phase))
        
        return {
            'total_wall_time_us': total_wall_time_us,
            'top_level_wall_time_us': top_level_wall_time_us,
            'by_phase': by_phase,
            'top_level_phases': list(self._top_level_phases),
            'phase_children': {k: list(v) for k, v in self._phase_children.items()},
            'phase_descendants': phase_descendants,
        }
    
    def get_phase_wall_time(self, phase: str) -> float:
        """Get total wall-clock time for a specific phase in microseconds."""
        return self._phase_totals.get(phase, 0)



