# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Graph rewrite framework for :class:`Route` objects.

A :class:`Route` is a DAG of hops (linear chain by default; arbitrary
DAG with the :attr:`RouteHop.succs` / :attr:`RouteHop.preds` extension).
Once a route is constructed, a series of correctness-preserving rewrite
passes can transform it into a more efficient equivalent form before
the dispatcher builds its per-layer plan.

The framework has three layers:

1. **DAG analyses** — pure functions over a :class:`Route` that
   return information without mutating the route:

   - :func:`successors_of` / :func:`predecessors_of` — adjacency lookups
   - :func:`topological_order` — Kahn's algorithm; raises on cycles
   - :func:`is_dag` — boolean
   - :func:`reachable_from_entry` — set of hop positions reachable
   - :func:`unreachable_hops` — hops not reachable from the entry
   - :func:`critical_path` — longest path through the DAG given
     per-hop cost estimates
   - :func:`activation_lifetimes` — for each hop, the latest hop that
     consumes its output (informs eager-free heuristics)

2. **Rewrite passes** — pure functions ``Route -> Route``. Each
   pass is correctness-preserving (the rewritten route computes the
   same logical function as the original) and idempotent (applying
   twice gives the same result as applying once). Concrete passes:

   - :class:`ValidateRoute` — checks DAG well-formedness; raises
     :class:`InvalidRouteError` with a clear message on violation.
     Always run first.
   - :class:`MergeAdjacentSameShardHops` — collapses consecutive
     hops on the same shard with no intervening cross-shard
     dependency.
   - :class:`DropEmptyHops` — removes hops with no layers (a
     degenerate state that can arise from upstream constructors).
   - :class:`SortHopsTopologically` — re-orders ``Route.hops`` so
     that each hop's predecessors precede it in the tuple. Required
     before the dispatcher walks the route.

3. **Composer** — :func:`apply_rewrites` runs a list of passes to
   a fixed point, exposing the diff to the caller for telemetry.

The framework is designed to be open-ended: new passes are just
functions, and the composer applies them in sequence. The current
set is minimal but the structure supports adding lower-level
optimizations (send fusion, dead-RECV elimination, etc.) later
without changing the API.

Conventions:

- All passes return a *new* :class:`Route`; the input is never
  mutated. :class:`Route` and :class:`RouteHop` are frozen
  dataclasses, so mutation is impossible by construction.
- Rewrites that change hop *positions* (merging, dropping)
  automatically remap any explicit ``succs`` / ``preds`` references
  so downstream consumers see a consistent DAG.
- Failures are loud: :func:`ValidateRoute` raises rather than
  silently producing garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from megatron.rl.inference.route_planner import Route, RouteHop


# ============================================================================
# Errors
# ============================================================================


class InvalidRouteError(ValueError):
    """Raised by :class:`ValidateRoute` when a route violates DAG
    invariants. The message identifies the specific violation
    (cycle, unreachable hop, dangling reduce, etc.).
    """


# ============================================================================
# DAG analyses
# ============================================================================


def successors_of(route: Route, hop_pos: int) -> Tuple[int, ...]:
    """All successor hop positions for ``hops[hop_pos]``. Delegates
    to :meth:`Route.successors` — re-exported here so analyses don't
    have to remember the implicit-linear-chain rule."""
    return route.successors(hop_pos)


def predecessors_of(route: Route, hop_pos: int) -> Tuple[int, ...]:
    """All predecessor hop positions for ``hops[hop_pos]``."""
    return route.predecessors(hop_pos)


def is_dag(route: Route) -> bool:
    """True iff the route's hop graph has no cycles. A linear
    (implicit-chain) route is trivially a DAG; this catches malformed
    explicit ``succs`` / ``preds`` that produce a cycle."""
    try:
        topological_order(route)
    except InvalidRouteError:
        return False
    return True


def topological_order(route: Route) -> Tuple[int, ...]:
    """Topological sort of hop positions via Kahn's algorithm.

    Returns hop positions in an order where every hop's predecessors
    appear before it. Raises :class:`InvalidRouteError` if the graph
    contains a cycle.

    For linear routes the result is just ``(0, 1, 2, ..., len-1)``
    — the same order they're already in.
    """
    n = len(route.hops)
    in_degree = [len(predecessors_of(route, i)) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    order: List[int] = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for succ in successors_of(route, i):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    if len(order) != n:
        cyclic = [i for i in range(n) if in_degree[i] > 0]
        raise InvalidRouteError(
            f"Route contains a cycle; hops with unresolved predecessors: "
            f"{cyclic}"
        )
    return tuple(order)


def reachable_from_entry(route: Route) -> Set[int]:
    """Set of hop positions reachable from the entry hop (position 0)
    via the successor relation. Used to detect unreachable hops that
    a malformed constructor might have left behind."""
    n = len(route.hops)
    if n == 0:
        return set()
    seen: Set[int] = set()
    stack = [0]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        for s in successors_of(route, i):
            if 0 <= s < n:
                stack.append(s)
    return seen


def unreachable_hops(route: Route) -> Tuple[int, ...]:
    """Hops not reachable from the entry hop. Empty for well-formed
    routes; non-empty means a constructor produced a disconnected
    fragment somewhere — almost certainly a bug."""
    n = len(route.hops)
    reachable = reachable_from_entry(route)
    return tuple(i for i in range(n) if i not in reachable)


def critical_path(
    route: Route, hop_cost_ms: Optional[Dict[int, float]] = None
) -> Tuple[float, Tuple[int, ...]]:
    """Longest path through the DAG by cumulative hop cost.

    Args:
        route: The route to analyze.
        hop_cost_ms: Optional per-hop cost estimate in milliseconds
            (e.g. from a profile). If omitted, every hop counts as
            1.0 (so the result is the longest unweighted path).

    Returns:
        ``(total_cost_ms, hop_positions_on_path)`` — the total
        cost along the critical path and the sequence of hop
        positions that traverses it.

    The critical path is the wall-time floor under perfect
    pipelining: no amount of parallelism can finish a request
    faster than the longest dependency chain.
    """
    n = len(route.hops)
    if n == 0:
        return (0.0, ())
    cost = [(hop_cost_ms or {}).get(i, 1.0) for i in range(n)]
    order = topological_order(route)
    # dp[i] = (max cost to reach end of hop i, predecessor on path)
    dp: List[Tuple[float, Optional[int]]] = [(0.0, None)] * n
    for i in order:
        best_pred_cost = 0.0
        best_pred: Optional[int] = None
        for p in predecessors_of(route, i):
            if dp[p][0] > best_pred_cost or best_pred is None:
                best_pred_cost = dp[p][0]
                best_pred = p
        dp[i] = (best_pred_cost + cost[i], best_pred)
    # Find the maximum dp[i] (path-terminal)
    end = max(range(n), key=lambda i: dp[i][0])
    # Reconstruct path
    path: List[int] = []
    cur: Optional[int] = end
    while cur is not None:
        path.append(cur)
        cur = dp[cur][1]
    path.reverse()
    return (dp[end][0], tuple(path))


def activation_lifetimes(route: Route) -> Dict[int, int]:
    """For each hop, the latest hop position that consumes its
    output (directly via the successor relation).

    Returns a dict mapping ``hop_pos → last_consumer_pos``. If a hop
    has no successors (the exit hop), its last_consumer is itself —
    the output is "alive" until the request completes. The dispatcher
    can use these lifetimes to insert eager-free hints (free the
    activation buffer as soon as the last consumer has finished).
    """
    n = len(route.hops)
    lifetimes: Dict[int, int] = {}
    for i in range(n):
        succs = successors_of(route, i)
        if succs:
            lifetimes[i] = max(succs)
        else:
            lifetimes[i] = i
    return lifetimes


# ============================================================================
# Rewrite passes
# ============================================================================


class RouteRewriter:
    """Base class for a single-pass rewrite. Override :meth:`apply`."""

    name: str = "base"

    def apply(self, route: Route) -> Route:  # pragma: no cover - abstract
        raise NotImplementedError


class ValidateRoute(RouteRewriter):
    """Check route well-formedness. Raises
    :class:`InvalidRouteError` on violation; returns the route
    unchanged otherwise.

    Checks:
        1. At least one hop.
        2. Every hop has at least one layer.
        3. Layer indices on each hop are strictly ascending.
        4. ``succs`` references are in-range.
        5. ``preds`` references are in-range.
        6. The graph is acyclic.
        7. All hops are reachable from the entry.
        8. Multi-input hops have a valid ``reduce_op``.
    """

    name = "validate"

    def apply(self, route: Route) -> Route:
        n = len(route.hops)
        if n == 0:
            raise InvalidRouteError("Route has no hops.")
        for i, hop in enumerate(route.hops):
            if len(hop.layer_indices) == 0:
                raise InvalidRouteError(
                    f"Hop {i} (shard={hop.shard_idx}) has no layers."
                )
            li = hop.layer_indices
            if any(li[k + 1] <= li[k] for k in range(len(li) - 1)):
                raise InvalidRouteError(
                    f"Hop {i} has non-strict-ascending layer_indices={li}"
                )
            if hop.succs is not None:
                bad = [s for s in hop.succs if s < 0 or s >= n]
                if bad:
                    raise InvalidRouteError(
                        f"Hop {i}'s succs={hop.succs} contains out-of-range "
                        f"positions {bad} (route has {n} hops)"
                    )
            if hop.preds is not None:
                bad = [p for p in hop.preds if p < 0 or p >= n]
                if bad:
                    raise InvalidRouteError(
                        f"Hop {i}'s preds={hop.preds} contains out-of-range "
                        f"positions {bad} (route has {n} hops)"
                    )
            if hop.preds is not None and len(hop.preds) > 1:
                if hop.reduce_op not in ("sum", "mean"):
                    raise InvalidRouteError(
                        f"Hop {i} has {len(hop.preds)} predecessors but "
                        f"reduce_op={hop.reduce_op!r}; expected 'sum' or 'mean'."
                    )
        if not is_dag(route):
            raise InvalidRouteError("Route hop graph contains a cycle.")
        u = unreachable_hops(route)
        if u:
            raise InvalidRouteError(
                f"Hops {u} are not reachable from the entry hop. "
                f"A malformed constructor probably left a disconnected "
                f"fragment in the route."
            )
        return route


class DropEmptyHops(RouteRewriter):
    """Remove hops with zero layers. Degenerate state — should never
    occur in a healthy route — but defensive cleanup catches
    constructor bugs.

    Empty hops have no layers to dispatch on, so they're pure
    pass-throughs. Removing them re-indexes downstream hops and
    re-maps any ``succs`` / ``preds`` references.
    """

    name = "drop_empty_hops"

    def apply(self, route: Route) -> Route:
        n = len(route.hops)
        keep = [i for i in range(n) if len(route.hops[i].layer_indices) > 0]
        if len(keep) == n:
            return route  # nothing to drop
        return _remap_hops(route, keep)


class MergeAdjacentSameShardHops(RouteRewriter):
    """Merge consecutive hops on the same shard with no intervening
    cross-shard work.

    When the planner emits ``hop_K`` and ``hop_K+1`` both on shard S
    and the only relationship between them is the implicit linear
    chain (``K+1.preds == (K,)``, ``K.succs == (K+1,)``), they can
    be collapsed into one hop with the union of their layer_indices.
    The merged hop inherits ``K``'s predecessors and ``K+1``'s
    successors.

    This is mostly a defensive cleanup — the planner already does
    this collapsing — but it makes the framework robust against
    routes constructed by hand or by future planners that don't
    pre-merge.
    """

    name = "merge_adjacent_same_shard"

    def apply(self, route: Route) -> Route:
        # Walk linearly; merge i+1 into i when conditions hold.
        # Repeat until no more merges fire (multi-step merges).
        prev_hops = route.hops
        while True:
            new_hops, merged = self._merge_one_pass(prev_hops, route)
            if not merged:
                return Route(hops=tuple(prev_hops))
            prev_hops = new_hops
            route = Route(hops=tuple(prev_hops))

    @staticmethod
    def _merge_one_pass(
        hops: Sequence[RouteHop], route: Route
    ) -> Tuple[List[RouteHop], bool]:
        n = len(hops)
        for i in range(n - 1):
            a, b = hops[i], hops[i + 1]
            if a.shard_idx != b.shard_idx:
                continue
            # Both must use implicit linear connectivity between them:
            #   a's only successor is b, and b's only predecessor is a
            a_succs = route.successors(i)
            b_preds = route.predecessors(i + 1)
            if a_succs != (i + 1,):
                continue
            if b_preds != (i,):
                continue
            # b's reduce_op must be None (it has one pred, so reduce
            # doesn't apply) — defensive.
            if b.reduce_op is not None and len(b_preds) <= 1:
                continue
            # Merge layer_indices, drop b.
            merged_layers = tuple(sorted(set(a.layer_indices) | set(b.layer_indices)))
            new_hop = RouteHop(
                shard_idx=a.shard_idx,
                layer_indices=merged_layers,
                succs=b.succs,
                preds=a.preds,
                reduce_op=a.reduce_op,
            )
            new_hops = list(hops[:i]) + [new_hop] + list(hops[i + 2:])
            return _remap_hops(Route(hops=tuple(new_hops)), list(range(len(new_hops)))).hops, True
        return list(hops), False


def _compute_downstream_costs(
    route: Route, hop_cost_ms: Dict[int, float]
) -> Dict[int, float]:
    """For each hop, the longest path from it to any exit (inclusive
    of the hop's own cost). Computed via reverse-topological DP.

    Used by :class:`PrioritizeFanOutByCost` to decide which
    successor of a fan-out hop should receive its activation first
    — the successor with the longest downstream chain is the
    critical-path bottleneck and starting it earliest minimizes
    overall wall time.
    """
    n = len(route.hops)
    cost = [hop_cost_ms.get(i, 1.0) for i in range(n)]
    downstream: Dict[int, float] = {}
    order = topological_order(route)
    for i in reversed(order):
        succs = successors_of(route, i)
        if not succs:
            downstream[i] = cost[i]
        else:
            downstream[i] = cost[i] + max(downstream[s] for s in succs)
    return downstream


class PrioritizeFanOutByCost(RouteRewriter):
    """Reorder a fan-out hop's ``succs`` list so successors with
    the longest downstream critical path come first.

    Motivation: at a hop with N successors, the dispatcher emits
    N ``send_hidden`` calls in ``succs`` order. With async NVSHMEM
    transport these sends run concurrently on the GPU, but **the
    receiver-side compute can only start once its activation has
    arrived**. So the wall-time floor on each branch is
    ``send_latency + downstream_critical_path``. Picking the
    longest downstream first means the longest branch starts its
    compute as early as possible — net wall time shrinks toward
    the longest branch's natural duration.

    For linear hops (one successor) this is a no-op. For DAG
    fan-out (e.g. MoE backbone → multiple expert shards) it picks
    the slowest expert to send to first.

    Args:
        hop_cost_ms: Per-hop cost estimate (typically from a
            profiling pass like ``pipelining_gap_probe``). Missing
            entries default to 1.0 unit-cost, so the pass is
            usable without profiling data (it just picks the
            unweighted-longest path).

    Composition: re-orders only ``succs``. ``preds`` references
    on receiver hops are untouched (the reduce_op semantics are
    commutative: ``sum`` and ``mean`` don't care about predecessor
    order).
    """

    name = "prioritize_fan_out_by_cost"

    def __init__(self, hop_cost_ms: Optional[Dict[int, float]] = None) -> None:
        self._costs = hop_cost_ms or {}

    def apply(self, route: Route) -> Route:
        if not route.is_dag():
            # Linear routes have at most one successor per hop —
            # nothing to reorder.
            return route
        downstream = _compute_downstream_costs(route, self._costs)
        new_hops: List[RouteHop] = []
        for hop in route.hops:
            if hop.succs is None or len(hop.succs) <= 1:
                new_hops.append(hop)
                continue
            # Sort succs by descending downstream cost so the
            # heaviest branch fires first.
            reordered = tuple(
                sorted(hop.succs, key=lambda s: -downstream[s])
            )
            if reordered == hop.succs:
                new_hops.append(hop)
            else:
                new_hops.append(replace(hop, succs=reordered))
        return Route(hops=tuple(new_hops))


class PrioritizeFanOutByTopology(RouteRewriter):
    """Reorder a fan-out hop's ``succs`` list so successors on the
    farthest network distance come first.

    Motivation: a cross-node NVSHMEM put pays a higher latency
    (e.g. IB ~30 µs) than an intra-node put (NVLink ~5 µs). If a
    hop fans to mixed-distance destinations, issuing the high-latency
    sends first hides their cost behind the lower-latency sends
    that follow. By the time the lower-latency sends queue, the
    high-latency ones are already in flight.

    Args:
        shard_node_id: Mapping from ``shard_idx`` to an opaque
            "node id" (any hashable). Shards with the same id are
            treated as on the same node (closer); different ids
            are treated as on different nodes (farther). The pass
            doesn't care about absolute distances, only the
            partition.

    For single-node deployments where every shard has the same
    node id, this pass is a no-op (no distance variance to
    exploit). Composes safely after :class:`PrioritizeFanOutByCost`
    — the secondary cost-based order becomes a tiebreaker within
    each distance class.

    Composition: re-orders only ``succs``; ``preds`` references
    on receiver hops are untouched.
    """

    name = "prioritize_fan_out_by_topology"

    def __init__(self, shard_node_id: Dict[int, object]) -> None:
        self._node_id = shard_node_id

    def apply(self, route: Route) -> Route:
        if not route.is_dag():
            return route
        new_hops: List[RouteHop] = []
        for hop in route.hops:
            if hop.succs is None or len(hop.succs) <= 1:
                new_hops.append(hop)
                continue
            src_node = self._node_id.get(hop.shard_idx)

            def _distance_class(succ_pos: int) -> int:
                # 1 if cross-node from src, 0 if same-node. Cross-node
                # comes first under descending sort.
                dst_shard = route.hops[succ_pos].shard_idx
                dst_node = self._node_id.get(dst_shard)
                if src_node is None or dst_node is None:
                    return 0
                return 1 if src_node != dst_node else 0

            reordered = tuple(sorted(hop.succs, key=lambda s: -_distance_class(s)))
            if reordered == hop.succs:
                new_hops.append(hop)
            else:
                new_hops.append(replace(hop, succs=reordered))
        return Route(hops=tuple(new_hops))


class SortHopsTopologically(RouteRewriter):
    """Re-order ``route.hops`` so the hop tuple is in topological
    order. The dispatcher walks ``hops`` in order, and the implicit
    linear chain (``hop_pos + 1`` as default successor) only makes
    sense if the order is topologically sound.

    For routes with no explicit ``succs`` / ``preds`` overrides this
    is a no-op (the construction order is already topological). For
    DAG routes with explicit edges, this normalizes the ordering and
    re-maps the edge references.
    """

    name = "sort_topologically"

    def apply(self, route: Route) -> Route:
        order = topological_order(route)
        if order == tuple(range(len(route.hops))):
            return route  # already topo
        return _remap_hops(route, list(order))


# ============================================================================
# Composer
# ============================================================================


@dataclass
class RewriteResult:
    """Outcome of running a rewrite pipeline. Useful for telemetry
    and unit tests that want to know what each pass changed."""

    rewritten: Route
    passes_applied: Tuple[str, ...]
    n_hops_before: int
    n_hops_after: int


def apply_rewrites(
    route: Route,
    passes: Optional[Sequence[RouteRewriter]] = None,
    *,
    verbose: bool = False,
) -> RewriteResult:
    """Run a sequence of rewrite passes against ``route``.

    Args:
        route: Input route.
        passes: List of :class:`RouteRewriter` instances to apply in
            order. ``None`` runs the default pipeline (validate, drop
            empty, merge same-shard, topo sort).
        verbose: If True, print a one-line summary per pass for
            debugging.

    Returns:
        :class:`RewriteResult` carrying the rewritten route + diff
        metadata.
    """
    if passes is None:
        passes = default_pipeline()
    n_before = len(route.hops)
    applied: List[str] = []
    for p in passes:
        before = route
        route = p.apply(route)
        applied.append(p.name)
        if verbose:
            print(
                f"[rewrite] {p.name}: hops {len(before.hops)} → {len(route.hops)}"
            )
    return RewriteResult(
        rewritten=route,
        passes_applied=tuple(applied),
        n_hops_before=n_before,
        n_hops_after=len(route.hops),
    )


def default_pipeline() -> List[RouteRewriter]:
    """The default sequence of passes for production use.

    Order matters: validate first (so later passes can assume DAG
    invariants); drop empty hops before merging (merge logic assumes
    every hop has layers); merge before topo sort (merge changes
    hop positions, so the final sort normalizes).
    """
    return [
        ValidateRoute(),
        DropEmptyHops(),
        MergeAdjacentSameShardHops(),
        SortHopsTopologically(),
        ValidateRoute(),  # post-condition check
    ]


# ============================================================================
# Internal helpers
# ============================================================================


def _remap_hops(route: Route, new_order: List[int]) -> Route:
    """Build a new Route from ``route.hops`` keeping only positions
    in ``new_order`` (and in that order). Updates any ``succs`` /
    ``preds`` references in the kept hops to point at the new
    positions.

    Caller is responsible for ensuring ``new_order`` is consistent
    (no dangling references after the remap).
    """
    old_to_new = {old: new for new, old in enumerate(new_order)}
    new_hops: List[RouteHop] = []
    for old_pos in new_order:
        hop = route.hops[old_pos]
        succs = (
            tuple(old_to_new[s] for s in hop.succs if s in old_to_new)
            if hop.succs is not None
            else None
        )
        preds = (
            tuple(old_to_new[p] for p in hop.preds if p in old_to_new)
            if hop.preds is not None
            else None
        )
        new_hops.append(
            RouteHop(
                shard_idx=hop.shard_idx,
                layer_indices=hop.layer_indices,
                succs=succs,
                preds=preds,
                reduce_op=hop.reduce_op,
            )
        )
    return Route(hops=tuple(new_hops))
