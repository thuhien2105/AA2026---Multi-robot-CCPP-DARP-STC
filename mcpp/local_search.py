"""Extension: min-max local search that improves any valid partition.

Because an STC tour of a region with ``s`` coarse cells has exactly ``4 s``
moves, the makespan of "partition + STC" equals ``4 * max_i s_i / v_i``
(``v_i`` = speed of robot ``i``; 1 for homogeneous teams). The local search
below therefore minimises the largest (speed-weighted) region size while
keeping every region connected.

Move = *transfer chain*. Let ``L`` be a region with maximal cost. We search
the region-adjacency graph from ``L`` by BFS for a region ``t`` whose cost
after receiving one cell stays strictly below ``cost(L)``. Along the found
path ``L = r_0 -> r_1 -> ... -> r_m = t`` every ``r_j`` hands one boundary
cell to ``r_{j+1}``; intermediate regions keep their size, ``L`` shrinks by
one and ``t`` grows by one.

A cell may leave region ``r`` only if it is not ``r``'s start cell and its
removal keeps ``r`` connected (it is not an articulation point - computed
with Tarjan's algorithm, then re-verified at execution time because earlier
transfers of the same chain may change the region). Among valid cells we
prefer the one with most neighbours inside the receiving region (smoother
boundaries -> fewer turns). Failed chains are rolled back and the offending
edge is blacklisted for the current round.

Termination: for homogeneous robots every successful chain decreases the
potential ``sum_i s_i^2`` by at least 2 (``s_L >= s_t + 2``), so the search
stops after O(|V|^2) chains; ``max_rounds`` additionally caps the work.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .graph import Cell, N4, articulation_points, is_connected
from .grid import CoarseMap
from .partition import region_sizes


@dataclass
class LSResult:
    assign: np.ndarray
    moves: int
    chains: int
    runtime: float
    max_before: float
    max_after: float


def _boundary_moves(assign: np.ndarray, region: int, starts: list[Cell]) -> dict[int, list[Cell]]:
    """For ``region``: receiving region -> candidate cells (non-articulation,
    not the start) that are 4-adjacent to the receiving region."""
    mask = assign == region
    if mask.sum() <= 1:
        return {}
    ap = articulation_points(mask)
    h, w = assign.shape
    start = starts[region]
    out: dict[int, list[Cell]] = {}
    rr, cc = np.nonzero(mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        if (r, c) == start or (r, c) in ap:
            continue
        seen = set()
        for dr, dc in N4:
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w:
                o = int(assign[a, b])
                if o >= 0 and o != region and o not in seen:
                    seen.add(o)
                    out.setdefault(o, []).append((r, c))
    return out


def improve_partition(assign: np.ndarray, cmap: CoarseMap, starts: list[Cell],
                      speeds: list[float] | None = None, max_rounds: int = 100000,
                      time_limit: float = 60.0) -> LSResult:
    """Min-max transfer-chain local search (see module docstring)."""
    t0 = time.perf_counter()
    assign = assign.copy()
    k = len(starts)
    v = np.ones(k) if speeds is None else np.asarray(speeds, dtype=float)
    sizes = region_sizes(assign, k).astype(float)
    max_before = float((sizes / v).max())
    moves = chains = 0
    h, w = assign.shape

    def nbr_count(cell: Cell, region: int) -> int:
        r, c = cell
        return sum(1 for dr, dc in N4 if 0 <= r + dr < h and 0 <= c + dc < w and assign[r + dr, c + dc] == region)

    for _ in range(max_rounds):
        if time.perf_counter() - t0 > time_limit:
            break
        cost = sizes / v
        L = int(np.argmax(cost))
        cache: dict[int, dict[int, list[Cell]]] = {}
        banned: set[tuple[int, int]] = set()
        improved = False
        while not improved:
            # BFS on the region graph restricted to edges with candidate cells
            parent = {L: -1}
            q = deque([L])
            target = -1
            while q and target < 0:
                a = q.popleft()
                if a not in cache:
                    cache[a] = _boundary_moves(assign, a, starts)
                for b in sorted(cache[a], key=lambda x: cost[x]):
                    if b in parent or (a, b) in banned:
                        continue
                    parent[b] = a
                    if (sizes[b] + 1) / v[b] < cost[L] - 1e-9:
                        target = b
                        break
                    q.append(b)
            if target < 0:
                break  # no improving chain from L -> local optimum
            path = [target]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            path.reverse()  # L ... target
            done: list[tuple[Cell, int, int]] = []
            ok = True
            for a, b in zip(path[:-1], path[1:]):
                cands = sorted(cache[a].get(b, []), key=lambda c_: -nbr_count(c_, b))
                moved = False
                for cell in cands:
                    if assign[cell] != a or nbr_count(cell, b) == 0:
                        continue
                    assign[cell] = b
                    if is_connected(assign == a):
                        done.append((cell, a, b))
                        moved = True
                        break
                    assign[cell] = a
                if not moved:
                    ok = False
                    banned.add((a, b))
                    break
            if not ok:
                for cell, a, _b in reversed(done):  # roll back the partial chain
                    assign[cell] = a
                for x in {x for c_, a_, b_ in done for x in (a_, b_)}:
                    cache.pop(x, None)
                continue
            sizes[path[0]] -= 1
            sizes[path[-1]] += 1
            moves += len(done)
            chains += 1
            improved = True
        if not improved:
            break
    return LSResult(assign, moves, chains, time.perf_counter() - t0, max_before, float((sizes / v).max()))
