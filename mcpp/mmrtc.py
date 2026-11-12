"""Min-max rooted tree cover (MMRTC) on the coarse grid graph.

Problem: given a connected graph G=(V,E) with edge weights and k roots,
find k trees T_1..T_k, T_i containing root r_i, whose union covers V and
that minimise max_i w(T_i). On our grids all edges have unit weight
(one coarse step), so w(T_i) = |V(T_i)| - 1 and the STC tour around T_i has
4 |V(T_i)| fine moves = 2 * (2 w(T_i) + 2), i.e. the tour costs about twice
the tree weight in coarse units - the classic "tour <= 2 x tree" bound.

Unlike a partition, trees may *overlap* (a vertex covered twice), which can
only help the min-max objective but creates redundant coverage and possible
inter-robot conflicts.

Algorithm: rooted 4-approximation of
    G. Even, N. Garg, J. Koenemann, R. Ravi, A. Sinha,
    "Min-max tree covers of graphs", Operations Research Letters 32, 2004.
re-implemented from the paper's description. For a guess B:

1. contract all roots into one vertex and compute an MST (with unit weights
   any spanning tree is minimum: we use the multi-source BFS forest). If its
   weight n-k exceeds k*B then B < OPT;
2. un-contract: a spanning forest with one tree per root;
3. cut each tree bottom-up into *pieces* of weight in [B, 2B) plus a residual
   tree around the root of weight < B (children subtrees of a vertex are
   grouped greedily; each group is closed as soon as its weight reaches B);
4. a piece may be served by root r if dist(r, piece) <= B; find a bipartite
   matching that serves every piece with distinct roots (Kuhn's augmenting
   paths). If none exists then B < OPT (Hall's theorem argument in the paper);
5. robot i gets residual_i + shortest path to its piece + piece:
   weight < B + B + 2B = 4B.

A binary search returns a B that is feasible while B-1 was proven
infeasible, so B <= OPT and the cover costs < 4 OPT - even without
monotonicity of the feasibility test.

``dedup=True`` adds a post-processing step (in the spirit of the
deduplication used by Tang & Ma, RA-L 2023): a vertex covered by several
trees is removed from the largest tree if it is neither that tree's root nor
an articulation point of it. This never increases any tree.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .graph import Cell, articulation_points, bfs_distances, bfs_tree, cells_to_mask, multi_source_labels
from .grid import CoarseMap


@dataclass
class TreeCoverResult:
    sets: list[set[Cell]]        # vertex set of each robot's tree
    bound_B: int                 # final guess; B <= OPT max tree weight, hence
                                 # 4 (B + 1) is a certified makespan lower bound
                                 # for any tree cover followed by STC
    max_weight: int              # max_i (|V_i| - 1) before dedup
    runtime: float
    feasibility_tests: int


def _kuhn_matching(adj: list[list[int]], n_right: int) -> list[int] | None:
    """Maximum bipartite matching saturating the left side, else ``None``.
    ``adj[p]`` = right vertices allowed for left vertex ``p``. O(V E)."""
    match_right = [-1] * n_right

    def try_augment(p: int, seen: list[bool]) -> bool:
        for r in adj[p]:
            if not seen[r]:
                seen[r] = True
                if match_right[r] < 0 or try_augment(match_right[r], seen):
                    match_right[r] = p
                    return True
        return False

    for p in range(len(adj)):
        if not try_augment(p, [False] * n_right):
            return None
    match_left = [-1] * len(adj)
    for r, p in enumerate(match_right):
        if p >= 0:
            match_left[p] = r
    return match_left


class _Forest:
    """BFS forest of the contracted graph (one tree per root)."""

    def __init__(self, cmap: CoarseMap, roots: list[Cell]):
        labels, parent = multi_source_labels(cmap.free, roots)
        h, w = cmap.shape
        self.cells = cmap.cells()
        self.children: dict[Cell, list[Cell]] = {c: [] for c in self.cells}
        self.parent: dict[Cell, Cell | None] = {}
        for c in self.cells:
            p = int(parent[c])
            if p < 0:
                self.parent[c] = None
            else:
                pc = (p // w, p % w)
                self.parent[c] = pc
                self.children[pc].append(c)
        self.roots = roots
        # Post-order of every tree (children before parents), iterative.
        self.post: list[list[Cell]] = []
        for r in roots:
            order, stack = [], [r]
            while stack:
                v = stack.pop()
                order.append(v)
                stack.extend(self.children[v])
            order.reverse()
            self.post.append(order)


def _decompose(forest: _Forest, B: int):
    """Step 3: pieces of weight in [B, 2B) and per-root residual trees."""
    piece_of_edge: dict[Cell, int] = {}   # child vertex -> piece id of edge (parent, child)
    pieces_top: list[tuple[Cell, list[Cell]]] = []
    for order in forest.post:
        resid: dict[Cell, int] = {}
        for v in order:
            acc, group = 0, []
            for c in forest.children[v]:
                acc += resid[c] + 1
                group.append(c)
                if acc >= B:
                    pid = len(pieces_top)
                    for g in group:
                        piece_of_edge[g] = pid
                    pieces_top.append((v, group))
                    acc, group = 0, []
            resid[v] = acc

    def reach(c: Cell) -> list[Cell]:
        """``c`` plus descendants reachable through uncut edges."""
        out, stack = [], [c]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(ch for ch in forest.children[x] if ch not in piece_of_edge)
        return out

    pieces = []
    for v, group in pieces_top:
        s = {v}
        for g in group:
            s.update(reach(g))
        pieces.append(s)
    residuals = [set(reach(r)) for r in forest.roots]
    return pieces, residuals


def mmrtc_even(cmap: CoarseMap, starts: list[Cell], dedup: bool = True) -> TreeCoverResult:
    """Rooted min-max tree cover, 4-approximation (see module docstring)."""
    t0 = time.perf_counter()
    k = len(starts)
    n = cmap.n_cells
    forest = _Forest(cmap, starts)
    dist = [bfs_distances(cmap.free, [r]) for r in starts]
    parents = [bfs_tree(cmap.free, r) for r in starts]
    tests = 0

    def feasible(B: int):
        nonlocal tests
        tests += 1
        if n - k > k * B:
            return None
        pieces, residuals = _decompose(forest, B)
        if len(pieces) > k:
            return None
        # nearest vertex of each piece from each root
        adj, nearest = [], []
        for P in pieces:
            pl = list(P)
            row, near = [], {}
            for j in range(k):
                dj = dist[j]
                best = min(pl, key=lambda c: dj[c])
                if dj[best] <= B:
                    row.append(j)
                    near[j] = best
            adj.append(row)
            nearest.append(near)
        match = _kuhn_matching(adj, k)
        if match is None:
            return None
        sets = [set(residuals[j]) for j in range(k)]
        for p, j in enumerate(match):
            # shortest path root_j -> nearest vertex of piece p
            x: Cell | None = nearest[p][j]
            while x is not None:
                sets[j].add(x)
                x = parents[j][x]
            sets[j].update(pieces[p])
        return sets

    # Every feasible guess yields a cover; we keep the one with the smallest
    # actual max tree. It is never worse than the cover of the final guess B,
    # so the 4 * OPT guarantee is preserved.
    best: list[set[Cell]] | None = None

    def keep(s: list[set[Cell]] | None) -> list[set[Cell]] | None:
        nonlocal best
        if s is not None and (best is None or max(map(len, s)) < max(map(len, best))):
            best = s
        return s

    lo = max(1, math.ceil(n / k) - 1)  # every cover needs max |V_i| >= n/k
    B = lo
    if keep(feasible(lo)) is None:
        hi = n
        assert keep(feasible(hi)) is not None, "B = n must be feasible"
        while hi - lo > 1:  # invariant: lo infeasible, hi feasible
            mid = (lo + hi) // 2
            if keep(feasible(mid)) is None:
                lo = mid
            else:
                hi = mid
        B = hi
    assert best is not None
    sets = best
    max_w = max(len(s) - 1 for s in sets)
    if dedup:
        sets = deduplicate(sets, starts, cmap.shape)
    return TreeCoverResult(sets, B, max_w, time.perf_counter() - t0, tests)


def deduplicate(sets: list[set[Cell]], starts: list[Cell], shape: tuple[int, int]) -> list[set[Cell]]:
    """Remove doubly covered vertices from the largest trees while keeping
    every tree connected and every vertex covered."""
    sets = [set(s) for s in sets]
    cover: dict[Cell, int] = {}
    for s in sets:
        for c in s:
            cover[c] = cover.get(c, 0) + 1
    changed = True
    while changed:
        changed = False
        for i in sorted(range(len(sets)), key=lambda j: -len(sets[j])):
            dup = [c for c in sets[i] if cover[c] > 1 and c != starts[i]]
            if not dup:
                continue
            ap = articulation_points(cells_to_mask(sets[i], shape))
            for c in dup:
                if c in ap:
                    continue
                sets[i].discard(c)
                cover[c] -= 1
                changed = True
                break  # articulation points change: recompute
            if changed:
                break
    return sets
