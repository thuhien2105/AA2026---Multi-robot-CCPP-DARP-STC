"""Spanning Tree Coverage (STC, Gabriely & Rimon 2001) for one robot.

Given a connected set ``V`` of coarse cells, STC

1. builds a spanning tree ``T`` of the coarse grid graph induced by ``V``;
2. circumnavigates ``T`` on the fine grid, which visits each of the
   ``4|V|`` fine sub-cells exactly once and returns to the start.

Circumnavigation is implemented by *cycle merging*: every coarse cell starts
as a 4-cycle over its sub-cells (TL-TR-BR-BL). For every tree edge between
cells ``a`` and ``b`` we delete the side of ``a`` facing ``b`` and the side of
``b`` facing ``a`` and add the two "bridging" edges. Removing one edge from
each of two disjoint cycles and reconnecting the four endpoints crosswise
always yields a single cycle, so by induction over the |V|-1 tree edges the
result is one Hamiltonian cycle of the 4|V| fine cells (invariant: after j
merges the fine graph is a disjoint union of |V|-j cycles, one per component
of the partially built tree). Each side of a coarse cell is shared with at
most one neighbour, hence an edge is never removed twice.

Total time O(|V|) for the tour plus the spanning-tree construction
(O(|V| log |V|) for Kruskal, O(|V|) for BFS/DFS).

The *shape* of the tree decides the number of turns of the tour; the
``horizontal`` / ``vertical`` / ``adaptive`` strategies are a simple
turn-reduction heuristic in the spirit of TMSTC* (long straight branches).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .graph import Cell, UnionFind

TREE_STRATEGIES = ("bfs", "dfs", "random", "horizontal", "vertical", "adaptive")


# --------------------------------------------------------------------------
# Spanning trees on a set of coarse cells
# --------------------------------------------------------------------------
def _cell_neighbors(cell: Cell, cells: set[Cell]) -> list[Cell]:
    r, c = cell
    return [n for n in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if n in cells]


def _grid_edges(cells: set[Cell]) -> list[tuple[Cell, Cell]]:
    """All undirected 4-adjacency edges (each listed once, row-major order)."""
    edges = []
    for (r, c) in sorted(cells):
        if (r, c + 1) in cells:
            edges.append(((r, c), (r, c + 1)))
        if (r + 1, c) in cells:
            edges.append(((r, c), (r + 1, c)))
    return edges


def spanning_tree(cells: set[Cell], root: Cell, strategy: str = "bfs",
                  rng: np.random.Generator | None = None) -> list[tuple[Cell, Cell]]:
    """Spanning tree (list of |V|-1 edges) of the grid graph induced by ``cells``.

    Strategies
    ----------
    bfs / dfs   graph search from ``root`` (O(|V|)).
    random      Kruskal with i.i.d. random edge weights (needs ``rng``).
    horizontal  Kruskal where horizontal edges are cheaper than vertical ones
                -> long horizontal "comb teeth", typically fewer turns.
    vertical    the transpose of ``horizontal``.
    """
    if root not in cells:
        raise ValueError("root must belong to the cell set")
    if strategy == "bfs":
        seen = {root}
        edges = []
        frontier: deque[Cell] = deque([root])
        while frontier:
            v = frontier.popleft()
            for u in _cell_neighbors(v, cells):
                if u not in seen:
                    seen.add(u)
                    edges.append((v, u))
                    frontier.append(u)
        if len(edges) != len(cells) - 1:
            raise ValueError("cell set is not connected")
        return edges
    if strategy == "dfs":
        edges = _dfs_tree(cells, root)
        if len(edges) != len(cells) - 1:
            raise ValueError("cell set is not connected")
        return edges

    edges = _grid_edges(cells)
    if strategy == "random":
        rng = rng or np.random.default_rng(0)
        weights = rng.random(len(edges))
    elif strategy == "horizontal":
        weights = np.array([0.0 if a[0] == b[0] else 1.0 for a, b in edges])
    elif strategy == "vertical":
        weights = np.array([0.0 if a[1] == b[1] else 1.0 for a, b in edges])
    else:
        raise ValueError(f"unknown tree strategy {strategy!r}")
    order = np.argsort(weights, kind="stable")
    uf = UnionFind(cells)
    tree = []
    for idx in order:
        a, b = edges[int(idx)]
        if uf.union(a, b):
            tree.append((a, b))
            if len(tree) == len(cells) - 1:
                break
    if len(tree) != len(cells) - 1:
        raise ValueError("cell set is not connected")
    return tree


def _dfs_tree(cells: set[Cell], root: Cell) -> list[tuple[Cell, Cell]]:
    """Depth-first spanning tree (iterative)."""
    seen = {root}
    edges = []
    stack = [(root, iter(_cell_neighbors(root, cells)))]
    while stack:
        v, it = stack[-1]
        for u in it:
            if u not in seen:
                seen.add(u)
                edges.append((v, u))
                stack.append((u, iter(_cell_neighbors(u, cells))))
                break
        else:
            stack.pop()
    return edges


# --------------------------------------------------------------------------
# Circumnavigation
# --------------------------------------------------------------------------
def stc_tour(cells: set[Cell], tree: list[tuple[Cell, Cell]], start: Cell) -> list[Cell]:
    """Closed coverage tour on the fine grid around ``tree``.

    ``start`` is a coarse cell; the tour starts and ends at its top-left
    sub-cell. Returns a list of fine cells of length ``4|V| + 1`` whose first
    and last element coincide.
    """
    adj: dict[Cell, set[Cell]] = defaultdict(set)

    def link(a: Cell, b: Cell) -> None:
        adj[a].add(b)
        adj[b].add(a)

    def unlink(a: Cell, b: Cell) -> None:
        adj[a].discard(b)
        adj[b].discard(a)

    def sub(cell: Cell) -> tuple[Cell, Cell, Cell, Cell]:
        r, c = cell
        return (2 * r, 2 * c), (2 * r, 2 * c + 1), (2 * r + 1, 2 * c), (2 * r + 1, 2 * c + 1)

    for cell in cells:  # one 4-cycle per coarse cell
        tl, tr, bl, br = sub(cell)
        link(tl, tr)
        link(tr, br)
        link(br, bl)
        link(bl, tl)
    for a, b in tree:  # merge the two cycles across each tree edge
        if b < a:
            a, b = b, a  # now a is left of / above b
        tla, tra, bla, bra = sub(a)
        tlb, trb, blb, brb = sub(b)
        if a[0] == b[0]:  # horizontal neighbours: a | b
            unlink(tra, bra)
            unlink(tlb, blb)
            link(tra, tlb)
            link(bra, blb)
        else:  # vertical neighbours: a above b
            unlink(bla, bra)
            unlink(tlb, trb)
            link(bla, tlb)
            link(bra, trb)

    s = sub(start)[0]
    tour = [s]
    prev, cur = None, s
    nxt = min(adj[s])  # deterministic direction choice
    while True:
        prev, cur = cur, nxt
        tour.append(cur)
        if cur == s:
            break
        a, b = tuple(adj[cur])
        nxt = a if a != prev else b
    if len(tour) != 4 * len(cells) + 1:  # pragma: no cover - guarded by tests
        raise RuntimeError("STC produced a non-Hamiltonian tour (tree not spanning?)")
    return tour


def count_turns(path: list[Cell], closed: bool = True) -> int:
    """Number of heading changes along a path (a U-turn counts as 2)."""
    if len(path) < 3:
        return 0
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])]
    pairs = list(zip(dirs[:-1], dirs[1:]))
    if closed and path[0] == path[-1]:
        pairs.append((dirs[-1], dirs[0]))
    turns = 0
    for d0, d1 in pairs:
        if d0 == d1:
            continue
        turns += 2 if (d0[0] == -d1[0] and d0[1] == -d1[1]) else 1
    return turns


def plan_region(cells: set[Cell], root: Cell, strategy: str = "adaptive",
                rng: np.random.Generator | None = None) -> tuple[list[Cell], list[tuple[Cell, Cell]]]:
    """STC tour for one robot. ``adaptive`` tries the horizontal and the
    vertical comb tree and keeps the tour with fewer turns."""
    if strategy == "adaptive":
        best = None
        for s in ("horizontal", "vertical"):
            tree = spanning_tree(cells, root, s, rng)
            tour = stc_tour(cells, tree, root)
            key = count_turns(tour)
            if best is None or key < best[0]:
                best = (key, tour, tree)
        assert best is not None
        return best[1], best[2]
    tree = spanning_tree(cells, root, strategy, rng)
    return stc_tour(cells, tree, root), tree
