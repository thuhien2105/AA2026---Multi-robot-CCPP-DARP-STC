"""Graph primitives on 4-connected grids: BFS, connected components,
articulation points (Tarjan) and a union-find structure.

All functions take a boolean mask ``mask[r, c]`` that marks the vertices of
an induced grid sub-graph; two vertices are adjacent iff they are 4-neighbours.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np
from scipy import ndimage

from .grid import FOUR_CONN

Cell = tuple[int, int]
N4: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def neighbors(cell: Cell, mask: np.ndarray) -> list[Cell]:
    """4-neighbours of ``cell`` that lie inside ``mask``."""
    r, c = cell
    h, w = mask.shape
    out = []
    for dr, dc in N4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            out.append((rr, cc))
    return out


def bfs_distances(mask: np.ndarray, sources: Iterable[Cell]) -> np.ndarray:
    """Multi-source BFS hop distance inside ``mask``; ``-1`` = unreachable.

    Time O(|V|), memory O(|V|).
    """
    dist = np.full(mask.shape, -1, dtype=np.int64)
    q: deque[Cell] = deque()
    for s in sources:
        if mask[s] and dist[s] < 0:
            dist[s] = 0
            q.append(s)
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = d
                q.append((rr, cc))
    return dist


def bfs_tree(mask: np.ndarray, root: Cell) -> dict[Cell, Cell | None]:
    """BFS parent pointers from ``root`` (``parent[root] is None``)."""
    parent: dict[Cell, Cell | None] = {root: None}
    q: deque[Cell] = deque([root])
    while q:
        v = q.popleft()
        for u in neighbors(v, mask):
            if u not in parent:
                parent[u] = v
                q.append(u)
    return parent


def multi_source_labels(mask: np.ndarray, sources: list[Cell]) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic Voronoi labelling by simultaneous BFS from all sources.

    Returns ``(labels, parent_index)``: ``labels[v]`` is the index of the
    source that reached ``v`` first (ties -> lower index, because sources are
    enqueued in index order), ``-1`` outside the mask / unreachable.
    ``parent_index[v]`` is the flat index of the BFS parent (``-1`` for roots),
    which makes the BFS forest available to callers (min-max tree cover).
    Each label class is connected because every vertex is attached to its BFS
    parent, which carries the same label.
    """
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int64)
    parent = np.full((h, w), -1, dtype=np.int64)
    q: deque[Cell] = deque()
    for i, s in enumerate(sources):
        if labels[s] >= 0:
            raise ValueError(f"duplicate source {s}")
        labels[s] = i
        q.append(s)
    while q:
        r, c = q.popleft()
        lab = labels[r, c]
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and labels[rr, cc] < 0:
                labels[rr, cc] = lab
                parent[rr, cc] = r * w + c
                q.append((rr, cc))
    return labels, parent


def count_components(mask: np.ndarray) -> int:
    """Number of 4-connected components of ``mask`` (C implementation)."""
    _, n = ndimage.label(mask, structure=FOUR_CONN)
    return int(n)


def is_connected(mask: np.ndarray) -> bool:
    return count_components(mask) <= 1


def articulation_points(mask: np.ndarray) -> set[Cell]:
    """Articulation points (cut vertices) of the grid graph induced by ``mask``.

    Iterative version of Tarjan's DFS low-link algorithm (no recursion limit
    issues on large regions). Time O(|V| + |E|) = O(|V|) on grids.
    A vertex ``v`` is a cut vertex iff
      * ``v`` is a DFS root with >= 2 DFS children, or
      * ``v`` is not a root and has a child ``u`` with ``low[u] >= disc[v]``.
    """
    disc: dict[Cell, int] = {}
    low: dict[Cell, int] = {}
    ap: set[Cell] = set()
    timer = 0
    rr, cc = np.nonzero(mask)
    for root in zip(rr.tolist(), cc.tolist()):
        if root in disc:
            continue
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        # Stack frames: (vertex, parent, neighbour list, next neighbour index)
        stack: list[list] = [[root, None, neighbors(root, mask), 0]]
        while stack:
            frame = stack[-1]
            v, par, nbrs, idx = frame
            if idx < len(nbrs):
                frame[3] += 1
                u = nbrs[idx]
                if u not in disc:
                    disc[u] = low[u] = timer
                    timer += 1
                    if v == root:
                        root_children += 1
                    stack.append([u, v, neighbors(u, mask), 0])
                elif u != par:
                    low[v] = min(low[v], disc[u])
            else:
                stack.pop()
                if par is not None:
                    low[par] = min(low[par], low[v])
                    if par != root and low[v] >= disc[par]:
                        ap.add(par)
        if root_children > 1:
            ap.add(root)
    return ap


class UnionFind:
    """Disjoint-set forest with union by rank and path halving.

    Amortised O(alpha(n)) per operation (inverse Ackermann).
    """

    def __init__(self, items: Iterable = ()):
        self.parent: dict = {}
        self.rank: dict = {}
        for x in items:
            self.add(x)

    def add(self, x) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a, b) -> bool:
        """Merge the sets of ``a`` and ``b``; return False if already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def cells_to_mask(cells: Iterable[Cell], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for c in cells:
        mask[c] = True
    return mask
