"""Grid maps, the coarse (2x2 "mega-cell") decomposition used by STC, map
loaders (MovingAI ``.map`` format) and seeded map generators.

Terminology used throughout the package
---------------------------------------
* **fine grid**  - the cell size equals the robot footprint / tool width.
  Coverage paths are sequences of 4-connected fine cells.
* **coarse grid** - every coarse cell ("mega-cell") is a 2x2 block of fine
  cells. Spanning Tree Coverage (STC) builds its spanning tree on the coarse
  grid and circumnavigates it on the fine grid.

A coarse cell is *usable* only if all four of its fine sub-cells are free.
Fine cells that belong to partially-occupied coarse cells can therefore not
be covered by STC; this loss is measured explicitly by the coverage metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

# Characters that MovingAI marks as traversable.
_PASSABLE = set(".GS")
# 4-connectivity structuring element for scipy.ndimage.label.
FOUR_CONN = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class GridMap:
    """Occupancy grid on the *fine* level. ``occ[r, c] == True`` is an obstacle."""

    occ: np.ndarray
    name: str = "map"

    @property
    def shape(self) -> tuple[int, int]:
        return self.occ.shape  # type: ignore[return-value]

    @property
    def free(self) -> np.ndarray:
        return ~self.occ

    @property
    def n_free(self) -> int:
        return int((~self.occ).sum())


@dataclass
class CoarseMap:
    """Coarse (2x2) decomposition of a :class:`GridMap`.

    Attributes
    ----------
    fine:      the underlying fine grid.
    free:      bool (h, w) - usable coarse cells (all 4 sub-cells free) that
               belong to the largest 4-connected component.
    raw_free:  bool (h, w) - usable coarse cells before component filtering.
    """

    fine: GridMap
    free: np.ndarray
    raw_free: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape  # type: ignore[return-value]

    @property
    def n_cells(self) -> int:
        return int(self.free.sum())

    def cells(self) -> list[tuple[int, int]]:
        """All usable coarse cells in row-major order."""
        rr, cc = np.nonzero(self.free)
        return list(zip(rr.tolist(), cc.tolist()))

    @staticmethod
    def sub_cells(cell: tuple[int, int]) -> list[tuple[int, int]]:
        """The four fine cells (TL, TR, BL, BR) of a coarse cell."""
        r, c = cell
        return [(2 * r, 2 * c), (2 * r, 2 * c + 1), (2 * r + 1, 2 * c), (2 * r + 1, 2 * c + 1)]

    def coverable_fine(self) -> np.ndarray:
        """bool mask of fine cells that STC can cover (sub-cells of usable cells)."""
        mask = np.zeros(self.fine.shape, dtype=bool)
        h, w = self.shape
        block = np.kron(self.free, np.ones((2, 2), dtype=bool))
        mask[: 2 * h, : 2 * w] = block
        return mask


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------
def build_coarse(grid: GridMap, keep_largest_component: bool = True) -> CoarseMap:
    """Group fine cells into 2x2 blocks; a block is usable iff fully free.

    Trailing rows/columns of odd-sized maps are dropped (they cannot form a
    complete 2x2 block). If ``keep_largest_component`` is set, only the largest
    4-connected component of usable blocks is kept, so that every robot can
    reach every coarse cell (a requirement of all partitioning methods).
    """
    H, W = grid.shape
    h, w = H // 2, W // 2
    f = grid.free[: 2 * h, : 2 * w]
    # A block is free iff all four sub-cells are free.
    raw = f[0::2, 0::2] & f[0::2, 1::2] & f[1::2, 0::2] & f[1::2, 1::2]
    free = raw.copy()
    if keep_largest_component and raw.any():
        lab, n = ndimage.label(raw, structure=FOUR_CONN)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            free = lab == int(np.argmax(sizes))
    return CoarseMap(fine=grid, free=free, raw_free=raw)


def upsample(grid: GridMap, factor: int = 2) -> GridMap:
    """Replace each cell by a ``factor x factor`` block.

    With ``factor=2`` every cell of the input map becomes exactly one coarse
    cell, i.e. the input map is interpreted as the *coarse* grid (the usual
    convention in DARP-STC papers, where the map resolution equals twice the
    robot footprint). No coverage is lost by the 2x2 decomposition then.
    """
    occ = np.kron(grid.occ.astype(np.uint8), np.ones((factor, factor), dtype=np.uint8)).astype(bool)
    return GridMap(occ=occ, name=grid.name)


def downsample(grid: GridMap, factor: int) -> GridMap:
    """Reduce resolution: a block of ``factor x factor`` cells becomes one cell,
    which is an obstacle if *any* of the original cells is an obstacle
    (conservative, keeps the robot collision-free)."""
    if factor <= 1:
        return grid
    H, W = grid.shape
    h, w = H // factor, W // factor
    occ = grid.occ[: h * factor, : w * factor].reshape(h, factor, w, factor).any(axis=(1, 3))
    return GridMap(occ=occ, name=f"{grid.name}-ds{factor}")


def load_movingai(path: str | Path) -> GridMap:
    """Parse a MovingAI benchmark ``.map`` file (``type octile`` header)."""
    path = Path(path)
    lines = path.read_text().splitlines()
    height = width = None
    i = 0
    while i < len(lines):
        tok = lines[i].strip().split()
        i += 1
        if not tok:
            continue
        if tok[0] == "height":
            height = int(tok[1])
        elif tok[0] == "width":
            width = int(tok[1])
        elif tok[0] == "map":
            break
    if height is None or width is None:
        raise ValueError(f"{path}: missing height/width header")
    rows = lines[i : i + height]
    if len(rows) != height:
        raise ValueError(f"{path}: expected {height} map rows, got {len(rows)}")
    occ = np.ones((height, width), dtype=bool)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row[:width]):
            occ[r, c] = ch not in _PASSABLE
    return GridMap(occ=occ, name=path.stem)


def from_ascii(text: str, name: str = "ascii") -> GridMap:
    """Build a map from a multi-line string: ``.`` free, anything else blocked.
    Handy for unit tests."""
    rows = [ln for ln in text.strip("\n").splitlines()]
    width = max(len(r) for r in rows)
    occ = np.ones((len(rows), width), dtype=bool)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            occ[r, c] = ch != "."
    return GridMap(occ=occ, name=name)


# --------------------------------------------------------------------------
# Seeded generators (all operate on the coarse level, then upsample x2 so
# that obstacles are aligned with the 2x2 blocks).
# --------------------------------------------------------------------------
def gen_open(h: int, w: int, name: str | None = None) -> GridMap:
    """Empty rectangular field of ``h x w`` coarse cells."""
    return upsample(GridMap(np.zeros((h, w), dtype=bool), name or f"open-{h}x{w}"))


def gen_random(h: int, w: int, density: float, seed: int, name: str | None = None) -> GridMap:
    """Uniformly random obstacles with the given density (coarse level)."""
    rng = np.random.default_rng(seed)
    occ = rng.random((h, w)) < density
    return upsample(GridMap(occ, name or f"random-{h}x{w}-{int(density * 100)}-s{seed}"))


def gen_rooms(h: int, w: int, room: int, seed: int, door: int = 2, name: str | None = None) -> GridMap:
    """Office-like floor plan: a lattice of rooms of size ``room`` separated by
    one-cell walls; every wall segment gets a door of width ``door`` with
    probability 0.8 (the room graph is kept connected by a random spanning
    tree whose walls always get a door)."""
    rng = np.random.default_rng(seed)
    occ = np.zeros((h, w), dtype=bool)
    step = room + 1
    for r in range(room, h, step):
        occ[r, :] = True
    for c in range(room, w, step):
        occ[:, c] = True
    nr, nc = (h + 1) // step + 1, (w + 1) // step + 1
    # Random spanning tree over rooms guarantees connectivity.
    seen = {(0, 0)}
    frontier = [((0, 0), d) for d in ((0, 1), (1, 0))]
    tree_walls = set()
    while frontier:
        idx = int(rng.integers(len(frontier)))
        (a, d) = frontier.pop(idx)
        b = (a[0] + d[0], a[1] + d[1])
        if not (0 <= b[0] < nr and 0 <= b[1] < nc) or b in seen:
            continue
        if b[0] * step >= h or b[1] * step >= w:
            continue
        seen.add(b)
        tree_walls.add((a, b))
        frontier += [(b, dd) for dd in ((0, 1), (1, 0), (0, -1), (-1, 0))]

    def open_door(a: tuple[int, int], b: tuple[int, int]) -> None:
        (ra, ca), (rb, cb) = sorted([a, b])
        if ra == rb:  # vertical wall between horizontally adjacent rooms
            wc = ca * step + room
            r0 = ra * step + int(rng.integers(0, max(1, room - door + 1)))
            occ[r0 : r0 + door, wc] = False
        else:  # horizontal wall
            wr = ra * step + room
            c0 = ca * step + int(rng.integers(0, max(1, room - door + 1)))
            occ[wr, c0 : c0 + door] = False

    for a in seen:
        for d in ((0, 1), (1, 0)):
            b = (a[0] + d[0], a[1] + d[1])
            if b in seen and ((a, b) in tree_walls or (b, a) in tree_walls or rng.random() < 0.8):
                open_door(a, b)
    return upsample(GridMap(occ, name or f"rooms-{h}x{w}-r{room}-s{seed}"))


def gen_maze(h: int, w: int, corridor: int, seed: int, name: str | None = None) -> GridMap:
    """Perfect maze (randomised DFS) with corridors of width ``corridor``
    and one-cell walls, cropped to ``h x w`` coarse cells."""
    rng = np.random.default_rng(seed)
    step = corridor + 1
    mr, mc = max(1, (h - 1) // step), max(1, (w - 1) // step)
    occ = np.ones((mr * step + 1, mc * step + 1), dtype=bool)

    def carve(cell: tuple[int, int]) -> None:
        r, c = cell
        occ[1 + r * step : 1 + r * step + corridor, 1 + c * step : 1 + c * step + corridor] = False

    start = (0, 0)
    carve(start)
    stack, seen = [start], {start}
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= r + dr < mr and 0 <= c + dc < mc and (r + dr, c + dc) not in seen]
        if not nbrs:
            stack.pop()
            continue
        nr_, nc_ = nbrs[int(rng.integers(len(nbrs)))]
        carve((nr_, nc_))
        # Remove the wall between (r, c) and (nr_, nc_).
        if nr_ == r:
            wc = 1 + min(c, nc_) * step + corridor
            occ[1 + r * step : 1 + r * step + corridor, wc] = False
        else:
            wr = 1 + min(r, nr_) * step + corridor
            occ[wr, 1 + c * step : 1 + c * step + corridor] = False
        seen.add((nr_, nc_))
        stack.append((nr_, nc_))
    full = np.ones((h, w), dtype=bool)
    hh, ww = min(h, occ.shape[0]), min(w, occ.shape[1])
    full[:hh, :ww] = occ[:hh, :ww]
    return upsample(GridMap(full, name or f"maze-{h}x{w}-c{corridor}-s{seed}"))


def sample_starts(cmap: CoarseMap, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw ``k`` distinct random start cells (coarse coordinates)."""
    cells = cmap.cells()
    if k > len(cells):
        raise ValueError(f"cannot place {k} robots on {len(cells)} cells")
    idx = rng.choice(len(cells), size=k, replace=False)
    return [cells[int(i)] for i in idx]
