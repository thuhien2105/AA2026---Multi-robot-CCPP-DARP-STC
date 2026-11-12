"""Baseline area-partitioning methods and partition utilities.

A *partition* is an integer array ``assign`` with the shape of the coarse grid:
``assign[v] = i`` means coarse cell ``v`` is covered by robot ``i`` and
``assign[v] = -1`` marks unusable cells. A partition is *valid* iff
  (1) every usable cell has exactly one owner,
  (2) robot ``i`` owns its start cell, and
  (3) every region is 4-connected (required by STC).
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .graph import Cell, N4, multi_source_labels
from .grid import FOUR_CONN, CoarseMap


def region_sizes(assign: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(assign[assign >= 0].ravel(), minlength=k)[:k]


def validate_partition(assign: np.ndarray, cmap: CoarseMap, starts: list[Cell]) -> None:
    """Raise ``AssertionError`` if ``assign`` violates (1)-(3)."""
    k = len(starts)
    assert np.all((assign >= 0) == cmap.free), "usable cells must be assigned exactly"
    assert assign.max() < k
    for i, s in enumerate(starts):
        assert assign[s] == i, f"robot {i} does not own its start cell"
        _, n = ndimage.label(assign == i, structure=FOUR_CONN)
        assert n == 1, f"region of robot {i} has {n} components"


# --------------------------------------------------------------------------
# Baseline 1: geodesic Voronoi (multi-source BFS)
# --------------------------------------------------------------------------
def voronoi_partition(cmap: CoarseMap, starts: list[Cell]) -> np.ndarray:
    """Each cell goes to the robot with the smallest BFS distance (ties: lower
    index). Always valid; ignores load balance. O(|V|)."""
    labels, _ = multi_source_labels(cmap.free, starts)
    return labels


# --------------------------------------------------------------------------
# Baseline 2: k-means clustering of cell coordinates + repair
# --------------------------------------------------------------------------
def kmeans_partition(cmap: CoarseMap, starts: list[Cell], rng: np.random.Generator | None = None,
                     n_iter: int = 100, init: str = "starts") -> np.ndarray:
    """Lloyd's k-means on the (Euclidean) coordinates of usable cells.

    ``init="starts"`` (default): centroid ``i`` is initialised at robot ``i``'s
    start, so cluster ``i`` naturally belongs to robot ``i`` (deterministic).
    ``init="++"``: k-means++ seeding (randomised); clusters are then matched
    to robots with the Hungarian algorithm (cost = squared start-centroid
    distance). This variant often puts a robot's start inside another
    cluster, which the repair turns into a near-empty region - it is kept only
    as a reference.

    In both cases start cells are forced into their robot's region and
    :func:`repair_connectivity` makes the regions connected (k-means ignores
    obstacles, so clusters can be split by walls). O(n_iter * k * |V|).
    """
    k = len(starts)
    cells = np.argwhere(cmap.free).astype(float)
    n = len(cells)
    if init == "starts":
        centers = np.array(starts, dtype=float)
    elif init == "++":
        rng = rng or np.random.default_rng(0)
        centers = np.empty((k, 2))
        centers[0] = cells[rng.integers(n)]
        d2 = ((cells - centers[0]) ** 2).sum(1)
        for j in range(1, k):
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
            centers[j] = cells[rng.choice(n, p=p)]
            d2 = np.minimum(d2, ((cells - centers[j]) ** 2).sum(1))
    else:
        raise ValueError(f"unknown init {init!r}")
    lab = np.zeros(n, dtype=np.int64)
    for it in range(n_iter):
        dist = ((cells[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(1)
        if it > 0 and np.array_equal(new, lab):
            break
        lab = new
        for j in range(k):
            pts = cells[lab == j]
            if len(pts):
                centers[j] = pts.mean(0)
    if init == "starts":
        cluster_to_robot = np.arange(k)
    else:
        st = np.array(starts, dtype=float)
        cost = ((st[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        cluster_to_robot = np.empty(k, dtype=np.int64)
        cluster_to_robot[cols] = rows
    assign = np.full(cmap.shape, -1, dtype=np.int64)
    ci = cells.astype(int)
    assign[ci[:, 0], ci[:, 1]] = cluster_to_robot[lab]
    for i, s in enumerate(starts):
        assign[s] = i
    return repair_connectivity(assign, cmap, starts)


# --------------------------------------------------------------------------
# Connectivity repair (used by k-means and as a safety net for DARP)
# --------------------------------------------------------------------------
def repair_connectivity(assign: np.ndarray, cmap: CoarseMap, starts: list[Cell]) -> np.ndarray:
    """Make every region connected.

    For each robot keep only the component that contains its start cell; all
    other ("orphan") cells are released and re-grown by a multi-source BFS
    from the kept regions (an orphan joins the region that reaches it first).
    The result is valid: kept components are connected and each grown cell is
    attached to an already connected region through its BFS parent.
    Requires the usable cells to be connected (guaranteed by
    :func:`mcpp.grid.build_coarse`). O(k |V|).
    """
    k = len(starts)
    out = np.full(cmap.shape, -1, dtype=np.int64)
    for i, s in enumerate(starts):
        lab, _ = ndimage.label(assign == i, structure=FOUR_CONN)
        if lab[s] > 0:
            out[lab == lab[s]] = i
        out[s] = i
    h, w = cmap.shape
    q: deque[Cell] = deque()
    rr, cc = np.nonzero(out >= 0)
    for r, c in zip(rr.tolist(), cc.tolist()):
        q.append((r, c))
    free = cmap.free
    while q:
        r, c = q.popleft()
        lab_ = out[r, c]
        for dr, dc in N4:
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w and free[a, b] and out[a, b] < 0:
                out[a, b] = lab_
                q.append((a, b))
    assert k == 0 or np.all((out >= 0) == free)
    return out
