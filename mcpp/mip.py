"""Exact MIP models for small instances (optimality-gap experiments).

Two problems on the coarse grid graph G=(V,E) with roots r_1..r_k:

* ``cover=False`` - min-max *rooted connected partition*: the optimum of any
  "partition + STC" method (DARP, Voronoi, k-means, local search), because
  the STC makespan equals 4 * max region size.
* ``cover=True``  - min-max *rooted tree cover* with unit vertex cost
  (overlaps allowed): a relaxation of the partition problem, i.e. a valid
  lower bound for the MMRTC branch with STC tours.

Single-commodity flow formulation, one commodity per robot::

    min  z
    s.t. sum_i x[v,i]  = 1   (partition)  or  >= 1  (cover)    for all v
         x[r_i, i]     = 1                                     for all i
         sum_{(u,v)} f[uv,i] - sum_{(v,u)} f[vu,i] = x[v,i]      for all i, v != r_i
         f[uv,i] <= (n-1) x[u,i],   f[uv,i] <= (n-1) x[v,i]      for all arcs, i
         sum_v x[v,i] <= z                                     for all i
         x binary, f >= 0

Flow can only originate at r_i and only travel through vertices owned by
robot i, so every owned vertex is connected to r_i inside region i.
Solved with HiGHS through ``scipy.optimize.milp`` (only the solver is
third-party; the model is written here).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .graph import Cell
from .grid import CoarseMap


@dataclass
class MIPResult:
    status: str
    objective: float | None   # best max region size found (coarse cells)
    dual_bound: float | None  # proven lower bound on the optimum
    assign: np.ndarray | None # (k x n) 0/1 matrix as a dict-friendly array
    cells: list[Cell]
    runtime: float

    @property
    def optimal(self) -> bool:
        return self.status == "optimal"


def solve_minmax(cmap: CoarseMap, starts: list[Cell], cover: bool = False,
                 time_limit: float = 60.0) -> MIPResult:
    t0 = time.perf_counter()
    cells = cmap.cells()
    idx = {c: j for j, c in enumerate(cells)}
    n, k = len(cells), len(starts)
    arcs = []
    for (r, c) in cells:
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in idx:
                arcs.append((idx[(r, c)], idx[nb]))
    A = len(arcs)
    nx, nf = n * k, A * k
    nvar = nx + nf + 1

    def X(v: int, i: int) -> int:
        return v * k + i

    def F(a: int, i: int) -> int:
        return nx + a * k + i

    Z = nvar - 1
    rows, cols, vals, lo, hi = [], [], [], [], []
    row = 0

    def add(entries: list[tuple[int, float]], lb: float, ub: float) -> None:
        nonlocal row
        for col, val in entries:
            rows.append(row)
            cols.append(col)
            vals.append(val)
        lo.append(lb)
        hi.append(ub)
        row += 1

    # (1) every vertex owned once (partition) / at least once (cover)
    for v in range(n):
        add([(X(v, i), 1.0) for i in range(k)], 1.0, np.inf if cover else 1.0)
    # (2) flow conservation
    in_arcs = [[] for _ in range(n)]
    out_arcs = [[] for _ in range(n)]
    for a, (u, v) in enumerate(arcs):
        out_arcs[u].append(a)
        in_arcs[v].append(a)
    root_idx = [idx[s] for s in starts]
    for i in range(k):
        for v in range(n):
            if v == root_idx[i]:
                continue
            ent = [(F(a, i), 1.0) for a in in_arcs[v]] + [(F(a, i), -1.0) for a in out_arcs[v]]
            ent.append((X(v, i), -1.0))
            add(ent, 0.0, 0.0)
    # (3) capacities: flow only through owned vertices
    cap = float(n - 1)
    for i in range(k):
        for a, (u, v) in enumerate(arcs):
            add([(F(a, i), 1.0), (X(u, i), -cap)], -np.inf, 0.0)
            add([(F(a, i), 1.0), (X(v, i), -cap)], -np.inf, 0.0)
    # (4) z >= region size
    for i in range(k):
        add([(X(v, i), 1.0) for v in range(n)] + [(Z, -1.0)], -np.inf, 0.0)

    Amat = coo_matrix((vals, (rows, cols)), shape=(row, nvar)).tocsr()
    c = np.zeros(nvar)
    c[Z] = 1.0
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[:nx] = 1.0
    for i, rv in enumerate(root_idx):
        lb[X(rv, i)] = 1.0
        if not cover:
            for j in range(k):
                if j != i:
                    ub[X(rv, j)] = 0.0
    integrality = np.zeros(nvar)
    integrality[:nx] = 1
    res = milp(c, constraints=LinearConstraint(Amat, lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "disp": False})
    status = {0: "optimal", 1: "time_limit"}.get(res.status, f"status{res.status}")
    obj = float(res.fun) if res.x is not None else None
    dual = getattr(res, "mip_dual_bound", None)
    if status == "optimal" and obj is not None:
        dual = obj
    xm = None
    if res.x is not None:
        xm = np.round(res.x[:nx]).reshape(n, k).T.astype(int)
    return MIPResult(status, obj, None if dual is None else float(dual), xm, cells,
                     time.perf_counter() - t0)


def mip_sets(res: MIPResult) -> list[set[Cell]]:
    """Vertex sets per robot from a MIP solution."""
    assert res.assign is not None
    return [{res.cells[v] for v in np.nonzero(res.assign[i])[0]} for i in range(res.assign.shape[0])]
