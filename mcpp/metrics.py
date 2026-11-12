"""Standard multi-robot CCPP metrics (Section 3.1 of the assignment).

All paths are lists of fine cells; one move = one time step = unit length.

* makespan          max_i L_i (moves of the longest closed tour)
* total_length      sum_i L_i
* balance           min_i L_i / max_i L_i      (1 = perfect)
* cv                std(L) / mean(L)           (coefficient of variation)
* coverage          |union of visited free cells| / |free fine cells|
                    (includes cells lost by the 2x2 STC discretisation)
* overlap           (sum_i |visited_i| - |union|) / |union| - redundant coverage
* turns / max_turns total and worst-robot number of heading changes
* time_makespan     max_i (L_i + tau * turns_i) / v_i - completion time when a
                    90-degree turn costs ``tau`` extra time units
* conflicts         vertex + swap conflicts when all robots execute their tours
                    synchronously (a finished robot waits at its last cell)
"""
from __future__ import annotations

import numpy as np

from .graph import Cell
from .grid import GridMap
from .stc import count_turns


def count_conflicts(paths: list[list[Cell]]) -> tuple[int, int]:
    """(vertex_conflicts, swap_conflicts) under synchronous execution.
    O(T k) with hashing, T = makespan."""
    if not paths:
        return 0, 0
    T = max(len(p) for p in paths)

    def pos(p: list[Cell], t: int) -> Cell:
        return p[t] if t < len(p) else p[-1]

    vertex = swap = 0
    for t in range(T):
        occupied: dict[Cell, int] = {}
        for p in paths:
            c = pos(p, t)
            occupied[c] = occupied.get(c, 0) + 1
        vertex += sum(n * (n - 1) // 2 for n in occupied.values() if n > 1)
        if t + 1 < T:
            moves = {}
            for i, p in enumerate(paths):
                a, b = pos(p, t), pos(p, t + 1)
                if a != b:
                    moves[(a, b)] = i
            swap += sum(1 for (a, b) in moves if (b, a) in moves) // 2
    return vertex, swap


def evaluate(grid: GridMap, paths: list[list[Cell]], speeds: list[float] | None = None,
             turn_cost: float = 0.5, conflicts: bool = True) -> dict[str, float]:
    k = len(paths)
    v = np.ones(k) if speeds is None else np.asarray(speeds, dtype=float)
    L = np.array([max(0, len(p) - 1) for p in paths], dtype=float)
    turns = np.array([count_turns(p, closed=True) for p in paths], dtype=float)
    free = grid.free
    visited_sets = [set(p) for p in paths]
    union: set[Cell] = set().union(*visited_sets) if visited_sets else set()
    union_free = {c for c in union if free[c]}
    n_free = grid.n_free
    sum_visits = sum(len(s) for s in visited_sets)
    out = {
        "makespan": float(L.max()),
        "time_makespan": float(((L + turn_cost * turns) / v).max()),
        "total_length": float(L.sum()),
        "balance": float(L.min() / L.max()) if L.max() > 0 else 1.0,
        "cv": float(L.std() / L.mean()) if L.mean() > 0 else 0.0,
        "coverage": len(union_free) / n_free if n_free else 1.0,
        "overlap": (sum_visits - len(union)) / len(union) if union else 0.0,
        "turns": float(turns.sum()),
        "max_turns": float(turns.max()),
        "invalid_cells": float(len(union) - len(union_free)),  # must be 0
    }
    if conflicts:
        vc, sc = count_conflicts(paths)
        out["vertex_conflicts"] = float(vc)
        out["swap_conflicts"] = float(sc)
    return out
