"""End-to-end multi-robot CCPP pipeline: partition / tree cover -> STC tours
-> metrics. This is the single entry point used by experiments and demos.

Methods
-------
voronoi       geodesic Voronoi partition + STC          (baseline, deterministic)
kmeans        k-means (seeded at robots) + repair + STC (baseline, deterministic)
darp          DARP partition + STC                      (main method, randomised)
darp_ls       DARP + min-max local search + STC         (extension)
voronoi_ls    Voronoi + min-max local search + STC      (extension / ablation)
mmrtc         Even et al. 4-approx tree cover + STC     (min-max tree cover branch)
mmrtc_dedup   ... + overlap deduplication
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .darp import DARPConfig, darp
from .graph import Cell
from .grid import CoarseMap
from .local_search import improve_partition
from .metrics import evaluate
from .mmrtc import mmrtc_even
from .partition import kmeans_partition, region_sizes, validate_partition, voronoi_partition
from .stc import plan_region

METHODS = ("voronoi", "kmeans", "darp", "darp_ls", "voronoi_ls", "mmrtc", "mmrtc_dedup")
RANDOMISED = {"darp", "darp_ls"}


@dataclass
class PlanResult:
    method: str
    sets: list[set[Cell]]                 # coarse cells covered by each robot
    paths: list[list[Cell]]               # closed fine-grid tours
    assign: np.ndarray | None             # partition (None for tree covers)
    metrics: dict[str, float] = field(default_factory=dict)


def _sets_from_assign(assign: np.ndarray, k: int) -> list[set[Cell]]:
    sets: list[set[Cell]] = [set() for _ in range(k)]
    rr, cc = np.nonzero(assign >= 0)
    for r, c in zip(rr.tolist(), cc.tolist()):
        sets[int(assign[r, c])].add((r, c))
    return sets


def plan(method: str, cmap: CoarseMap, starts: list[Cell], seed: int = 0,
         darp_cfg: DARPConfig | None = None, tree: str = "adaptive",
         speeds: list[float] | None = None, turn_cost: float = 0.5,
         conflicts: bool = True, validate: bool = True) -> PlanResult:
    rng = np.random.default_rng(seed)
    k = len(starts)
    extra: dict[str, float] = {}
    portions = None
    if speeds is not None:
        portions = list(np.asarray(speeds, dtype=float) / np.sum(speeds))

    t0 = time.perf_counter()
    assign = None
    if method in ("voronoi", "voronoi_ls"):
        assign = voronoi_partition(cmap, starts)
    elif method == "kmeans":
        assign = kmeans_partition(cmap, starts, rng)
    elif method in ("darp", "darp_ls"):
        res = darp(cmap, starts, darp_cfg, rng, portions=portions)
        assign = res.assign
        extra.update(darp_success=float(res.success), darp_iterations=float(res.iterations))
    elif method in ("mmrtc", "mmrtc_dedup"):
        tc = mmrtc_even(cmap, starts, dedup=(method == "mmrtc_dedup"))
        sets = tc.sets
        extra.update(mmrtc_B=float(tc.bound_B), mmrtc_tests=float(tc.feasibility_tests))
    else:
        raise ValueError(f"unknown method {method!r}")
    if method.endswith("_ls"):
        ls = improve_partition(assign, cmap, starts, speeds=speeds)
        assign = ls.assign
        extra.update(ls_moves=float(ls.moves), ls_chains=float(ls.chains))
    t_part = time.perf_counter() - t0

    if assign is not None:
        if validate:
            validate_partition(assign, cmap, starts)
        sets = _sets_from_assign(assign, k)

    t1 = time.perf_counter()
    paths = [plan_region(sets[i], starts[i], tree, rng)[0] for i in range(k)]
    t_path = time.perf_counter() - t1

    m = evaluate(cmap.fine, paths, speeds=speeds, turn_cost=turn_cost, conflicts=conflicts)
    m.update(extra)
    m["runtime_partition"] = t_part
    m["runtime_paths"] = t_path
    m["runtime"] = t_part + t_path
    sizes = np.array([len(s) for s in sets])
    m["max_region"] = float(sizes.max())
    m["lower_bound"] = float(4 * np.ceil(cmap.n_cells / k))  # makespan LB for STC tours
    return PlanResult(method, sets, paths, assign, m)


def plan_darp_pair(cmap: CoarseMap, starts: list[Cell], seed: int = 0,
                   darp_cfg: DARPConfig | None = None, tree: str = "adaptive",
                   speeds: list[float] | None = None, turn_cost: float = 0.5,
                   conflicts: bool = True) -> tuple[PlanResult, PlanResult]:
    """Run DARP once and return both ``darp`` and ``darp_ls`` results.

    Saves one DARP run per experiment; the reported ``darp_ls`` runtime is
    DARP time + local-search time, exactly as if run separately.
    """
    rng = np.random.default_rng(seed)
    k = len(starts)
    portions = None if speeds is None else list(np.asarray(speeds, float) / np.sum(speeds))
    t0 = time.perf_counter()
    res = darp(cmap, starts, darp_cfg, rng, portions=portions)
    t_darp = time.perf_counter() - t0
    t1 = time.perf_counter()
    ls = improve_partition(res.assign, cmap, starts, speeds=speeds)
    t_ls = time.perf_counter() - t1
    out = []
    for name, assign, t_part in (("darp", res.assign, t_darp), ("darp_ls", ls.assign, t_darp + t_ls)):
        validate_partition(assign, cmap, starts)
        sets = _sets_from_assign(assign, k)
        t2 = time.perf_counter()
        paths = [plan_region(sets[i], starts[i], tree, np.random.default_rng(seed))[0] for i in range(k)]
        t_path = time.perf_counter() - t2
        m = evaluate(cmap.fine, paths, speeds=speeds, turn_cost=turn_cost, conflicts=conflicts)
        m.update(darp_success=float(res.success), darp_iterations=float(res.iterations),
                 runtime_partition=t_part, runtime_paths=t_path, runtime=t_part + t_path,
                 max_region=float(max(len(s) for s in sets)),
                 lower_bound=float(4 * np.ceil(cmap.n_cells / k)))
        if name == "darp_ls":
            m.update(ls_moves=float(ls.moves), ls_chains=float(ls.chains))
        out.append(PlanResult(name, sets, paths, assign, m))
    return out[0], out[1]


__all__ = ["plan", "plan_darp_pair", "PlanResult", "METHODS", "RANDOMISED", "region_sizes"]
