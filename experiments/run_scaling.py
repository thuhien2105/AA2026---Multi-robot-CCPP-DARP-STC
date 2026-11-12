"""Empirical complexity: runtime vs problem size (log-log), profiling and
peak memory. Runs single-process for clean timings.

For every size n = s x s coarse cells (open field and 10 % random obstacles),
k = 4 robots and ``reps`` start scenarios we time
  * geodesic Voronoi              theory O(n)
  * STC tours (given partition)   theory O(n log n) (Kruskal sort) / O(n)
  * DARP: total and per iteration theory O(k n) per iteration
  * local search (from Voronoi)   theory O(#chains * n)
  * MMRTC (Even et al.)           theory O(k n log n)
and fit the slope of log(time) vs log(n).

Usage:  python experiments/run_scaling.py [--sizes 8 16 ... 128] [--reps 5]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
import tracemalloc

import numpy as np
from common import RESULTS, environment_info, save_json, write_csv

from mcpp.darp import DARPConfig, darp
from mcpp.grid import build_coarse, gen_open, gen_random, sample_starts
from mcpp.local_search import improve_partition
from mcpp.mmrtc import mmrtc_even
from mcpp.partition import voronoi_partition
from mcpp.planner import _sets_from_assign
from mcpp.stc import plan_region

K = 4


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def run_size(kind: str, s: int, rep: int) -> dict:
    g = gen_open(s, s) if kind == "open" else gen_random(s, s, 0.10, seed=500 + s)
    cm = build_coarse(g)
    st = sample_starts(cm, K, np.random.default_rng([77, s, rep]))
    vor, t_vor = timed(voronoi_partition, cm, st)
    sets = _sets_from_assign(vor, K)
    _, t_stc = timed(lambda: [plan_region(sets[i], st[i], "adaptive") for i in range(K)])
    dres, t_darp = timed(darp, cm, st, DARPConfig(max_iter=2000), np.random.default_rng(rep))
    ls, t_ls = timed(improve_partition, vor, cm, st)
    _, t_mm = timed(mmrtc_even, cm, st, False)
    return dict(kind=kind, side=s, n=cm.n_cells, rep=rep, t_voronoi=t_vor, t_stc=t_stc,
                t_darp=t_darp, darp_iterations=dres.iterations, darp_success=float(dres.success),
                t_darp_per_iter=t_darp / max(1, dres.iterations), t_ls=t_ls, ls_chains=ls.chains,
                t_ls_per_chain=t_ls / max(1, ls.chains), t_mmrtc=t_mm)


def fit_slope(n: np.ndarray, t: np.ndarray) -> float:
    ok = t > 0
    return float(np.polyfit(np.log(n[ok]), np.log(t[ok]), 1)[0])


def peak_memory(s: int) -> dict:
    """Peak Python heap (tracemalloc) of each component on an s x s map."""
    cm = build_coarse(gen_random(s, s, 0.10, seed=500 + s))
    st = sample_starts(cm, K, np.random.default_rng(1))
    vor = voronoi_partition(cm, st)
    sets = _sets_from_assign(vor, K)
    jobs = {
        "voronoi": lambda: voronoi_partition(cm, st),
        "stc": lambda: [plan_region(sets[i], st[i], "adaptive") for i in range(K)],
        "darp": lambda: darp(cm, st, DARPConfig(max_iter=300)),
        "local_search": lambda: improve_partition(vor, cm, st),
        "mmrtc": lambda: mmrtc_even(cm, st, False),
    }
    out = {"side": s, "n": cm.n_cells}
    for name, fn in jobs.items():
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out[f"peak_mb_{name}"] = peak / 2**20
    return out


def profile_darp_pipeline(s: int) -> str:
    cm = build_coarse(gen_random(s, s, 0.10, seed=500 + s))
    st = sample_starts(cm, 8, np.random.default_rng(3))
    pr = cProfile.Profile()
    pr.enable()
    res = darp(cm, st, DARPConfig(max_iter=2000), np.random.default_rng(0))
    ls = improve_partition(res.assign, cm, st)
    sets = _sets_from_assign(ls.assign, 8)
    for i in range(8):
        plan_region(sets[i], st[i], "adaptive")
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(18)
    return f"DARP+LS+STC, random map {s}x{s} coarse ({cm.n_cells} cells), k=8, " \
           f"DARP iterations={res.iterations}, success={res.success}\n\n" + buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48, 64, 96, 128])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    out = RESULTS / "scaling"
    rows = []
    for kind in ("open", "random"):
        for s in args.sizes:
            for rep in range(args.reps):
                rows.append(run_size(kind, s, rep))
            print(f"{kind} {s}x{s}: done", flush=True)
    write_csv(rows, out / "scaling.csv")
    slopes = {}
    for kind in ("open", "random"):
        rr = [r for r in rows if r["kind"] == kind]
        n = np.array([r["n"] for r in rr], dtype=float)
        for col in ("t_voronoi", "t_stc", "t_darp", "t_darp_per_iter", "t_ls", "t_ls_per_chain", "t_mmrtc"):
            slopes[f"{kind}:{col}"] = fit_slope(n, np.array([r[col] for r in rr]))
    mem = [peak_memory(s) for s in (32, 64, 128)]
    write_csv(mem, out / "memory.csv")
    (out / "profile.txt").write_text(profile_darp_pipeline(64))
    save_json({"slopes": slopes, "env": environment_info()}, out / "scaling_summary.json")
    print("slopes:", {k: round(v, 2) for k, v in slopes.items()})


if __name__ == "__main__":
    main()
