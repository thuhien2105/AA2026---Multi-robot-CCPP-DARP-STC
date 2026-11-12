"""Optimality gap on small instances against exact MIP optima.

For every instance we solve
  * MIP-P: min-max rooted connected partition  -> optimum of every
           "partition + STC" method (makespan = 4 * max region);
  * MIP-C: min-max rooted tree cover (overlap allowed) -> optimum of the
           "tree cover + STC" family (a relaxation of MIP-P);
and report  gap = makespan / (4 * OPT) - 1  for each heuristic. If a MIP hits
the time limit, its proven dual bound is used instead (the gap is then an
upper bound on the true gap) and the row is flagged.

Usage:  python experiments/run_optgap.py [--time-limit 120] [--workers 14]
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
from common import RESULTS, coarse_from_spec, environment_info, map_name, save_json, scenario_starts, write_csv

from mcpp.darp import DARPConfig
from mcpp.mip import solve_minmax
from mcpp.planner import RANDOMISED, plan, plan_darp_pair

SMALL_MAPS = [
    {"gen": "open", "h": 6, "w": 6, "name": "S-open-6x6"},
    {"gen": "open", "h": 7, "w": 8, "name": "S-open-7x8"},
    {"gen": "random", "h": 8, "w": 8, "density": 0.15, "seed": 11, "name": "S-random-8x8-a"},
    {"gen": "random", "h": 8, "w": 8, "density": 0.15, "seed": 12, "name": "S-random-8x8-b"},
    {"gen": "rooms", "h": 8, "w": 8, "room": 3, "seed": 13, "name": "S-rooms-8x8"},
    {"gen": "maze", "h": 9, "w": 9, "corridor": 2, "seed": 14, "name": "S-maze-9x9"},
]
METHODS = ["voronoi", "kmeans", "darp", "darp_ls", "voronoi_ls", "mmrtc", "mmrtc_dedup"]
SCEN_SEED = 99


def job(args):
    spec, k, scen, tl, n_seeds = args
    cm = coarse_from_spec(spec)
    name = map_name(spec)
    st = scenario_starts(cm, k, scen, SCEN_SEED, name)
    mp = solve_minmax(cm, st, cover=False, time_limit=tl)
    mc = solve_minmax(cm, st, cover=True, time_limit=tl)
    base = dict(map=name, n_cells=cm.n_cells, k=k, scenario=scen,
                mip_part=mp.objective, mip_part_bound=mp.dual_bound, mip_part_optimal=float(mp.optimal),
                mip_part_time=mp.runtime, mip_cover=mc.objective, mip_cover_bound=mc.dual_bound,
                mip_cover_optimal=float(mc.optimal), mip_cover_time=mc.runtime,
                trivial_lb=float(np.ceil(cm.n_cells / k)))
    rows = []
    cfg = DARPConfig(max_iter=2000)
    for seed in range(n_seeds):
        res = {}
        d, dl = plan_darp_pair(cm, st, seed, cfg, conflicts=False)
        res["darp"], res["darp_ls"] = d.metrics, dl.metrics
        for m in METHODS:
            if m in res or (m not in RANDOMISED and seed > 0):
                continue
            res[m] = plan(m, cm, st, seed, cfg, conflicts=False).metrics
        for m, met in res.items():
            if m not in RANDOMISED and seed > 0:
                continue
            rows.append({**base, "method": m, "seed": seed, "makespan": met["makespan"],
                         "max_region": met["max_region"], "overlap": met["overlap"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    jobs = [(spec, k, sc, args.time_limit, args.seeds) for spec in SMALL_MAPS for k in (2, 3, 4)
            for sc in range(args.scenarios)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = [r for rr in pool.imap_unordered(job, jobs) for r in rr]
    rows.sort(key=lambda r: (r["map"], r["k"], r["scenario"], r["method"], r["seed"]))
    out = RESULTS / "optgap"
    write_csv(rows, out / "optgap.csv")
    save_json({"env": environment_info(), "time_limit": args.time_limit, "maps": SMALL_MAPS,
               "wall_time_s": time.time() - t0}, out / "env.json")
    print(f"{len(jobs)} instances, {len(rows)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
