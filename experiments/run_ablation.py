"""Design ablations and parameter sensitivity, on the *tuning* instance set
(generated maps with their own seeds - disjoint from the MovingAI report set).

A. DARP parameters / design choices: learning rate c, adaptive step size,
   distance (Euclidean vs geodesic), connectivity term on/off, its amplitude
   nu, random perturbation level.             -> darp_ablation.csv
B. Spanning-tree strategy of STC (turns, turn-weighted completion time) on
   the DARP+LS partitions.                     -> tree_ablation.csv
C. Heterogeneous team (speeds 1,1,2,2): equal shares vs speed-proportional
   DARP shares vs + speed-aware local search.  -> hetero.csv

Every randomised configuration runs 30 times per instance.
Usage:  python experiments/run_ablation.py [--workers 14] [--seeds 30]
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
from common import RESULTS, coarse_from_spec, environment_info, map_name, save_json, scenario_starts, write_csv

from mcpp.darp import DARPConfig, darp
from mcpp.local_search import improve_partition
from mcpp.metrics import evaluate
from mcpp.partition import region_sizes
from mcpp.planner import _sets_from_assign
from mcpp.stc import TREE_STRATEGIES, count_turns, plan_region

TUNING_MAPS = [
    {"gen": "open", "h": 24, "w": 24, "name": "T-open-24"},
    {"gen": "rooms", "h": 32, "w": 32, "room": 5, "seed": 101, "name": "T-rooms-32"},
    {"gen": "maze", "h": 32, "w": 32, "corridor": 3, "seed": 102, "name": "T-maze-32"},
    {"gen": "random", "h": 32, "w": 32, "density": 0.15, "seed": 103, "name": "T-random-32"},
]
SCEN_SEED = 7  # tuning scenarios use a different base seed than the report set

DARP_VARIANTS = {
    "default (c=0.02)": {},
    "c=0.005": {"learning_rate": 0.005},
    "c=0.01": {"learning_rate": 0.01},
    "c=0.05": {"learning_rate": 0.05},
    "c=0.1": {"learning_rate": 0.1},
    "adaptive c (0.05, p=100)": {"learning_rate": 0.05, "adaptive_lr": True, "patience": 100},
    "geodesic distance": {"distance": "geodesic"},
    "no connectivity term": {"connectivity": False},
    "nu=0.1": {"cc_variation": 0.1},
    "no random noise": {"random_level": 0.0},
    "noise=1e-2": {"random_level": 1e-2},
}


def darp_job(job):
    spec, k, scen, variant, seeds = job
    cm = coarse_from_spec(spec)
    st = scenario_starts(cm, k, scen, SCEN_SEED, map_name(spec))
    cfg = DARPConfig(max_iter=2000, **DARP_VARIANTS[variant])
    rows = []
    for seed in seeds:
        r = darp(cm, st, cfg, np.random.default_rng(seed))
        sizes = region_sizes(r.assign, k)
        rows.append(dict(map=map_name(spec), k=k, scenario=scen, variant=variant, seed=seed,
                         success=float(r.success), iterations=r.iterations, runtime=r.runtime,
                         imbalance=float(sizes.max() / (cm.n_cells / k)),
                         makespan=4.0 * sizes.max(), lower_bound=4.0 * np.ceil(cm.n_cells / k)))
    return rows


def tree_job(job):
    spec, k, scen = job
    cm = coarse_from_spec(spec)
    st = scenario_starts(cm, k, scen, SCEN_SEED, map_name(spec))
    r = darp(cm, st, DARPConfig(max_iter=2000), np.random.default_rng(0))
    assign = improve_partition(r.assign, cm, st).assign
    sets = _sets_from_assign(assign, k)
    rows = []
    for strat in TREE_STRATEGIES:
        for seed in (range(30) if strat == "random" else [0]):
            rng = np.random.default_rng(seed)
            paths = [plan_region(sets[i], st[i], strat, rng)[0] for i in range(k)]
            m = evaluate(cm.fine, paths, conflicts=False)
            rows.append(dict(map=map_name(spec), k=k, scenario=scen, strategy=strat, seed=seed,
                             turns=m["turns"], max_turns=m["max_turns"], makespan=m["makespan"],
                             time_makespan=m["time_makespan"]))
    return rows


def hetero_job(job):
    spec, scen, seeds = job
    speeds = [1.0, 1.0, 2.0, 2.0]
    k = len(speeds)
    cm = coarse_from_spec(spec)
    st = scenario_starts(cm, k, scen, SCEN_SEED, map_name(spec))
    portions = list(np.array(speeds) / sum(speeds))
    rows = []
    for seed in seeds:
        eq = darp(cm, st, DARPConfig(max_iter=2000), np.random.default_rng(seed))
        pr = darp(cm, st, DARPConfig(max_iter=2000), np.random.default_rng(seed), portions=portions)
        ls = improve_partition(pr.assign, cm, st, speeds=speeds)
        for name, assign in (("DARP equal shares", eq.assign), ("DARP speed shares", pr.assign),
                             ("DARP speed shares + LS", ls.assign)):
            sets = _sets_from_assign(assign, k)
            paths = [plan_region(sets[i], st[i], "adaptive")[0] for i in range(k)]
            m = evaluate(cm.fine, paths, speeds=speeds, conflicts=False)
            L = np.array([len(p) - 1 for p in paths]) / np.array(speeds)
            rows.append(dict(map=map_name(spec), scenario=scen, seed=seed, variant=name,
                             time_makespan=m["time_makespan"], move_time_makespan=float(L.max()),
                             lower_bound=4.0 * cm.n_cells / sum(speeds)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenarios", type=int, default=3)
    args = ap.parse_args()
    out = RESULTS / "ablation"
    t0 = time.time()
    seeds = list(range(args.seeds))
    chunks = [seeds[i:i + 10] for i in range(0, len(seeds), 10)]
    darp_jobs = [(spec, k, sc, v, ch) for spec in TUNING_MAPS for k in (4, 8)
                 for sc in range(args.scenarios) for v in DARP_VARIANTS for ch in chunks]
    tree_jobs = [(spec, k, sc) for spec in TUNING_MAPS for k in (2, 4, 8) for sc in range(5)]
    het_jobs = [(spec, sc, ch) for spec in TUNING_MAPS for sc in range(5) for ch in chunks]
    with Pool(args.workers) as pool:
        print(f"A: {len(darp_jobs)} DARP jobs", flush=True)
        rows = [r for rr in pool.imap_unordered(darp_job, darp_jobs) for r in rr]
        write_csv(sorted(rows, key=lambda r: (r["variant"], r["map"], r["k"], r["scenario"], r["seed"])),
                  out / "darp_ablation.csv")
        print(f"B: tree strategies ({time.time() - t0:.0f}s)", flush=True)
        rows = [r for rr in pool.imap_unordered(tree_job, tree_jobs) for r in rr]
        write_csv(rows, out / "tree_ablation.csv")
        print(f"C: heterogeneous team ({time.time() - t0:.0f}s)", flush=True)
        rows = [r for rr in pool.imap_unordered(hetero_job, het_jobs) for r in rr]
        write_csv(rows, out / "hetero.csv")
    save_json({"env": environment_info(), "wall_time_s": time.time() - t0,
               "darp_variants": DARP_VARIANTS, "tuning_maps": TUNING_MAPS}, out / "env.json")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
