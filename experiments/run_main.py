"""Main benchmark: all methods x maps x team sizes x start scenarios.

Deterministic methods run once per instance; randomised methods (k-means,
DARP, DARP+LS) run ``seeds`` times per instance (>= 30 as required).
Output: ``results/<name>/runs.csv`` (one row per run) and ``env.json``.

Usage:  python experiments/run_main.py --config configs/main.json
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

from common import (RESULTS, coarse_from_spec, environment_info, load_config, map_name,
                    save_json, scenario_starts, write_csv)

from mcpp.darp import DARPConfig
from mcpp.planner import RANDOMISED, plan, plan_darp_pair


def run_job(job: dict) -> list[dict]:
    spec, k, scen, cfg = job["spec"], job["k"], job["scenario"], job["cfg"]
    cmap = coarse_from_spec(spec)
    name = map_name(spec)
    starts = scenario_starts(cmap, k, scen, cfg["scenario_seed"], name)
    darp_cfg = DARPConfig(**cfg.get("darp", {}))
    tree = cfg.get("tree", "adaptive")
    base = dict(map=name, n_cells=cmap.n_cells, k=k, scenario=scen)
    rows = []
    methods = cfg["methods"]
    for seed in job["seeds"]:
        done = set()
        if "darp" in methods or "darp_ls" in methods:
            for res in plan_darp_pair(cmap, starts, seed, darp_cfg, tree):
                if res.method in methods:
                    rows.append({**base, "method": res.method, "seed": seed, **res.metrics})
            done |= {"darp", "darp_ls"}
        for m in methods:
            if m in done:
                continue
            if m not in RANDOMISED and seed != job["seeds"][0]:
                continue  # deterministic: once per instance
            res = plan(m, cmap, starts, seed, darp_cfg, tree)
            rows.append({**base, "method": m, "seed": seed, **res.metrics})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.json")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = RESULTS / cfg["name"]
    seeds = list(range(cfg["seeds"]))
    chunk = cfg.get("seed_chunk", 10)
    jobs = []
    for spec in cfg["maps"]:
        for k in cfg["k"]:
            for scen in range(cfg["scenarios"]):
                for i in range(0, len(seeds), chunk):
                    jobs.append(dict(spec=spec, k=k, scenario=scen, seeds=seeds[i:i + chunk], cfg=cfg))
    # Every finished job is appended to a JSON-lines checkpoint, so an
    # interrupted run (Ctrl-C, out of memory, reboot) resumes where it stopped.
    # Delete runs_partial.jsonl to start from scratch.
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "runs_partial.jsonl"
    rows, done = [], set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            rec = json.loads(line)
            done.add(tuple(rec["key"]))
            rows.extend(rec["rows"])
    todo = [j for j in jobs if _job_key(j) not in done]
    workers = args.workers or cfg.get("workers", 4)
    print(f"{len(jobs)} jobs ({len(jobs) - len(todo)} restored from checkpoint), workers={workers}")
    t0 = time.time()
    with Pool(workers, maxtasksperchild=20) as pool, ckpt.open("a") as fck:
        for i, (key, r) in enumerate(pool.imap_unordered(_job_wrapper, todo), 1):
            rows.extend(r)
            fck.write(json.dumps({"key": list(key), "rows": r}) + "\n")
            fck.flush()
            if i % max(1, len(todo) // 20) == 0:
                print(f"  {i}/{len(todo)} jobs done ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda r: (r["map"], r["k"], r["scenario"], r["method"], r["seed"]))
    write_csv(rows, out / "runs.csv")
    save_json({"config": cfg, "env": environment_info(), "wall_time_s": time.time() - t0}, out / "env.json")
    print(f"wrote {len(rows)} rows to {out / 'runs.csv'} in {time.time() - t0:.0f}s")


def _job_key(job: dict) -> tuple:
    return (map_name(job["spec"]), job["k"], job["scenario"], job["seeds"][0])


def _job_wrapper(job: dict) -> tuple[tuple, list[dict]]:
    key = _job_key(job)
    # Deterministic methods run only in the chunk that holds seed 0.
    if job["seeds"][0] != 0:
        cfg = dict(job["cfg"])
        cfg["methods"] = [m for m in cfg["methods"] if m in RANDOMISED]
        job = {**job, "cfg": cfg}
    return key, run_job(job)


if __name__ == "__main__":
    main()
