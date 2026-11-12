"""Shared helpers for all experiment scripts: map specs, seeded scenarios,
parallel execution, environment logging and CSV output."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcpp.grid import (CoarseMap, GridMap, build_coarse, downsample, gen_maze, gen_open,  # noqa: E402
                       gen_random, gen_rooms, load_movingai, sample_starts, upsample)

DATA = ROOT / "data" / "movingai"
RESULTS = ROOT / "results"


def load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def map_from_spec(spec: dict) -> GridMap:
    """Build a map from a JSON spec.

    MovingAI maps: ``{"movingai": "room-32-32-4.map", "mode": "coarse"}``.
    ``mode="coarse"`` interprets each map cell as one coarse STC cell (2x2
    robot footprints, no discretisation loss); ``mode="fine"`` interprets each
    map cell as one robot footprint (partially blocked 2x2 blocks are lost).
    Optional ``"downsample": f`` lowers the resolution first.
    Generators: ``{"gen": "rooms", "h": 32, "w": 32, "room": 5, "seed": 1}``.
    """
    if "movingai" in spec:
        g = load_movingai(DATA / spec["movingai"])
        if spec.get("downsample", 1) > 1:
            g = downsample(g, spec["downsample"])
        if spec.get("mode", "coarse") == "coarse":
            g = upsample(g, 2)
        return g
    kind = spec["gen"]
    if kind == "open":
        return gen_open(spec["h"], spec["w"])
    if kind == "random":
        return gen_random(spec["h"], spec["w"], spec["density"], spec["seed"])
    if kind == "rooms":
        return gen_rooms(spec["h"], spec["w"], spec["room"], spec["seed"])
    if kind == "maze":
        return gen_maze(spec["h"], spec["w"], spec["corridor"], spec["seed"])
    raise ValueError(f"unknown map spec {spec}")


def map_name(spec: dict) -> str:
    if "name" in spec:
        return spec["name"]
    if "movingai" in spec:
        base = Path(spec["movingai"]).stem
        return base if spec.get("mode", "coarse") == "coarse" else f"{base}[fine]"
    return map_from_spec(spec).name


@lru_cache(maxsize=64)
def _coarse_cached(spec_json: str) -> CoarseMap:
    return build_coarse(map_from_spec(json.loads(spec_json)))


def coarse_from_spec(spec: dict) -> CoarseMap:
    """Cached per process (maps are re-used by many jobs)."""
    return _coarse_cached(json.dumps(spec, sort_keys=True))


def scenario_starts(cmap: CoarseMap, k: int, scenario: int, base_seed: int, map_id: str) -> list:
    """Deterministic start positions for (map, k, scenario)."""
    h = sum(map_id.encode()) % 100003  # stable across runs/platforms
    rng = np.random.default_rng([base_seed, h, k, scenario])
    return sample_starts(cmap, k, rng)


def environment_info() -> dict:
    """Hardware / software description required by the assignment (3.3)."""
    import matplotlib
    import scipy
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:  # total RAM on Windows / Linux, best effort
        if sys.platform == "win32":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory;"
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.split("\n")
            info["ram_gb"] = round(int(out[0].strip()) / 2**30, 1)
            info["cpu_name"] = out[1].strip()
        elif Path("/proc/meminfo").exists():
            kb = int(Path("/proc/meminfo").read_text().split()[1])
            info["ram_gb"] = round(kb / 2**20, 1)
    except Exception:  # pragma: no cover - informational only
        pass
    return info


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
