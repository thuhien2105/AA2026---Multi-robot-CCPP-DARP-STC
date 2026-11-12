"""Interactive-free demo: plan coverage for k robots on one map, print the
metrics of every method, save a side-by-side figure and (optionally) a GIF
animation of the synchronised execution.

Examples
--------
    python demo.py                                    # room-32-32-4, k = 4
    python demo.py --map maze-32-32-4.map --k 6 --gif
    python demo.py --gen rooms --size 24 --k 3 --methods darp darp_ls mmrtc_dedup
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcpp.darp import DARPConfig
from mcpp.grid import build_coarse, gen_maze, gen_open, gen_random, gen_rooms, load_movingai, sample_starts, upsample
from mcpp.planner import METHODS, plan
from mcpp.viz import METHOD_LABELS, animate_solution, plot_solution, plt

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="room-32-32-4.map", help="MovingAI map in data/movingai/")
    ap.add_argument("--fine", action="store_true",
                    help="interpret map cells as robot footprints (default: as coarse STC cells)")
    ap.add_argument("--gen", choices=["open", "rooms", "maze", "random"], help="use a generated map instead")
    ap.add_argument("--size", type=int, default=24, help="coarse size of generated maps")
    ap.add_argument("--k", type=int, default=4, help="number of robots")
    ap.add_argument("--seed", type=int, default=0, help="seed for start positions and randomised methods")
    ap.add_argument("--methods", nargs="+", default=["voronoi", "kmeans", "darp", "darp_ls", "mmrtc_dedup"],
                    choices=METHODS)
    ap.add_argument("--tree", default="adaptive", help="STC spanning-tree strategy")
    ap.add_argument("--gif", action="store_true", help="also write a GIF of the last method")
    ap.add_argument("--out", default="results/demo", help="output folder")
    args = ap.parse_args()

    if args.gen:
        s = args.size
        g = {"open": lambda: gen_open(s, s), "rooms": lambda: gen_rooms(s, s, 5, args.seed),
             "maze": lambda: gen_maze(s, s, 3, args.seed), "random": lambda: gen_random(s, s, 0.15, args.seed)}[args.gen]()
    else:
        g = load_movingai(ROOT / "data" / "movingai" / args.map)
        if not args.fine:
            g = upsample(g, 2)
    cm = build_coarse(g)
    starts = sample_starts(cm, args.k, np.random.default_rng(args.seed))
    print(f"map {g.name}: fine {g.shape}, coarse cells n = {cm.n_cells}, k = {args.k}, starts = {starts}")
    print(f"lower bound on makespan: {4 * int(np.ceil(cm.n_cells / args.k))} moves\n")
    header = f"{'method':<20}{'makespan':>9}{'total':>8}{'balance':>9}{'turns':>7}{'cover%':>8}{'overlap':>8}{'confl':>6}{'time[s]':>9}"
    print(header)
    print("-" * len(header))
    results = []
    for m in args.methods:
        r = plan(m, cm, starts, args.seed, DARPConfig(), args.tree)
        x = r.metrics
        conf = x["vertex_conflicts"] + x["swap_conflicts"]
        print(f"{METHOD_LABELS[m]:<20}{x['makespan']:>9.0f}{x['total_length']:>8.0f}{x['balance']:>9.3f}"
              f"{x['turns']:>7.0f}{100 * x['coverage']:>8.1f}{x['overlap']:>8.3f}{conf:>6.0f}{x['runtime']:>9.3f}"
              + ("" if "darp_success" not in x else f"   (DARP converged: {bool(x['darp_success'])}, "
                                                    f"{x['darp_iterations']:.0f} it.)"))
        results.append(r)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    n = len(results)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.4 * rows * g.shape[0] / g.shape[1] + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        plot_solution(g, r.sets, r.paths, starts, ax=ax,
                      title=f"{METHOD_LABELS[r.method]}\nmakespan {r.metrics['makespan']:.0f}, "
                            f"balance {r.metrics['balance']:.2f}, turns {r.metrics['turns']:.0f}")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    png = out / f"{g.name}_k{args.k}_s{args.seed}.png"
    fig.savefig(png, dpi=130)
    print(f"\nfigure: {png}")
    if args.gif:
        r = results[-1]
        gif = animate_solution(g, r.paths, out / f"{g.name}_k{args.k}_{r.method}.gif",
                               title=METHOD_LABELS[r.method])
        print(f"animation: {gif}")


if __name__ == "__main__":
    main()
