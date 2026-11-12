"""Turn the raw CSVs into report tables (Markdown), statistical tests and
figures. Writes ``results/REPORT.md`` and ``results/figures/*.png``.

Statistics (assignment section 3.3):
  * randomised methods: per-instance median over the 30 runs, then
    median [IQR] over instances;
  * "A better than B" claims use the paired Wilcoxon signed-rank test on
    per-instance medians (two-sided, alpha = 0.05).

Usage:  python experiments/analyze.py
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from common import RESULTS, map_from_spec, read_csv
from scipy.stats import wilcoxon

from mcpp.grid import build_coarse, load_movingai
from mcpp.viz import METHOD_COLORS, METHOD_LABELS, PALETTE, TEXT_2, apply_style, plt

FIG = RESULTS / "figures"
ORDER = ["voronoi", "kmeans", "darp", "darp_ls", "voronoi_ls", "mmrtc", "mmrtc_dedup"]


def med_iqr(x, fmt="{:.3f}") -> str:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return "-"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{fmt.format(q2)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def md_table(header: list[str], rows: list[list]) -> str:
    out = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def per_instance(rows, keys=("map", "k", "scenario"), metric="makespan"):
    """{method: {instance: median over seeds}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get(metric, "") == "":
            continue
        acc[r["method"]][tuple(r[k] for k in keys)].append(float(r[metric]))
    return {m: {i: float(np.median(v)) for i, v in d.items()} for m, d in acc.items()}


def paired_test(a: dict, b: dict) -> tuple[int, float, float, float]:
    """(n, median of a-b, fraction a<b, p-value) over common instances."""
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    d = x - y
    if len(keys) == 0 or np.all(d == 0):
        return len(keys), 0.0, 0.0, 1.0
    p = float(wilcoxon(x, y, zero_method="wilcox").pvalue)
    return len(keys), float(np.median(d)), float(np.mean(d < 0)), p


# ---------------------------------------------------------------- main ----
def analyze_main(rep: list[str]) -> None:
    path = RESULTS / "main" / "runs.csv"
    if not path.exists():
        rep.append("_main benchmark not run yet_\n")
        return
    rows = read_csv(path)
    env = json.loads((RESULTS / "main" / "env.json").read_text())
    for r in rows:
        r["ratio_lb"] = r["makespan"] / r["lower_bound"]
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    rep.append("## 1. Main benchmark (MovingAI maps)\n")
    rep.append(f"Maps: {', '.join(sorted({r['map'] for r in rows}))}; k = "
               f"{sorted({int(r['k']) for r in rows})}; {env['config']['scenarios']} start scenarios "
               f"per (map, k); {env['config']['seeds']} runs per randomised method and instance. "
               f"Wall time {env['wall_time_s'] / 60:.1f} min on {env['env'].get('cpu_name', env['env']['processor'])}.\n")
    rep.append("`ratio_lb` = makespan / (4 * ceil(n/k)); 1.0 means the trivial lower bound is met. "
               "Values: median [IQR] over instances of the per-instance median over runs.\n")
    metrics = [("ratio_lb", "{:.3f}"), ("balance", "{:.3f}"), ("turns", "{:.0f}"),
               ("coverage", "{:.3f}"), ("overlap", "{:.3f}"), ("vertex_conflicts", "{:.0f}"),
               ("runtime", "{:.3f}")]
    tab = []
    for m in methods:
        line = [METHOD_LABELS[m]]
        for met, fmt in metrics:
            pi = per_instance([r for r in rows if r["method"] == m], metric=met).get(m, {})
            line.append(med_iqr(list(pi.values()), fmt))
        tab.append(line)
    rep.append(md_table(["method", "makespan / LB", "balance (min/max)", "total turns", "coverage",
                         "overlap", "vertex conflicts", "runtime [s]"], tab))
    # DARP convergence
    d = [r for r in rows if r["method"] == "darp"]
    by = defaultdict(list)
    for r in d:
        by[(r["map"], int(r["k"]))].append(r["darp_success"])
    maps = sorted({r["map"] for r in rows})
    ks = sorted({int(r["k"]) for r in rows})
    rep.append("\n**DARP success rate** (connected + |k_i - f_i| <= 2 within the iteration budget):\n")
    rep.append(md_table(["map"] + [f"k={k}" for k in ks],
                        [[mp] + [f"{100 * np.mean(by[(mp, k)]):.0f}%" for k in ks] for mp in maps]))
    rep.append(f"\nOverall DARP success: {100 * np.mean([r['darp_success'] for r in d]):.1f}% of {len(d)} runs.\n")
    # per-map makespan table
    rep.append("\n**Makespan / LB per map** (median over scenarios and k):\n")
    pi_all = per_instance(rows, metric="ratio_lb")
    tab = []
    for mp in maps:
        tab.append([mp] + [f"{np.median([v for i, v in pi_all[m].items() if i[0] == mp]):.3f}" for m in methods])
    rep.append(md_table(["map"] + [METHOD_LABELS[m] for m in methods], tab))
    # statistical tests
    rep.append("\n**Paired Wilcoxon signed-rank tests** (per-instance medians; negative median "
               "difference = first method better):\n")
    tests = [("darp", "voronoi"), ("darp", "kmeans"), ("darp_ls", "darp"), ("darp_ls", "voronoi_ls"),
             ("darp", "mmrtc_dedup"), ("mmrtc_dedup", "mmrtc"), ("darp_ls", "mmrtc_dedup")]
    tab = []
    for metric in ("makespan", "turns", "runtime"):
        pi = per_instance(rows, metric=metric)
        for a, b in tests:
            if a in pi and b in pi:
                n, md, win, p = paired_test(pi[a], pi[b])
                verdict = ("A better" if md < 0 else "B better") if p < 0.05 else "no significant difference"
                tab.append([metric, METHOD_LABELS[a], METHOD_LABELS[b], n, f"{md:.3g}", f"{100 * win:.0f}%",
                            f"{p:.2e}", verdict])
    rep.append(md_table(["metric", "A", "B", "n", "median(A-B)", "A<B", "p", "conclusion"], tab))
    plot_main(rows, methods, maps, ks)


def plot_main(rows, methods, maps, ks) -> None:
    apply_style()
    pi = per_instance(rows, metric="ratio_lb")
    ncol = 3
    nrow = int(np.ceil(len(maps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.4 * nrow), sharey=True, squeeze=False)
    for ax, mp in zip(axes.ravel(), maps):
        for m in methods:
            y = [np.median([v for i, v in pi[m].items() if i[0] == mp and int(i[1]) == k]) for k in ks]
            ax.plot(ks, y, marker="o", color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.axhline(1.0, color=TEXT_2, lw=1, ls="--")
        ax.set_title(mp)
        ax.set_xticks(ks)
        ax.set_xlabel("robots k")
    for ax in axes[:, 0]:
        ax.set_ylabel("makespan / lower bound")
    for ax in axes.ravel()[len(maps):]:
        ax.axis("off")
    h, lab = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Makespan relative to the lower bound 4*ceil(n/k) (median over 5 scenarios)", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "main_makespan_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # runtime per method vs k (log scale), aggregated over maps
    pr = per_instance(rows, metric="runtime")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in methods:
        y = [np.median([v for i, v in pr[m].items() if int(i[1]) == k]) for k in ks]
        ax.plot(ks, y, marker="o", color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("robots k")
    ax.set_ylabel("runtime [s] (median, log)")
    ax.set_title("Planning time (partition + STC)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "main_runtime_vs_k.png", dpi=150)
    plt.close(fig)

    # turns: total turns relative to Voronoi
    pt = per_instance(rows, metric="balance")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    data = [list(pt[m].values()) for m in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="#0b0b0b"))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(methods) + 1), [METHOD_LABELS[m] for m in methods], rotation=25, ha="right")
    ax.set_ylabel("load balance  min L / max L")
    ax.set_title("Load balance per instance (1 = perfect)")
    fig.tight_layout()
    fig.savefig(FIG / "main_balance_box.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- scaling ----
THEORY = {"t_voronoi": "O(n)", "t_stc": "O(n log n)", "t_darp_per_iter": "O(k n)",
          "t_darp": "O(I k n), I = #iterations", "t_ls": "O(C n), C = #chains",
          "t_ls_per_chain": "O(n)", "t_mmrtc": "O(k n log n)"}


def analyze_scaling(rep: list[str]) -> None:
    path = RESULTS / "scaling" / "scaling.csv"
    if not path.exists():
        rep.append("_scaling experiment not run yet_\n")
        return
    rows = read_csv(path)
    summ = json.loads((RESULTS / "scaling" / "scaling_summary.json").read_text())
    rep.append("## 2. Empirical complexity (log-log)\n")
    rep.append("k = 4, sizes 8x8 ... 128x128 coarse cells, 5 scenarios each; slope = least-squares "
               "slope of log(time) vs log(n). A slope of 1 means linear growth.\n")
    tab = [[c, THEORY[c], f"{summ['slopes']['open:' + c]:.2f}", f"{summ['slopes']['random:' + c]:.2f}"]
           for c in THEORY]
    rep.append(md_table(["component", "theory", "slope (open)", "slope (random 10%)"], tab))
    mem_path = RESULTS / "scaling" / "memory.csv"
    if mem_path.exists():
        mem = read_csv(mem_path)
        cols = [c for c in mem[0] if c.startswith("peak_mb_")]
        rep.append("\n**Peak Python heap (tracemalloc) [MB]**, random map, k = 4:\n")
        rep.append(md_table(["n"] + [c.replace("peak_mb_", "") for c in cols],
                            [[int(m["n"])] + [f"{m[c]:.2f}" for c in cols] for m in mem]))
    rep.append("\nProfile of DARP+LS+STC: see `results/scaling/profile.txt`.\n")
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind in zip(axes, ("open", "random")):
        rr = [r for r in rows if r["kind"] == kind]
        n = np.array([r["n"] for r in rr])
        for j, col in enumerate(("t_voronoi", "t_stc", "t_darp_per_iter", "t_darp", "t_ls", "t_mmrtc")):
            t = np.array([r[col] for r in rr])
            ns = np.unique(n)
            med = np.array([np.median(t[n == v]) for v in ns])
            ax.plot(ns, med, marker="o", color=PALETTE[j], label=f"{col[2:]} (slope {summ['slopes'][kind + ':' + col]:.2f})")
        ref = np.unique(n).astype(float)
        ax.plot(ref, 1e-5 * ref, color=TEXT_2, ls="--", lw=1, label="O(n) reference")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n = number of coarse cells")
        ax.set_title(f"{kind} map")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("time [s]")
    fig.tight_layout()
    fig.savefig(FIG / "scaling_loglog.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ ablation ----
def analyze_ablation(rep: list[str]) -> None:
    path = RESULTS / "ablation" / "darp_ablation.csv"
    if not path.exists():
        rep.append("_ablation not run yet_\n")
        return
    rows = read_csv(path)
    rep.append("## 3. Design ablation & parameter sensitivity (tuning set)\n")
    rep.append("Generated tuning maps (open, rooms, maze, random; disjoint from the report set), "
               "k in {4, 8}, 3 scenarios, 30 runs each.\n")
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    tab = []
    base = per_instance([dict(r, method=r["variant"]) for r in rows], metric="imbalance")
    for v in variants:
        rr = [r for r in rows if r["variant"] == v]
        maze = [r for r in rr if "maze" in r["map"]]
        n, md, win, p = paired_test(base[v], base["default (c=0.02)"]) if v != "default (c=0.02)" else (0, 0, 0, 1)
        tab.append([v, f"{100 * np.mean([r['success'] for r in rr]):.0f}%",
                    f"{100 * np.mean([r['success'] for r in maze]):.0f}%",
                    f"{np.median([r['iterations'] for r in rr]):.0f}",
                    med_iqr([r["imbalance"] for r in rr]), f"{np.median([r['runtime'] for r in rr]):.3f}",
                    "-" if v == "default (c=0.02)" else f"{p:.1e}"])
    rep.append(md_table(["DARP variant", "success", "success (maze)", "median iters",
                         "max region / fair share", "median runtime [s]", "Wilcoxon p vs default (imbalance)"], tab))
    # learning-rate sensitivity figure
    apply_style()
    lrs = [("c=0.005", 0.005), ("c=0.01", 0.01), ("default (c=0.02)", 0.02), ("c=0.05", 0.05), ("c=0.1", 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    maps = sorted({r["map"] for r in rows})
    for j, mp in enumerate(maps):
        sr = [100 * np.mean([r["success"] for r in rows if r["variant"] == v and r["map"] == mp]) for v, _ in lrs]
        it = [np.median([r["iterations"] for r in rows if r["variant"] == v and r["map"] == mp]) for v, _ in lrs]
        axes[0].plot([c for _, c in lrs], sr, marker="o", color=PALETTE[j], label=mp)
        axes[1].plot([c for _, c in lrs], it, marker="o", color=PALETTE[j], label=mp)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("learning rate c")
    axes[0].set_ylabel("success rate [%]")
    axes[1].set_ylabel("median iterations")
    axes[1].set_yscale("log")
    axes[0].set_title("DARP convergence vs c")
    axes[1].set_title("DARP iterations vs c")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "ablation_learning_rate.png", dpi=150)
    plt.close(fig)

    tpath = RESULTS / "ablation" / "tree_ablation.csv"
    if tpath.exists():
        tr = read_csv(tpath)
        rep.append("\n**STC spanning-tree strategy** (same DARP+LS partitions; makespan in moves is "
                   "identical for every tree, 4|V|):\n")
        strategies = list(dict.fromkeys(r["strategy"] for r in tr))
        pi = per_instance([dict(r, method=r["strategy"]) for r in tr], metric="turns")
        pt = per_instance([dict(r, method=r["strategy"]) for r in tr], metric="time_makespan")
        tab = []
        for s in strategies:
            rel = [pi[s][i] / pi["bfs"][i] for i in pi[s]]
            n, md, win, p = paired_test(pi[s], pi["bfs"]) if s != "bfs" else (0, 0, 0, 1)
            tab.append([s, med_iqr(list(pi[s].values()), "{:.0f}"), f"{np.median(rel):.3f}",
                        med_iqr(list(pt[s].values()), "{:.0f}"), "-" if s == "bfs" else f"{p:.1e}"])
        rep.append(md_table(["tree", "total turns", "turns / BFS", "completion time (turn cost 0.5)",
                             "Wilcoxon p vs BFS (turns)"], tab))
    hpath = RESULTS / "ablation" / "hetero.csv"
    if hpath.exists():
        hr = read_csv(hpath)
        rep.append("\n**Heterogeneous team** (speeds 1, 1, 2, 2; time = moves / speed):\n")
        variants = list(dict.fromkeys(r["variant"] for r in hr))
        pi = per_instance([dict(r, method=r["variant"], k=4) for r in hr], keys=("map", "scenario"),
                          metric="move_time_makespan")
        lb = {(r["map"], r["scenario"]): r["lower_bound"] for r in hr}
        tab = []
        for v in variants:
            rel = [pi[v][i] / lb[i] for i in pi[v]]
            n, md, win, p = paired_test(pi[v], pi["DARP equal shares"]) if v != "DARP equal shares" else (0, 0, 0, 1)
            tab.append([v, med_iqr(rel), "-" if v == "DARP equal shares" else f"{p:.1e}"])
        rep.append(md_table(["variant", "time makespan / LB (LB = 4n / sum v)", "Wilcoxon p vs equal shares"], tab))


# -------------------------------------------------------------- optgap ----
def analyze_optgap(rep: list[str]) -> None:
    path = RESULTS / "optgap" / "optgap.csv"
    if not path.exists():
        rep.append("_optimality-gap experiment not run yet_\n")
        return
    rows = read_csv(path)
    rep.append("## 4. Optimality gap vs exact MIP (small instances)\n")
    inst = {}
    for r in rows:
        inst[(r["map"], r["k"], r["scenario"])] = r
    n_inst = len(inst)
    opt_p = sum(r["mip_part_optimal"] for r in inst.values())
    opt_c = sum(r["mip_cover_optimal"] for r in inst.values())
    rep.append(f"{n_inst} instances (n = 36..64 cells, k = 2..4). MIP-P solved to optimality on {opt_p:.0f}, "
               f"MIP-C on {opt_c:.0f}; otherwise the proven dual bound is used (gap = upper bound).\n")
    strict = sum(1 for r in inst.values() if r["mip_cover"] != "" and r["mip_part"] != ""
                 and r["mip_cover"] < r["mip_part"] - 1e-6)
    rep.append(f"Tree cover strictly better than partition optimum (MIP-C < MIP-P) on {strict} instances.\n")
    tab = []
    for m in ORDER:
        rr = [r for r in rows if r["method"] == m]
        if not rr:
            continue
        pi = defaultdict(list)
        for r in rr:
            pi[(r["map"], r["k"], r["scenario"])].append(r["max_region"])
        gp, gc = [], []
        for i, v in pi.items():
            ins = inst[i]
            bp = ins["mip_part"] if ins["mip_part_optimal"] else ins["mip_part_bound"]
            bc = ins["mip_cover"] if ins["mip_cover_optimal"] else ins["mip_cover_bound"]
            gp.append(np.median(v) / bp - 1)
            gc.append(np.median(v) / bc - 1)
        gp, gc = np.array(gp), np.array(gc)
        tab.append([METHOD_LABELS[m], f"{100 * gp.mean():.1f}%", f"{100 * np.median(gp):.1f}%",
                    f"{100 * gp.max():.1f}%", f"{100 * np.mean(gp < 1e-9):.0f}%",
                    f"{100 * gc.mean():.1f}%", f"{100 * gc.max():.1f}%"])
    rep.append(md_table(["method", "mean gap vs MIP-P", "median", "max", "optimal",
                         "mean gap vs MIP-C", "max vs MIP-C"], tab))
    mt = [r["mip_part_time"] for r in inst.values()]
    rep.append(f"\nMIP-P solve time: median {np.median(mt):.2f}s, max {np.max(mt):.1f}s.\n")


# ------------------------------------------------------- coverage loss ----
def analyze_coverage_loss(rep: list[str]) -> None:
    """Fine-mode interpretation: fraction of free cells STC can cover."""
    rep.append("## 5. STC discretisation loss (map cell = robot footprint)\n")
    rep.append("If every MovingAI cell is one robot footprint, STC can only use fully free 2x2 "
               "blocks in the largest connected component; the remaining free cells are lost.\n")
    tab = []
    for f in sorted((RESULTS.parent / "data" / "movingai").glob("*.map")):
        g = load_movingai(f)
        cm = build_coarse(g)
        cov = cm.coverable_fine().sum() / g.n_free
        raw = 4 * cm.raw_free.sum() / g.n_free
        tab.append([f.stem, g.n_free, f"{100 * raw:.1f}%", f"{100 * cov:.1f}%"])
    rep.append(md_table(["map", "free cells", "in free 2x2 blocks", "coverable (largest component)"], tab))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rep = ["# Experimental results\n",
           "_Generated by `experiments/analyze.py` from the CSV files in `results/`._\n"]
    analyze_main(rep)
    analyze_scaling(rep)
    analyze_ablation(rep)
    analyze_optgap(rep)
    analyze_coverage_loss(rep)
    (RESULTS / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
