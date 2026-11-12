"""Plotting: partitions + coverage tours, coverage animations and a shared
chart style for the experiment figures (static PNGs for the report/video).

Colours follow a fixed categorical order (never cycled); identity is never
colour-alone (legends/labels are always drawn)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402

from .graph import Cell  # noqa: E402
from .grid import GridMap  # noqa: E402

# Validated categorical palette (light mode), fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
OBSTACLE = "#3b3a37"

# One fixed slot per method: colour follows the entity, never its rank.
METHOD_COLORS = {
    "voronoi": PALETTE[0], "kmeans": PALETTE[1], "darp": PALETTE[2], "darp_ls": PALETTE[3],
    "voronoi_ls": PALETTE[4], "mmrtc": PALETTE[5], "mmrtc_dedup": PALETTE[6], "mip": PALETTE[7],
}
METHOD_LABELS = {
    "voronoi": "Voronoi+STC", "kmeans": "k-means+STC", "darp": "DARP+STC",
    "darp_ls": "DARP+LS+STC", "voronoi_ls": "Voronoi+LS+STC", "mmrtc": "MMRTC (Even 4-apx)",
    "mmrtc_dedup": "MMRTC+dedup", "mip": "MIP optimum",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 10,
        "legend.frameon": False, "axes.titleweight": "bold",
    })


def _blend(color: str, alpha: float) -> tuple[float, float, float]:
    """Colour blended with the surface (light region tint)."""
    c, s = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(alpha * c + (1 - alpha) * s)


def plot_solution(grid: GridMap, sets: list[set[Cell]], paths: list[list[Cell]],
                  starts: list[Cell], title: str = "", ax=None, show_paths: bool = True,
                  path_width: float = 1.2):
    """Region tints (coarse cells) + fine-grid tours + start markers."""
    apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * grid.shape[0] / grid.shape[1] + 0.4))
    H, W = grid.shape
    img = np.ones((H, W, 3)) * np.array(to_rgb(SURFACE))
    img[grid.occ] = to_rgb(OBSTACLE)
    k = len(sets)
    count = np.zeros((H, W), dtype=int)
    for i, s in enumerate(sets):
        col = _blend(PALETTE[i % len(PALETTE)], 0.35)
        for (r, c) in s:
            img[2 * r:2 * r + 2, 2 * c:2 * c + 2] = col
            count[2 * r:2 * r + 2, 2 * c:2 * c + 2] += 1
    img[count > 1] = to_rgb("#b9b8b3")  # overlapping coverage (tree covers)
    ax.imshow(img, interpolation="nearest", origin="upper")
    if show_paths:
        for i, p in enumerate(paths):
            arr = np.array(p)
            ax.plot(arr[:, 1], arr[:, 0], color=PALETTE[i % len(PALETTE)], lw=path_width,
                    label=f"robot {i + 1}" if k <= 8 else None)
    for i, (r, c) in enumerate(starts):
        ax.plot(2 * c, 2 * r, marker="o", ms=8, mfc=PALETTE[i % len(PALETTE)], mec=SURFACE, mew=2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if show_paths and k <= 8:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(k, 4), fontsize=8)
    return ax


def animate_solution(grid: GridMap, paths: list[list[Cell]], out: str | Path,
                     frames: int = 120, fps: int = 20, title: str = "") -> Path:
    """GIF of the synchronised execution: covered cells get the robot colour,
    robots are drawn as dots. ``frames`` frames span the whole makespan."""
    apply_style()
    H, W = grid.shape
    T = max(len(p) for p in paths)
    k = len(paths)
    base = np.zeros((H, W), dtype=int)  # 0 free, 1 obstacle, 2+i covered by i
    base[grid.occ] = 1
    cmap = ListedColormap([SURFACE, OBSTACLE] + [_blend(PALETTE[i % 8], 0.55) for i in range(k)])
    fig, ax = plt.subplots(figsize=(5.5, 5.5 * H / W + 0.5))
    im = ax.imshow(base, cmap=cmap, vmin=0, vmax=k + 1, interpolation="nearest")
    dots = [ax.plot([], [], "o", ms=7, mfc=PALETTE[i % 8], mec=TEXT, mew=1)[0] for i in range(k)]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    txt = ax.set_title(title)
    steps = np.linspace(0, T - 1, frames).astype(int)
    state = base.copy()
    last = [0] * k

    def update(fi):
        t = steps[fi]
        for i, p in enumerate(paths):
            for tt in range(last[i], min(t + 1, len(p))):
                state[p[tt]] = 2 + i
            last[i] = min(t + 1, len(p))
            r, c = p[min(t, len(p) - 1)]
            dots[i].set_data([c], [r])
        im.set_data(state)
        txt.set_text(f"{title}  t = {t}/{T - 1}")
        return [im, *dots, txt]

    anim = animation.FuncAnimation(fig, update, frames=len(steps), blit=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out
