"""DARP - Divide Areas based on Robots' initial Positions.

Re-implementation from the description in
    A. C. Kapoutsis, S. A. Chatzichristofis, E. B. Kosmatopoulos,
    "DARP: Divide Areas Algorithm for Optimal Multi-Robot Coverage Path
    Planning", J. Intelligent & Robotic Systems 86, 2017.
(The authors' code, github.com/alice-st/DARP, was consulted only to check the
behaviour of the connectivity term; no code was copied.)

Idea
----
Each robot ``i`` owns an *evaluation matrix* ``E_i`` (initially the distance
from its start to every cell). Every iteration

1. ``A(v) = argmin_i E_i(v)``                       (assignment)
2. ``k_i = |{v : A(v) = i}|``                         (region sizes)
3. ``m_i = 1 + c * (k_i - f_i) / f_i``                (load-balance term,
   ``f_i`` = fair share, ``c`` = learning rate): over-loaded robots get more
   expensive, under-loaded ones cheaper;
4. if region ``i`` is disconnected, a connectivity matrix ``C_i`` rewards
   cells close to the component ``R_i`` that holds the robot and penalises
   cells close to the other components ``Q_i``:
       ``C_i = normalise(dist(v, R_i) - dist(v, Q_i)) in [1-nu, 1+nu]``;
5. ``E_i <- E_i * m_i * C_i * (1 + random perturbation)``
   (implemented additively on ``log E_i`` to avoid overflow).

Our ``m_i`` uses the *relative* deviation ``(k_i - f_i)/f_i`` (the paper uses
``c (k_i - f_i)``) so that one learning rate works for all map sizes; the
authors' code uses yet another, error-proportional rule. The default
``c = 0.02`` was chosen on separate tuning maps (see ``experiments/``);
an optional step-size decay (``adaptive_lr``) is evaluated in the ablation.

The loop stops when every region is connected and ``|k_i - f_i| <= thr``.
If that does not happen within ``max_iter`` iterations, ``thr`` is relaxed by
one cell (as in the original), up to ``max_threshold``. If DARP still fails,
the best *connected* assignment seen is returned (or, if none, the last one
after :func:`mcpp.partition.repair_connectivity`) and ``success=False`` is
reported - convergence is not guaranteed in general.

Complexity: one iteration costs O(k |V|) (argmin over k matrices, k
connected-component labellings, and up to k Euclidean distance transforms,
each linear in |V|).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .graph import Cell, bfs_distances
from .grid import FOUR_CONN, CoarseMap
from .partition import repair_connectivity


@dataclass
class DARPConfig:
    max_iter: int = 2000          # iterations per threshold level
    learning_rate: float = 0.02   # c in m_i = 1 + c (k_i - f_i) / f_i
    cc_variation: float = 0.01    # nu: amplitude of the connectivity term
    random_level: float = 1e-4    # amplitude of the multiplicative noise
    max_threshold: int = 2        # largest tolerated |k_i - f_i| (cells)
    distance: str = "euclidean"   # "euclidean" (original) or "geodesic"
    connectivity: bool = True     # use the connectivity matrix C_i
    adaptive_lr: bool = False     # halve c when the imbalance stagnates
    patience: int = 50            # iterations without improvement before halving
    min_lr: float = 1e-3          # floor of the adaptive learning rate


@dataclass
class DARPResult:
    assign: np.ndarray
    success: bool
    iterations: int
    threshold: int
    runtime: float
    history: list[float] = field(default_factory=list)  # max |k_i - f_i| per iteration


def _connectivity_matrix(region: np.ndarray, start: Cell, free: np.ndarray, nu: float) -> np.ndarray | None:
    """``C_i`` for a disconnected region, ``None`` if the region is connected."""
    lab, n = ndimage.label(region, structure=FOUR_CONN)
    if n <= 1:
        return None
    r_mask = lab == lab[start]
    q_mask = region & ~r_mask
    # distance_transform_edt measures the distance to the nearest zero, so
    # pass the complement of the target set.
    d_r = ndimage.distance_transform_edt(~r_mask)
    d_q = ndimage.distance_transform_edt(~q_mask)
    raw = d_r - d_q
    vals = raw[free]
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return None
    return (raw - lo) * (2.0 * nu / (hi - lo)) + (1.0 - nu)


def darp(cmap: CoarseMap, starts: list[Cell], cfg: DARPConfig | None = None,
         rng: np.random.Generator | None = None, portions: list[float] | None = None,
         record_history: bool = False) -> DARPResult:
    """Run DARP on the usable cells of ``cmap``.

    ``portions`` (summing to 1) allow unequal target shares, e.g. proportional
    to robot speeds for heterogeneous teams. Default: equal shares.
    """
    cfg = cfg or DARPConfig()
    rng = rng or np.random.default_rng(0)
    t0 = time.perf_counter()
    free = cmap.free
    k = len(starts)
    n = int(free.sum())
    if len(set(starts)) != k or any(not free[s] for s in starts):
        raise ValueError("start cells must be distinct usable cells")
    p = np.full(k, 1.0 / k) if portions is None else np.asarray(portions, dtype=float)
    if abs(p.sum() - 1.0) > 1e-6:
        raise ValueError("portions must sum to 1")
    fair = p * n

    # ---- initial evaluation matrices --------------------------------------
    h, w = cmap.shape
    E = np.empty((k, h, w))
    if cfg.distance == "euclidean":
        rr, cc = np.mgrid[0:h, 0:w]
        for i, (r, c) in enumerate(starts):
            E[i] = np.hypot(rr - r, cc - c)
    elif cfg.distance == "geodesic":
        for i, s in enumerate(starts):
            d = bfs_distances(free, [s]).astype(float)
            d[d < 0] = d.max() + 1
            E[i] = d
    else:
        raise ValueError(f"unknown distance {cfg.distance!r}")
    # The multiplicative updates are applied in the log domain:
    # log E_i <- log E_i + log m_i + log C_i + log(noise). The argmin is the
    # same, but compounded multipliers can no longer overflow (a robot whose
    # multiplier keeps shrinking would otherwise drive E to 0 / inf).
    E = np.log(E + 1.0)  # +1: strictly positive distances
    E[:, ~free] = np.inf
    start_idx = tuple(np.array(starts).T)

    threshold = 0 if np.allclose(fair, np.round(fair)) else 1
    max_threshold = max(cfg.max_threshold, threshold)  # run at least one level
    best_connected: tuple[float, np.ndarray] | None = None
    history: list[float] = []
    total_iter = 0
    assign = None
    lr = cfg.learning_rate
    while threshold <= max_threshold:
        best_level = np.inf      # best imbalance in the current threshold level
        since_best = 0
        for _ in range(cfg.max_iter):
            total_iter += 1
            # 1) assignment -------------------------------------------------
            assign = np.argmin(E, axis=0)
            assign[~free] = -1
            assign[start_idx] = np.arange(k)
            sizes = np.bincount(assign[free], minlength=k)
            dev = sizes - fair
            max_dev = float(np.abs(dev).max())
            if record_history:
                history.append(max_dev)
            # 2) connectivity -----------------------------------------------
            C = []
            all_connected = True
            for i in range(k):
                Ci = _connectivity_matrix(assign == i, starts[i], free, cfg.cc_variation)
                if Ci is not None:
                    all_connected = False
                C.append(Ci)
            if all_connected:
                if best_connected is None or max_dev < best_connected[0]:
                    best_connected = (max_dev, assign.copy())
                if max_dev <= threshold:
                    return DARPResult(assign, True, total_iter, threshold,
                                      time.perf_counter() - t0, history)
            # 3) step-size control -----------------------------------------
            # Large steps converge fast but oscillate around the fair share;
            # halving c whenever max|k_i - f_i| has not improved for
            # ``patience`` iterations damps the oscillation (design choice,
            # evaluated in the ablation study).
            if max_dev < best_level:
                best_level, since_best = max_dev, 0
            else:
                since_best += 1
                if cfg.adaptive_lr and since_best >= cfg.patience and lr > cfg.min_lr:
                    lr = max(cfg.min_lr, lr * 0.5)
                    since_best = 0
            # 4) update ---------------------------------------------------
            m = 1.0 + lr * dev / fair
            log_m = np.log(np.maximum(m, 1e-3))
            for i in range(k):
                Ei = E[i]
                Ei += log_m[i]
                if cfg.connectivity and C[i] is not None:
                    Ei += np.log(C[i])
                if cfg.random_level > 0:
                    Ei += np.log(1.0 - cfg.random_level
                                 + 2.0 * cfg.random_level * rng.random((h, w)))
            # A common additive shift (= scalar rescaling of E) keeps the
            # argmin unchanged and the numbers small.
            E -= E[:, free].min()
        threshold += 1

    # ---- failure: best connected assignment, else repaired last one ---------
    if best_connected is not None:
        out = best_connected[1]
    else:
        out = repair_connectivity(assign, cmap, starts)
    return DARPResult(out, False, total_iter, max_threshold, time.perf_counter() - t0, history)
