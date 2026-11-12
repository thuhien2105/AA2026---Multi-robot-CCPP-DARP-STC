"""Unit tests for the partitioning / tree-cover algorithms and the MIP models."""
import itertools

import numpy as np
import pytest

from mcpp.darp import DARPConfig, darp
from mcpp.graph import articulation_points, cells_to_mask, count_components
from mcpp.grid import build_coarse, gen_maze, gen_open, gen_random, gen_rooms, sample_starts
from mcpp.local_search import improve_partition
from mcpp.mip import mip_sets, solve_minmax
from mcpp.mmrtc import _kuhn_matching, mmrtc_even
from mcpp.partition import (kmeans_partition, region_sizes, repair_connectivity,
                            validate_partition, voronoi_partition)
from mcpp.planner import METHODS, plan

MAPS = {
    "open": lambda: gen_open(10, 12),
    "rooms": lambda: gen_rooms(17, 17, 4, seed=7),
    "maze": lambda: gen_maze(17, 17, 3, seed=8),
    "random": lambda: gen_random(14, 14, 0.2, seed=9),
}


def _instance(name, k, seed=0):
    cm = build_coarse(MAPS[name]())
    return cm, sample_starts(cm, k, np.random.default_rng(seed))


# ------------------------------------------------------------ partitions ----
@pytest.mark.parametrize("name", MAPS)
@pytest.mark.parametrize("k", [2, 3, 5])
def test_baselines_are_valid(name, k):
    cm, st = _instance(name, k)
    validate_partition(voronoi_partition(cm, st), cm, st)
    validate_partition(kmeans_partition(cm, st), cm, st)
    validate_partition(kmeans_partition(cm, st, np.random.default_rng(1), init="++"), cm, st)


def test_repair_reassigns_orphans():
    cm, st = _instance("open", 2)
    bad = voronoi_partition(cm, st)
    # create an island of robot 0 inside robot 1's region
    r1 = np.argwhere(bad == 1)
    far = tuple(r1[len(r1) // 2])
    if far != st[1]:
        bad[far] = 0
    validate_partition(repair_connectivity(bad, cm, st), cm, st)


@pytest.mark.parametrize("name", MAPS)
@pytest.mark.parametrize("k", [2, 4])
def test_darp_valid_and_balanced(name, k):
    cm, st = _instance(name, k, seed=3)
    res = darp(cm, st, DARPConfig(), np.random.default_rng(0))
    validate_partition(res.assign, cm, st)
    sizes = region_sizes(res.assign, k)
    if res.success:
        assert np.abs(sizes - cm.n_cells / k).max() <= res.threshold + 1e-9


def test_darp_open_field_is_exactly_balanced():
    cm = build_coarse(gen_open(8, 8))
    st = [(0, 0), (0, 7), (7, 0), (7, 7)]
    res = darp(cm, st, DARPConfig(), np.random.default_rng(0))
    assert res.success and region_sizes(res.assign, 4).tolist() == [16, 16, 16, 16]


def test_darp_portions():
    cm = build_coarse(gen_open(8, 8))
    st = [(0, 0), (7, 7)]
    res = darp(cm, st, DARPConfig(), np.random.default_rng(0), portions=[0.25, 0.75])
    assert res.success
    assert np.abs(region_sizes(res.assign, 2) - np.array([16, 48])).max() <= 1


# ---------------------------------------------------------- local search ----
@pytest.mark.parametrize("name", MAPS)
@pytest.mark.parametrize("k", [2, 4, 6])
def test_local_search_never_worse_and_valid(name, k):
    cm, st = _instance(name, k, seed=5)
    start = voronoi_partition(cm, st)
    res = improve_partition(start, cm, st)
    validate_partition(res.assign, cm, st)
    assert region_sizes(res.assign, k).max() <= region_sizes(start, k).max()
    assert res.max_after == region_sizes(res.assign, k).max()


def test_local_search_reaches_perfect_balance_on_open_field():
    cm = build_coarse(gen_open(6, 6))
    st = [(0, 0), (0, 1), (0, 2)]  # Voronoi is very unbalanced here
    res = improve_partition(voronoi_partition(cm, st), cm, st)
    assert region_sizes(res.assign, 3).max() == 12


# --------------------------------------------------------- tree cover -------
def test_kuhn_matching():
    assert _kuhn_matching([[0, 1], [0]], 2) == [1, 0]
    assert _kuhn_matching([[0], [0]], 2) is None


@pytest.mark.parametrize("name", MAPS)
@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("dedup", [False, True])
def test_mmrtc_cover_is_valid_and_within_4B(name, k, dedup):
    cm, st = _instance(name, k, seed=2)
    res = mmrtc_even(cm, st, dedup=dedup)
    union = set().union(*res.sets)
    assert union == set(cm.cells())
    for i, s in enumerate(res.sets):
        assert st[i] in s
        assert count_components(cells_to_mask(s, cm.shape)) == 1
    assert res.max_weight < 4 * res.bound_B
    assert res.bound_B >= int(np.ceil(cm.n_cells / k)) - 1


# ------------------------------------------------------------------ MIP -----
def _brute_force_minmax(cm, st, cover):
    """Enumerate all assignments (partition) / covers on a tiny instance."""
    cells = cm.cells()
    k = len(st)
    best = None
    if not cover:
        for lab in itertools.product(range(k), repeat=len(cells)):
            assign = np.full(cm.shape, -1)
            for c, l in zip(cells, lab):
                assign[c] = l
            if any(assign[s] != i for i, s in enumerate(st)):
                continue
            if all(count_components(assign == i) == 1 for i in range(k)):
                val = max(region_sizes(assign, k))
                best = val if best is None else min(best, val)
    else:  # each cell gets a non-empty subset of robots
        subsets = [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]
        for lab in itertools.product(subsets, repeat=len(cells)):
            sets = [{c for c, l in zip(cells, lab) if i in l} for i in range(k)]
            if any(st[i] not in sets[i] for i in range(k)):
                continue
            if all(count_components(cells_to_mask(s, cm.shape)) == 1 for s in sets):
                val = max(len(s) for s in sets)
                best = val if best is None else min(best, val)
    return best


@pytest.mark.parametrize("cover", [False, True])
@pytest.mark.parametrize("starts", [[(0, 0), (1, 2)], [(0, 1), (1, 1)], [(0, 0), (0, 1), (1, 2)]])
def test_mip_matches_brute_force(cover, starts):
    cm = build_coarse(gen_open(2, 3))
    res = solve_minmax(cm, starts, cover=cover, time_limit=30)
    assert res.optimal
    assert res.objective == pytest.approx(_brute_force_minmax(cm, starts, cover))
    sets = mip_sets(res)
    for i, s in enumerate(sets):
        assert starts[i] in s and count_components(cells_to_mask(s, cm.shape)) == 1


def test_mip_on_l_shape_cover_beats_partition():
    # Plus-shaped map where sharing the centre helps tree covers.
    from mcpp.grid import from_ascii, upsample
    g = upsample(from_ascii("#.#\n...\n#.#\n"))
    cm = build_coarse(g)
    st = [(0, 1), (2, 1)]
    part = solve_minmax(cm, st, cover=False).objective
    cov = solve_minmax(cm, st, cover=True).objective
    assert cov <= part


# --------------------------------------------------------------- planner ----
@pytest.mark.parametrize("method", METHODS)
def test_planner_end_to_end(method):
    cm, st = _instance("rooms", 3, seed=11)
    res = plan(method, cm, st, seed=0)
    m = res.metrics
    assert m["invalid_cells"] == 0
    assert m["coverage"] == pytest.approx(cm.n_cells * 4 / cm.fine.n_free)
    assert m["makespan"] >= m["lower_bound"]
    if res.assign is not None:  # disjoint regions -> no inter-robot conflicts
        assert m["overlap"] == 0 and m["vertex_conflicts"] == 0 and m["swap_conflicts"] == 0
        assert m["makespan"] == 4 * m["max_region"]
