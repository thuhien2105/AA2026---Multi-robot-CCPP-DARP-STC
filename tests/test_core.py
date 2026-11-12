"""Unit tests for the core building blocks (grid, graph primitives, STC)."""
import numpy as np
import pytest
from scipy import ndimage

from mcpp.graph import (UnionFind, articulation_points, bfs_distances, cells_to_mask,
                        count_components, multi_source_labels)
from mcpp.grid import (build_coarse, downsample, from_ascii, gen_maze, gen_open, gen_random,
                       gen_rooms, load_movingai, upsample)
from mcpp.metrics import count_conflicts, evaluate
from mcpp.stc import TREE_STRATEGIES, count_turns, plan_region, spanning_tree, stc_tour


# ----------------------------------------------------------------- grid ----
def test_coarse_requires_fully_free_blocks():
    g = from_ascii("""
....
.#..
....
....
""")
    cm = build_coarse(g)
    # top-left block contains the obstacle -> unusable
    assert cm.raw_free.tolist() == [[False, True], [True, True]]
    assert cm.n_cells == 3
    cov = cm.coverable_fine()
    assert cov.sum() == 12 and not cov[1, 1]


def test_largest_component_kept():
    g = upsample(from_ascii("..#...\n..#...\n"))
    cm = build_coarse(g)
    assert cm.n_cells == 6  # right part (3x2) beats left part (2x2)
    assert count_components(cm.free) == 1


def test_up_and_downsample_roundtrip():
    g = gen_random(10, 12, 0.2, seed=3)
    assert g.shape == (20, 24)
    assert np.array_equal(downsample(g, 2).occ, g.occ[::2, ::2])


@pytest.mark.parametrize("gen", [
    lambda: gen_open(8, 9), lambda: gen_rooms(20, 20, 4, seed=1),
    lambda: gen_maze(21, 21, 3, seed=2), lambda: gen_random(16, 16, 0.2, seed=5)])
def test_generators_are_deterministic_and_nonempty(gen):
    a, b = gen(), gen()
    assert np.array_equal(a.occ, b.occ)
    assert build_coarse(a).n_cells > 10


def test_load_movingai(tmp_path):
    p = tmp_path / "t.map"
    p.write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\nT.G\n")
    g = load_movingai(p)
    assert g.occ.tolist() == [[False, True, False], [True, False, False]]


# ---------------------------------------------------------------- graph ----
def test_bfs_and_voronoi_labels():
    mask = np.ones((3, 5), dtype=bool)
    mask[1, 1:4] = False
    d = bfs_distances(mask, [(0, 0)])
    assert d[2, 4] == 6 and d[1, 2] == -1
    lab, _ = multi_source_labels(mask, [(0, 0), (2, 4)])
    for i in range(2):
        assert count_components(lab == i) == 1


def _brute_articulation(mask):
    base = count_components(mask)
    out = set()
    for r, c in zip(*np.nonzero(mask)):
        m = mask.copy()
        m[r, c] = False
        if count_components(m) > base:
            out.add((int(r), int(c)))
    return out


@pytest.mark.parametrize("seed", range(15))
def test_articulation_points_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    mask = rng.random((7, 8)) < 0.65
    assert articulation_points(mask) == _brute_articulation(mask)


def test_union_find():
    uf = UnionFind(range(6))
    assert uf.union(0, 1) and uf.union(2, 3) and uf.union(1, 3)
    assert not uf.union(0, 2)
    assert uf.find(0) == uf.find(3) != uf.find(4)


# ------------------------------------------------------------------ STC ----
def _random_connected_cells(rng, h=6, w=7, p=0.7):
    while True:
        mask = rng.random((h, w)) < p
        lab, n = ndimage.label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            comp = lab == sizes.argmax()
            if comp.sum() >= 3:
                return {(int(r), int(c)) for r, c in zip(*np.nonzero(comp))}


@pytest.mark.parametrize("strategy", [s for s in TREE_STRATEGIES if s != "adaptive"])
@pytest.mark.parametrize("seed", range(10))
def test_spanning_tree_is_spanning(strategy, seed):
    rng = np.random.default_rng(seed)
    cells = _random_connected_cells(rng)
    root = min(cells)
    tree = spanning_tree(cells, root, strategy, rng)
    assert len(tree) == len(cells) - 1
    uf = UnionFind(cells)
    for a, b in tree:
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        assert uf.union(a, b), "tree contains a cycle"


@pytest.mark.parametrize("strategy", TREE_STRATEGIES)
@pytest.mark.parametrize("seed", range(10))
def test_stc_tour_visits_every_subcell_exactly_once(strategy, seed):
    rng = np.random.default_rng(seed)
    cells = _random_connected_cells(rng)
    root = sorted(cells)[len(cells) // 2]
    tour, _ = plan_region(cells, root, strategy, rng)
    assert tour[0] == tour[-1] == (2 * root[0], 2 * root[1])
    assert len(tour) == 4 * len(cells) + 1
    body = tour[:-1]
    assert len(set(body)) == len(body)  # no fine cell visited twice
    expected = {(2 * r + dr, 2 * c + dc) for r, c in cells for dr in (0, 1) for dc in (0, 1)}
    assert set(body) == expected
    for a, b in zip(tour[:-1], tour[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1  # 4-connected moves


def test_single_cell_tour():
    assert stc_tour({(3, 4)}, [], (3, 4)) == [(6, 8), (6, 9), (7, 9), (7, 8), (6, 8)]


def test_turn_count():
    square = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
    assert count_turns(square) == 4
    straight = [(0, 0), (0, 1), (0, 2)]
    assert count_turns(straight, closed=False) == 0
    uturn = [(0, 0), (0, 1), (0, 0)]
    assert count_turns(uturn, closed=False) == 2


def test_comb_tree_reduces_turns_on_open_rectangle():
    cells = {(r, c) for r in range(4) for c in range(12)}
    # A wide rectangle: long horizontal teeth need far fewer turns than
    # many short vertical ones; "adaptive" must pick the better comb.
    t_h = count_turns(plan_region(cells, (0, 0), "horizontal")[0])
    t_v = count_turns(plan_region(cells, (0, 0), "vertical")[0])
    t_a = count_turns(plan_region(cells, (0, 0), "adaptive")[0])
    assert t_h < t_v and t_a == min(t_h, t_v)


# -------------------------------------------------------------- metrics ----
def test_conflict_counting():
    a = [(0, 0), (0, 1), (0, 2)]
    b = [(0, 2), (0, 1), (0, 0)]  # meet at (0, 1) at t=1
    assert count_conflicts([a, b]) == (1, 0)
    c = [(0, 0), (0, 1)]
    d = [(0, 1), (0, 0)]  # swap
    assert count_conflicts([c, d]) == (0, 1)


def test_evaluate_disjoint_tours():
    g = gen_open(2, 4)
    cm = build_coarse(g)
    t1 = stc_tour({(0, 0), (1, 0)}, [((0, 0), (1, 0))], (0, 0))
    t2 = plan_region({(r, c) for r in range(2) for c in range(1, 4)}, (0, 1), "bfs")[0]
    m = evaluate(cm.fine, [t1, t2])
    assert m["coverage"] == 1.0 and m["overlap"] == 0.0
    assert m["makespan"] == 24 and m["total_length"] == 32
    assert m["vertex_conflicts"] == 0 and m["invalid_cells"] == 0

