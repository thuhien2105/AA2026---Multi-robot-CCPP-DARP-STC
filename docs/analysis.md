# Algorithm analysis notes (Topic 11)

These notes cover the six mandatory analysis points of the assignment
(Section 3.2) for the algorithms implemented in `mcpp/`. They are a starting
point for the report: **every team member must be able to re-derive and
explain the parts they present** (assignment Section 5.4). Statements marked
*(cite)* come from the literature and should be checked against the original
papers before they are quoted.

---

## 1. Formal problem definition and hardness

**Input.**
* A 4-connected occupancy grid. Its fine cells have the size of the robot
  footprint, and each coarse cell is a 2x2 block of fine cells.
* The coarse grid graph `G = (V, E)`. Its vertices are the coarse cells
  whose four sub-cells are all free (we keep the largest connected
  component), and its edges join 4-adjacent coarse cells. `n = |V|`.
* `k` robots with distinct start cells `r_1, ..., r_k` in `V`, and optional
  speeds `v_i > 0` (default `v_i = 1`).

**Output.** For each robot a closed walk `P_i` on the fine grid that starts
and ends at the top-left sub-cell of `r_i`. Together the walks must visit
every sub-cell of every `v` in `V`.

**Objective.** Minimise the makespan `max_i |P_i| / v_i`, where `|P_i|` is
the number of unit moves. Secondary metrics are the total length, the number
of turns, overlap and conflicts.

**Two restricted versions, both solved by STC on each robot's cell set:**

| version | decision variables | constraint | STC makespan |
|---|---|---|---|
| min-max rooted connected **partition** (MMRCP) | `V = V_1 ⊔ ... ⊔ V_k` | `V_i` connected, `r_i ∈ V_i` | `4 · max_i |V_i| / v_i` |
| min-max rooted **tree cover** (MMRTC) | trees `T_i ⊆ G`, `∪ V(T_i) = V` | `r_i ∈ T_i` (overlap allowed) | `4 · max_i |V(T_i)| / v_i` |

The partition version is exactly what DARP, Voronoi, k-means and our local
search optimise. The tree-cover version relaxes it: every partition is a
tree cover. So `OPT_MMRTC ≤ OPT_MMRCP`, and the MIP experiment confirms
that the inequality can be strict.

**Lower bound used throughout.** A closed walk of length `L` visits at most
`L` distinct cells, and there are `4n` fine cells to visit. Hence
`makespan ≥ 4n / Σ v_i`. With unit speeds and STC tours (length a multiple
of 4) this becomes `LB = 4 · ceil(n / k)`.

**Hardness.**
* *Min-max tree cover is NP-hard* (cite: Even et al., ORL 2004). It is
  NP-hard already for `k = 2`, by a reduction from PARTITION. Sketch: given
  integers `a_1..a_m` with sum `2S`, build a hub `h` joined to both roots by
  zero-weight edges and attach one leaf per item by an edge of weight `a_j`.
  Each leaf edge must belong to some tree, so a cover with
  `max_i w(T_i) ≤ S` exists iff the items can be split into two halves of sum
  `S`. Because PARTITION is only *weakly* NP-hard and the construction needs
  edge weights, this sketch does not by itself cover our unit-weight grids
  (replacing an edge of weight `a_j` by a path of `a_j` cells is only a
  pseudo-polynomial blow-up). Hardness on unit-weight graphs follows from the
  balanced connected partition result below.
* *Balanced connected partition of graphs is NP-hard* (cite: Dyer & Frieze,
  1985, for partitioning planar bipartite graphs into equal connected parts).
  MMRCP with roots contains it as the decision question "is
  `max |V_i| = n/k` achievable?".
* Single-robot *milling / lawn-mowing with holes* is NP-hard on general grid
  polygons (cite: Arkin, Fekete & Mitchell 2000). In our 2x2 STC model with
  one robot the problem becomes trivial (STC is optimal: `4n` moves visit
  `4n` cells). The difficulty of the model is concentrated in the *partition
  / cover* step.

---

## 2. Correctness, guarantees and pseudocode

### 2.1 STC tour by cycle merging (`mcpp/stc.py`)

```
STC(V_i, root):
  T <- spanning tree of G[V_i]          # BFS / DFS / Kruskal(weights)
  F <- one 4-cycle (TL-TR-BR-BL) per cell of V_i
  for each tree edge (a, b) of T:       # a left of / above b
      delete the side of a facing b and the side of b facing a
      add the two bridging edges (TR_a-TL_b, BR_a-BL_b)  (or vertical analogue)
  return the cycle of F traversed from TL(root)
```

**Invariant.** After `j` merges, `F` is a disjoint union of `|V_i| - j`
cycles, one per component of the partially built forest. *Proof:* removing
one edge from each of two disjoint cycles and reconnecting the four endpoints
crosswise yields exactly one cycle. Each side of a cell faces at most one
neighbour, so no edge is deleted twice. The two cycles being merged are
different because `T` is acyclic, so the endpoints of a tree edge always lie
in different components. After `|V_i| - 1` merges, `F` is a single
Hamiltonian cycle of the `4|V_i|` sub-cells.

**Guarantees.** STC is complete and gives a closed tour of exactly
`4|V_i|` moves, visiting each sub-cell exactly once. This is optimal for the
cell set `V_i`. *Tour vs tree:* with one coarse step = 2 fine moves, the tree
weight is `w(T) = 2(|V_i| - 1)` fine units, and the tour length is
`4|V_i| = 2 w(T) + 4`. This is the classical "tour ≤ 2 × tree" bound (Euler
tour of the doubled tree), up to the additive constant from going round the
root cell. The tree *shape* does not change the length, but it does change
the number of turns. That is why the horizontal / vertical / adaptive comb
trees are an ablation target.

**Loss with partially occupied coarse cells.** A coarse cell with one
blocked sub-cell is unusable, so its free sub-cells are never covered.
`results/REPORT.md` §5 quantifies this loss on the MovingAI maps when the map
resolution equals the robot footprint.

### 2.2 DARP (`mcpp/darp.py`)

```
DARP(G, starts, c, nu, rho, max_iter, max_thr):
  E_i(v) <- dist(r_i, v) + 1                      # Euclidean or BFS
  thr <- 0 if n/k integer else 1
  while thr <= max_thr:
    repeat max_iter times:
      A(v) <- argmin_i E_i(v);  A(r_i) <- i
      k_i <- |A^{-1}(i)|;  dev_i <- k_i - f_i
      for i: if A^{-1}(i) disconnected: C_i <- connectivity matrix (Sec. 2.2)
      if all regions connected and max_i |dev_i| <= thr: return A   # success
      E_i <- E_i * (1 + c dev_i / f_i) * C_i * U(1 - rho, 1 + rho)
    thr <- thr + 1
  return best connected assignment seen (else repair(A))          # failure
```

* **Invariant.** `A` is always a total assignment: every usable cell has
  exactly one owner, and every robot owns its start. The returned partition
  is always valid. On success this holds by the stopping test. On failure,
  either a connected assignment seen earlier is returned, or `repair`
  (proved below) is applied.
* **Stopping condition.** Success, or the iteration budget
  `(max_thr - thr_0 + 1) · max_iter` is exhausted. The algorithm therefore
  always terminates.
* **Guarantee type.** DARP is a heuristic. It has *no convergence guarantee
  and no approximation guarantee*. Without the connectivity and noise terms,
  `A` is a multiplicatively weighted Voronoi diagram. The update is a
  multiplicative-weights / dual-ascent step on the weights. Because sizes
  are integers and boundaries move in discrete jumps, the iteration can
  oscillate. In maze-like maps a region can also "jump" over a neighbour
  along a corridor and become disconnected. Our tuning experiments show both
  effects; the maze map has the lowest success rate.
* **Convergence conditions** (empirical, see the ablation). Small `c`
  converges more reliably but slowly. Large `c` oscillates. Halving `c` on
  stagnation did *not* help. The connectivity term matters most when the
  distance is Euclidean, because it ignores walls.

### 2.3 Connectivity repair (`mcpp/partition.py::repair_connectivity`)

Keep the component of each region that contains its start, then grow all
kept regions simultaneously by multi-source BFS. *Validity:* the kept parts
are connected. Every re-assigned cell is attached to an already connected
region through its BFS parent, which has the same label. Every cell is
reached because `G` is connected.

### 2.4 Min-max local search (extension, `mcpp/local_search.py`)

* **Move.** A transfer chain `L = r_0 → r_1 → ... → r_m = t` along the
  region-adjacency graph. Each `r_j` hands one boundary cell to `r_{j+1}`.
  The cell must not be `r_j`'s start and must not be an articulation point
  (Tarjan), and connectivity is re-verified at execution time.
* **Invariant.** The partition stays valid and the maximum (speed-weighted)
  region size never increases.
* **Termination** (unit speeds). A chain is only executed if
  `s_t ≤ s_L - 2`. Then the potential `Φ = Σ_i s_i²` changes by
  `2(s_t - s_L) + 2 ≤ -2`. Since `0 ≤ Φ ≤ n²`, there are at most `n²/2`
  chains.
* **Guarantee.** The result is a local optimum w.r.t. transfer chains: no
  chain from a maximum region to a region that can absorb a cell exists. No
  global guarantee.

### 2.5 Min-max rooted tree cover, 4-approximation (`mcpp/mmrtc.py`)

```
feasible(B):                              # Even et al. 2004, rooted version
  F <- MST of G with all roots contracted (unit weights: multi-source BFS forest)
  if w(F) = n - k > k B: return FAIL        # certifies B < OPT
  cut each tree of F bottom-up into pieces of weight in [B, 2B) + residual < B at the root
  bipartite graph: piece P -- root r  iff  dist(r, P) <= B
  if no matching saturates all pieces: return FAIL   # certifies B < OPT (Hall)
  robot i <- residual_i ∪ path(r_i, matched piece) ∪ matched piece      # weight < 4B
binary search smallest B with feasible(B) and FAIL(B - 1)
```

* **Piece weights.** When a group is closed, its weight before the last
  child was `< B`. The child's residual is `< B`, so adding it with its edge
  gives `< B + B`. Hence pieces have weight in `[B, 2B)`, and at most
  `(n - k)/B ≤ k` of them exist.
* **Approximation.** Robot `i`'s tree has weight
  `< B + B + 2B = 4B`. The binary search returns a `B` such that `B - 1` was
  certified infeasible, so `B - 1 < OPT` and therefore `B ≤ OPT`. This
  argument does not need monotonicity of the test. In tour terms, our tour
  is `4(w+1) ≤ 16B`, while every tree-cover+STC solution has makespan
  `≥ 4(OPT+1) ≥ 4(B+1)`. The ratio is `< 4`, and `4(B+1)` is a *certified
  lower bound* computed for free.
* We keep the best cover produced by any feasible guess. It is never worse
  than the final one, so the guarantee is preserved. Deduplication only
  removes vertices from trees, so it never increases any tree.

### 2.6 Exact MIP (`mcpp/mip.py`)

Single-commodity flow per robot. The root `r_i` is the only source. Every
owned vertex consumes one unit, and arcs carry robot-`i` flow only between
vertices owned by robot `i` (capacity `(n-1)x`). Hence every owned vertex is
connected to `r_i` inside the region. Conversely, every connected rooted
region admits such a flow (along a spanning tree). The model is exact. The
`= 1` coverage rows give MMRCP; `≥ 1` gives MMRTC with unit vertex cost. The
model is validated against brute-force enumeration in
`tests/test_algorithms.py`.

---

## 3. Time and memory complexity

`n` = coarse cells, `k` = robots, `I` = DARP iterations, `C` = local-search
chains.

| component | time | memory | data structures |
|---|---|---|---|
| coarse map + largest component | O(N) (fine cells) | O(N) | `scipy.ndimage.label` (two-pass union-find) |
| geodesic Voronoi | O(n) | O(n) | FIFO queue (multi-source BFS) |
| k-means | O(it · k · n) | O(k n) | dense distance matrix |
| STC tree: BFS / DFS | O(n) | O(n) | queue / stack, hash set |
| STC tree: Kruskal (comb / random) | O(n log n) | O(n) | sort + union-find (rank, path halving, α(n)) |
| STC tour (cycle merging) | O(n) | O(n) | hash map adjacency, degree 2 |
| DARP, one iteration | O(k n) | O(k n) | k evaluation matrices; k CC labellings; ≤ k exact EDTs (linear-time, Maurer et al.) |
| DARP, total | O(I k n), `I ≤ (max_thr+1) · max_iter` | O(k n) | |
| local search, one chain | O(k² + Σ_j n_{r_j}) ≤ O(k n) | O(n) | Tarjan articulation points (iterative DFS), cached per region |
| local search, total | O(C k n), `C ≤ n²/2` | O(n) | |
| MMRTC preprocessing | O(k n) | O(k n) | k BFS distance arrays + parent maps |
| MMRTC, one feasibility test | O(k n + k³) | O(n) | post-order cut, Kuhn matching on ≤ k pieces |
| MMRTC total | O(k n log n) | O(k n) | binary search over B ∈ [n/k, n] |
| dedup | O(d · n) | O(n) | Tarjan per removal, `d` = duplicated cells |
| MIP | exponential worst case; ≈ 5nk variables, ≈ (4k+1)n + 8kn rows | | HiGHS branch & cut |
| conflict counting | O(T k) | O(k) | hash map per time step |

## 4. Empirical verification

`experiments/run_scaling.py` measures runtime for `n` from 64 to 16384,
fits log–log slopes and profiles the DARP+LS+STC pipeline. Results are in
`results/REPORT.md` §2, `results/figures/scaling_loglog.png` and
`results/scaling/profile.txt`. Compare each fitted slope with the theoretical
exponent in the table above. For DARP, separate the per-iteration cost
(theory: linear) from the number of iterations (no theory; depends on the
map).

## 5. Design choices and ablation

`experiments/run_ablation.py` studies the following on a separate tuning
set:
* DARP learning rate `c`
* adaptive step size
* Euclidean vs geodesic distance
* the connectivity term (on/off and amplitude `ν`)
* the noise level
* the STC spanning-tree strategy (turns)
* heterogeneous speeds (speed-proportional shares, speed-aware LS)

Results are in `results/REPORT.md` §3.

## 6. Optimality gap

`experiments/run_optgap.py` solves both MIPs exactly on 90 small instances
(36–64 cells, k = 2..4). It reports the gap of every heuristic against the
partition optimum (MIP-P) and the tree-cover optimum (MIP-C). Results are in
`results/REPORT.md` §4.
