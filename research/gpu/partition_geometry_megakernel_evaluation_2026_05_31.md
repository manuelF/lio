# Partition geometry & single-launch "mega-kernel" evaluation — heme (2026-05-31)

**Status: EVALUATED — partitioning restructuring REJECTED as a wall lever on this
hardware; g2g *kernel* work is defensible only on slow GPUs / large systems.**

Question posed: is the cube/sphere grid-partitioning structure in
`g2g/regenerate_partition.cpp` now blocking GPU performance? Two concrete
proposals were raised:

1. **Bigger groups** — coarser cubes so each GPU launch carries more work and
   the per-group fixed costs are amortized over fewer, larger launches.
2. **Single mega-launch** — collect all groups into one array and process them
   in a single kernel, with the block/thread mapping absorbing the per-launch
   cost.

Short answer: **on the RTX 3080 Ti, heme is overlap-bound and the entire g2g
path has a hard ~4 % wall ceiling. The GPU is already ~100 % busy during g2g, so
there is no starvation for "more work" to fill, and the per-launch overhead the
mega-kernel would remove lives in a 0.5 % slice.** Neither proposal can move the
wall here, and both perturb the float32 reduction order that heme's SCF
trajectory is documented to be pathologically sensitive to. The honest
conclusion is that heme-on-3080-Ti is near its floor; the exposed levers that
remain are CPU/BLAS (diagonalization 28 %, DIIS/base-change setup 21 %), not g2g
geometry.

The one caveat the user correctly anticipated ("don't overfit to recent GPUs")
is real and is treated explicitly in §4: on a GTX-1080-class GPU or much larger
systems, g2g stops being overlap-hidden and *does* become the binding pole — but
even there the lever is **density-kernel efficiency, not partition geometry.**

---

## 1. Measured baseline (heme.in, restart, open-shell NUNP=4, 72 iters, 18.70 s)

`LIO_OVERLAP_INT3LU_G2G=1`, hybrid `cuda=1 cpu=1`, fgm=-1 caching.

Per-iteration overlap section (steady state):

```
[overlap] int3lu = 84.1 ms   g2g = 94.5 ms   idle = 10.3 ms
```

The two halves run as concurrent OpenMP sections; the section wall ≈ max(int3lu,
g2g) ≈ **94.5 ms ≈ g2g**. int3lu finishes ~10 ms early and idles. Over 72 iters
this is the `Coulomb fit + Fock` bucket = 6.88 s. Full iteration breakdown:

| Iteration sub-cost | Time | % of Iteration | Exposed? |
|---|---|---|---|
| Fock integrals (int3lu ∥ g2g overlap) | 6.91 s | 40.9 % | only the ~10 ms/iter g2g overhang |
| **SCF - Fock Diagonalization** | 4.80 s | 28.4 % | **fully** |
| **SCF acceleration setup** (BChange 1.80 + DIIS commut 1.05 + update_emat 0.69) | 3.58 s | 21.2 % | **fully** |
| SCF - MOC base change | 0.34 s | 2.0 % | fully |
| SCF acceleration | 0.50 s | 2.9 % | fully |

### The ceiling

Because g2g is overlapped with int3lu and beats it by only 10.4 ms
(94.5 − 84.1, matching the 10.3 ms idle exactly), **the maximum wall any g2g
optimization can return is ~10 ms/iter** until int3lu (fixed CPU work) becomes
the pole. That is **0.74 s over 72 iters = 4.0 % of total runtime.** Every
microsecond shaved off g2g *below* 84 ms/iter is invisible — it just widens the
int3lu idle.

This is the number that governs everything below: partitioning lives entirely
inside g2g, so its wall budget is ≤4 %.

---

## 2. The GPU is already ~100 % busy during g2g (refutes "feed it more work")

nsys on a short run (`nmax=3`) gives a clean kernel census. Decoding the
instance counts (fgm caching makes `compute_functions`/`compute_weights` fire
once per group, so their count = group count):

- `gpu_compute_functions` = **187 instances ⇒ 187 GPU groups.**
- `gpu_compute_density_opened<float,0,0>` (big, group_m>128) = 896 = **112 × 8**
- `gpu_compute_density_opened<float,0,1>` (small, group_m≤128) = 600 = **75 × 8**
- 112 + 75 = **187** ✓  → 8 `solve()` calls captured (7 with RMM + 1 energy-only)
- `gpu_scatter_rmm_open` = 1309 = **187 × 7** ✓ (RMM solves only)
- `gpu_gather_rdm_open` = 1496 = **187 × 8** ✓ (every solve)

Total GPU kernel time in the capture ≈ **0.764 s**. Divided by the 8 solve
calls = **95 ms/iter — which matches the measured 94.5 ms g2g wall.** Summed
kernel time ≈ section wall means **inter-launch gaps are negligible: the GPU is
saturated during the g2g phase.**

The "array of groups in one kernel" proposal is, at root, a *utilization*
argument — fill a starved GPU. The measurement says the GPU is not starved.
There is no idle GPU time for a mega-kernel to reclaim. Its only remaining
benefit would be removing per-launch overhead, which §3 shows is a rounding
error.

### Kernel time distribution

| Kernel | % of GPU | Avg | Med | Max | Notes |
|---|---|---|---|---|---|
| `density_opened<0,0>` (big) | **81.9 %** | 698 µs | 475 µs | **9.38 ms** | stddev 1.07 ms > mean — a few giant launches dominate |
| cuBLAS sgemm (∑ RMM-update GEMMs) | ~7.7 % | — | — | — | `gpu_update_rmm_cublas` path |
| `gpu_compute_weights` (1-time) | 2.6 % | — | — | — | cached |
| `transpose` (1-time) | ~1.6 % | — | — | — | cached |
| `density_opened<0,1>` (small) | **0.5 %** | 6.7 µs | — | — | 600 launches, launch-overhead-bound |
| gather/scatter/accumulate (open) | ~2.2 % | 2–4 µs | — | — | per-group glue |

---

## 3. Proposal-by-proposal

### 3a. Bigger groups (coarser cubes)

**Rejected.** Three independent reasons:

1. **No throughput headroom to capture.** The big-group launches that own 82 %
   of GPU time already run 475 µs–9.4 ms and already saturate the device
   (prior profiling: 48 % occupancy / 72 % SM throughput, register-limited at
   ~24 active warps/SM — see `density_opened_profile_2026_05_25`). A saturated,
   reg-bound kernel does not get faster with more work per launch; it just runs
   longer per launch. Merging only reduces the *count* of launches.

2. **The work itself grows superlinearly.** `PointGroup::cost()` =
   `10·npts·group_m·(1+group_m)/2`. Density is O(npts·group_m²). A bigger cube
   means a bigger bounding box ⇒ more basis functions pass the significativity
   test ⇒ larger group_m ⇒ **more than proportional** FLOPs. The cube geometry
   exists precisely to *bound* group_m. Coarsening it trades a negligible
   launch-count saving for a quadratic work increase.

3. **It perturbs the SCF trajectory (predicted, high-confidence).** A larger box
   admits more *marginally* significative functions — ones whose contribution at
   a given point is near-zero but nonzero in float32. Each adds a tiny term to
   the per-point density reduction `ρ = Σ_i Σ_j P_ij f_i f_j`, changing the
   float32 rounding. heme's SCF is documented as Lyapunov-divergent in iteration
   count under *any* ulp-level Fock perturbation (thread count, FMA emission,
   DIIS solver choice — see `heme_dgels_rank_deficiency_2026_05_28` and
   `heme_diis_stability_dead_ends_2026_05_28`). The mechanism here is the same
   class of perturbation, so the expected outcome is an iteration-count change.
   At ~235 ms/iter, **a single extra iteration (+0.4 % of 72) costs more wall
   than the theoretical ceiling of the entire optimization (4 %) divides across
   ~9 iters.** This is a predicted risk by analogy, not measured for geometry
   specifically — but it is asymmetric: the upside is ≤4 %, the downside is one
   bad geometry away from +10–90 iters.

### 3b. Single mega-launch over an array of groups

**Rejected.** It targets the wrong cost.

- The overhead it removes (per-launch ~5–10 µs) is **<2 % of each big-group
  launch** (698 µs avg). For the 82 % slice it is noise.
- The only place launch overhead is comparable to kernel time is the *small*
  groups: `density_opened<0,1>`, 6.7 µs kernel vs ~5 µs launch. But that whole
  class is **0.5 % of GPU time.** Eliminating *all* of its launch overhead saves
  <0.3 % of GPU time ⇒ <0.012 % of wall.
- A mega-kernel that handles heterogeneous group_m (here: 64 small + 112 big,
  group_m spanning ≤128 to several hundred) needs either a worst-case-sized
  block (kills occupancy for the small groups) or a persistent-kernel
  work-queue with dynamic group fetch (hundreds of LOC of fragile state,
  divergent control flow across groups in a warp). CUDA Graphs were the
  lighter-weight version of this same idea and were already tried and
  **rejected** (`path4_cuda_graphs_2026_05_20`: net-neutral wall, +300 LOC of
  capture/replay state). Multi-stream concurrent group execution was also
  **rejected** (`multi_stream_gpu_workers_2026_05_21`: 3× slower from host-side
  driver/cuBLAS/OMP contention).
- It also re-orders the global Fock `atomicAdd` accumulation across groups →
  same float32-trajectory risk as 3a.

### 3c. Geometry change (cubes/spheres → something else)

**Rejected as a perf lever.** The cube/sphere split is a *work-minimization and
load-balancing* structure (minimize total Σ npts·group_m² while keeping per-group
function sets local). It is not a throughput knob. Any alternative tessellation
changes the per-group function membership and therefore the float32 reduction
order — same convergence risk, for a sub-4 % theoretical reward that the
saturation measurement says is mostly unreachable anyway.

The one genuine inefficiency the data *does* show is the **9.38 ms max vs 475 µs
median** spread: a handful of giant-group_m groups serialize on the single GPU
worker thread and form a tail. The fix for that is *splitting* the giant groups
(the opposite of proposal 1), not merging — but that also reorders FP, and the
tail is a small fraction of the 82 % slice. Not worth the convergence risk for
≤4 %.

---

## 4. Hardware/size scoping (the "don't overfit to recent GPUs" caveat)

The 4 % ceiling is **specific to this GPU and this system size.** g2g (94.5 ms)
sits only **12 % above** int3lu (84.1 ms) — a razor-thin margin that flips on
slower hardware or larger problems:

- **GTX 1080 (≈3.7× less FP32, still supported):** density is the FP32-bound
  kernel; g2g scales ~linearly with GPU throughput ⇒ g2g ≈ 300–350 ms/iter while
  int3lu (fixed CPU work) stays ~84 ms. g2g becomes the **dominant exposed
  pole**, not a sliver. The overlap stops hiding it.
- **Much larger systems:** density is O(npts·group_m²); group_m and npts both
  grow with system size, so g2g outgrows int3lu's scaling and again dominates.

**So the user's instinct is not wrong in general — it is wrong for this
benchmark.** The correct split:

- **g2g *kernel* efficiency** (density_opened register pressure, the GEMM-vs-hand
  RMM path, the giant-group tail) is a **defensible target on slow GPUs and big
  systems**, where it is the binding cost.
- **Partition *geometry/launch* restructuring** (the three proposals above) is
  **weak everywhere**: a saturated GPU doesn't speed up with bigger or fused
  groups, it only changes FP-reduction order. The geometry is not the
  bottleneck on any hardware — the density kernel's arithmetic is.

If future work pursues g2g for the slow-GPU/large-system regime, the prior art
already maps the dead ends: TF32/tensor cores (`tensor_core_feasibility_2026_05_31`),
point-batching (`density_opened_point_batch_dead_end_2026_05_25`), float2 α/β
tex fusion (done), single-pointer template (done). The remaining unexplored
density lever is reducing the register footprint further to lift occupancy past
the 50 % reg wall — independent of partitioning.

---

## 5. What is actually exposed on heme-3080-Ti (for completeness, not g2g)

The wall is now CPU/BLAS-bound inside the iteration:

- **Fock Diagonalization 4.80 s (28 %)** — documented BASIS-LOCKED for heme;
  dsyevr-full and cuSOLVER both perturb the near-degenerate Fe-d subspace and
  2× the iteration count (`heme_scf_hotpath_exhausted_2026_05_29`).
- **SCF acceleration setup 3.58 s (21 %)** — BChange AO→ON 1.80 s, DIIS commut
  1.05 s, update_emat 0.69 s. update_emat already got the stride-1 mirror
  (`heme_dgels_rank_deficiency_2026_05_28`, −22 %). BChange and commut are DGEMM
  base-changes; whether they can be fused or kept resident across iters is the
  open question, but it is a Fortran/BLAS lever, not g2g.

These are larger and fully exposed, but the memory record shows the SCF math
path is itself largely constrained by the same heme convergence sensitivity.
**heme on this box is genuinely close to its floor.** The most useful next move
is to benchmark any candidate against the *median iteration count across
OMP∈{1,4,6,8}* (~100 iters), not the lucky-default 72, so a "speedup" that
merely got a favorable DIIS dice-roll is not mistaken for real progress.

---

## Verdict

| Proposal | Wall ceiling (3080 Ti) | Convergence risk | Verdict |
|---|---|---|---|
| Bigger groups | ≤4 %, mostly unreachable (GPU saturated) | high (predicted) | **Reject** |
| Single mega-launch | <0.02 % (overhead is 0.5 % slice) | medium | **Reject** |
| Geometry change | ≤4 %, unreachable | high | **Reject** |
| g2g density-kernel efficiency | ~4 % here; **dominant on GTX 1080 / large systems** | low if FP-order preserved | **Defer to slow-HW/large-system regime** |

The partitioning structure is **not** the blocker. On this GPU the blocker is
that g2g is already overlap-hidden under int3lu, and the GPU is already
saturated during the part that isn't hidden.
