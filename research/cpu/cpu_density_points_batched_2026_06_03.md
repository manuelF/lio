# CPU GGA density: points-batched / points-vectorized rewrite — DONE 2026-06-03

**Status:** DONE — shipped (unconditional, no toggle).
**Impact:** HIGH for CPU-only builds. fosfatoQMMM g2g **533 ms → ~250 ms/iter (−53%)**,
total wall **22.9 s → 15.4 s (−33%)** on 5800X3D (15 threads, `LIO_OVERLAP_INT3LU_G2G=1`).

## Context

CPU-only build (`make cuda=0 cpu=1`), fosfato QM/MM closed-shell, 27 iters.
Run: `OMP_NUM_THREADS=15 LIO_OVERLAP_INT3LU_G2G=1 liosolo -i fos.in -c fos.xyz -b basis -v`.

In a CPU-only build there are no GPU groups, so the entire g2g XC integration runs as
`PointGroupCPU::solve_closed/opened`. Under the int3lu/g2g overlap, the int3lu section
(73 ms) hides trivially under g2g (533 ms) — g2g is the long pole and the int3lu thread
sits idle ~460 ms/iter. The overlap's OMP cap (`phys*3/4 = 6`) does **not** bind: the
`Partition::solve` outer loop uses `num_threads(cpu_threads+gpu_threads)` explicitly,
where `cpu_threads = max_threads` (=15) captured at `g2g_init`. So g2g already runs on
15 outer threads; a thread sweep (6→15 cap) confirmed it (533→518 ms, noise). **Thread
allocation was not the lever.**

## Root cause (perf, `--call-graph dwarf`)

`cpu_compute_density_gga` = **60.7 %** of total CPU samples. `compute_functions` only 5.9 %.
The kernel's inner loop is a **float reduction** (`w += fv[j]*rmm`). The g2g CPU build is
`-O3 -march=native` **without `-ffast-math`/`-fassociative-math`**, so GCC cannot reassociate
a float reduction and the loop runs **scalar** — one FMA/cycle while AVX2+FMA could do 8-wide.
The existing `#pragma GCC ivdep` only waives memory-dependence, not reduction reassociation.

## Fix

Reformulate so the inner loop runs over the independent **points** dimension (an axpy, not a
reduction). `cpu_compute_density_gga_batch` (in `g2g/cpu/cpu_kernels.h`) computes pd + the 9
gradient/hessian terms for a whole group at once, vectorized over points. Each point's `w_k`
vectors still accumulate over `j` in order `0..i`, and outputs over `i` in order `0..m` — so
SIMD only runs distinct points in distinct lanes; it never reorders a single point's sum.

**Cache discipline is the whole game.** The first attempt transposed *all* points
(`10·m·np ≈ 300 KB`) and streamed it `m` times → blew L1 → **3× slower** (1500 ms). The
shipped version **tiles points in blocks of B=8** (one AVX-256 float vector): per-tile
function data (`10·m·8`) stays L1-resident and `rmm` is the hot stationary operand reused
across all `i`. This keeps the kernel compute-bound (as the scalar version was) while
exposing SIMD. `solve_opened` calls it twice (α, β).

Per-group reused scratch buffers (`density_scratch_a/b` on `PointGroupCPU`) avoid per-iteration
heap churn (matters under BOMD/geom-opt; wall-neutral on a single SCF).

## Correctness / convergence

**Not bit-exact** — FMA-contraction codegen freedom in the per-`i` output expressions
(`igx*w + Wgx*Fi`: compiler picks which product to fuse, differently in scalar vs vector
form) shifts results ~ulp. This is the FP-order class the project knows well (see
convergence notes): energy/basin stable, iter count is the noisy signal.

- fosfato: 27 iters, Total energy −2341.457194 vs `.ok` −2341.457156 (Δ 3.8e-5, **tol 1.5e-4**). Energy/Forces/Mulliken/Dipole all PASS. (Closer to `.ok` than the prior code, which gave 26 iters.)
- Fe3H2O6 (open-shell, restart): Energy PASS, −1610.6073609 vs `.ok` −1610.6073625 (Δ 1.6 µHa); 9 vs 10 iters (basin correct — the expected, acceptable iter shift).
- Full `run_travis_test.py` CPU subset: **all pass**.

## After (perf re-profile)

Density `27.6 %` (was 60.7), `compute_functions`+`cpu_eval_gto_shell` ~20 %, OMP barriers ~11 %,
memset/page-zero ~9 % (mostly `compute_functions` array `.zero()` each iter), transpose 2 %.

## Follow-on: function-value caching in CPU-only mode — DONE 2026-06-03

After the density rewrite, re-profile showed `compute_functions`+`cpu_eval_gto_shell` ≈ 20 %
(recomputed every SCF iter though geometry is fixed) + ~9 % memset/page-faults from the
per-iter array alloc/zero it drives. **Bit-exact** lever (function values are geometry-only).

The caching machinery already existed (`inGlobal` flag → skip recompute in `compute_functions`;
`Partition::compute_functions` precompute pass in `init.cpp:369`; skip-dealloc in `solve_*`) but
was gated on `GPU_KERNELS`, so **only hybrid builds cached their CPU groups** — CPU-only
(`!GPU_KERNELS`) recomputed+deallocated every iter. Since `CPU_RECOMPUTE=0` by default in *both*
builds, the fix is two guard edits making caching depend on `CPU_RECOMPUTE` alone:
- `g2g/cpu/functions.cpp`: `#if !CPU_RECOMPUTE && GPU_KERNELS` → `#if !CPU_RECOMPUTE`
- `g2g/cpu/iteration.cpp` (both dealloc blocks): `#if CPU_RECOMPUTE or !GPU_KERNELS` → `#if CPU_RECOMPUTE`

This makes CPU-only behave exactly like the already-validated CPU-groups-in-hybrid path. The two
edits are a **provable no-op for every `(CPU_RECOMPUTE, GPU_KERNELS)` combination except `(0,0)`**
(CPU-only), so hybrid is byte-identical and needs no re-test.

**Result:** g2g per-iter **246 → ~120 ms (−51 %)**, total wall **15.3 → 11.5 s (−25 %)**, 27 iters.
Cumulative with the density rewrite: **g2g 533 → 120 ms (−77 %), wall 22.9 → 11.5 s (−50 %)**.

**Bit-exactness proven** (not just tolerance): at `OMP_NUM_THREADS=1` (no `rebalance`/overlap FP-order
noise) cached and `cpu_recompute=1` reference both give exactly **−2148.6508600** (26 iters). The
OMP=15 run-to-run wobble (~1 µHa) is the pre-existing timing-dependent `rebalance()` reordering,
independent of caching.

**Invalidation is structurally safe:** `g2g_reload_atom_positions_` (called at the top of *every*
SCF and TD entry — `SCF.f90:259`, `TD.f90:571`) calls `compute_new_grid` **unconditionally** →
`regenerate_partition` → `clear()` deletes all `PointGroup` objects → fresh `new PointGroupCPU`
(`inGlobal=false`). A `PointGroupCPU` object cannot survive a grid reload, so a stale cache across
geometries is impossible. Validated empirically: a 5-step open-shell **Ehrenfest** run (nuclei
moving, grid reloaded each step) converges cleanly with finite/sane energies at every geometry.

**Memory:** caching keeps all groups' function arrays resident simultaneously (vs. ~15 concurrent).
For fosfato that's ~1.3 GB (≈330k pts × ~80 funcs × 11 arrays × 4 B) against 49 GB free — a non-issue.
No CPU memory-budget system added (RAM headroom is 30×). For a system large enough to matter, the
fallback is a `cpu_recompute=1` build.

Validation: fosfato (Energy/Forces/Mulliken/Dipole PASS), Fe3H2O6 open-shell PASS, full
`run_travis_test.py` CPU subset PASS, 5-step Ehrenfest multi-geometry PASS.

## Follow-on: default CPU thread count — DONE 2026-06-03

`g2g/init.cpp` sized `cpu_threads` from `max_threads`, which **defaulted to 1 unless
`OMP_NUM_THREADS` was set** (`int max_threads = 1; if (getenv("OMP_NUM_THREADS")) max_threads
= omp_get_max_threads();`). So out of the box g2g ran single-threaded on CPU-only (185% CPU,
31 s vs 12 s) and `cpu_threads = max_threads - gpu_threads` went to **0** on hybrid builds.

Fix: honor `OMP_NUM_THREADS` when set; otherwise default to `detect_physical_cores()` (not
all logical CPUs). Per-iter on 8C/16T Zen3: 8 threads = 432 ms, 16 = 464 ms, 6 = 462 ms — the
XC kernels are FP/cache-bound and SMT oversubscription costs ~7%, so physical-core count is the
efficient default. `detect_physical_cores()` falls back to the logical count, so it's never
worse than the libgomp default. Affects only the `OMP_NUM_THREADS`-unset path; explicit settings
are honored exactly. Validated: CPU-only (fosfato/Fe3H2O6/travis) and hybrid (fosfato/Fe3H2O6,
unset → 7CPU+1GPU and explicit OMP=15 → 14CPU+1GPU) all PASS.

## Next levers (open)

1. OMP barrier ~10 % hints at residual load imbalance across the 15 bins; check `rebalance`.
2. The per-point XC functional loop (`calc_ggaCS_in`) is now a larger relative share — `libxc`-free
   `pointxc` path; could batch/vectorize but it's `pow`/`exp`-heavy.

## Files
- `g2g/cpu/cpu_kernels.h` — `cpu_compute_density_gga_batch` (old per-point `cpu_compute_density_gga` kept as the reference spec; no live callers).
- `g2g/cpu/iteration.cpp` — `solve_closed`/`solve_opened` GGA branches rewired.
- `g2g/partition.h` — `density_scratch_a/b` members on `PointGroupCPU`.
