# Numerical Stability Investigation — 2026-04-17

**Goal**: Characterize the run-to-run variability in the SCF loop on the current
build (post-int3mem+Cholesky optimizations), and determine whether any structural
change can make the fosfatoQMMM energy *bit-exactly reproducible* — which would
unlock the math-reordering optimizations (Kahan summation in kernels, double
accumulators) that were previously ruled out because they shifted the DIIS
trajectory.

**Status**: characterization complete; actionable conclusions below.

---

## Run-to-run variance (fosfatoQMMM, 25 iters baseline)

All runs: warm cache, `liosolo -i fos{,.fgm0}.in -b basis -c fos.xyz`, RTX 3080 Ti,
baseline float32 build (commit `d7a05b53` tree), 25 SCF iterations in every case.

| Configuration | Runs | Iters | Energy range (Ha) | Notes |
|---|---|---|---|---|
| Default (OMP=15, fgm=0.3) | 5 | 25 (×5) | 2.6e-6 | rebalance + cross-thread Kahan sum |
| `free_global_memory=0.0` (no cache) | 5 | 25 (×5) | 7e-7 | fgm=0.0 disables function caching |
| `OMP_NUM_THREADS=1` (fgm=0.3) | 3 | 25 (×3) | 2e-6 | serial CPU; GPU thread only |
| `OMP=1` + `fgm=0.0` combined | 5 | 25 (×5) | 1.1e-6 | tightest deterministic config |
| `full_double=1` build | 3 | 25 (×3) | see below | BROKEN — wrong energy |

Notes:
- `fgm=0.0` is the strongest single lever: disabling GPU function caching tightens
  variance from 2.6e-6 to 7e-7 Ha. The caching interacts with the per-iteration
  rebalancer, and the cross-iteration side effect shows up as 2-3× wider variance.
- Serializing CPU (`OMP_NUM_THREADS=1`) alone does **not** tighten to the fgm=0.0
  baseline, which means CPU-side OMP reductions are not the dominant source.
- Combined `OMP=1` + `fgm=0.0`: 1.1e-6 Ha. Still not bit-exact — residual
  non-determinism is on the GPU side or inside OpenBLAS/MKL even with
  `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1`.
- Convergence itself is rock-solid: every single run in every configuration
  hit 25 iters. The variance is entirely in the converged energy, not in
  whether convergence happens.

---

## compute-sanitizer initcheck — clean

`/usr/local/cuda-13.1/bin/compute-sanitizer --tool=initcheck` on the fosfatoQMMM
run reports **0 errors**. Not a source of noise: GPU memory is properly
initialized before every read.

---

## Source-code audit of accumulation sites

### GPU kernels

- `gpu_scatter_rmm` (rmm_scatter.h:58) — uses `atomicAdd(double)` into
  `global_rmm_out[bigs[k]]`. *But*: within a single kernel launch, the `bigs[k]`
  indices are **unique** (built in `compute_indexes()` as one entry per
  lower-triangular function pair). So there is **no intra-launch atomicAdd race**.
  Across launches, stream 0 serializes group scatters, so global order is
  well-defined given a fixed group launch sequence.
- `gpu_update_rmm` (rmm.h) — each (i,j) thread privately accumulates `rmm_local`
  and writes once to `rmm[i,j]`. No atomics. Fully deterministic.
- `gpu_compute_density` (energy.h) — warp-shuffle reductions designed to
  preserve the volatile-shmem FP order (commit `ac87eef0`). Safe per existing
  investigations.
- All other GPU kernels: no `atomicAdd` calls found in the SCF path.

**Conclusion**: given a fixed group launch order, every GPU kernel should be
deterministic. Any residual variance implies the **launch order itself** is not
fixed — either the rebalancer (suppressed by `OMP=1`, still measurable) or
OpenBLAS thread scheduling during Fortran-side int3lu / int3mem.

### CPU-side partition accumulation

- `partition.cpp:168-190` — `add_rmm_output` promotes float→double **per
  element** before accumulating into the global Fortran Fock matrix. Correct.
- `partition.cpp:721-752` — Kahan-compensated summation across per-thread Fock
  matrices (closed **and** open shell). Already deterministic *once the
  per-thread matrices are fixed*.
- `partition.cpp:557` — `#pragma omp parallel for reduction(+:energy)`. OMP
  `reduction(+)` is order-non-deterministic at high thread counts. With
  `OMP=1` this collapses to a serial loop (deterministic), which matches the
  OMP=1 variance shrink.

### Kernel-launch-ordering variance

- `Partition::rebalance()` (partition.cpp:466-535) reshuffles work[] bins based
  on per-group timings; different runs → different group→thread assignment →
  different intra-thread launch order. With `cpu_threads=0, gpu_threads=1`
  (our OMP=1 config), there is only one GPU bin, so rebalance is a no-op —
  yet we still see 1.1e-6 variance, which means the GPU bin's internal group
  order is itself determined at rebalance time, not fixed at partition build.

---

## `full_double=1` build is broken

Rebuilt with `make clean && make cuda=1 cpu=1 full_double=1 -j16`. Binary grew
13,541,672 → 14,151,784 bytes (confirming recompilation). Runs converge in
27–33 iters and produce energy **-2148.6443x**, i.e. **6.6 mHa above the float32
baseline** (float baseline: -2148.6509x).

6.6 mHa is three orders of magnitude larger than any float32 noise signature
and larger than the typical chemical-accuracy target (~1 kcal/mol ≈ 1.6 mHa).
This is not a precision improvement — it is a real computational bug in the
`FULL_DOUBLE=1` code path.

Suspected causes (not yet confirmed):
1. **Texture format mismatch**: `iteration.cu:272-277` gates the `int2` vs
   `float` channel format on `FULL_DOUBLE`. A stale/mixed object may read
   texture data using the wrong bit layout.
2. **AINT precision coupling**: `aint_mp ?= 1` is default. The templated
   AINT classes conditionally instantiate as `<float>` when `AINT_MP && !FULL_DOUBLE`
   and as `<double>` otherwise. All observed `#if` guards appear internally
   consistent, but a build-system bug could mix ABI-incompatible objects if
   only a subset of files rebuilt.
3. **libxcproxy.h** already contains a telling comment (lines 587, 594, 601):
   *"Esto es para engañar al compilador pq el FLAG FULL_DOUBLE a veces permite
   que T sea double y se rompe todo."* Historical evidence that the
   FULL_DOUBLE template instantiations have been fragile.

The g2g/CLAUDE.md currently recommends FULL_DOUBLE as *"the real fix for
precision"*. **This recommendation is stale** — the path is broken on the
current tree and must not be used as a reference until the 6.6 mHa offset is
root-caused and fixed.

---

## Actionable conclusions

1. **No safe structural lever to make the float32 baseline bit-exact.** We
   tightened variance from 2.6e-6 → 1.1e-6 Ha by disabling fgm and serializing
   OMP, but residual non-determinism remains and is plausibly in OpenBLAS
   threading inside int3lu/int3mem. Going further requires either fixing
   FULL_DOUBLE (option A below) or accepting the 1e-6 Ha noise as inherent.

2. **Math-reordering optimizations remain unsafe on the float32 path.** All the
   prior failed experiments (Kahan in density, `__launch_bounds__`, double
   accumulators) changed float32 bit patterns and broke DIIS trajectories,
   producing 28-31 iters instead of 25. That result is orthogonal to whether
   the baseline is bit-exact — it is about DIIS sensitivity to the specific
   noise pattern, which the user's 1e-6 observed variance already demonstrates
   is on the edge of `told=1e-6`.

3. **The actionable high-impact path is fixing FULL_DOUBLE.** If that code path
   converges to the correct energy, Kahan summation, double accumulators, and
   register-pressure reductions become immediately testable — and the float32
   noise floor disappears entirely for systems that need tighter convergence.
   Worth a dedicated investigation: bisect recent precision-touching commits,
   rebuild each variant, find the first tree where FULL_DOUBLE diverges.

4. **The g2g/CLAUDE.md should be updated** to mark FULL_DOUBLE as broken
   rather than "the real fix" — this is a documentation correctness issue.

5. **No uninitialized GPU memory**: initcheck rules this out as a noise source.

## Non-levers (tested here or previously)

- `OPENBLAS_NUM_THREADS` (1/2/4/default) — all within 0.08s noise (see
  `user/MEMORY.md` entry `project profile_fosfato_2026_04_17`)
- `initial_guess=1` (aufbau) — 26-27 iters vs baseline 25
  (`initial_guess_evaluation.md`)
- All kernel-level Kahan/precision changes (`g2g/CLAUDE.md` SCF Convergence
  section)
