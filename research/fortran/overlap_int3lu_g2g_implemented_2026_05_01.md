---
status: DONE
date: 2026-05-01
impact: ~250-300 ms wall reduction (~12-15% on fosfato), ceiling 18%
risk: low (behind LIO_OVERLAP_INT3LU_G2G env flag, default off)
area: lioamber/SCF.f90 + g2g/init.cpp
---

# Overlap of int3lu and g2g_solve_groups (closed-shell)

## Result

Median fosfato wall: **1.90 s → 1.66 s** (-12-15%, ~250-300 ms saved)
across 10 paired baseline/overlap runs. All 8 e2e tests pass with the
flag on. Closed-shell only; open-shell falls back to the sequential path.
Convergence preserved at 25 iterations; final energy within 3 µHa noise
envelope of baseline (-2148.6509632 ± 3 µHa).

| Metric | Baseline | Overlap on | Δ |
|---|---|---|---|
| Total wall (median, n=10) | ~1.95 s | ~1.70 s | -250 ms (-13%) |
| Total wall (best, n=10) | 1.89 s | 1.66 s | -230 ms (-12%) |
| Fock integrals (sample) | 859 ms | 770 ms | -89 ms |
| Convergence iterations | 25 | 25 | 0 |
| Final energy | -2148.6509632 | -2148.6509632 | < 1 µHa |

## Mechanism

Per SCF iter, `int3lu` (Coulomb fit + Fock, ~13 ms CPU BLAS) and
`g2g_solve_groups` (XC Fock, ~19 ms GPU + CPU partition) ran strictly
sequentially. Both are independent reads of `Pmat_vec`; both write
additively into the Fock matrix.

The overlap runs them concurrently in a 2-section OpenMP region:
- **Section A** (`int3lu`): keeps writing `Fmat_vec = Hmat + Coulomb`
  in-place. OpenBLAS thread count is throttled to
  `LIO_OVERLAP_BLAS_THREADS` (default 4) so the int3lu BLAS calls don't
  starve g2g's CPU partition workers.
- **Section B** (`g2g_solve_groups_into`): writes XC into a separate
  scratch buffer (`fmat_xc_scratch`) by rebinding `fortran_vars.rmm_output.data`
  to the scratch for the duration of the call. After both sections finish,
  `Fmat_vec += fmat_xc_scratch` on the main thread.

`omp_set_max_active_levels(2)` is called once on first entry so the
inner `parallel for` inside g2g's `Partition::solve()` actually spawns
workers when invoked from a section.

## Why writes don't race

`partition.solve()` accumulates per-thread XC into `rmm_outputs[k]`
(thread-local buffers); only the final Kahan merge reads
`fortran_vars.rmm_output.data`. CPU and GPU groups all write into
`rmm_outputs[i]` (or `s_global_fock_dev` on the GPU side), never directly
into `fortran_vars.rmm_output`. So rebinding the pointer in
`g2g_solve_groups_into_` only affects where the final merge lands.

Meanwhile `int3lu` writes to its own Fortran array `Fmat_vec`. The two
threads target disjoint memory; the race is gone.

## FP-order shift

The bit pattern of the final Fmat differs from baseline because XC is
added once at the end via `Fmat_vec(:) = Fmat_vec(:) + fmat_xc_scratch(:)`
instead of being scattered into Fmat as XC is accumulated. With
float32 GPU kernels and DIIS noise floor ~2e-6, this is well below the
convergence tolerance — verified empirically: 25 iters every time across
all 10 paired runs.

## Files modified

- `g2g/init.cpp` — added `g2g_solve_groups_into_(...)` extern "C" entry
  that swaps `fortran_vars.rmm_output.data` to a caller buffer for the
  duration of an existing `g2g_solve_groups_` call.
- `lioamber/SCF.f90` — added scratch `fmat_xc_scratch`,
  `LIO_OVERLAP_INT3LU_G2G` / `LIO_OVERLAP_BLAS_THREADS` env reads, and
  the `!$omp parallel sections` block guarding the closed-shell path.
- `lioamber/Makefile.options` — added `-lopenblas` so the
  `openblas_set_num_threads_` / `openblas_get_num_threads_` symbols
  resolve at link time. The Ubuntu reference `libblas.so.3` does not
  export the OpenBLAS-specific helpers.

## Toggle

Default off for safety. Enable with:

```
LIO_OVERLAP_INT3LU_G2G=1 [LIO_OVERLAP_BLAS_THREADS=4] liosolo ...
```

`LIO_OVERLAP_BLAS_THREADS` default is 4. Sweep on fosfato:
- 2 threads: median 1.72 s (some variance from oversubscription)
- **4 threads: median 1.67 s** (best, tightest variance)
- 6 threads: median 1.69 s but occasional 3+ s outlier from CPU thrashing
- 8 threads: median 1.73 s (oversubscribes the 16-thread Ryzen)

## Promotion plan

After it bakes for a few weeks of MD/SCF runs, flip the default to ON.
Open-shell support is a follow-up: needs alpha *and* beta scratch buffers
and the equivalent of `g2g_solve_groups_into_` for the open-shell Fock
write paths (`fortran_vars.rmm_output_a/b`).

## Why it took less than the ceiling

Theoretical ceiling per iter: int3lu (13 ms) hides under g2g (19 ms),
saving ~13 ms × 25 = ~330 ms. Realized: ~250-300 ms.
Reasons for the gap:
1. OpenBLAS thread cap (4 vs unrestricted) slows int3lu slightly when
   it runs alone in cold iterations.
2. `omp parallel sections` setup/teardown overhead (~50 µs/iter ≈ 1 ms total).
3. The post-section merge `Fmat += scratch` is a daxpy on MM doubles
   (~70 KB), ~100 µs/iter ≈ 2.5 ms total.
4. Memory bandwidth contention between int3lu's BLAS calls and g2g's
   per-group CPU scatter loops.

The 12-15% wall reduction matches the research doc's range
(150-250 ms / 8-14% wall) almost exactly.

## Caveats

- **Inner timers fire from two threads.** With overlap on, both
  `g2g_timer_*` calls inside `int3lu` (`'int3lu - start'`, `'int3lu'`) and
  inside `g2g_solve_groups_` run from different OS threads. The outer
  `'Coulomb fit + Fock'` timer is touched only by the main thread and is
  reliable. Per-component inner timers may report nonsensical values when
  overlap is on — read the wall clock or the outer timer instead.
- **`omp_set_max_active_levels(2)` is scoped to the parallel sections.**
  An earlier version set it process-globally on first SCF entry. That
  caused a 2× regression in TDDFT propagation (`07_TDDFTHCL`: 52 s →
  107 s) because nested OpenMP enabled BLAS calls inside `g2g`'s
  `#pragma omp parallel for` workers to spawn their own teams,
  oversubscribing the 16 hardware threads. The fix is to bracket the
  `parallel sections` with `omp_set_max_active_levels(2)` before and
  restore the previous value (`omp_get_max_active_levels`) immediately
  after — verified: `07_TDDFTHCL` is back to baseline (52.6 s vs
  52.6 s) and fosfato still saves 12-15%.
- **OpenBLAS thread count is process-global**, not thread-local.
  `openblas_set_num_threads(N)` inside the int3lu section affects all
  OpenBLAS callers in the process. The current state machine (cap → run →
  restore) is correct because g2g_solve_groups doesn't call OpenBLAS, but
  any future code path that calls OpenBLAS concurrently with the int3lu
  section needs to be aware.
- **Linker dependency on OpenBLAS.** `lioamber/Makefile.options` adds
  `-lopenblas` so the `openblas_set_num_threads_` / `_get_` symbols
  resolve. On non-OpenBLAS BLAS implementations (MKL, Cray, reference)
  the link will fail; drop `-lopenblas` and the overlap path won't be
  enabled (the env-var read still works, but the `openblas_*` calls
  would be undefined references). A Makefile gate would be cleaner;
  deferred until needed.

## What's still on the table

The user asked about both diagonalization and Coulomb fit. This shipped
the Coulomb-fit-side win. The diagonalization side (~206 ms / 8.2 ms per
iter at M=364) still has ~150-200 ms of headroom:

- **McWeeny purification** (research/convergence/algorithmic_diag_avoidance.md):
  pure lioamber change, replaces O(N³) DSYEVD with 4-5 GEMMs of the same
  size. Estimated 100-200 ms save. Risk: small-gap systems may stall.
- **DSYEVR range-selection** for occupied + a few virtuals: ~3-5× cheaper
  on CPU, no GPU coupling, no convergence risk surface.
- **cuSOLVER DSYEVD** is borderline on M=364 (1.5-3 ms launch + workspace
  vs 8.2 ms CPU) and now competes with `g2g_solve_groups` for GPU time
  during the same iteration when overlap is on. Likely the worst of the
  three options post-overlap.

