# Concurrent α/β Fock diagonalization — DEAD END

**Date:** 2026-06-22
**Status:** REJECTED (measured worse), reverted
**Case:** `12_Zn_timers` (multiZn, open-shell, M=2600). HW: Ryzen 7 5800X3D (8c/16t) + RTX 3080 Ti.

## Idea

The open-shell SCF diagonalizes the alpha and beta Fock matrices **sequentially**
(`fock_aop%Diagon_datamat` then `fock_bop%Diagon_datamat`), each at full OpenBLAS threads.
Diag is the dominant CPU pole (~1.55 s/iter, 2× DSYEVD(2600)). The two diagonalizations are
independent → run them **concurrently** in two OMP sections to overlap them.

## Why it looked promising (and the trap)

A DSYEVD thread-scaling microbench showed it **flat**: 0.557 s @1 thread vs 0.578 s @16 →
apparently serial-bound, so 2 concurrent single-threaded solves should ≈ halve the phase. A
determinism check also showed DSYEVD is **bit-identical across thread counts** (Δeigenvalue =
Δeigenvector = 0), so overlapping would be numerically free.

**The trap:** the microbench matrix was diagonally dominant (`A[i][i]=i+1`, off-diag `0.01·sin`).
For such a matrix the tridiagonal divide-and-conquer converges trivially, so DSYEVD does almost
no D&C work and barely uses threads. The **real Fock matrix** has clustered eigenvalues; its
DSYEVD spends real time in the threaded tridiagonalization (DSYTRD, BLAS-2/3) and the D&C, and
**does** benefit from threads.

## Measured result (3 variants, all worse)

Baseline sequential diag: **3.18 s / 2 iters (1.59 s/iter)**.

| Variant | diag (2 iters) |
|---|---|
| Sequential, full threads (baseline) | **3.18 s** |
| 2 OMP sections, OpenBLAS uncapped | 5.12 s |
| 2 sections, OpenBLAS pinned to 1 | 5.19 s |
| 2 sections, nested levels + cap = cores/2 (4 each) | 5.20 s |

Concurrency was **~1.6× slower** in every configuration. Inside an OMP parallel region OpenBLAS
will not spawn a nested team by default, so each section's DSYEVD ran ~1-thread (slow on the
real matrix); enabling nesting + capping to 4 threads each didn't recover it either (libgomp
nested-team + OpenBLAS pool interaction adds overhead, and the real DSYEVD wants the full core
count, not half). Bit-exactness held throughout (`02_Fe3H2O6` PASS), confirming the determinism
finding — but the perf premise was wrong.

## Lesson

**Never characterize a library kernel's thread scaling on a synthetic matrix whose structure
differs from the real workload.** DSYEVD scaling is matrix-dependent: flat on easy spectra,
threaded on hard ones. The real multiZn Fock DSYEVD uses the cores well, so sequential-at-full-
threads is already near-optimal and splitting cores between the two spins loses.

## What stands

Diag remains at the practical floor (see [[scf_diag_lapack_floor_2026_06_16]]): DSYEVD,
basis-locked, subset-DSYEVR rejected. The reverted change touched `SCF.f90` (diag block) and
`matrix_diagon_dsyevd.f90` (a `threadprivate` workspace that would have been needed for
concurrent calls). The BChange in-place win ([[bchange_inplace_congruence_2026_06_22]]) is
unrelated and stands (committed 7dee6077).
