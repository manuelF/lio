# Fortran Codebase (lioamber)

Research on modernizing and optimizing the Fortran 90 layer that drives
the SCF loop, handles I/O, and calls into the C++/CUDA engine.

**2026-04-08: This is now the highest-impact optimization area.** On RTX 3080 Ti,
Fortran-side CPU work dominates at 47% of wall time (converger/DIIS 26%, int3lu 9%,
int3mem 12%), while g2g GPU+CPU kernels are only 11%. See `research/INDEX.md` for
the full time breakdown.

**2026-04-08 profiling update:** Detailed per-iteration profiling (fosfatoQMMM, M=364)
shows the SCF loop is now **BLAS/LAPACK-bound**. Top CPU symbols are OpenBLAS kernels
(dgemv 9.4%, dgemm 3.9%, sgemv 4.4%). Allocation hoisting done in Dens_build,
Diagon_datamat, DSYEVD workspace, and int3lu. No measurable wall-time change at M=364
but eliminates heap churn for larger systems. Further Fortran speedups require either
algorithmic changes (fewer DGEMMs per iter) or GPU offload.

## Files

| File | Status | Summary |
|------|--------|---------|
| [blas_optimization.md](blas_optimization.md) | DONE | BLAS replacements + allocation hoisting; SCF loop is BLAS-bound at M=364 |
| [fortran_modernization.md](fortran_modernization.md) | OPEN | Replace global mutable state, improve allocatable arrays, remove pre-F90 patterns |
| [unit_testing_strategy.md](unit_testing_strategy.md) | REF | Unit test infrastructure audit; 4 existing test programs |
| [ecp_optimizations.md](ecp_optimizations.md) | OPEN | Effective Core Potential initialization; 3 center-combination routines |
