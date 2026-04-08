# Fortran Codebase (lioamber)

Research on modernizing and optimizing the Fortran 90 layer that drives
the SCF loop, handles I/O, and calls into the C++/CUDA engine.

**2026-04-08: This is now the highest-impact optimization area.** On RTX 3080 Ti,
Fortran-side CPU work dominates at 47% of wall time (converger/DIIS 26%, int3lu 9%,
int3mem 12%), while g2g GPU+CPU kernels are only 11%. See `research/INDEX.md` for
the full time breakdown.

## Files

| File | Status | Summary |
|------|--------|---------|
| [blas_optimization.md](blas_optimization.md) | MOSTLY DONE | BLAS replacements in Fortran (converger, fock_commuts, restart_coef done; some LOW items open) |
| [fortran_modernization.md](fortran_modernization.md) | OPEN | Replace global mutable state, improve allocatable arrays, remove pre-F90 patterns |
| [unit_testing_strategy.md](unit_testing_strategy.md) | REF | Unit test infrastructure audit; 4 existing test programs |
| [ecp_optimizations.md](ecp_optimizations.md) | OPEN | Effective Core Potential initialization; 3 center-combination routines |
