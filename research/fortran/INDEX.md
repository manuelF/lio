# Fortran Codebase (lioamber)

Research on modernizing and optimizing the Fortran 90 layer that drives
the SCF loop, handles I/O, and calls into the C++/CUDA engine.

## Files

| File | Status | Summary |
|------|--------|---------|
| [blas_optimization.md](blas_optimization.md) | PARTIAL | BLAS replacements in Fortran (converger_subs done, others open) |
| [fortran_modernization.md](fortran_modernization.md) | OPEN | Replace global mutable state, improve allocatable arrays, remove pre-F90 patterns |
| [unit_testing_strategy.md](unit_testing_strategy.md) | REF | Unit test infrastructure audit; 4 existing test programs |
| [ecp_optimizations.md](ecp_optimizations.md) | OPEN | Effective Core Potential initialization; 3 center-combination routines |
