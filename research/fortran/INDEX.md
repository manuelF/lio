# Fortran Codebase (lioamber)

Research on modernizing and optimizing the Fortran 90 layer that drives
the SCF loop, handles I/O, and calls into the C++/CUDA engine.

**2026-04-17 status:** Three rounds of Fortran-side wins landed since the
April 8 analysis: converger direct P'_ON (2 DGEMMs/iter, commit 7325eeb3),
int3mem OpenMP parallelization (286ms, commit 7229cad3), and Cholesky for G
matrix (116ms, commit de2d2f21). Warm wall 3.63s → **~1.84s**. Remaining
Fortran work is dominated by `int3lu` (10.7ms/iter × 25 = 268ms, 15% wall),
which is strictly sequential with `g2g solve`. The top remaining lever in
this area is **overlapping int3lu with g2g solve** — see
[overlap_int3lu_g2g.md](overlap_int3lu_g2g.md).

Perf flat top (post-optimizations): OpenMP barrier wait 40% (mostly
legitimate worker idle during serial Fortran), `cpu_compute_density_gga` 18%,
OpenBLAS dgemv/dgemm 20%, int3mem outlined OpenMP fns ~1.8%.

## Files

| File | Status | Summary |
|------|--------|---------|
| [overlap_int3lu_g2g.md](overlap_int3lu_g2g.md) | DESIGN | Original design doc — superseded by implementation note below |
| [overlap_int3lu_g2g_implemented_2026_05_01.md](overlap_int3lu_g2g_implemented_2026_05_01.md) | DONE | Closed-shell overlap shipped behind `LIO_OVERLAP_INT3LU_G2G=1`. Fosfato 1.90→1.66s (-12-15%, ~250-300ms saved); 25 iters preserved; all 8 e2e PASS |
| [overlap_autotune_2026_05_03.md](overlap_autotune_2026_05_03.md) | DONE | Auto-tunes OMP/BLAS threads from physical core count; `g2g/hardware_topo.{h,cpp}` module; 187-subtest unit test; no manual env vars needed |
| [int3lu_gpu_offload_evaluation.md](int3lu_gpu_offload_evaluation.md) | REJECTED | Same ceiling as CPU/GPU overlap, 3-5× the effort; revisit at M≥1500 |
| [full_scf_port_evaluation.md](full_scf_port_evaluation.md) | DEFERRED | 440-565ms ceiling but 11-14 weeks effort; revisit at M≥1500 or MD throughput. First step: try `cuda=2` build |
| [blas_optimization.md](blas_optimization.md) | DONE | BLAS replacements + allocation hoisting; SCF loop is BLAS-bound at M=364 |
| [fortran_modernization.md](fortran_modernization.md) | OPEN | Replace global mutable state, improve allocatable arrays, remove pre-F90 patterns |
| [unit_testing_strategy.md](unit_testing_strategy.md) | REF | Unit test infrastructure audit; 4 existing test programs |
| [ecp_optimizations.md](ecp_optimizations.md) | OPEN | Effective Core Potential initialization; 3 center-combination routines |
