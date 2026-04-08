# BLAS Optimization Opportunities

**Status:** MOSTLY DONE — all hot-path operations use BLAS. Allocation hoisting done.
**Last updated:** 2026-04-08

Audit of Fortran code for operations that can be replaced with optimized BLAS calls.
With OpenBLAS installed, these replacements gain AVX2 vectorization + OpenMP threading.

## 2026-04-08 Profiling Results (fosfatoQMMM, M=364, RTX 3080 Ti)

Detailed per-iteration profiling of the SCF hot loop (25 iters, 3.77s wall):

| Component | Avg/iter | % Wall | Dominant BLAS |
|-----------|----------|--------|---------------|
| int3lu | 23.7ms | 12.5% | DGEMV(804×15K), SGEMV(804×21K), DSPMV(804) |
| g2g_solve | 22.3ms | 11.8% | (GPU+CPU partition) |
| diag+base | 13.2ms | 7.0% | DSYEVD(364), DGEMM(364³) |
| converger | 10.5ms | 5.5% | 5× DGEMM(364³), DDOT, DGELSS |
| unpack+copy | 0.9ms | 0.5% | (array copy) |
| int3mem | 437ms | 9.2% | (once, pre-loop) |

**Key finding:** At M=364, the Fortran SCF loop is **BLAS/LAPACK-bound**. The perf profile
shows `dgemv_kernel_4x4` (9.4%), `dgemm_kernel_ZEN` (3.9%), `sgemv_kernel` (4.4%) as the
top non-idle CPU symbols. Allocation hoisting (Dens_build, Diagon_datamat, DSYEVD workspace,
int3lu temporaries) eliminates heap churn but doesn't measurably affect wall time at this
system size. Benefits grow with M.

### Allocation hoisting (DONE, 2026-04-08)

| File | Change |
|------|--------|
| `Dens_build.f90` | Eliminated coef_occ copy + dens_mat temp; DGEMM writes directly to data_AO |
| `Diagon_datamat.f90` | Eliminated per-call M×M Dmat allocation |
| `matrix_diagon_dsyevd.f90` | Persistent DSYEVD workspace (SAVE); removed M² NaN check loop |
| `subm_int3lu.f90` | Persistent work arrays (Rc, aux, rho_gathered, terms_d/s, etc.) |

## CRITICAL (in SCF hot path, called every iteration)

### fock_commuts.f90 — 4 triple-nested loops → 4 DGEMM
- **Status**: DONE

### SCF.f90:613,658 — matmul(Xmat, morb_coefon) → DGEMM
- **Called**: 1× per SCF iter (closed-shell), 2× per SCF iter (open-shell)
- **Status**: DONE

## HIGH (in SCF loop or called frequently)

### restart_coef.f90 — Triple-nested density build → DGEMM/SGEMM
- **Replace**: `dens = factor * C * C^T` via DGEMM('N','T',...)
- **All 4 variants**: cd (double closed), cs (single closed), od (double open), os (single open)
- **Status**: DONE

### converger_subs.f90:340-349 — DIIS Fock accumulation → DAXPY
- **Replace**: Loop over k with DAXPY(M*M, bcoef(k), fockm(1,1,k), 1, suma_w, 1)
- **Status**: DONE

### mathsubs/commutator.f — 7 MATMUL overloads → xGEMM
- **Replace**: dd→DGEMM, zz/zd/dz→ZGEMM, cc/cd/dc→CGEMM
- **Used by**: TD-DFT via Commut_data_c/Commut_data_r in typedef_operator
- **Status**: DONE

### SCF.f90:334-337,515-522,808-811 — Manual E1 dot product loops
- **Replace**: DDOT. O(N) operation, negligible perf impact.
- **Status**: SKIP (low priority)

### TD.f90:679-682,864-867 — Manual E1 dot product loops
- **Replace**: DDOT. Same as SCF.f90.
- **Status**: SKIP (low priority)

## MEDIUM (less frequent or smaller matrices)

### properties.f90:114-121 — Triple-nested Lowdin charge → DGEMM
- **Replace**: DGEMM for S^½ * rho * S^½, then extract diagonal
- **Status**: DONE

### properties.f90:184-197 — Triple-nested Fukui function
- **Note**: O(M²×nDeg) where nDeg=1-3. Converting to DGEMM would add O(M³) step.
- **Status**: SKIP (would be slower with DGEMM)

### propagators.f90:39-40 — MATMUL in Magnus propagator
- **Replace**: CGEMM/ZGEMM (ifdef TD_SIMPLE)
- **Status**: DONE

### ehrensubs/calc_forceDS.f90:40-53 — Chain of 8 MATMUL
- **Replace**: ZGEMM chain with complex copies of real matrices
- **Status**: DONE

### ehrensubs/ehrendyn_prep.f90:53-54,69-70 — MATMUL for basis transforms
- **Replace**: ZGEMM + DGEMM with temporary buffers for aliasing
- **Status**: DONE

## LOW

### SCF_aux.f90:111-117 — Conditional column scaling
- **Replace**: Per-column DSCAL
- **Status**: TODO

### mathsubs/basechange.f — Hand-written triple loops
- **Note**: Already has `basechange_gemm.f` DGEMM variant; may be dead code
- **Status**: SKIP (DGEMM variant exists)
