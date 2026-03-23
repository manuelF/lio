# BLAS Optimization Opportunities

Audit of Fortran code for operations that can be replaced with optimized BLAS calls.
With OpenBLAS installed, these replacements gain AVX2 vectorization + OpenMP threading.

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
