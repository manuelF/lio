# BLAS Optimization Opportunities

Audit of Fortran code for operations that can be replaced with optimized BLAS calls.
With OpenBLAS installed, these replacements gain AVX2 vectorization + OpenMP threading.

## CRITICAL (in SCF hot path, called every iteration)

### fock_commuts.f90:21-57 — 4 triple-nested loops
- **Current**: Hand-written i,j,k loops for X^T*F*X (base change) and X^T*F*P*Y (commutator)
- **Replace**: 4× DGEMM calls
- **Note**: Currently dead code (converger_subs.f90 inlines its own DGEMM base changes),
  but should be fixed for any future callers
- **Status**: DONE

### SCF.f90:613,658 — matmul(Xmat, morb_coefon)
- **Current**: gfortran MATMUL intrinsic (5-10× slower than DGEMM without -fexternal-blas)
- **Replace**: DGEMM('N','N', M_f, M_f, M_f, 1.0D0, Xmat, M_f, morb_coefon, M_f, 0.0D0, morb_coefat, M_f)
- **Called**: 1× per SCF iter (closed-shell), 2× per SCF iter (open-shell)
- **Status**: DONE

## HIGH (in SCF loop or called frequently)

### SCF.f90:334-337,515-522,808-811 — Manual E1 dot product loops
- **Current**: `do kk=1,MM; E1 = E1 + Pmat_vec(kk)*Hmat_vec(kk); enddo`
- **Replace**: `E1 = DDOT(MM, Pmat_vec, 1, Hmat_vec, 1)` (5 sites total)
- **Note**: Lines 334-337 are outside SCF loop (1× call); 515-522 inside loop (~25× calls);
  808-811 post-convergence (1× call). O(N) operation, negligible perf impact.
- **Status**: TODO (low priority — minimal performance benefit)

### TD.f90:679-682,864-867 — Manual E1 dot product loops
- **Current**: Same pattern as SCF.f90
- **Replace**: DDOT
- **Status**: TODO

### restart_coef.f90:34-40,61-67,100-105 — Triple-nested density build
- **Current**: `rho(i,j) += C(i,k)*C(j,k)` triple loop
- **Replace**: DGEMM('N','T', M, M, NCO, factor, C, M, C, M, 0.0D0, rho, M)
- **Status**: TODO

### converger_subs.f90:340-349 — DIIS Fock accumulation triple loop
- **Current**: `fock_w(i,j) += bcoef(k)*fockm(i,j,k)` triple loop
- **Replace**: Loop over k with DAXPY(M*M, bcoef(k), fockm(1,1,k), 1, fock_w, 1)
- **Status**: TODO

### mathsubs/commutator.f (all 7 overloads) — MATMUL pairs
- **Current**: `MP=MATMUL(MA,MB); MN=MATMUL(MB,MA); MC=MP-MN`
- **Replace**: DGEMM/ZGEMM (or single DGEMM + antisymmetric fill for symmetric inputs)
- **Note**: May be dead code like fock_commuts; verify callers
- **Status**: TODO

## MEDIUM (less frequent or smaller matrices)

### properties.f90:114-121 — Triple-nested Lowdin charge calculation
- **Replace**: DGEMM + diagonal extraction
- **Status**: TODO

### properties.f90:184-197 — Triple-nested Fukui function
- **Replace**: DGEMM
- **Status**: TODO

### propagators.f90:39-40 — MATMUL in Magnus propagator
- **Replace**: ZGEMM
- **Status**: TODO

### ehrensubs/calc_forceDS.f90:40-53 — Chain of 4 MATMUL
- **Replace**: ZGEMM chain
- **Status**: TODO

### ehrensubs/ehrendyn_prep.f90:53-54,69-70 — MATMUL for basis transforms
- **Replace**: ZGEMM
- **Status**: TODO

## LOW

### SCF_aux.f90:111-117 — Conditional column scaling
- **Replace**: Per-column DSCAL
- **Status**: TODO

### mathsubs/basechange.f — Hand-written triple loops
- **Note**: Already has `basechange_gemm.f` DGEMM variant; may be dead code
- **Status**: SKIP (DGEMM variant exists)
