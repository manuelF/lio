# Converger (DIIS/Damping) — Performance and Correctness Analysis

**File:** `lioamber/converger_subs.f90` (279 lines)
**Data:** `lioamber/converger_data.f90` (21 lines)
**Called from:** `SCF.f90` lines 589–594 (alpha) and 631–637 (beta)

## Overview

The `conver` subroutine implements two convergence accelerators:
- **Damping** (iterations 1–2): `F_new = (F + λ·F_old) / (1+λ)`
- **DIIS** (iteration 3+): Pulay's Direct Inversion of the Iterative Subspace

Selection is controlled by `conver_criter`: 1=always damp, 2=damp then DIIS,
3=damp until `good < good_cut` then DIIS.

### DIIS Implementation (lines 187–274)

1. Store Fock matrix `F'` in ON basis → `fockm(:,:,ndiis,spin)` (line 135)
2. Compute commutator `[F',P']` → `FP_PFm(:,:,ndiis,spin)` (lines 132–134)
3. Build B-matrix: `B(i,j) = Tr(e_i · e_j)` where `e_i = [F'_i, P'_i]` (lines 209–224)
4. Solve augmented system `B·c = [0,...,0,-1]` via DGELS (lines 252–258)
5. Build extrapolated Fock: `F = Σ c_k · F'_k` (lines 262–270)

---

## Performance Issues

### P1. Memory allocation every `conver` call — CONFIRMED

**Line 97:**
```fortran
allocate(fock00(M_in,M_in), fock(M_in,M_in), rho(M_in,M_in), work(1000))
```
**Lines 108–109:** `suma(M,M)`, `diag1(M,M)`, `scratch1(M,M)`, `scratch2(M,M)`
**Line 188:** `EMAT(ndiist+1, ndiist+1)`

That's up to **7 M×M matrices + 1 small EMAT** allocated per SCF iteration.
For M=86: 7 × 86² × 8 bytes = 412 KB of heap allocation per call.

**Fix:** Move `fock00`, `fock`, `rho`, `suma`, `diag1`, `scratch1`, `scratch2`
to `converger_data` as module-level allocatables. Allocate once in
`converger_init`, reuse across calls. The `work` array (1000 doubles = 8 KB)
can also be persistent.

**Impact:** Medium. Eliminates ~400 KB of malloc/free per iteration.

### P2. History buffer O(M²) copy per iteration — CONFIRMED

**Lines 119–122:**
```fortran
do jj = ndiis-(ndiist-1), ndiis-1
   fockm(:,:,jj,spin)  = fockm(:,:,jj+1,spin)    ! M×M copy
   FP_PFm(:,:,jj,spin) = FP_PFm(:,:,jj+1,spin)   ! M×M copy
enddo
```

When history is full (`ndiist == ndiis`), this shifts `ndiis-1` matrices,
totaling `2 × (ndiis-1) × M²` element copies. For M=86, ndiis=30:
2 × 29 × 7396 = 429,000 copies per iteration.

**Fix:** Use a circular buffer with a head index. Replace the array shift with:
```fortran
integer :: head_idx  ! in converger_data
head_idx = mod(head_idx, ndiis) + 1
fockm(:,:,head_idx,spin) = new_fock
```
Then map logical index `k` to physical: `phys = mod(head_idx + k - 2, ndiis) + 1`.

**Impact:** Medium. Eliminates O(ndiis × M²) data movement per iteration.

### P3. `matmuldiag` + diagonal sum → eliminate intermediate M×M matrix

**Lines 214–223:** The B-matrix element computation:
```fortran
call matmuldiag(scratch1, scratch2, diag1, M_in)   ! O(M²): computes only diagonal of A·B
EMAT(ndiist,kk) = 0.0d0
do ii = 1, M_in
   EMAT(ndiist,kk) = EMAT(ndiist,kk) + diag1(ii,ii)
enddo
```

Note: the previous analysis claimed `matmuldiag` computes a "full matrix
product" — this is **wrong**. `matmuldiag` (in `linear_algebra/matmuldiag.f90`)
only computes the diagonal: `C(i,i) = Σ_k A(i,k)·B(k,i)`, which is O(M²).

However, the whole sequence computes `Tr(A·B) = Σ_i Σ_k A(i,k)·B(k,i)`.
This can be done directly without the M×M `diag1` intermediate matrix:
```fortran
EMAT(ndiist,kk) = 0.0d0
do ii = 1, M_in
do kk2 = 1, M_in
   EMAT(ndiist,kk) = EMAT(ndiist,kk) + scratch1(ii,kk2)*scratch2(kk2,ii)
enddo
enddo
```

**Impact:** Low. Eliminates one M×M allocation but the computation is the same.

### P4. DIIS Fock extrapolation triple loop — can use DGEMV

**Lines 262–270:** Weighted sum of ndiist matrices. Reshaping `fockm` to
M²×ndiist and using DGEMV would leverage BLAS. For ndiist ≤ 30 and M ≤ 300,
the improvement is marginal.

**Impact:** Low.

---

## Correctness Issues

### C1. DGELS work sizing — CORRECT

**Lines 252–258.** Uses workspace query (LWORK=-1), caps at MIN(1000, optimal).
The DIIS system is at most 31×31; optimal LWORK ~200. Safe.

### C2. EMAT2 persistence logic — CORRECT but fragile

**Lines 191–206.** Saves B-matrix to `EMAT2` and restores next iteration with
shifting for the oldest-vector-dropped case. Verified correct. Would be cleaner
with circular buffer indexing.

### C3. DGELS for symmetric system — MARGINALLY SUBOPTIMAL

The augmented DIIS B-matrix is symmetric. DSYSV (symmetric solver) would be
more appropriate than DGELS (general least-squares). For 31×31 systems, the
difference is microseconds. Not worth changing.

---

## CRITICAL: DIIS and Float32 GPU Noise

See `g2g/CLAUDE.md` "SCF Convergence and Numerical Precision" for detailed
findings. Key points:

1. **ndiis=30 is necessary** despite literature defaults of 6–12. The float32
   GPU noise floor (~2e-6 rho diff) means DIIS needs more history vectors.
   Reducing to ndiis=8 causes convergence to stall permanently.

2. **Do NOT modify DIIS without full E2E testing** on both fosfatoQMMM
   (closed-shell, 25 iters) and Fe3H2O6 (open-shell, restart).

---

## Summary Table

| ID | Type | Impact | Effort | Description |
|----|------|--------|--------|-------------|
| P1 | Perf | Medium | Medium | Per-call allocation of 7 M×M matrices |
| P2 | Perf | Medium | Medium | History buffer O(M²) shift → circular buffer |
| P3 | Perf | Low | Low | Eliminate diag1 intermediate matrix |
| P4 | Perf | Low | Low | Fock extrapolation → DGEMV |
| C1 | Correct | — | — | DGELS work size: correct as-is |
| C2 | Correct | — | — | EMAT2 persistence: correct, fragile |
| C3 | Correct | Negligible | Low | DGELS → DSYSV for symmetric system |
