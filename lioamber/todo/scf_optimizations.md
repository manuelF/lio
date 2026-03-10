# SCF Loop — Performance and Correctness Analysis

**File:** `lioamber/SCF.f90` (953 lines)
**Called from:** `lioamber/liomain.f90`

## Overview

The `SCF` subroutine implements the self-consistent field loop. Each iteration:
1. Builds the Fock matrix (Coulomb + XC via g2g, 1e integrals via Fortran)
2. Applies convergence acceleration (damping or DIIS, via `conver`)
3. Diagonalizes the Fock matrix to get MO coefficients and orbital energies
4. Constructs the new density matrix from occupied MOs
5. Checks convergence: `rho_diff < told` AND `energy_diff < Etold`

Loop structure: `do 999 while (condition)` ... `999 continue` (lines 466–749).

---

## Corrections to Previous Analysis

### ~~"Systemic Memory Leaks"~~ — FALSE

The previous version claimed that `fock_a0`, `rho_a0`, `fock_a`, `rho_a`,
`morb_energy`, `morb_coefat` (allocated at lines 168–182) are leaked because
they lack explicit `deallocate` calls.

**This is incorrect.** These are local `allocatable` variables inside
`subroutine SCF` (declared at lines 92–101). Fortran 95+ guarantees automatic
deallocation of local allocatables when the subroutine returns (F2003 §6.3.3.1).
All compilers used by LIO (gfortran, ifort) implement this correctly. There is
no memory leak.

### ~~"Pmat_en_wgt is O(M³) — use DGEMM"~~ — MISLEADING

The previous version claimed the Pmat_en_wgt calculation (lines 835–881) is
O(M³). It is actually **O(M² × NCO)** where NCO is the number of occupied
orbitals (typically M/4 to M/2).

The loop iterates over M(M+1)/2 packed elements (outer) × NCO occupied orbitals
(inner). For fosfatoQMMM (M=86, NCO=21): 86×87/2 × 21 ≈ 78,000 ops. A DGEMM
approach (`C * diag(ε) * C^T`) has the same asymptotic complexity. For M < 300,
the manual loop is competitive with DGEMM due to lower overhead.

**Verdict:** Not a bottleneck for typical systems. Only worth DGEMM for M > 1000.

---

## Performance Issues

### P1. `xnano(M,M)` allocated/deallocated every iteration

**Lines 694, 732.** This M×M temporary is only used for the convergence check
(lines 722–731). The code's own TODO (line 692) already flags this.

**Fix:** Allocate once before the loop or compute the convergence metric directly
from `rho_a` / `Pmat_vec` without the intermediate matrix.

**Impact:** Low (59 KB for M=86), but easy cleanup.

### P2. Memory allocation inside `conver` (called every iteration)

**File:** `converger_subs.f90`, line 97:
```fortran
allocate(fock00(M_in,M_in), fock(M_in,M_in), rho(M_in,M_in), work(1000))
```
Plus `suma`, `diag1`, `scratch1`, `scratch2` (lines 108–109) and `EMAT`
(line 188). That's up to 8 matrices of size M×M allocated per SCF iteration.

**Fix:** Move to module-level persistent workspace in `converger_data`. See
`converger_optimizations.md` for details.

**Impact:** Medium. For M=86, this is ~470 KB of heap allocation per iteration.

### P3. Manual dot products instead of BLAS DDOT

**Lines 335–337, 516–517, 809–811:** Energy accumulation uses manual loops.
Could use `DDOT(MM, Pmat_vec, 1, Hmat_vec, 1)`.

**Impact:** Negligible for typical MM < 10000.

---

## Correctness Issues

### C1. Convergence metric overcounts diagonal by √2

**Lines 722–730:**
```fortran
do jj=1,M
do kk=jj,M        ! includes kk==jj (diagonal)
  del = xnano(jj,kk) - Pmat_vec(kk+(M2-jj)*(jj-1)/2)
  del = del * sq2    ! sq2 = sqrt(2.0) applied to ALL elements
  good = good + del**2
```

For off-diagonal (kk>jj), √2 correctly accounts for symmetric (j,k)+(k,j).
For diagonal (kk==jj), it should be 1.0, not √2. This overcounts diagonal
contributions by 2×, inflating `good` by ~2.3% (86 diagonal / 3741 total for
fosfatoQMMM). Both baseline and optimized code have this bug, so it doesn't
affect relative convergence behavior.

### C2. DIIS sensitivity to float32 GPU noise — CRITICAL CONTEXT

**See `g2g/CLAUDE.md` "SCF Convergence and Numerical Precision" section.**

Any change to GPU kernel FP operations (Kahan summation, `__launch_bounds__`,
instruction reordering) changes the float32 noise pattern that feeds DIIS.
This can shift convergence by 6+ iterations. The current `ndiis=30` default
is necessary (unlike literature's typical 8–12) because DIIS needs more
history vectors to push through the float32 noise floor (~2e-6 rho diff).

**Do NOT change `ndiis` or GPU kernel numerics without full E2E testing.**

### C3. `do 999 while` / `999 continue` — cosmetic

**Lines 466, 749.** Pre-F90 loop construct. Works correctly. Modernize to
`do while (...) / end do` for readability. Code's own TODO (line 734) flags this.

### C4. Stale comment about unused `morb_coefon`

**Lines 82–83:** Comment says this variable might be unused. It IS used at
lines 599, 611, 613, 643, 656, 658. Remove the misleading comment.

---

## Summary Table

| ID | Type | Impact | Effort | Description |
|----|------|--------|--------|-------------|
| P1 | Perf | Low | Low | `xnano` per-iteration alloc — hoist out |
| P2 | Perf | Medium | Medium | `conver` per-iteration allocs — make persistent |
| P3 | Perf | Negligible | Low | Dot products → DDOT |
| C1 | Correct | Negligible | Low | √2 diagonal overcounting (~2%) |
| C2 | Correct | **Critical** | — | DIIS + float32 noise sensitivity (documented) |
| C3 | Maint | — | Low | Modernize loop construct |
| C4 | Maint | — | Trivial | Remove stale comment |
