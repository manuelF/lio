# Effective Core Potential (ECP) — Performance and Correctness Analysis

**File:** `lioamber/intECP.f90` (2081 lines)
**Data:** `lioamber/liomods/ECP_mod.f90`
**Called from:** `SCF.f90` (once per SCF, before the iteration loop)

## Overview

ECP integrals replace inner-shell electrons with a pseudopotential. Three
routines handle different center combinations:

| Routine | Centers | Lines | Nesting depth |
|---------|---------|-------|---------------|
| `intECPAAA` | 1-center (A=A=A) | 126–196 | 5 levels |
| `intECPAAB` | 2-center (A=A≠B) | 308–432 | 7 levels |
| `intECPABC` | 3-center (A≠B≠C) | 665–780 | 8 levels |

ECP is NOT computed every SCF iteration — it's called once during initialization
(`tipodecalculo=1,2,3` in `intECP`, line 26). The Fock contribution is stored in
`VBAC` and reused. This means **ECP optimization is startup cost, not per-iteration.**

---

## Performance Issues

### P1. Deep loop nesting in intECPABC — CONFIRMED

**Lines 708–763:** The 3-center routine has 8 levels of nesting:
```
DO i=1,M                        ! basis function i
DO j=1,i                        ! basis function j (triangular)
   DO k=1, natom                ! atoms
      IF (nuc(i)≠k AND nuc(j)≠k)   ! filter
         DO ki=1, ecptypes      ! ECP types
            IF (IzECP(k)==ZlistECP(ki))  ! filter
               DO ii=1, ncont(i)         ! contractions of i
                  IF (cutoff check)
               DO ji=1, ncont(j)         ! contractions of j
                  IF (cutoff check)
                     CALL ABC_LOCAL + ABC_SEMILOCAL  ! heavy computation
```

The innermost computation (`ABC_LOCAL`, `ABC_SEMILOCAL`) contains additional
nested loops over angular momentum components (binomial expansion).

Complexity: O(M² × natom × ecptypes × ncont² × angular_terms). For large
systems with many ECP atoms, this can be the startup bottleneck.

**Fix options (in priority order):**
1. **Parallelize outer loop with OpenMP:** The i-loop is embarrassingly parallel
   since each (i,j) pair writes to a distinct `VBAC(pos)`. Add
   `!$OMP PARALLEL DO PRIVATE(...) REDUCTION(+:VBAC)` on the i-loop.
2. **Precompute distance-dependent cutoffs:** The `a(i,ii)*(dxi²+dyi²+dzi²)`
   check (line 729) is recomputed for every (j, ji) pair. Precompute and store
   per (i, ii, k) combination.
3. **Batch angular integral computation:** `ABC_LOCAL` and `ABC_SEMILOCAL`
   internally loop over angular momentum — precomputing angular integrals
   for each (lxi,lyi,lzi,lxj,lyj,lzj) pair could avoid redundant work.

**Impact:** High for systems with many ECP atoms. Medium for typical use (few
heavy atoms). Only affects startup, not per-SCF-iteration.

### P2. SEARCH_NAN overhead

The `SEARCH_NAN` calls (scattered in intECP.f90) loop over the entire ECP matrix.
These should only run when `ecp_debug` is true.

**Impact:** Low (already gated by debug flags in most cases).

---

## Correctness Issues

### C1. Hardcoded angular momentum limit — CONFIRMED

**`Ucoef` function** (referenced in ECP analysis): contains `if (l .GT. 4) stop`.
This limits ECP to s, p, d, f, g shells. Higher angular momenta (h, i) would
crash. This is adequate for all current pseudopotentials in common use.

**Verdict:** Acceptable for production. Document the limitation.

### C2. Cutoff logic complexity

**Lines 729–735:** Three nested cutoff checks:
```fortran
IF (.NOT.cutECP .OR. (a(i,ii)*dist_i² .LE. cut3_0)) THEN    ! line 729
IF (.NOT.cutECP .OR. (a(j,ji)*dist_j² .LE. cut3_0)) THEN    ! line 731
IF (.NOT.cutECP .OR. (Distcoef .LT. cutecp3)) THEN           ! line 735
```

When `cutECP=.false.`, ALL integrals are computed (no screening). When true,
three separate thresholds (`cut3_0`, `cutecp3`) control screening. The variable
`cutecp3` guards against `0 * NaN` from exponentially small terms (line 737
comment). This is correct but the threshold values should be validated for
each pseudopotential set.

### C3. `exp(-Distcoef)` applied after accumulation — potential overflow

**Line 747:**
```fortran
acum = acum + ABC * Cnorm(j,ji) * exp(-Distcoef)
```

If `Distcoef` is very large (distant atoms), `exp(-Distcoef)` underflows to
zero — safe. If `Distcoef` is very negative (shouldn't happen since it's a
sum of positive terms), `exp(-Distcoef)` overflows. The positivity of
`Distcoef` is guaranteed by construction (line 733: sum of `a*dist²` terms).

**Verdict:** Safe. No fix needed.

---

## Summary Table

| ID | Type | Impact | Effort | Description |
|----|------|--------|--------|-------------|
| P1 | Perf | High | High | Deep nesting — OpenMP parallelize outer loop |
| P2 | Perf | Low | Low | SEARCH_NAN gated by debug flag |
| C1 | Correct | — | — | l≤4 limit: acceptable, document it |
| C2 | Correct | — | — | Cutoff logic: correct, thresholds need validation |
| C3 | Correct | — | — | exp(-Distcoef) safe by construction |

## Key Context

ECP integrals are computed **once** (not per SCF iteration). Optimization
effort should be proportional to the fraction of total runtime spent in ECP
initialization. For typical systems with few heavy atoms, this is < 1%.
For large ECP-heavy systems (e.g., lanthanide clusters), P1 becomes important.
