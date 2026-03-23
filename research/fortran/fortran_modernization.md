# Fortran Codebase Modernization

**Scope:** Entire `lioamber/` directory
**Language:** Fortran 90/95 with some pre-F90 patterns

## Overview

The `lioamber` codebase dates to 1992 and has been incrementally updated. Most
code uses F90 features (`allocatable`, `module`, `implicit none`), but several
pre-F90 patterns remain. This document prioritizes modernization by impact.

---

## High Priority (Affects Correctness/Maintainability)

### M1. Global mutable state in `garcha_mod`

**File:** `lioamber/liomods/garcha_mod.f` (109 lines)

This module is the central data store for the entire program: basis set data,
simulation parameters, density/Fock matrices, atomic coordinates, ECP flags,
GPU options, geometry optimization state, Ehrenfest dynamics state, etc.

Nearly every subroutine `use`s `garcha_mod` and reads/writes its variables
directly. This creates:
- **Implicit coupling:** Changes to any variable can silently affect distant code
- **Testing difficulty:** Unit tests must initialize the entire global state
- **Thread safety:** OpenMP threads sharing garcha_mod state is fragile

**Recommended approach:** Split into focused modules:
- `basis_data` (already partially done — `basis_data.f90` exists)
- `scf_data` (density, Fock, convergence parameters)
- `output_data` (verbose, write flags, file names)
- `gpu_data` (gpu_level, assign_all_functions, etc.)
- `dynamics_data` (Ehrenfest state, forces)

**Impact:** Very high for maintainability and testability.
**Effort:** High (touches every file).
**Risk:** Medium (mechanical refactoring, but widespread changes).

### M2. `do NNN while` / `NNN continue` loops

**Files:** `SCF.f90` (line 466), `TD.f90` (line 276)

Pre-F90 numbered-label loops. Replace with:
```fortran
! Before:
do 999 while (condition)
   ...
999 continue

! After:
do while (condition)
   ...
end do
```

**Impact:** Readability. No functional change.
**Effort:** Low.

### M3. Fixed-format Fortran source

**File:** `garcha_mod.f` — uses column-based format with continuation markers
(`>` in column 6). This is the only remaining fixed-format file in core code.

**Fix:** Convert to free-format `.f90` (already partially done — other modules
are `.f90`).

**Impact:** Consistency. Enables standard editor support.
**Effort:** Low (one file).

---

## Medium Priority (Affects Performance Indirectly)

### M4. Packed storage ↔ full matrix conversions

**Functions:** `spunpack`, `sprepack`, `spunpack_rho`, `sprepack_rho`
(`lioamber/packed_storage.f90`)

The code stores density/Fock matrices in lower-triangular packed vectors
(`Pmat_vec`, `Fmat_vec`) for memory efficiency, but repeatedly unpacks to
full M×M matrices for computation, then repacks. Each conversion is O(M²).

In `SCF.f90`, per iteration:
- 2–4 `spunpack`/`spunpack_rho` calls (lines 531–539)
- 2–4 `sprepack` calls (lines 703–714)
- Total: ~8 × M² element copies

**Options:**
1. **Standardize on full matrices internally.** Memory cost: M² vs M(M+1)/2
   (factor ~2×). For M=86: 59 KB vs 30 KB — negligible.
2. **Use LAPACK packed-storage routines (DSPEV, DSPMV).** Avoids unpacking
   but limits BLAS options.

**Impact:** Low for small M. Medium for M > 500.
**Effort:** Medium.

### M5. Operator type unification

**File:** `lioamber/typedef_operator/` directory

The `operator` and `sop` types partially encapsulate matrix operations and
basis changes. However, many routines still pass raw arrays and dimensions.

**Current state:**
- `operator` handles AO↔ON basis changes, Fock diagonalization
- `sop` handles overlap matrix operations
- SCF.f90 mixes `operator` calls with direct array manipulation

**Ideal:** All matrix operations through `operator` methods, with CPU/GPU
dispatch handled internally.

**Impact:** High for maintainability. Enables transparent GPU offloading.
**Effort:** High.

---

## Low Priority (Cosmetic / Long-term)

### M6. `iso_c_binding` for C++ interop

Current Fortran↔C++ bindings use manual `extern "C"` functions with trailing
underscores (e.g., `g2g_init_`). This works but is fragile:
- Name mangling depends on compiler
- No type checking across the boundary
- Assumes specific calling conventions

Using `iso_c_binding` with `bind(C, name="g2g_init")` would provide:
- Portable, compiler-independent bindings
- Type checking via `interface` blocks
- No reliance on underscore conventions

**Impact:** Robustness. Prevents subtle bugs from type mismatches.
**Effort:** Medium (many binding points).
**Risk:** Low (mechanical changes).

### M7. Error handling centralization

Currently: `stop` statements scattered throughout the code. ECP routines
use `stop` on angular momentum overflow, CUBLAS routines use `stop` on
allocation failure, etc.

**Ideal:** Central `error_handler` module with severity levels, that can
optionally write restart files before stopping.

**Impact:** Low for correctness. Useful for production robustness.
**Effort:** Medium.

### M8. Lowercase keyword standardization

Mixed case throughout: `INTEGER`, `Integer`, `integer`. Modern convention
is lowercase. Not worth changing unless doing a comprehensive reformatting
pass (which risks git blame noise).

---

## Summary Table

| ID | Priority | Impact | Effort | Description |
|----|----------|--------|--------|-------------|
| M1 | High | Very High | High | Split garcha_mod into focused modules |
| M2 | High | Readability | Low | Replace numbered loops with do/end do |
| M3 | High | Consistency | Low | Convert garcha_mod.f to .f90 |
| M4 | Medium | Low–Medium | Medium | Reduce packed↔full conversions |
| M5 | Medium | High | High | Unify operator type usage |
| M6 | Low | Robustness | Medium | iso_c_binding for C++ interop |
| M7 | Low | Robustness | Medium | Central error handling |
| M8 | Low | Cosmetic | Low | Lowercase keywords |
