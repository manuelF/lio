# Unit Testing Strategy for lioamber

## Current State

### Existing test infrastructure

**Location:** `lioamber/utests/`

Four test programs exist, built by `Makefile.utests`:

| Test | File | Tests | Status |
|------|------|-------|--------|
| test-converger | `test-converger.f90` | 3 tests: init, damping, DIIS activation | Compiles; links against lioamber objects |
| test-mathsubs | `test-mathsubs.f90` | 2 tests: real and complex basis change | Compiles; standalone |
| test-propagators | `test-propagators.f90` | 2 tests: Magnus with commuting/non-commuting | Compiles; large dependency chain |
| test-properties | `test-properties.f90` | 6 tests: Mulliken, Löwdin, degeneration, softness, Fukui (stubs) | Compiles; includes properties.f90 |

**Test runner:** `runtests.sh` runs all `.x` binaries sequentially.

**Build system:** `Makefile.utests` links test programs against pre-compiled
objects from `lioamber/obj/`. Requires a full lioamber build first.

### Strengths
- Tests exist and run
- Good coverage of properties and math utilities
- Converger test validates damping arithmetic (expected value 16.666...)
- Magnus test validates commuting-matrices invariant

### Weaknesses
- **No test runner with pass/fail counting.** Tests print PASSED/FAILED but
  nothing aggregates results or returns a nonzero exit code on failure.
- **No automated invocation from `make check`.** The root Makefile runs E2E
  tests only.
- **Heavy dependencies.** `test-converger` needs the `operator` type, which
  pulls in g2g linkage. `test-propagators` needs half of lioamber.
- **No numerical regression tests** for SCF convergence behavior.
- **Fukui tests are stubs** (empty subroutines).

---

## Recommended Testing Priorities

### Priority 1: Pure Math / Utility Functions (LOW dependency, HIGH value)

These functions have no dependencies on garcha_mod, operator types, or g2g:

| Function | File | What to test |
|----------|------|-------------|
| `spunpack` / `sprepack` | `packed_storage.f90` | Round-trip: pack then unpack = identity. Off-diagonal ×2 handling. Boundary: M=1. |
| `spunpack_rho` / `sprepack_rho` | `packed_storage.f90` | Round-trip with off-diagonal ÷2 correction. Verify asymmetry handling. |
| `matmuldiag` | `linear_algebra/matmuldiag.f90` | Known A·B product diagonal. Identity matrix. Anti-symmetric matrix (Tr should be zero). |
| `basechange_d_gemm` / `basechange_z_gemm` | `mathsubs.f90` | Identity transform. Known rotation. Unitary invariance of trace. |
| `mulliken_calc` / `lowdin_calc` | `properties.f90` | Already tested. Extend: trace conservation, large M. |

**How to test:** Each test is a standalone program that links only against the
specific `.o` file. No g2g dependency.

**Template:**
```fortran
program test_packed_storage
   implicit none
   integer, parameter :: M = 4
   integer :: MM
   real*8 :: full(M,M), packed(M*(M+1)/2), roundtrip(M,M)

   MM = M*(M+1)/2

   ! Build symmetric matrix
   full = reshape([1,2,3,4, 2,5,6,7, 3,6,8,9, 4,7,9,10], [M,M])

   ! Pack then unpack
   call sprepack('L', M, packed, full)
   call spunpack('L', M, packed, roundtrip)

   if (maxval(abs(full - roundtrip)) < 1.0d-15) then
      write(*,*) 'PASSED: sprepack/spunpack round-trip'
   else
      write(*,*) 'FAILED: sprepack/spunpack round-trip'
      stop 1
   end if
end program
```

### Priority 2: Converger / DIIS Internals (MEDIUM dependency)

Current test-converger validates init and damping but not DIIS correctness.

**New tests:**
| Test | Description |
|------|-------------|
| DIIS with known solution | Build a sequence of Fock matrices that converge to a known fixed point. Verify DIIS accelerates convergence vs. pure damping. |
| B-matrix symmetry | After building EMAT, verify `EMAT(i,j) == EMAT(j,i)` for all i,j. |
| Circular buffer equivalence | When implementing P2 (converger_optimizations.md), test that circular buffer produces identical bcoef as the current shift-based implementation. |
| DIIS coefficient constraint | Verify `sum(bcoef(1:ndiist)) == 1.0` (Lagrange constraint). |

**Dependencies:** Requires `operator` type (for BChange_AOtoON) — heavy.
**Workaround:** Extract the pure-math DIIS solver (lines 188–258 of
converger_subs.f90) into a separate subroutine that takes raw matrices
instead of `operator` types. This makes it independently testable.

### Priority 3: Convergence Metric (LOW dependency, HIGH correctness value)

The rho_diff calculation (`SCF.f90` lines 722–731) is critical for convergence
behavior and has the known √2 diagonal bug (see scf_optimizations.md C1).

**Test:**
```fortran
! Extract convergence metric into a function:
! function rho_diff(M, new_rho, old_packed) result(good)
!
! Tests:
! 1. Zero difference → good = 0
! 2. Known difference → verify against manual calculation
! 3. Symmetric matrix: good should equal Frobenius norm / M
! 4. Diagonal-only difference: verify √2 overcounting quantitatively
```

**Dependencies:** None (pure math on arrays).

### Priority 4: Propagator Correctness (HIGH dependency, HIGH value for TD-DFT)

Current test-propagators only tests commuting matrices (trivial case).

**New tests:**
| Test | Description |
|------|-------------|
| Unitarity | After Magnus propagation, verify `Tr(ρ²) ≈ Tr(ρ₀²)` (purity). |
| Trace conservation | `Tr(ρ_new) == Tr(ρ_old)` (particle conservation). |
| Hermiticity | `ρ_new(i,j) == conj(ρ_new(j,i))` after propagation. |
| Short-time limit | For small dt, verify `ρ(t+dt) ≈ ρ(t) - i·dt·[F,ρ]` (first-order expansion). |
| Energy conservation | For time-independent F, verify `Tr(F·ρ)` is constant across steps. |

**Dependencies:** Links against propagators.o + mathsubs.o. The CPU-only
`magnus` subroutine can be tested without CUBLAS.

### Priority 5: ECP Integral Validation (HIGH dependency, specialized)

ECP integrals are mathematically complex and hard to test in isolation.

**Approach:** Compare against reference values from another code (e.g., Gaussian,
NWChem) for a small system (1–2 atoms with ECP). Store reference `VBAC` matrix
and verify LIO reproduces it to within tolerance.

---

## Test Infrastructure Improvements

### 1. Exit code on failure

All tests should `stop 1` (or `error stop`) on any failure, not just print
"FAILED". This enables CI integration:
```fortran
if (.not. passed) then
   write(*,*) 'FAILED: test description'
   error stop 1
end if
```

### 2. Aggregate test runner

Replace `runtests.sh` with a script that:
- Runs each test
- Captures exit code
- Prints summary (N passed, M failed)
- Returns nonzero if any test failed

### 3. Integration with `make check`

Add `lioamber/utests` to the root Makefile's `check` target:
```makefile
check: liblio
	cd lioamber/utests && make -f Makefile.utests && make -f Makefile.utests run
	cd test && ./new_tests.py
```

### 4. Reduce dependencies for testability

The biggest barrier to testing is that converger and propagator subroutines
depend on the `operator` type, which depends on g2g. To enable lightweight
unit tests:

1. **Extract pure-math kernels** from `conver` (DIIS solver, B-matrix builder,
   damping) into standalone subroutines that take raw arrays.
2. **Create mock operator** with minimal implementation (just stores M×M matrix,
   returns it on Gets_data_AO/ON) for test-converger.
3. **Test CPU propagators** (`magnus` vs `cumagnusfac`) — the CPU version in
   propagators.f90 has no CUBLAS dependency.

---

## Recommended Implementation Order

1. **Add `error stop 1` to all existing tests** — 30 min, immediate CI value
2. **Write packed_storage round-trip tests** — 1 hour, zero dependencies
3. **Write matmuldiag tests** — 30 min, validates DIIS inner product
4. **Extract and test rho_diff function** — 1 hour, validates convergence metric
5. **Extract and test DIIS solver** — 2 hours, validates B-matrix + DGELS
6. **Add propagator conservation tests** — 2 hours, validates TD-DFT
7. **Add ECP reference tests** — 4+ hours, requires external reference data
