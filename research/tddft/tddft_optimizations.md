# TD-DFT Propagation — Performance and Correctness Analysis

**File:** `lioamber/TD.f90` (1537 lines)
**Propagators:** `lioamber/propagators.f90`
**Called from:** `lioamber/liomain.f90`

## Overview

The `TD` subroutine performs real-time density matrix propagation for TD-DFT.
Two propagators are available:
- **Verlet** (`propagator=1`): Leapfrog-like scheme using `[F,ρ]` commutator
- **Magnus** (`propagator=2`): Baker-Campbell-Hausdorff (BCH) expansion to
  order N (default `NBCH=10`)

The time loop (`do 999 istep = 1, ntdstep`, line 276) can run for thousands
of steps, making per-step overhead critical.

---

## Performance Issues

### P1. Per-step memory allocation in propagators — CONFIRMED, CRITICAL

Both `td_verlet_cu` and `td_magnus_cu` allocate local matrices every time step:

**`td_verlet_cu` (line 1079):**
```fortran
allocate(rho(M_f, M_f, dim3), rho_aux(M_f,M_f,dim3))
```
2 complex matrices × M_f² × dim3. Never explicitly deallocated (relies on
Fortran auto-dealloc on subroutine return).

**`td_magnus_cu` (line 1166–1167):**
```fortran
allocate(rho(M_f,M_f,dim3), rho_aux(M_f,M_f,dim3), &
         fock_aux(M_f,M_f,dim3), fock(M_f,M_f,dim3))
```
4 matrices: 2 complex (rho, rho_aux) + 2 real (fock_aux, fock).

**`cumagnusfac` (propagators.f90, lines 164, 170–173):**
```fortran
allocate(Omega1(M,M))                                    ! complex M×M
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPOmega)      ! GPU alloc
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPPrev)       ! GPU alloc
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPNext)       ! GPU alloc
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPRho)        ! GPU alloc
```
1 host + 4 GPU allocations PER call! With NBCH=10 commutator iterations inside,
the GPU allocs are particularly expensive (CUBLAS_ALLOC maps to cudaMalloc).

**`td_bc_fock` (non-CUBLAS path, line 1263):**
```fortran
allocate(fock(M_f,M_f), Xtemp(M_f,M_f), fock_0(M,M))
```
3 real matrices per step.

**Total per step (Magnus + CUBLAS):** ~6 host matrices + 4 GPU allocations.
For M=140 with complex*16: 6 × 140² × 16 = 1.88 MB host + 4 × 140² × 16 =
1.25 MB GPU per step. Over 10,000 steps: 18 GB of cumulative allocation.

**Fix:** Move all propagator workspace to a persistent module (analogous to
`converger_data`). Allocate once at TD initialization (`td_allocate_all`,
line 151), pass workspace pointers to propagator subroutines. For GPU memory,
allocate persistent device buffers in `td_allocate_cublas`.

**Impact:** Very High. This is the single biggest performance fix for TD-DFT.

### P2. Magnus BCH: CUBLAS_ALLOC/FREE per call

**`cumagnusfac` (propagators.f90, lines 170–173, 255–258):**
```fortran
! Per call:
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPOmega)   ! cudaMalloc
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPPrev)
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPNext)
stat = CUBLAS_ALLOC(M*M, SIZEOF_COMPLEX, devPRho)
...
call CUBLAS_FREE(devPOmega)                             ! cudaFree
call CUBLAS_FREE(devPRho)
call CUBLAS_FREE(devPNext)
call CUBLAS_FREE(devPPrev)
```

4 `cudaMalloc` + 4 `cudaFree` per time step. `cudaMalloc` involves kernel driver
calls and can take 100+ µs each. For 10,000 steps: 400,000+ µs = 400+ ms of
pure allocation overhead.

**Fix:** Pre-allocate 4 persistent device buffers at TD init. Pass them to
`cumagnusfac`. The pointer-swap trick (lines 243–245) already exists and works
with persistent buffers.

**Impact:** High. Eliminates ~0.4–1.0 seconds for 10K-step propagation.

### P3. Redundant basis changes (AO ↔ ON)

Each TD step involves:
1. Get `rho` in ON basis from operator (line 1081/1170)
2. Compute Fock in AO basis (via g2g/integrals)
3. Transform Fock AO→ON (line 1243+ via `td_bc_fock`)
4. Propagate in ON basis
5. Store result back in operator

Steps 1 and 5 each involve an M×M operator copy. Steps 3 involves a full
basis change (2 matrix multiplies = 2 × O(M³)).

**This is inherent to the algorithm** — XC integration requires AO basis,
propagation requires ON basis. Cannot be eliminated, only optimized by ensuring
all basis changes use CUBLAS when available.

**Impact:** Inherent cost, already uses CUBLAS when available.

---

## Correctness Issues

### C1. No explicit deallocation in propagator subroutines

`td_verlet_cu` and `td_magnus_cu` allocate local arrays but never deallocate.
Fortran auto-deallocation on return handles this correctly, but it means the
deallocator runs every step. Combined with P1, this is purely a performance
issue, not a correctness bug.

### C2. CUBLAS error handling: `stop` on failure

`cumagnusfac` calls `stop` on any CUBLAS error (lines 177–179, 183–186, etc.).
In a production TD-DFT run, a CUBLAS error mid-propagation means lost work
with no recovery. Consider writing a restart file before stopping, or at least
flushing output buffers.

### C3. Complex precision controlled by preprocessor

`#ifdef TD_SIMPLE` switches between `complex*8` (single) and `complex*16`
(double). The Magnus propagator involves NBCH=10 matrix multiplies per step
× thousands of steps — single precision drift over long propagations is a
known risk. The default (double) is appropriate for production.

---

## Summary Table

| ID | Type | Impact | Effort | Description |
|----|------|--------|--------|-------------|
| P1 | Perf | **Very High** | Medium | Per-step host matrix allocation in propagators |
| P2 | Perf | **High** | Medium | Per-step CUBLAS_ALLOC/FREE in cumagnusfac |
| P3 | Perf | Inherent | — | AO↔ON basis changes (2× O(M³) per step) |
| C1 | Correct | — | — | Auto-dealloc: correct but slow |
| C2 | Correct | Low | Low | CUBLAS error → stop (no restart file) |
| C3 | Correct | — | — | TD_SIMPLE precision risk documented |
