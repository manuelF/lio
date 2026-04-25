---
status: OPEN
date: 2026-04-25
impact: 40-80 ms (2-4% wall on fosfatoQMMM)
risk: low-medium
area: lioamber (typedef_operator) + cuSOLVER
---

# Reduced-precision diagonalization for early SCF iters

## Idea

Today every SCF iter does double-precision `DSYEVD` on the M×M Fock matrix
(~10 ms × 25 = 250 ms wall). For early iters where DIIS error
`‖[F,P]‖ > 1e-3`, the float64 accuracy of eigenvectors is **wasted** —
the Fock matrix itself is changing by O(1e-2) per iter, so
eigenvector precision below ~1e-5 is overkill.

Plan:
- iters where `good > 1e-3`: use `DSYEVR` selected-range, OR
  single-precision `SSYEVD`, OR cuSOLVER `Ssyevd` on GPU.
- iters where `good ≤ 1e-3`: full double `DSYEVD` (current path).

Single-precision diag is 2-3× faster on Intel MKL and ~5× faster on
cuSOLVER. cuSOLVER also runs concurrently with CPU work for free.

## Where it lives

**lioamber side, plus optional cuSOLVER hook.**

- **`lioamber/typedef_operator/matrix_diagon_dsyevd.f90`** — current
  diagonalizer. Single subroutine wrapping LAPACK `DSYEVD`.
- **`lioamber/typedef_operator/typedef_operator.f90`** —
  `Diagon_datamat` method.
- **`lioamber/SCF.f90:607,649`** — call sites (closed/open).
- **`g2g/cublasinit.cpp`** — cuBLAS handle init; cuSOLVER would need a
  parallel handle.

## Files to modify

1. **New file `lioamber/typedef_operator/matrix_diagon_ssyevd.f90`** —
   `SSYEVD` wrapper, accepts `real*8` Fock, casts to `real*4`,
   diagonalizes, casts eigvecs back to `real*8`.

2. **`lioamber/typedef_operator/typedef_operator.f90`** — add
   `Diagon_datamat_sp` method, plus a wrapper that picks based on a
   precision-level argument.

3. **`lioamber/SCF.f90`**:
   ```fortran
   if (good > 1.0d-3) then
      call fock_aop%Diagon_datamat_sp(morb_coefon, morb_energy)
   else
      call fock_aop%Diagon_datamat(morb_coefon, morb_energy)
   end if
   ```

4. **`lioamber/init_lio.f90`** — namelist option `diag_mixed_precision`
   (default `.false.`).

5. *(Optional, plan B)* **GPU diag via cuSOLVER**:
   - New file `g2g/cuda/cusolver_diag.cu` — extern "C" entry point
     `g2g_cusolver_dsyevd_(F, eigvals, eigvecs, M)`.
     Uses `cusolverDnDsyevd` (double) or `cusolverDnSsyevd` (single).
   - Fortran calls into it through a new method on `Operator`.
   - Pros: runs concurrent with `int3lu` on CPU when paired with the
     `overlap_int3lu_g2g` plan.
   - Cons: extra D2H transfer of eigenvectors back to Fortran each iter
     (M² doubles ≈ 60 KB for M=86, negligible).

## Why it works

DIIS uses eigenvectors only to:
1. Diagonalize Fock and produce a new density `P = 2 C C^T` (occupied).
2. Compute orbital energies for energy convergence test.

Both are tolerant of single-precision noise:
- The density `P` is then overwritten by the next iter's int3lu+g2g
  output (well, P is what *feeds* int3lu, not vice versa — but the
  rebuild from any P with O(1e-7) noise gives a Fock with O(1e-7)
  noise, well below the float32 grid noise floor anyway).
- Orbital energies are not used in the convergence test (only `good`
  density-RMS is); they're reported for the user.

The orthogonality of `C` from `SSYEVD` is good to ~1e-7. After
back-transforming to AO, density built as `2 C^T C` has trace error
~M × 1e-7 ≈ 1e-5 — within `told` slack.

## Math correctness

- Eigenvalue equation `F C = ε C S` is solved approximately. Eigenvalue
  error is bounded by `‖F‖ × ε_machine_single ≈ 1e-7 × 1` Ha = 1e-7 Ha.
- Density built from `C_sp` (single-prec eigvecs cast to double) has
  RMS error ~1e-6, so the next int3lu rebuilds Fock with that noise
  level — already below the GPU float32 noise floor that DIIS handles.
- Final 2-3 iters use full double, so converged density is double-precision
  quality.

## Validation

1. **Unit**: random symmetric `F`, run `DSYEVD` and `SSYEVD`-then-cast,
   check `‖C^T_sp F C_sp − diag(ε_sp)‖_F < 1e-5`.
2. **E2E**: full suite with `diag_mixed_precision=.true.`. All 30 tests;
   energies within `1e-6 Ha`; iter count ±2.
3. **Open-shell**: 02_Fe3H2O6 — single-prec α/β diag is the riskiest case.
4. **TDDFT**: 07_TDDFTHCL — TDDFT uses ground-state orbitals with high
   accuracy. Verify the **last** iter's diag is full double (the one that
   feeds the propagator).
5. **Benchmark**: ≥ 30 ms wall improvement on fosfato.

## What could kill this

- LAPACK `SSYEVD` is rarely linked into LIO builds; may need to extend
  the linker config in `lioamber/Makefile`. Cheap to fix.
- If `good` doesn't drop below 1e-3 quickly enough, single-prec is used
  for too many iters and converged eigenvalues drift; mitigation is to
  set the threshold to `1e-2` and only use single-prec for iters 1-10.
- Mixed cuSOLVER + LAPACK paths add complexity. The Fortran-LAPACK path
  alone is sufficient as a first cut.

## Comparison with other levers

| Lever | Estimated save | Risk | Effort |
|---|---|---|---|
| XC grid schedule | 150-250 ms | low | medium |
| McWeeny purification | 100-200 ms | medium | medium |
| **This (mixed-prec diag)** | **40-80 ms** | **low** | **low** |
| DIIS step skip | 80-150 ms | low | low |

Smaller payoff than the top levers but **trivial to implement** if
`SSYEVD` linking works out — could be a 1-day prototype.

## Recommendation

Implement after the XC-grid schedule. If McWeeny purification lands
first, this becomes redundant for SCF (purification doesn't diagonalize
at all). Keep this in mind as a fallback if McWeeny is rejected on
robustness grounds.
