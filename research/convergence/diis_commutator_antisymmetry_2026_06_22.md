# DIIS commutator exact-antisymmetry — DONE 2026-06-22

## Summary

The DIIS error matrix is the commutator `e = [F',P'] = F'P' − P'F'` in the orthonormal
(ON) basis (`lioamber/typedef_operator/Commut_data.f90`, `Commut_data_r`). It was computed
as **two full M³ DGEMMs** (`AB = F'P'`, then `AB += −P'F'`).

Both `F'` and `P'` are **symmetric** in the ON basis, so `P'F' = (F'P')^T` — *with the same
inner k-summation order*. Replaced the second DGEMM with an O(M²) antisymmetrization of the
first product: `e = AB − AB^T` (exact-zero diagonal, exact antisymmetry off-diagonal).

```fortran
call DGEMM('N','N',N,N,N, 1d0, F', N, P', N, 0d0, AB_BAmat, N)  ! AB = F'P'
do jj=1,N;  AB_BAmat(jj,jj)=0
  do ii=jj+1,N
     aij=AB_BAmat(ii,jj); aji=AB_BAmat(jj,ii)
     AB_BAmat(ii,jj)=aij-aji;  AB_BAmat(jj,ii)=aji-aij
  enddo
enddo
```

## Two wins, not one

**(1) Halves the commutator BLAS-3.** multiZn (12_Zn_timers, M=2600, open shell):
`Conv setup - DIIS commut` **1.55 s → 0.92 s (−40%)** over the run (2 spins × the O(M²)
antisymmetrize partially offsets the removed DGEMM). The commutator was ~13% of the SCF
iteration; this is a real CPU-side reduction on the multiZn pole (BChange/diag/DIIS).

**(2) Tames heme's DIIS chaos.** Forcing the error matrix to be *exactly* antisymmetric
removes the ulp-level diagonal/asymmetry noise that the near-singular DGELS EMAT pivot
amplifies (the root cause in [[heme_dgels_rank_deficiency_2026_05_28]]). Heme
(13_Heme, open, NUNP=4), iteration count, **OMP-median across {1,4,6,8}** per the project rule:

| OMP | Baseline (2-DGEMM) | Exact-antisym |
|-----|--------------------|---------------|
| 1   | 145 | 117 |
| 4   | 118 | 139 |
| 6   | **311** | 145 |
| 8   | 206 | 160 |
| **median** | **175.5** | **142 (−19%)** |
| spread | 118–311 (193) | 117–160 (43) |

The variance collapse is the principled signal (holds across *all four* OMP settings, not a
lucky default): the worst-case 311-iter trajectory is gone. Same DFT basin everywhere
(E spread ~26 µHa = FP32-XC noise).

## Correctness / validation

Not bit-identical to the 2-DGEMM form (F'/P' are symmetric only up to the ulp asymmetry left
by their DGEMM base changes), so this is a numerical-path change — hence validated on the
chaos canary. e2e: **02_Fe3H2O6, 01_OxyMol (open), 00_agua, 03_fosfatoQMMM (closed) all PASS**
(fosfato Energy too). Closed shell uses the same commutator (total density symmetric) and is
covered by fosfato. No toggle — ships unconditionally.

## Why this is safe where other DIIS-path changes were rejected

Tikhonov / |c|-cap / DGELSD / FULL_DOUBLE ([[heme_diis_stability_dead_ends_2026_05_28]]) all
*added* bias or changed the LS solution. This change instead makes the error matrix
numerically *cleaner* (its exact mathematical structure) without altering what DIIS optimizes
— it removes noise rather than adding regularization. That is why it improves rather than
perturbs the trajectory.

## Not pursued (measured floored)

Same investigation confirmed the rest of the multiZn CPU iteration is BLAS-floored:
BChange (8× M³ DGEMM/iter, `U^T A U` congruence — flop-minimal; flat at OMP 8 vs 16 so
cores already saturated) and Fock diag (DSYEVD; **NCO/M = 1500/2600 = 0.58** so subset/range
DSYEVR cannot win, and DSYEVR is anyway documented to perturb heme). The remaining lever
for those is iter-count (this change) or convergence research, not BLAS substitution.
