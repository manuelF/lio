# SCF Fock diagonalization is at the OpenBLAS LAPACK floor (fosfato, M=364)

**Status: ANALYSIS / DEAD-END for the eigensolver. 2026-06-16.**

## Context

Per-iter breakdown on fosfato (closed-shell, M=364, NCO=99, hybrid build,
`LIO_OVERLAP_INT3LU_G2G=1`):

```
[iter] 23  fock=10.1  build=0.3  accel=4.0  diag=7.1  rest=1.5  tot=23.0 ms
```

The `diag` timer (`SCF.f90`, `t_diag_w`) wraps **only**
`fock_aop%Diagon_datamat` → `matrix_diagon_dsyevd` → LAPACK `DSYEVD('V','L')`.
At ~7 ms it is ~30% of the iteration, second only to `fock` (overlapped
int3lu+g2g). The obvious idea — "we only need the NCO=99 occupied eigenvectors,
so compute a subset instead of all 364" — is **wrong at this size**. Measured
below.

## Method

Dumped the real ON-basis Fock matrix at SCF iter 15 (`this%data_ON` in
`Diagon_datamat`, temporary instrumentation, reverted) and benchmarked every
LAPACK symmetric-eigensolver path on it, wall-clock, 50 reps, OpenBLAS-OpenMP
at 8 and 4 threads (5800X3D, 8 physical cores). In-app `diag`≈7.2 ms matches the
bench `DSYEVD`≈6.4 ms (the rest is allocate/copy jitter + an occasional D&C
deflation spike).

## Results (real fosfato Fock, M=364, want lowest NCO+1=100)

| Routine | Time | vs DSYEVD | Note |
|---|---|---|---|
| **DSYEVD 'V'** (divide & conquer, current) | **6.4 ms** | 1.00× | what we ship |
| DSYEVD 'N' (eigenvalues only) | 3.0 ms | — | eigenvector half ≈ 3.4 ms |
| DSYTRD only (reduction to tridiagonal) | 1.5 ms | — | the unavoidable floor |
| DSYEVR 'I' (lowest 100 vecs, MRRR) | 12.2 ms | **0.52×** | 2× *slower* |
| DSYEVX 'I' (lowest 100 vecs, bisection+invit) | 12.3 ms | **0.52×** | 2× *slower*, eigval err 7e-14 |
| DSYEV 'V' (plain QR) | 32.0 ms | 0.20× | 5× *slower* |
| DSYEVD_2STAGE 'V' | — | — | rejects 'V'; eigenvalues-only |
| SSYEVD 'V' (single precision) | 4.6 ms | 1.42× | unsafe — see below |

## Why "compute only the occupied subspace" loses

`DSYEVD` is `DSYTRD` (reduce to tridiagonal, 1.5 ms) + `DSTEDC`
(divide & conquer on the tridiagonal — computes **all** eigenpairs, the bulk) +
`DORMTR` (back-transform, BLAS-3, fast). The reduction is *not* the bottleneck;
the tridiagonal eigenvector assembly is. To compute only a subset you must
replace D&C with MRRR (`DSTEMR`, DSYEVR) or bisection+inverse-iteration
(`DSTEBZ`/`DSTEIN`, DSYEVX). Both carry large fixed costs (whole-spectrum
bisection to locate the range; reorthogonalization within eigenvalue clusters)
and parallelize worse than D&C. At M=364 the subset routines are 2× slower
**even though they do less arithmetic**. There is no LAPACK decomposition that
back-transforms only 99 vectors *and* gets the tridiagonal eigenvectors cheaper
than D&C does for all of them.

Crossover where subset solvers win is much larger M (≈1000+) and/or much smaller
NCO/M; not fosfato-class systems.

## Other levers ruled out

- **Reference vs tuned LAPACK** — already optimal. `libblas.so.3` *and*
  `liblapack.so.3` both resolve through update-alternatives to
  `openblas-openmp` (verified with `readlink -f` + `nm -D`). Not reference
  netlib. No portable swap available; MKL would be faster but is not a
  deployable assumption (and would overfit to Intel).
- **Single precision (SSYEVD)** — only 1.42×, and it perturbs eigenvectors at
  the 1e-6 level. The whole heme convergence story is ulp-sensitivity of the
  Fock/DIIS path (see `research/convergence/` heme notes); a 1e-6 perturbation
  to the occupied vectors would change heme's chaotic trajectory length far more
  than thread-count already does. Cannot ship unconditionally (no-toggle rule),
  cannot gate safely. Dead.
- **GPU eigensolver** — cuSOLVER `dsyevd` at M=364 is 2.4× slower than CPU
  (already measured, `diis_circular_buffer_2026_05_24`); launch overhead
  dominates at this size; would also overfit to recent GPUs. Dead.
- **Two-stage reduction (DSYTRD_2STAGE / DSYEVD_2STAGE)** — only benefits
  eigenvalues-only; the band→full eigenvector back-transform negates the gain
  for `'V'`. The library's `DSYEVD_2STAGE` rejects `JOBZ='V'` outright.
- **Overlap diag with the GPU** — impossible. The SCF loop is strictly
  sequential: `fock(ρ_N)` [int3lu‖g2g] → DIIS → `diag` → `ρ_{N+1}`. The next
  XC/Coulomb needs `ρ_{N+1}`, which is the *output* of diag. No legal
  cross-phase or cross-iteration overlap exists (consistent with
  `initial_guess_blas_2026_06_14`).

## Conclusion

The per-iteration diagonalization is at the OpenBLAS divide-and-conquer floor
for M=364 and cannot be reduced by an eigensolver swap, a precision change, or
GPU offload without either a slowdown or a convergence-perturbing risk that the
no-toggle + heme-safety constraints forbid. This upgrades the long-standing
"diag hard wall" assertion to a benchmark-backed result.

**The only remaining headroom is to do fewer full diagonalizations**, not to
make each one faster:

1. **Reduce iteration count** (the documented active project,
   `convergence/scf_iteration_reduction_strategies_2026_06_14`) — each iter
   saved removes a whole 7 ms diag. This is the high-value path.
2. **Pseudo-diagonalization / subspace reuse** near convergence (reuse previous
   MOs, annihilate only the occ–virt Fock block, Stewart-style) — real per-iter
   headroom (could drop the tail iters' diag to ~2 ms) but produces *approximate*
   eigenvectors → changes the SCF trajectory → exactly the class of perturbation
   that has repeatedly doubled heme's iter count. Would need open/closed gating
   or a convergence-tail guard and careful median-over-OMP validation before it
   could be considered. Not pursued under the current no-toggle constraint.
