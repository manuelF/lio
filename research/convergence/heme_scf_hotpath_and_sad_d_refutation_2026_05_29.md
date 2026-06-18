# Heme SCF hot-path profile + SAD-d refutation + diagonalization basis-lock

**Status**: investigation DONE 2026-05-29. Empirical answer to "implement
SAD-d / speed up the heme SCF Fortran side." Conclusion: **SAD-d refuted as a
heme lever; per-iter Fortran/BLAS hot paths are exhausted; the one apparent
remaining lever (diagonalization) is basis-locked by Fe-d near-degeneracy.**

Evaluates [`heme_scf_advanced_methods_2026_05_28.md`](heme_scf_advanced_methods_2026_05_28.md) §4 (SAD-d).
Build: `make -j cuda=1 cpu=1` (hybrid). Run command:
`LIO_OVERLAP_INT3LU_G2G=1 liosolo -i heme.in -c heme.xyz -b DZVP -v`.
Hardware: Ryzen 7 5800X3D (8C/16T) + RTX 3080 Ti.

## 1. SAD-d is refuted on two independent counts

### (a) It cannot run on the benchmark
`heme.in` sets `vcinp = t`. With VCINP true:
- `drive.f90:86` loads the density from `restart.in` (71,925 lines).
- `SCF.f90:353` guards `get_initial_guess` on `(.not. VCINP)`, so it is
  **skipped entirely**.

SAD-d lives in the initial-guess path (`initial_guess.f90`). As configured it
would be dead code — it would silently no-op while the restart density drives
the run. You'd only notice by watching iter counts.

### (b) A better diagonal atomic guess makes heme *worse*, not better
SAD-d's own stated precondition (advanced-methods §4 risks): "verify SAD-d
materially beats aufbau." Measured, all converging to the same basin
(−2829.6326, charge=−1, nunp=4, DZVP):

| Start condition | iters |
|-----------------|-------|
| **restart (`vcinp=t`, the benchmark)** | **72** |
| 1e guess (`initial_guess=0`) | 202 |
| aufbau d-shell guess (`initial_guess=1`) | 391 |

`initial_guess_aufbau` (`initial_guess.f90:154`) already builds a d-shell-aware
superposition-of-atomic-densities diagonal — exactly SAD-d's family — and it is
the **worst** of the three (391). The restart density wins because it carries
off-diagonal/bonding structure; any diagonal atomic-density guess (aufbau *or*
SAD-d) lacks it and lands far from the basin. SAD-d would realistically land in
aufbau's 200–400 range, never near the restart's 72. The premise "a better
initial guess skips the wild d-shell phase" is contradicted: the better-than-1e
d-shell guess (aufbau) is 2× *more* iters than 1e, not fewer.

**Do not implement SAD-d for heme.** ~800 LOC for a guaranteed regression vs
restart and a no-op on the configured benchmark.

## 2. Per-iteration profile (72-iter restart run, ~239 ms/iter)

g2g section timers, `Iteration` = 17.23 s / 72:

| Phase | % of iter | Nature |
|-------|-----------|--------|
| Fock integrals (int3lu ‖ g2g) | 41.7% | CPU/GPU overlapped: int3lu=96ms ‖ g2g=94ms, idle=3ms — **balanced** |
| Fock diagonalization | 27.7% | pure CPU LAPACK `dsyevd`, 2×/iter (α,β) on M_f≈382 |
| SCF acceleration setup | 21.4% | BChange AOtoON 11.5%, DIIS commut 5.7%, update_emat 4.0% |
| MOC base change | 2.0% | DGEMM |
| SCF acceleration | 2.7% | DGEMM |

`perf record --call-graph dwarf` self-time across all threads:
- **gomp_barrier_wait_end 40.5% + gomp_team_barrier_wait_end 16.2% = ~56% OMP barrier spin** (idle wait)
- dgemm_kernel_ZEN 21% (real compute; includes dsyevd's dlaed3 back-transform)
- int3lu uses `dgemv_` (BLAS2, memory-bound)

## 3. Why each path is closed

- **Fock integrals (41.7%)** — already overlapped (`LIO_OVERLAP_INT3LU_G2G=1`)
  and balanced (96 ‖ 94 ms). int3lu is BLAS2/dgemv-bound but speeding it yields
  **zero wall**: g2g (GPU) is the equal/longer pole. Closed.
- **Thread count is NOT a wall lever.** Sweep of `OPENBLAS_NUM_THREADS`:

  | threads | iters | wall |
  |---------|-------|------|
  | 16 | 72 | 20.1 s |
  | 8  | 72 | 35.6 s (SMT cross-contention anomaly) |
  | 4  | 72 | 19.5 s |
  | 2  | 72 | 19.7 s |

  Wall is flat 19.5–20 s (ignoring the 8T anomaly): the 56% barrier-spin is
  *free* idle on spare SMT cores, not wall cost. Iters are rock-stable at 72
  because the overlap path pins int3lu BLAS to 4 threads, fixing the
  FP-sensitive Coulomb-fit summation order. Confirms the long-standing
  "OPENBLAS_NUM_THREADS not a lever" note.
- **Base change / DIIS commutator** — already optimal BLAS3 (`basechange_gemm`
  = 2 DGEMMs; commutator DGEMM; `update_emat` stride-1 mirror). Worked over in
  prior commits.
- **Fock diagonalization (27.7%) — basis-locked (measured).** `dsyevd` computes
  the full M_f≈382 spectrum though SCF needs only ~96 occ + LUMO. The obvious
  lever is a subset solver (`dsyevr`/`dsyevx` index range), but a subset is
  necessarily a *different eigenvector basis* in degenerate subspaces. Test:
  routed `matrix_diagon_d` through the existing `matrix_diagon_dsyevr`
  (MRRR, **full spectrum**, range `'A'`) to isolate basis-choice from FP-tiling:

  | solver | iters | basin | wall |
  |--------|-------|-------|------|
  | dsyevd (D&C, baseline) | 72 | −2829.6325926 | 19.5 s |
  | dsyevr (MRRR, full) | **119** | −2829.6326015 | 35.9 s |

  +65% iters from a basis swap alone. This **measures** the cuSOLVER-dsyevd
  failure mode (reverted 2026-05-24, "doubled heme iters via basis-choice
  perturbation on near-degenerate Fe-d"): heme tolerates FP-tiling shifts
  *within* dsyevd's algorithm (thread sweep) but **not** a different solver's
  basis. A dsyevr *subset* is the same basis change → same or worse. Closed.

## 4. Bottom line

The heme SCF per-iteration Fortran/BLAS hot paths are **exhausted** on this
hardware: Fock integrals are GPU-bound under the overlap; base change / DIIS
are already BLAS3; thread count is absorbed by spare cores; and the
diagonalization cannot be reduced because heme's Fe-d near-degeneracy makes the
eigenvector basis load-bearing for convergence (dsyevr-full: 72→119 iters).

Iter count itself remains the dominant wall variable (Lyapunov-chaotic, basin
stable — see prior dead-end sweeps), and no DIIS-level or guess-level lever
reduces it without regression. Future heme wall wins must come from the GPU
side (Fock integrals / XC) or an algorithmic SCF change (direct minimization,
advanced-methods §1), not from CPU BLAS retuning.

## Cross-references
- [`heme_scf_advanced_methods_2026_05_28.md`](heme_scf_advanced_methods_2026_05_28.md) — the four candidate methods (SAD-d = §4, refuted here).
- [`heme_diis_stability_dead_ends_2026_05_28.md`](heme_diis_stability_dead_ends_2026_05_28.md) — Lyapunov-divergent iter count, no algebraic fix.
- MEMORY: heme DGELS rank-deficiency; cuSOLVER diag revert (2026-05-24).
