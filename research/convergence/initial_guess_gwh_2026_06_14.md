# Initial-guess family (A3 GWH, A1/A2 SAD) — measured on LIO cases

**Status:** A3 (GWH) IMPLEMENTED as selectable option (not default); A1/A2 not
pursued. **Date:** 2026-06-14. Tier 1 of
[`scf_iteration_reduction_strategies_2026_06_14.md`](scf_iteration_reduction_strategies_2026_06_14.md).

## TL;DR

The cheap initial guesses do **not** beat LIO's existing 1e core-Hamiltonian
guess on the measurable LIO cases — for these main-group QM/MM systems the
core guess is already near-optimal, so SCF iteration count is **not
guess-limited**. GWH was implemented and kept as a *selectable* option
(`initial_guess = 2`) but is **not** the default. Family A is effectively a dead
end as a universal iteration-count win here; the lever for these systems is the
accelerator (Family C) or, for MD, cross-step extrapolation (B1 ASPC, shipped
separately).

## What was measured (fresh closed-shell, `LIO_OVERLAP_INT3LU_G2G=1`)

Iterations to convergence by guess (`initial_guess`: 0 = 1e core / default,
1 = aufbau, 2 = GWH):

| Case | 1e core (default) | aufbau | GWH (K=1.75) |
|------|-------------------|--------|--------------|
| fosfato (03) | **24** | 25 | 27 |
| agua (00)    | **14** | —  | 16 |
| QMinPcharges (06) | 11 | — | **9** |

GWH is **mixed**: it helps the point-charge case (11→9) but regresses both agua
(14→16) and fosfato (24→27). aufbau is also slightly worse than core on fosfato
(25 vs 24). Under the no-toggle rule ("ships unconditionally or stays out", and
nothing that regresses a case is acceptable as a default), GWH cannot be the
default. It is retained only as a user-selectable `initial_guess = 2`.

## GWH implementation (shipped, opt-in)

`lioamber/initial_guess.f90`, new `initial_guess_gwh`. Builds the effective Fock

    F_ii = H_ii ,  F_ij = 0.5 * K * S_ij * (H_ii + H_jj)  (i /= j) ,  K = 1.75

from the core-Hamiltonian diagonal and the overlap, then diagonalizes it in the
orthonormal basis (`F' = X^T F X`, dsyevd) exactly like `initial_guess_1e` and
builds the density from the occupied block. `get_initial_guess` now also receives
`Smat` (threaded through from `SCF.f90`). Guess-only: cannot change the converged
result, only the starting point.

## Why the core guess is already good here

These are closed-shell main-group systems with a sizeable HOMO-LUMO gap; the
core-Hamiltonian eigenvectors already place electrons sensibly and DIIS recovers
quickly. GWH's empirical e-e interpolation (tuned historically for EHT/minimal
bases) does not consistently improve on that for LIO's gaussian bases, and can
overshoot. This also tempers the catalogue's optimistic "25 → ~15-18" SAD
estimate for fosfato: the headroom from a better *guess* on these systems is
small, because the guess is not the bottleneck — the DIIS convergence path is.

## A1/A2 (SAD/SAP) — not pursued

SAD/SAP are higher effort (atomic-density tables or per-element atomic SCF / SAP
potential tables) and the GWH + aufbau evidence shows limited guess headroom on
the systems we can measure. SAD/SAP remain theoretically better for *fresh*
transition-metal runs, but LIO's heme case is VCINP-restart (skips the guess
entirely, see `heme_scf_hotpath_and_sad_d_refutation_2026_05_29.md`), so they
would only help fresh TM single-points — a narrow payoff for the effort. Deferred.

## Takeaway for sequencing

Family A is exhausted as a cheap universal win for the measurable LIO cases. The
remaining iteration-count levers are **B1 ASPC** (cross-step reuse, the MD
multiplier — see `scf_density_extrapolation_aspc_2026_06_14.md`) and **Family C**
(better accelerator: EDIIS staging / preconditioned DIIS), which is the only
lever that can move a *single-point* case like fosfato further.
