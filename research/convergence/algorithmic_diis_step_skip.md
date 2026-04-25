---
status: OPEN
date: 2026-04-25
impact: 80-150 ms (4-7% wall on fosfatoQMMM)
risk: low
area: lioamber (SCF.f90 + converger_subs)
---

# DIIS-step Fock-build skip

## Idea

DIIS extrapolates a new Fock matrix `F̃ = Σ c_k F_k` from history. If the
extrapolation moves very little (`‖F̃ − F_n‖ < δ`), it predicts that the
next density `P̃` will also move very little (`P̃ ≈ P_n`). In that case,
the Fock matrix we'd build from `P̃` is almost identical to the one we
already have — so **skip the int3lu + g2g_solve_groups rebuild for that
iter** and just diagonalize `F̃` directly.

This isn't an approximation: DIIS already commits to using `F̃` regardless
of what the rebuilt Fock would be. The only question is whether to
compute the next iter's `F_{n+1}` from a P that's nearly identical to
P_n. If we skip, the next iter's DIIS history will use the same
`F_skip = F̃` slot — DIIS naturally tolerates this (it's just one more
near-parallel error vector, which DGELSS already handles via SVD rank
truncation, see `scf_data_flow.md` §5.1).

## Where it lives

**lioamber side, no GPU touch.**

- **`lioamber/SCF.f90:467-748`** — main SCF loop.
- **`lioamber/converger_subs.f90`** — `conver` subroutine; this is where
  DIIS coefficients are computed and `F̃` is produced.
- **`lioamber/converger_data.f90`** — DIIS state (history buffers).

## Files to modify

1. **`lioamber/converger_subs.f90`**:
   - After computing DIIS coefficients and the extrapolated Fock, expose
     a `predicted_step_norm` output: `‖F̃ − F_current‖_F / M`.
   - Or: expose the largest-magnitude DIIS coefficient and the
     residual-norm prediction `‖[F̃, P]‖`.

2. **`lioamber/SCF.f90`** — at the top of the loop, BEFORE int3lu:
   ```fortran
   if (niter > 4 .and. last_diis_step_norm < skip_threshold &
       .and. consecutive_skips < max_skips) then
      ! Skip Fock rebuild; reuse Fmat_vec from previous iter.
      ! conver still runs DIIS extrapolation to get F̃ for diag.
      consecutive_skips = consecutive_skips + 1
   else
      call int3lu(...)
      call g2g_solve_groups(0, Ex, 0)
      consecutive_skips = 0
   end if
   ```
   - `skip_threshold` ~ `told × 10` (e.g., 1e-5).
   - `max_skips = 1` (don't skip two iters in a row — bounds drift).

3. **`lioamber/init_lio.f90`** — namelist option
   `diis_skip_threshold` (default 0.0 = disabled), `diis_max_skips`
   (default 1).

## Why it's a free lever

A skipped iter saves the **entire** Fock build:
- int3lu: ~10.7 ms
- g2g_solve_groups: ~16 ms GPU + ~13 ms CPU = ~29 ms
- Total: ~40 ms saved per skipped iter

Empirically on fosfato, iters 22-25 are "polishing" — DIIS is already
near-converged but waiting for the density convergence metric. 2-3 of
these can typically be skipped: ~80-120 ms.

## Math correctness

DIIS provides `F̃` as the best linear combination of historical Focks
for the current density, with error `[F̃, P_n] ≈ 0`. Diagonalizing `F̃`
gives a new `P_{n+1}` that, by construction, is what the SCF wants
**given the current Fock landscape**. Skipping the rebuild means the
*next* DIIS iter uses an unchanged Fock, so the next error vector
`[F_n, P_{n+1}]` may not improve — that's the cost.

Net effect: typically adds 1 iter to compensate for 1-2 skipped iters.
Gain = (skipped × 40 ms) − (extra × 40 ms). Positive when skipped > extra.

In practice, skipping iters near-convergence (where the rebuilt Fock
would barely differ) is positive-sum because the "extra" iter is itself
near-convergence and very cheap to diag.

## Validation

1. **E2E**: full `./run_tests.py` with `diis_skip_threshold=1e-5`. All
   30 tests; energies within `1e-6 Ha`. Track iter count delta — must be
   `≤ +2` for any system; converged total skipped + extra net-positive.
2. **Iteration log**: instrument `verbose=2` to print
   `[iter N] SKIPPED: predicted_step=X` lines so we can audit skip
   patterns post-hoc.
3. **Convergence robustness**: 02_Fe3H2O6 (open-shell, hardest case)
   must converge within nmax=50.
4. **DIIS history pollution check**: monitor max `|c_k|` after a skip —
   should stay bounded by the existing 10,000 fallback in
   `converger_subs.f90` (see `scf_data_flow.md` §5.1).
5. **Benchmark**: ≥ 50 ms wall improvement on fosfato 10-run median.

## What could kill this

- If DIIS history quality degrades from skipped iters, convergence may
  stall. Mitigation: cap `max_skips` and monitor `‖[F̃,P]‖` after each
  skip — if it doesn't decrease, force a rebuild next iter.
- If float32 noise in Fmat_vec from `g2g` is the actual driver of
  near-convergence behavior (per `scf_data_flow.md` §4.3), skipping
  rebuilds means we don't get fresh noise patterns for DIIS to "search"
  — could increase iter count more than expected. Counter: with
  `ndiis=30` history is already large; one frozen Fock won't dominate.
- TBDFT mode — the `build_chimera_TBDFT` and `extract_rhoDFT` calls
  manipulate Fock and density between Fock-build and conver; verify
  these still work when Fock is reused.

## Recommendation

Cheap to prototype, low risk, modest payoff. Implement after the
XC-grid schedule lands. Default OFF; enable per-namelist for benchmark
runs first; promote to default if 30-test suite is clean.
