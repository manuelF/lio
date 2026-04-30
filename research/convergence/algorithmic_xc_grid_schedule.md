---
status: REJECTED
date: 2026-04-25
impact: NEGATIVE (+8 iters on fosfatoQMMM, energy off 0.5 mHa)
risk: realised
area: g2g
---

# Coarse → fine XC grid schedule

## Spike result (2026-04-25) — REJECTED

One-day spike: hardcoded `g2g_new_grid` schedule before the per-iter
`g2g_solve_groups` in `lioamber/SCF.f90`, gated by `good`:
- `good > 1e-3` → SMALL (50)
- `1e-5 < good ≤ 1e-3` → MEDIUM (116)
- `good ≤ 1e-5` → BIG (194)

fosfatoQMMM:
- Baseline: 25 iters, −2148.6509632 Ha, ~2.85 s wall
- Schedule: **33 iters (+8)**, **−2148.6514517 Ha (Δ = 488 µHa)**, 2.80 s wall

The +8 iter penalty entirely consumes the per-iter XC savings, and the
final energy is 488 µHa off baseline — far outside the 1 µHa target.
DIIS does not absorb the Fock discontinuity at each grid switch: each
switch effectively pollutes the 30-vector DIIS history with a Fock
matrix from a different grid, which (combined with the rebalancer
`timeforgroup` reset triggered by `partition.regenerate()`) shifts the
SCF trajectory enough to cost iters. Same root cause documented in
`reproducibility_investigation_2026_04_19.md` for fgm caching variation.

Conclusion: **even with full partition caching the iter cost would
still apply** — caching only removes the 195 ms regenerate overhead per
switch, not the DIIS perturbation. Killing this lever entirely.

Pivot to `algorithmic_diis_step_skip.md` (80–150 ms, lioamber-only, no
DIIS perturbation: just reuses an iter's Fock when DIIS already
predicts a tiny step).

# Original proposal (kept for context)

## Idea

## Idea

XC integration error doesn't need to be tighter than the SCF residual at any
given iteration. Today, every SCF iter integrates Exc and Vxc on the **same
fixed grid** (Becke partition × radial × angular). Replace this with a schedule:

| SCF state | Grid | Cost factor |
|---|---|---|
| `good > 1e-3` (early iters, ~1-15) | Coarse (e.g. SG-0: 23 radial × 110 angular) | ~3-4× cheaper |
| `1e-5 < good < 1e-3` (mid iters) | Standard | 1× |
| `good < 1e-5` (last 2-3 iters) | Fine (production) | same as today |

Final energy is computed with the production grid, so accuracy is preserved.
DFT codes that ship this: ORCA (`Grid1`/`Grid2`/`Grid5`), Q-Chem (`XC_GRID
SG-0` early), Gaussian (`Grid=Coarse,Fine,UltraFine`).

## Where it lives

**g2g side (C++/CUDA), with one Fortran toggle.**

- `g2g/init.cpp` — `g2g_reload_atom_positions_` is where the partition is
  built; grid parameters come from `FortranVars` (atom positions, basis,
  `Iexch`). The partition is rebuilt only on geometry changes.
- `g2g/partition.cpp` — `Partition::regenerate()` constructs `PointGroup`
  objects with current grid spec.
- `g2g/regenerate_partition.cpp` — actual point generation.
- `g2g/common.h` — block size constants, group thresholds.
- `lioamber/SCF.f90:497` — call site `g2g_solve_groups(0, Ex, 0)`.

The `0` first argument is currently a fock-only flag. We piggyback on it (or
add a new entrypoint) to pass a grid level.

## Files to modify

1. **`g2g/init.h`** — add a new `g2g_set_grid_level_(int level)` extern "C"
   binding. Levels: 0=coarse, 1=standard, 2=fine.
2. **`g2g/partition.h/cpp`** — store grid level on the partition; on level
   change, mark partition stale and call `regenerate()`. Cache **multiple**
   partitions (one per level) to avoid rebuilding mid-SCF — the grid points
   for SG-0 vs production are independent and both fit in VRAM.
3. **`g2g/regenerate_partition.cpp`** — parameterize the radial/angular
   counts on `grid_level`. Today these come from a fixed table in
   `g2g/common.h`.
4. **`lioamber/SCF.f90`** — inside the SCF loop, before `g2g_solve_groups`:
   ```fortran
   if (good > 1.0d-3) then
      call g2g_set_grid_level(0)   ! coarse
   else if (good > 1.0d-5) then
      call g2g_set_grid_level(1)   ! standard
   else
      call g2g_set_grid_level(2)   ! fine
   end if
   ```
   Also expose `XC_GRID_SCHEDULE = .true./.false.` namelist option in
   `lioamber/init_lio.f90` for backward compatibility.

## Why it's a real lever (numbers)

Fosfato profile: `g2g_solve_groups` is ~16 ms GPU + ~13 ms CPU partition per
iter × 25 iters = ~400 ms / ~325 ms. If 20 of 25 iters use a 3× cheaper grid:
- GPU: 16→5 ms × 20 + 16 × 5 = 180 ms (vs 400 ms today) → **220 ms saved**
- CPU partition: similar relative scaling → ~70 ms saved
- Realistic with switching overhead and final-grid SCF "rebound" iters:
  **150-250 ms (7-12% wall)**.

## Math correctness

DFT total energy is **variational** in the density: the converged density
`P*` minimizes E[P]. If the Vxc used during SCF is integrated on a coarse
grid, the converged `P*_coarse` differs from `P*_fine` by O(grid_error_in_Vxc).
But:

1. We re-do the **last 2-3 iters on the fine grid**, so the final density and
   total energy are computed on the production grid.
2. The SCF trajectory is allowed to deviate during early iters — DIIS
   tolerates this; the only risk is convergence stalling at the grid switch.
3. Empirically, the energy error from a coarse-grid intermediate density is
   ~O(1e-6 Ha) once the final grid is applied — well below `told=1e-6`
   absolute density tolerance.

**Caveat**: DIIS extrapolates Fock matrices that came from *different
grids*. The error vector commutator `[F,P]` then mixes Vxc from coarse and
fine grids — values aren't comparable. Mitigation: **clear DIIS history at
each grid switch** (`call converger_clear_history`).

## Validation

1. **Unit**: add a g2g unit test that builds the same `PointGroup` with two
   levels and checks the integrated density `∫ρ` differs by ≤ 1% (coarse) or
   ≤ 0.01% (fine).
2. **E2E correctness**: run the full `./run_tests.py` suite. Every test must
   produce final energy within `1e-6 Ha` of baseline. The schedule must NOT
   change converged energies — only the trajectory.
3. **Convergence robustness**: run 30 random small molecules (use the test
   set in `test/LIO_test/`); none may regress in iteration count by more than
   2 iters. If any system fails to converge, disable the schedule for that
   case.
4. **Open-shell guard**: 02_Fe3H2O6 must remain converged with no NaN. The
   coarse grid for spin-density GGA can underintegrate near nuclei; verify
   no spin contamination drift.
5. **Benchmark gate**: fosfatoQMMM 10-run median wall must drop ≥100 ms vs
   baseline (2.09 s).
6. **Reproducibility**: `./run_unit.py --sanitize=racecheck` after the change
   (per project rule).

## Implementation order

1. Plumb `g2g_set_grid_level` API + cache multiple partitions in g2g.
2. Add the Fortran toggle, default OFF.
3. Run E2E suite with toggle ON. Tune the `good` thresholds.
4. Enable by default once 30-system regression suite is clean.

## What could kill this

- If the partition rebuild itself dominates (~13 ms CPU partition is
  per-iter; multiple cached partitions avoid this).
- If DIIS history pollution causes systems to fail to converge after the
  grid switch — fallback is to clear history every switch (acceptable
  cost: ~1 extra iter).
- If the fine-grid "rebound" eats most of the savings — only happens if
  the coarse grid is too coarse. SG-0 → SG-1 transition is well-tested
  in literature.
