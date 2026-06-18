# B1 — ASPC cross-step density extrapolation (SHIPPED)

**Status:** IMPLEMENTED & validated, 2026-06-14. Tier 2 of
[`scf_iteration_reduction_strategies_2026_06_14.md`](scf_iteration_reduction_strategies_2026_06_14.md)
— the highest-aggregate-payoff strategy for LIO's QM/MM MD mission.

## TL;DR

A new module `lioamber/scf_extrapolation.f90` extrapolates the SCF starting
density from the last few **converged** densities along an MD / geometry
trajectory (Kolafa ASPC predictor), replacing the plain VCINP "reuse the single
last density" guess. On a water steepest-descent trajectory this cut total SCF
iterations **153 → 75 (−51%)**, dropping per-step iterations from a flat 14 to
4–7 once history fills. Guess-only: the fixed-geometry result is unchanged. It is
a **no-op for single points and TD** (one SCF call → no history), so the entire
single-point e2e suite is bit-identical.

## Mechanism

For k stored converged densities (i = 1 newest), predict

    P_pred = sum_{i=1..k} B_i * P(t+1-i) ,   B_i = (-1)^(i+1) * C(k,i) .

These are the time-reversible polynomial-extrapolation (ASPC predictor)
coefficients; they satisfy `sum_i B_i = 1`, so the predicted density **conserves
the electron count** `tr(P S) = N` exactly (every stored P has the same trace).
The order ramps with available history, capped at `max_order = 4` (coefficients
4, −6, 4, −1).

**No purification needed.** The prediction need not be idempotent: it is only a
*guess*. The first SCF half-iteration rebuilds an idempotent density by
diagonalization, so McWeeny purification (which matters for XL-BOMD, where the
propagated density enters the forces directly) is unnecessary here. Skipping it
removes the trickiest correctness hazard (idempotency under the non-orthogonal S
metric and the LIO occupation-factor convention).

## Implementation

- `lioamber/scf_extrapolation.f90` — ring buffer of the last `max_order`
  converged densities (closed: total `Pmat_vec`; open: `rhoalpha`/`rhobeta`
  separately, total = sum). `scf_extrap_predict` overwrites the guess;
  `scf_extrap_store` pushes a converged density.
- `SCF.f90`: `predict` called once before the iteration loop (after the VCINP /
  initial-guess setup); `store` called in Finalize **only on successful
  convergence** (so a failed step never poisons the history).
- Gated on `stored >= 2` (≥2 densities) and on matching system size / spin
  layout — this covers single points and step 1 automatically. It is **not**
  gated on `npas`, so it benefits both the internal geometry optimizer
  (`do_steep`, which calls SCF directly without bumping `npas`) and real MD
  (AMBER/GROMACS/`liomd.x`, which increment `npas` via `liomain`).
- Build: `OBJECTS += scf_extrapolation.o` and a `SCF.o : scf_extrapolation.o`
  dependency in `Makefile.depends`.

## Validation (water, steepest descent, `lineal_search=f`, 11 SCF solves)

Per-step SCF iterations:

```
ASPC OFF: 14 14 14 14 14 14 14 14 14 14 13   (153 total)
ASPC ON : 14 14  7  4  5  5  6  5  5  5  5    ( 75 total, -51%)
```

Steps 1–2 are identical (0 then 1 stored density → no extrapolation); from step 3
the extrapolation drops the count to 4–7/step, matching the ASPC literature's
"10–15 → 3–5 iters/step". Converged energies track within ~1–3 µHa of the
non-extrapolated trajectory — sub-`tolD` drift that is inherent to *any* guess
change on a force-driven trajectory (each step's geometry depends on the previous
converged density, which the SCF criterion fixes only to tolerance). The
fixed-geometry converged result is unchanged.

Single points: bit-identical (fosfato 24 iters / −2148.6510063, agua 14 iters) —
ASPC is a pure no-op when SCF is called once.

## Bug found during bring-up (documented to avoid repeats)

The reallocation guard was written
`if (vlen /= MM .or. hist_open .neqv. openshell .or. .not. allocated(hist_t))`.
In Fortran **`.neqv.` binds looser than `.or.`**, so this parsed as
`(vlen/=MM .or. hist_open) .neqv. (openshell .or. .not.allocated)` and evaluated
to `.false.`, skipping the allocation — `hist_t` stayed unallocated and the array
copy segfaulted. Fix: parenthesize `(hist_open .neqv. openshell)`. Also: closed
shell allocates `rhoalpha`/`rhobeta` to size 1 (see `drive.f90`), so those dummy
args must be **assumed-shape `(:)`** and only touched on the open-shell branch.

## No-toggle / safety

Ships unconditionally on (per the no-toggle rule). It is provably guess-only
(cannot change a fixed-geometry result), a no-op for the whole single-point/TD
e2e suite, and the worst case on a non-smooth trajectory is a few extra
iterations — never a wrong result, since the SCF self-corrects. The conservative
order-4 cap and the converged-only store bound the downside.

## Next in this family

- **B3** (extrapolate F or the ON-basis P′ instead of AO P) — cheap A/B once this
  infra exists; F can be smoother across steps.
- **B2 XL-BOMD** — the steady-state upgrade (near 1-iter SCF, no energy drift) but
  it changes the MD integrator and needs purification + dissipation tuning;
  sequence after this.
