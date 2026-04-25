---
status: VERIFIED — already done, NOT a lever
date: 2026-04-25
impact: 0 ms (the optimization is already in place)
area: lioamber
---

# QM/MM 1-electron Fock cache — already in place

## Hypothesis

In QM/MM with frozen MM positions/charges (typical SCF — only QM density
varies), the QM/MM electrostatic Fock contribution
`V_μν^QM-MM = Σ_A q_A ∫ φ_μ φ_ν / |r − R_A| dr` is constant across SCF
iterations (depends only on basis functions and MM coords, not on the
QM density). If LIO rebuilt this matrix every SCF iter on the QM/MM
benchmark, caching it would save ms-scale time per iter.

## Investigation

`lioamber/SCF.f90` was inspected for any per-iter call to QM/MM 1-e
integrals. Found:

| Line | Call | Inside SCF loop? |
|---|---|---|
| 308 | `int1(En, Fmat_vec, Hmat_vec, Smat, ...)` | **No** — initialization, before line 467 |
| 319 | `intsol(Pmat_vec, Hmat_vec, ..., E1s, Ens, .true.)` | **No** — initialization |
| 325 | `aint_qmmm_fock(E1s, Ens)` | **No** — initialization |
| 806 | `int1(...)` | **No** — post-loop, for energy decomposition |
| 812 | `aint_qmmm_fock(E1s, Etrash)` | **No** — post-loop |

The SCF loop starts at line 467. **All 1-electron and QM/MM contributions
are computed once before the loop and accumulated into `Hmat_vec`.**

Inside the loop, the only Fock-builders are:
- `int3lu` (line 487): does `Fmat_vec(:) = Hmat_vec(:)` first
  (`subm_int3lu.f90:178`), then **adds** Coulomb J. So it picks up the
  cached `Hmat_vec` (which already contains kinetic + nuclear-attraction
  + ECP + intsol + aint_qmmm_fock contributions).
- `g2g_solve_groups` (line 497): adds XC.
- Optional `field_calc` (line 513) for reaction field — not the QM/MM
  electrostatic case.

## Conclusion

The cache hypothesis is already satisfied by LIO's current code structure.
The QM/MM 1-electron contribution is built once per SCF run and reused
through `Hmat_vec`. **No optimization opportunity here.**

Documented to prevent re-investigation.

## Where to look if the assumption breaks

If a future change introduces in-loop rebuilds of the 1-electron Fock
(e.g. for moving MM charges in real-time TD-DFT, or polarizable
embedding), the optimization would re-emerge:
- `lioamber/ehrensubs/` — Ehrenfest dynamics may need to recompute
  `intsol`/`aint_qmmm_fock` per timestep. That's a different call site
  (per MD step, not per SCF iter), so still not relevant here.
- Polarizable QM/MM (not currently in LIO) would require per-iter
  rebuilds.

## Files inspected

- `lioamber/SCF.f90` lines 290-340 (pre-loop init), 467-748 (loop body)
- `lioamber/faint_cpu/subm_int3lu.f90` line 178 (`Fmat = Hmat`)
- `g2g/init.cpp` `g2g_solve_groups_` entry point — confirms it only
  adds XC, never rebuilds 1-electron terms
