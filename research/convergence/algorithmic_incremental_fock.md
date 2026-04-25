---
status: OPEN
date: 2026-04-25
impact: 80-150 ms (4-7% wall on fosfatoQMMM)
risk: medium (numerical drift)
area: lioamber (int3lu) + g2g (rmm_update)
---

# Incremental Fock build via density-difference

## Idea

The Coulomb (J) and XC (Vxc) Fock contributions today are rebuilt **from
scratch** every SCF iteration from the full density `P_n`. But under DIIS,
`ΔP = P_n − P_{n−1}` shrinks exponentially — by iter 15 it's ~1e-4 of `P_n`
in norm. We can build incrementally:

```
F_n  = h + J[P_n]       + Vxc[P_n]
     = (h + J[P_{n-1}] + Vxc[P_{n-1}])  +  J[ΔP] + ΔVxc
     = F_{n-1}  −  Vxc[P_{n-1}]_grid_part  +  ΔJ  +  Vxc[P_n]_grid_part
```

Two flavors:

**(a) RI-J incremental fitting.** LIO uses density-fitted Coulomb (`int3lu`,
auxiliary basis Md). The bottleneck is `Rc = cool · ρ_packed` (DGEMV
`Md × kknumd`). When `ρ` only changes by ΔP, we can compute
`Rc_new = Rc_old + cool · Δρ_packed`. Same FLOPs **unless we screen** small
Δρ entries: if `|Δρ_i| < threshold`, drop column `i` of `cool` from the
GEMV. Near convergence ~70% of Δρ entries are below threshold → DGEMV
becomes 30% of full cost.

**(b) Periodic resync.** Every K iterations (e.g., K=5) do a full rebuild
to flush accumulated drift. Late iters (DIIS converging) skip rebuilds.

## Where it lives

- **`lioamber/faint_cpu/subm_int3lu.f90`** (lines 99-265) — main int3lu
  routine. Currently builds `Rc_w` from full `rho` every call (lines
  150-170). This is where (a) goes.
- **`lioamber/faint_cpu/subm_int3mem.f90`** — builds `cool`, `cools` (the
  fitted ERI tensors). These are precomputed once at SCF start.
- **`lioamber/SCF.f90:487`** — call site for int3lu.
- **`g2g/cuda/iteration.cu`** — the XC accumulation kernel
  (`gpu_update_rmm`). Incremental Vxc would mean accumulating
  `ΔVxc = Vxc[P_n] − Vxc[P_{n-1}]` rather than full Vxc. Harder because
  Vxc is non-linear in P (XC is non-linear functional). Skip (b) for the
  XC side — see `algorithmic_xc_taylor_linearization.md` for the proper
  Vxc-incremental approach.

## Files to modify

For (a) — Coulomb-side density-difference fitting:

1. **`lioamber/faint_cpu/subm_int3lu.f90`**:
   - Add module-level state: `rho_prev_w(MM)`, `Rc_prev_w(Md)`, `valid_cache`.
   - On first call (or after geometry/density-reset events), do the full
     GEMV as today and cache `Rc_prev_w = Rc_w`, `rho_prev_w = rho`.
   - On subsequent calls: compute `dr_w(:) = rho - rho_prev_w`; build a
     compact index list of entries with `|dr_w(i)| > tau` (typical
     `tau = 1e-7 × max|dr|`); call `dgemv` with the **submatrix** of
     `cool` corresponding to those columns; add to `Rc_prev_w`.
   - Update `rho_prev_w = rho`, `Rc_prev_w = Rc_w` after.
   - Force a full rebuild every K iters (configurable; default K=5) to
     bound numerical drift.
   - Same logic for the single-precision `cools` path.

2. **`lioamber/init_lio.f90`** — namelist option
   `int3lu_incremental` (default `.false.`), `int3lu_resync_every`
   (default `5`), `int3lu_drho_tau` (default `1e-7`).

3. **`lioamber/SCF.f90`** — at SCF start, `call int3lu_invalidate_cache()`
   to force iter-1 to do a full rebuild. Also after MD/Ehrenfest steps
   that move atoms (caller already calls `aint_new_step`; piggyback).

## Why this is more modest than it looks

LIO's int3lu is **already RI-fitted** — there are no 4-index ERIs in the
SCF loop. The per-iter Coulomb work is two GEMVs:
- `Rc = cool · ρ_gathered` — 10.7 ms total but ~6 ms is this GEMV
- `Fmat += cool^T · af` — ~3 ms

Only the first GEMV benefits from density-difference (the second one
operates on `af`, the fitting coefficients, which **change uniformly** so
sparsity in `af` differences is poor).

Realistic savings: 5-7 ms × 25 iters × 60% applicability = **80-100 ms**.

## Math correctness

The density-fitted Coulomb is exactly linear in ρ:
```
J_μν[ρ] = Σ_PQ (μν|P) [G^-1]_PQ Σ_λσ (Q|λσ) ρ_λσ
       = (cool^T)_PQ_μν · af  where  af = G^-1 · Rc, Rc = cool · ρ
```
So `Rc[ρ] = Rc[ρ_prev] + cool · (ρ − ρ_prev)` is **exact** with no
truncation. The screening `|Δρ_i| < τ` introduces error
`O(τ × ‖cool‖_1)` in `Rc`, which propagates linearly into `af` and
`Fmat`. With `τ = 1e-7` and typical `‖cool‖_1 ~ 1`, Fock error is
~1e-7 — well below the float32 noise floor that DIIS already tolerates.

## Validation

1. **Unit**: extend `test/test_int3lu` with an "incremental" mode test —
   call `int3lu` 5× with slowly varying `rho`, compare final `Fmat` to
   non-incremental rebuild within `1e-10`.
2. **Drift bound**: instrument the cache to log `‖Fmat_inc − Fmat_full‖_∞`
   every iter when verbose=2; assert ≤ 1e-7 before resync.
3. **E2E**: `./run_tests.py` — all 30 tests; energies within `1e-6 Ha`.
4. **Convergence**: iter count must not regress by more than 1 on fosfato
   (25 baseline). If a system needs +3 iters, reduce `tau` or `K`.
5. **Open-shell**: 02_Fe3H2O6 — special attention. The α/β densities
   change at different rates; both must be tracked separately.
6. **Reproducibility guard**: ndiis=30 history is built from `Fmat_inc`
   not `Fmat_full` — verify DIIS still converges within nmax=50.

## What could kill this

- The savings (~100 ms) are smaller than the XC-grid lever (200 ms) and
  comparable to the work to implement and validate.
- If `tau` has to be lowered to 1e-9 to keep convergence robust, the
  screening rate drops and savings vanish.
- LIO's RI-J path is already O(N²·Md) cheap for the M=86 fosfato case;
  this lever scales worse (relative gain shrinks) for larger systems.
  Counter-argument: it scales **better** absolutely on bigger systems,
  where int3lu time is in the seconds.

## Recommended ordering

Do XC-grid schedule first (bigger lever, lower risk, shorter implementation).
Revisit this only if profiling shows int3lu still in the top 3 hotspots
after that.
