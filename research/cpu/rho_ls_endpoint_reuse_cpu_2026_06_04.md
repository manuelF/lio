# Rho linear-search endpoint-density reuse (CPU) — DONE 2026-06-04

**Status:** DONE — shipped (unconditional; CPU-only / non-libxc fast path, automatic
recompute fallback for GPU groups and libxc builds).
**Impact:** MEDIUM-HIGH on `11_LinearSearchRho` (open-shell, `Rho_LS=2`). Median wall
**1.65–1.84× across OMP={4,6,8}** on 5800X3D. Per-iteration speedup is robust (~1.7×);
net wall inherits the case's chaotic iteration count (same chaos affects the baseline).

## Context

`11_LinearSearchRho` / `fe3h2o.in` (open-shell, charge=3, DIIS off, `told=1e-8`).
Run: `LIO_OVERLAP_INT3LU_G2G=1 liosolo -i fe3h2o.in -b DZVP -c fe3h2o.xyz -v`.

CPU-only profile: `LS - XC g2g` = **42.5s of ~51s wall (83%)**. The line search evaluates
the XC energy at ~675 points along `rho(lambda) = (1-lambda)*P0 + lambda*P1` (53 cycles ×
11 lambda + endpoints), and `give_me_energy`'s `g2g_solve_groups(1,Ex,0)` recomputes the full
grid-point density from the trial matrix every time. perf: `cpu_compute_density_gga_batch`
85% of that, the XC functional (`calc_ggaOS`, `cbrt`/`log`) only ~15%.

## Fix

The grid density (and its gradients) is **linear** in the density matrix, so it is computed
once at each endpoint (P0, P1) and **blended per lambda**; only the cheap non-linear functional
runs each lambda. New CPU path:
- `Partition::ls_set_endpoints(p0,p1,...)` → for each CPU group, `get_rmm_input(source)` +
  `cpu_compute_density_gga_batch` into cached per-group endpoint buffers (`ls_d0a/ls_d1a` +
  `_b` for open shell). Returns 1 if usable, 0 to fall back (GPU groups / libxc).
- `Partition::ls_energy(lambda)` → blends endpoints and evaluates the functional.
- Fortran (`converger_ls.f90`): `rho_linear_calc` calls `g2g_ls_set_endpoints[_open]` once per
  line search; `give_me_energy` takes `(ls_lambda, ls_reuse)` and swaps the
  `g2g_solve_groups` call for `g2g_ls_energy(lambda)` when reuse is active. int3lu (Coulomb,
  4%) and E1 stay per-lambda on the Fortran-blended matrix.

Density computes drop ~675 → ~200 (≈50 line searches × 4 endpoints for open shell).
`LS - XC g2g` 42.5s → **10.9s (~4×)**; net LS 44.4s → 26.3s (endpoint setup ~14s is outside
the `LS - XC g2g` timer).

## The precision trap (resolved)

`give_me_energy`'s recompute is float, and near convergence the E(lambda) curve flattens to
~1e-6 variation — below float noise. Three findings:
1. **Float blend coefficients** (`(scalar_type)(1-lambda)`) inject ~6e-6 curve error →
   line search diverges (3900+ evals, no convergence). **Blend coefficients must be double.**
2. With **double blend** of float-precision endpoint densities, the endpoint oracle
   (lambda=1 blend == stored endpoint == recompute of P1) matches to **2e-10**, interior
   lambda matches recompute to ~1e-6. Converges in-band.
3. The residual ~1e-6 is the float density kernel itself, and **cannot** be made to match the
   recompute's float trajectory without recomputing (different rounding by construction).
   Per the advisor + `heme_diis_stability_dead_ends`, precision-promotion (double endpoints)
   is a documented dead end for this Lyapunov-chaotic system: it lands on a *different*
   chaotic trajectory, not a calmer one, while eroding the win. Not pursued.

So the shipped kernel: **float endpoint densities, double blend coefficients, float functional.**

## Validation (OMP-median, per `rho_ls_density_reuse` note — case is chaotic)

Energy always correct: Total energy −1720.35616x vs `.ok` −1720.356161 (Δ ~6e-6 ≪ tol 1.5e-4),
across all runs/OMP. Median wall (3×/2×/1× samples):

| OMP | baseline median | reuse median | speedup |
|-----|-----------------|--------------|---------|
| 8 (default) | 53.1s (iters 43–52) | 32.2s (iters 50–85) | 1.65× |
| 6 | 54.6s | 31.5s | 1.73× |
| 4 | ~79s | ~43s | 1.84× |

Per-iter: reuse ~0.62 s/iter vs baseline ~1.10 s/iter (~1.7×, robust). Iteration count is
chaotic for **both** (baseline 43–52, reuse 50–85 at OMP=8); median iters are comparable
(~51–52), so the per-iter win converts to median wall. Worst-case reuse draw (85 iters)
≈ baseline median wall — does not regress below it. Full `run_travis_test.py` CPU subset PASS.

## Files
- `g2g/partition.h` — `PointGroupCPU::ls_d0a/d1a/d0b/d1b`, `ls_compute_density`,
  `ls_energy_at_lambda`; `Partition::ls_set_endpoints`, `ls_energy`, `ls_open`.
- `g2g/cpu/iteration.cpp` — implementations + `extern "C"` `g2g_ls_set_endpoints[_open]_`,
  `g2g_ls_energy_`. (Uses the global `::partition`, not the vestigial `G2G::partition`.)
- `lioamber/converger_subs/converger_ls.f90` — endpoint setup + `give_me_energy` reuse path.

## Next levers (open)
- Endpoint setup (~14s, 200 density computes) is now the largest LS cost; open shell needs
  4 endpoints/search. The per-lambda blend+functional (10.9s) is mostly the functional.
- The iteration-count chaos caps the realizable wall win; a convergence-stabilising change
  (better guess / EDIIS in the wild phase) is the orthogonal lever, not g2g-side.
