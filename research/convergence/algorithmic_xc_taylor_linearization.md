---
status: OPEN — speculative
date: 2026-04-25
impact: 60-120 ms (3-6% wall on fosfatoQMMM)
risk: high (changes XC bit-pattern → DIIS sensitivity)
area: g2g
---

# Vxc Taylor linearization for late SCF iters

## Idea

Vxc[P] is non-linear in P (XC is a non-linear functional), so we can't just
add Δ contributions like for Coulomb. But near convergence, ΔP is small and
Vxc admits a Taylor expansion:

```
Vxc(P + ΔP) ≈ Vxc(P) + ∫ f_xc(r,r') · Δρ(r') dr'
```

where `f_xc = δVxc/δρ` is the **XC kernel** (the same object TDDFT uses for
linear response). If we cache `Vxc(P)` and the grid-point density at iter K,
all subsequent iters that have `‖ΔP‖ < ε` can skip the full Vxc recomputation
and apply only the linear correction.

For GGA, `f_xc` includes gradient-gradient mixing terms, so the kernel-times-
density operation is more involved than LDA, but every TDDFT/CPKS code
already has it.

## Where it lives

**g2g side, deep in the kernel stack.**

- **`g2g/cuda/iteration.cu`** — main GPU iteration driver. Today calls
  `gpu_compute_density` (evaluates ρ on grid from P) then
  `gpu_compute_density_derivs` (∇ρ for GGA) then a XC functional call then
  `gpu_update_rmm` (accumulates Vxc into Fock).
- **`g2g/cuda/kernels/density.h`** — density kernel (ρ at grid points).
- **`g2g/cuda/kernels/energy.h`** — XC functional evaluation (Becke,
  Perdew, LYP, etc).
- **`g2g/cuda/kernels/rmm.h`** — `gpu_update_rmm` (scatter Vxc → Fock).
- **`g2g/partition.cpp`** — `solve_groups` orchestrates the per-group
  pipeline.

## Files to modify

This is a substantial refactor — sketch only:

1. **New per-`PointGroupGPU` cache** (in `g2g/pointgroup.h`):
   - `rho_grid_cache` — float array, ρ at each grid point.
   - `vxc_grid_cache` — float array, Vxc(ρ) at each grid point.
   - Plus gradient caches for GGA: `nabla_rho_cache`, `vxc_grad_cache`.
   - `cache_iter` — int, SCF iter when populated.
   - `cache_valid` — bool.

2. **`g2g/cuda/iteration.cu`**:
   - Accept a new "linearized mode" flag from Fortran
     (`g2g_solve_groups_linear(...)`).
   - In linear mode: skip `gpu_compute_density` (use cached ρ + Δρ via
     low-cost kernel); evaluate `f_xc · Δρ` with a new kernel
     `gpu_apply_xc_kernel` that combines the XC kernel matrix-element
     evaluation with `gpu_update_rmm` directly.

3. **New kernel `g2g/cuda/kernels/xc_kernel_apply.h`**:
   - Inputs: cached ρ, ∇ρ, ΔP (RMM-difference), grid weights.
   - Output: Fock contribution `ΔF_xc`.
   - For LDA: `ΔF_xc_μν = Σ_pts w · f_xc_LDA(ρ) · χ_μ χ_ν · Δρ`.
   - For GGA: includes gradient terms. Reuse the gradient-of-density
     code that already exists for forces.

4. **Fortran caller (`SCF.f90`)** picks linear vs full mode based on
   `good < 1e-4`. Switching back to full mode for the last 1-2 iters
   commits the final Vxc.

## Why it's tempting

`gpu_compute_density` is ~46% of GPU time (per `roofline_gpu_compute_density.md`).
`gpu_update_rmm` is another 30%. If 5 of 25 iters skip the heavy density
evaluation and use linear-response instead, we save:
- density skip: 16 ms × 5 × 0.5 (kernel apply ~0.5×) = 40 ms
- rmm-update structurally similar but with cached Vxc table: ~20 ms

Net: ~60 ms. Best case with 8 linearized iters: ~120 ms.

## Math correctness

This is mathematically a **second-order SCF** technique — equivalent to one
Newton step using the XC Hessian. Provably correct for `‖ΔP‖ → 0`. The
risk is that:
1. The linear approximation introduces a **systematic bias** in Vxc that
   shifts the converged density by O(‖ΔP‖²).
2. Mitigation: **always** redo the last 2 iters with full Vxc to commit
   the final density.

For GGA functionals, the kernel `f_xc^GGA` includes ∇·(coupling × ∇Δρ)
terms — these are well-documented in TDDFT references (Casida 1995,
van Leeuwen 1999) but require careful coding around boundary conditions.

## Why it's risky

Per `scf_data_flow.md` §4: **any change to GPU float32 patterns in Vxc
disrupts DIIS**. Kahan summation has been ruled out for exactly this
reason. A new linearized Vxc kernel produces fundamentally different
bit-patterns from the cubature kernel — even if mathematically equivalent
in double precision, in float32 the noise structure differs.

This means:
- DIIS may need re-tuning (`ndiis`, damping, GOLD).
- Some systems that converged in 25 iters may take 30+ with linear mode.
- Open-shell systems are extra fragile (per `Fe3H2O6` history).

This idea should NOT be implemented until:
1. The XC-grid schedule (`algorithmic_xc_grid_schedule.md`) is in place
   and proven (lower-risk variant of the same general lever).
2. Full-double-precision GPU build (`precision=1`) is validated as a
   baseline — that mode eliminates the float32-noise interference and
   makes linearization safer.

## Validation

1. **Unit**: build the XC kernel application kernel in isolation;
   compare `Vxc(P + ΔP) − Vxc(P)` vs analytic `f_xc · ΔP` for small
   random ΔP — agreement to O(‖ΔP‖²).
2. **E2E**: full 30-test suite. Energies within `1e-6 Ha`. Iteration
   count regressions ≤ +3 globally; no NaN on open-shell.
3. **Bias check**: run 5 systems with linear mode ON for last K iters
   varying K=2,4,6,8; report converged-energy drift vs full mode.
   Acceptable: |ΔE| < 1e-6 Ha.
4. **DIIS robustness**: 02_Fe3H2O6 must converge with linear mode.
5. **Reproducibility**: `./run_unit.py --sanitize=racecheck`.
6. **Benchmark**: ≥ 50 ms wall improvement on fosfatoQMMM with no test
   regression; otherwise rejected.

## Recommendation

**Defer.** Schedule order: (1) XC grid schedule → (2) FULL_DOUBLE
validation → (3) revisit this. Until precision=1 is the validated default,
any change to float32 Vxc bit patterns is high-risk for marginal gain.
