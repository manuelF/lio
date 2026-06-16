---
name: td-xc-launch-overhead-group-merge
description: RT-TDDFT XC Fock build is host-launch-bound (GPU 18% util, 58 launches/step); merging grid groups into one cube for the TD partition cuts wall -18%.
status: DONE — shipped (TD-only single-cube merge + pool-init cache fix)
impact: high (-18.4% wall on 07_TDDFTHCL chloride.in 50000 steps)
metadata:
  type: project
  workload: 07_TDDFTHCL/chloride.in (open shell, propagator=2, ntdstep=50000)
  date: 2026-06-16
  hardware: RTX 3080 Ti (Ampere SM 8.6), CUDA 13.1 build (12.0 runtime path)
---

# RT-TDDFT XC Fock build — launch-overhead diagnosis + grid-group merge

## Diagnosis (the important part)

Per Magnus TD step on chloride.in the only Fock build is in `predictor`
(`lioamber/propagators.f90`): `int3lu` + `g2g_solve_groups(0,Ex,0)` (XC, COMPUTE_RMM).
`td_calc_energy` skips the Fock build for Magnus steps (gated on `is_lpfrg`), so
there is exactly **one XC eval per step**.

Per-step timer split (50000 steps): **`TD - Pred XC` = 337µs/step = 72% of the step**.
Magnus BCH + inner Magnus (host ZGEMM) = 97µs (21%); everything else 7%.

nsys on 400 steps shows the XC is **NOT GPU-compute-bound**:

| metric | value |
|---|---|
| GPU kernel execution | 134 ms |
| GPU **utilization** | **18.2%** (idle 82%) |
| kernel launches | 23114 = **57.8/step** |
| host `cudaLaunchKernel` | 189 ms |

HCl's grid was split into ~9 cube/sphere groups (because of the grid geometry),
each launching ~7 kernels (gather_rdm, density_opened, accumulate_point, dgmm,
sgemm, splitK, scatter_rmm). The GPU finishes the tiny XC math quickly and waits
for the host to issue the next launch. **The lever is launch count, not kernel
speed.** This refutes the older "XC kernel is the lever" note for small TDDFT.

## Fix 1 — merge grid groups for the TD partition only

For small M every group already overlaps ~all M functions, so collapsing the
integration prism into a single cube does not inflate the `npts*M^2` density work
but cuts the launch count ~Ngroups-fold. Implemented in
`g2g/regenerate_partition.cpp`: when `td_merge_groups` is set and `m <= 80`,
temporarily enlarge the global `little_cube_size` to cover the whole bounding
prism (restored after the cube loop). All three call sites plus
`classify_functions.cpp` read the global, so overriding it keeps the build
consistent.

Scoped to TD via `g2g_set_td_merge(.true.)`/`(.false.)` around
`g2g_reload_atom_positions` in `td_integration_setup` (`lioamber/TD.f90`). The
pre-TD SCF keeps its fine partition — **the SCF/heme partition is never merged**,
preserving DIIS/float32 convergence (verified: heme energy within run-to-run
noise of the unmerged binary; the heme `.ok` is independently stale by ~2.4e-4).

## Fix 2 — pool re-init (the load-bearing half)

The merge alone gave only -5% because the merged cube failed `tryAlloc` and
recomputed its basis functions **every step** (`compute_functions` = 588
instances vs 12 baseline). Root cause: `GlobalMemoryPool::init()` had
`if (_init) return;`, so the budget was pinned to the first (SCF) partition and
TD's freshly-computed `effective_fgm` was discarded. Made `init()` re-apply the
per-device budget on every call (`g2g/global_memory_pool.cpp`); it runs after
`Partition::clear()` frees old groups and before any new `tryAlloc`, so
overwriting is safe. After the fix `compute_functions` = 9 (one-time, cached).
This also fixes function caching for any non-merged TD/multi-geometry partition
that differs in size from the SCF one.

## Result (chloride.in, 50000 steps, clean)

| metric | baseline | merged+fix | Δ |
|---|---|---|---|
| wall | 25.6 s | 20.9 s | **-18.4%** |
| TD - TD Step | 25.18 s | 20.41 s | -18.9% |
| TD - Pred XC | 16.86 s | 11.77 s | -30.2% |
| launches/step | 57.8 | ~42 | -28% |
| compute_functions | 12 | 9 | cached |

All e2e pass (agua, Fe3H2O6 open, fosfato, ECP, 05/07 TDDFT incl. dipole).

## Caveat — not bit-exact

The merge changes the float32 reduction order. No-field ground-state propagation
energy drift over 50k steps: baseline 2e-5 Ha, merged **7.3e-4 Ha** (≈0.46
kcal/mol, sub-chemical). This is the documented partition-geometry FP32 noise; it
was the explicitly-accepted tradeoff. If a tighter trajectory is ever needed, a
less aggressive merge (cap to 2–4 groups instead of 1) would trade some launch
savings for smaller reorder.

## Update — sphere fold (commit 5a7bd661)

After the cube merge, the per-atom spheres (sphere_radius=0.5) were still ~3.5
extra groups (~36 launches/step, GPU 11%). Zeroing sphere_radius in the same TD
merge window folds them into the cube → **~1 group/step, ~13 launches/step**.
TD-step 20.0→17.1s (-15%), TD-Pred XC 11.7→8.8s (-25%); cumulative wall
25.6→17.6s (**-31%** vs pre-merge baseline). 07 TD dipole within 3.6e-5 (tol
1e-3). No-field 50k energy drift grows to 1.7e-3 Ha (cube-only was 7.3e-4) — the
dipole observable is unaffected.

## What is left (re-profiled at 17.6s wall)

The group-count lever is now exhausted (1 group). Per-step split:

| phase | /step | share | nature |
|---|---|---|---|
| Pred XC (g2g) | 176µs | 51% | host-launch-bound, ~13 launches, GPU 8% util |
| Magnus BCH (final) | 49µs | 14% | host ZGEMM, NBCH=10 |
| Pred inner Magnus | 49µs | 14% | host ZGEMM, NBCH=10 (predictor only) |
| Pred field / rho rebuild / ON→AO | ~30µs | 9% | host |

Next levers, in order:
1. ~~**Inner-Magnus order**~~ — DONE (commit 30575bd2). The predictor's inner
   Magnus only builds a density for the corrector and runs a half step, so it
   needs far fewer BCH terms than the final propagation. Cut to `max(4,NBCH/2)`
   (=5): inner Magnus 2.43→1.25s (-48%), TD-step 17.2→15.9s, wall 16.4s. 50k
   dipole within 1.6e-5 (tol 1e-3); sweep {2..10} all sub-1e-3, order 4-5 ≈ full.
   The *final* Magnus (also 14%) keeps full NBCH — reducing it shifts the actual
   dynamics, not just the predictor, so leave it.
2. **XC eval frequency** — still 1 full XC build/step. Reusing/extrapolating the
   XC Fock on alternating steps could ~halve it; biggest risk to dynamics.
3. **CUDA-graph the ~13-launch XC sequence** (constant across steps) — now C++,
   but at 8% util the host-launch headroom is large; prior graph attempt was on
   a different path.

## Update — field_calc wasted dipole (commit 95afd376)

`field_calc` (called every step from the predictor and td_calc_energy) computed
the electronic dipole unconditionally, but `dipxyz` only feeds the field-energy
term after the no-field early-return. On a field-free run (delta-kick spectrum)
that dipole() was pure waste. Reordered it after the field-strength check.
Bit-exact. TD-Pred field 0.475→0.005s, TD-Step Energy 0.478→0.052s, wall
16.4→15.9s. 05_TDDFTField (has a field) still passes.

## Cumulative (this line of work)

baseline 25.6 → group-merge 20.9 → sphere-fold 17.6 → inner-Magnus 16.4 →
field_calc 15.9s (**-38%**).

Remaining per-step split (15.9s wall): Pred XC 57% · final Magnus BCH 16% · inner
Magnus 8% · int3lu 3% · base changes (rho ON→AO, rho rebuild, Fock→ON) 6%.

The clean Fortran wins are now spent. Remaining levers:
- **XC (57%)** — host-launch-bound, 1 group, GPU ~8% util. Only via XC eval
  frequency (Fortran, but couples to Pmat_vec/dipole; risky) or CUDA-graphing the
  fixed launch sequence (C++; biggest clean win ~17%).
- **final Magnus (16%)** — NBCH=10 is load-bearing: with Cl core states
  ‖Ω‖dt≈16, the BCH series needs all terms, so neither order-reduction nor
  norm-early-exit is safe (unlike the inner predictor, whose error is corrected).
- **int3lu/XC overlap** — DEAD on the TD path: per-step OMP team spawn dwarfs the
  ~0.45s saving (see [[td_overlap_int3lu_reverted_2026_05_19]]).
