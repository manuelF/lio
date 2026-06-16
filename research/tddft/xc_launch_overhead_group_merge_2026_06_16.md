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

## What is left

XC is still 74% of the (smaller) step and still host-bound at the remaining ~42
launches/step (1 cube + 2 spheres + leapfrog). Next levers: fold the per-atom
spheres into the merged cube too; or CUDA-graph the fixed per-step launch
sequence (partition is constant across TD steps) — prior graph attempt was on a
different path and rejected as net-neutral, but at 11.5% util the headroom here
is much larger.
