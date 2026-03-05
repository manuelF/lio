# Bug: `free_global_memory > 0` causes wrong SCF results

## Goal

Enable the `free_global_memory` parameter (`fgm`) to cache GPU function values
(φ_μ(r), ∇φ_μ(r), ∇∇φ_μ(r)) across SCF iterations, eliminating redundant recomputation.
This is expected to give a significant speedup by eliminating the `gpu_compute_functions` +
transpose work (~18% of GPU time in fosfatoQMMM) for the majority of groups per iteration.

## The Bug

Setting `free_global_memory = 0.8` (in the `[g2g]` namelist of the `.in` input file)
causes wrong SCF energies in fosfatoQMMM (`test/LIO_test/03_fosfatoQMMM`):

| Setting | Final energy | SCF iters |
|---|---|---|
| `fgm=0.0` (default) | −2148.65 A.U. | 27 |
| `fgm=0.8` (bugged) | −1629.40 A.U. | ~15 |

The wrong run converges prematurely to a completely different energy.

## Test Files

- `test/LIO_test/03_fosfatoQMMM/fos_fgm.in` — same as `fos.in` but with `free_global_memory = 0.8`
- `test/LIO_test/03_fosfatoQMMM/fos_fgm001.in` — `fgm=0.001` (crashes with pool accounting assert)

Run with:
```bash
cd test/LIO_test/03_fosfatoQMMM
../../liosolo/liosolo fos_fgm.in
```

## Math Background: XC Fock Matrix Pipeline

Each GPU group (PointGroupGPU) contributes to the XC Fock matrix via:

```
1. gpu_compute_functions(forces, gga)
   → fills: function_values[M × P]  (basis function values at P points)
             gradient_values[M × P] (gradients, x/y/z components as vec4)
             hessian_values[M × P]  (second derivatives)

2. transpose kernels (streams 1/2, async):
   → fills: function_values_transposed[P × M]
             gradient_values_transposed[P × M]
             hessian_values_transposed[P × M]

3. get_rmm_input(rmm_input_cpu)
   → gathers: P_μν  (density matrix elements for this group's local basis)
              from fortran_vars.rmm_output (the global density matrix)
              stored as lower-triangle [M × M] padded matrix

4. cudaMemcpy2DToArrayAsync → rmm_cuArray → rmm_tex (density matrix as texture)

5. gpu_compute_density<<<P, 64>>>(..., rmm_tex, FVT, GVT, HVT, point_weights_gpu, ...)
   → for each point p:
       ρ(p)     = Σ_{μ,ν} P_{μν} φ_μ(p) φ_ν(p)         (closed-shell electron density)
       ∇ρ(p)   = 2 Σ_{μ,ν} P_{μν} φ_μ(p) ∇φ_ν(p)      (density gradient)
       ∇²ρ(p)  = similar with hessian
   → fills: partial_densities_gpu[P × B]  (B = ceil(M / 2*64) block rows)
             dxyz_gpu[P × B], dd1_gpu[P × B], dd2_gpu[P × B]

6. gpu_accumulate_point<<<ceil(P/256), 256>>>(..., partial_densities_gpu, dxyz_gpu, ...)
   → for each point p:
       accumulates block rows → scalar ρ(p), ∇ρ(p)
       calls XC functional (LDA/GGA) to get:
           f_XC(p) = v_XC(ρ(p)) * w(p)   (XC potential × integration weight)
   → fills: factors_gpu[P]

7. gpu_update_rmm<<<M, M>>>(..., factors_gpu, function_values_non_transposed, ...)
   → computes: F^XC_{μν} = Σ_p f_XC(p) φ_μ(p) φ_ν(p)
   → fills: rmm_output_gpu[M × M]

8. cudaMemcpy: rmm_output_gpu → rmm_output_host (pinned)

9. CPU (post-sync): add_rmm_output() scatters rmm_output_host into global Fock matrix
```

Notation:
- M = `group_m` = `total_functions()` for this group (S + 3P + 6D basis functions)
- P = `number_of_points` = integration points in this group
- B = `block_height = ceil(M / 2*DENSITY_BLOCK_SIZE)` = ceil(M/128)
- FVT/GVT/HVT = function/gradient/hessian values transposed

## `inGlobal` Caching Mechanism

When `fgm > 0`, the `GlobalMemoryPool` tracks available GPU memory. In `compute_functions()`:

```cpp
// Attempt to reserve GPU memory for this group's function data
if (0 == GlobalMemoryPool::tryAlloc(size_in_gpu())) {
    this->inGlobal = true;  // reserved; will NOT free these matrices after solve
}
```

At the start of compute_functions():
```cpp
if (this->inGlobal) return;  // CACHE HIT: skip recomputation
```

When `inGlobal=true`, at end of `solve_closed()`:
```cpp
if (!(this->inGlobal)) {
    function_values.deallocate();
    gradient_values.deallocate();
    hessian_values_transposed.deallocate();
    function_values_transposed.deallocate();
    gradient_values_transposed.deallocate();
}
```
So these matrices persist in GPU memory when `inGlobal=true`.

Also in `solve_closed()`, point weights are only re-uploaded if not already cached:
```cpp
if (!point_weights_gpu.is_allocated() || !this->inGlobal) {
    // fill and upload point_weights_gpu
}
```

`rmm_input_cpu_cache` (density matrix staging buffer) is persistent and always refilled:
```cpp
HostMatrix<scalar_type>& rmm_input_cpu = rmm_input_cpu_cache;
if (!rmm_input_cpu.is_allocated()) {
    // allocate (once)
}
get_rmm_input(rmm_input_cpu);  // ALWAYS refills from fortran_vars.rmm_output
```
Then always re-uploaded: `cudaMemcpy2DToArrayAsync(rmm_cuArray, ...)`.

## Confirmed Diagnostics

### Setup

Added diagnostic code to `g2g/cuda/iteration.cu` (solve_closed) and `g2g/partition.cpp` (solve).
The `static int` counters increment across all group calls globally (so "grp 1" = 1st group
processed, "grp 2" = 2nd group processed, in the first solve_groups call).

Test command:
```bash
cd test/LIO_test/03_fosfatoQMMM
../../liosolo/liosolo fos.in       # fgm=0.0 reference
../../liosolo/liosolo fos_fgm.in   # fgm=0.8 bugged
```

### [XC_FOCK] — Total XC Fock contribution (partition.cpp, in solve())

Printed in the closed-shell compute_rmm accumulation block (after parallel region, before
scatter into fortran_vars.rmm_output). Counter ≤ 3.

| fgm | call | sum | abs_sum |
|---|---|---|---|
| 0.0 | 1 | −2.62×10² | 6.16×10² |
| 0.8 | 1 | **−5.73×10⁷** | **6.37×10⁷** |

XC Fock is **~200,000× wrong** with fgm=0.8 from the very first solve_groups call.

### [fvt] — function_values_transposed (after cudaMemcpy2DToArrayAsync, synced)

| fgm | grp | nelems | m | npts | sum | abs | inG |
|---|---|---|---|---|---|---|---|
| 0.0 | 2 | ... | 38 | 236 | same | same | 0 |
| 0.8 | 2 | ... | 38 | 236 | same | same | **1** |

function_values_transposed: **IDENTICAL** between runs. ✓

Note: `inG=1` with fgm=0.8 means `inGlobal=true` — function values were cached.
With fgm=0.0, `inG=0` — computed fresh each time.

### [gvt] — gradient_values_transposed valid elements

Total-abs differs hugely (uninitialized padding in fgm=0.0), but **valid-element-only** abs:
```
grp 2: 6.464e+00  (SAME for both fgm=0.0 and fgm=0.8)
```
Confirmed: gradient_values_transposed (valid region) is **IDENTICAL**. ✓

### [factors] — factors_gpu after gpu_accumulate_point

| fgm | grp | npts | m | sum | abs | inG |
|---|---|---|---|---|---|---|
| 0.0 | 2 | 236 | 38 | **−1.071** | ... | 0 |
| 0.8 | 2 | 236 | 38 | **−0.814** | ... | 1 |

**DIFFERENT!** The XC factors diverge between runs for group 2.
Groups 1 and 3 have factors ≈ 0 in both runs (near-zero density → zero XC correction).

### [rmm_in] — rmm_input_cpu checksum (after get_rmm_input, before zeroing padding)

Partial data — only fgm=0.8 was captured before run was interrupted:

| fgm | grp | m | sum | abs |
|---|---|---|---|---|
| 0.0 | 2 | 38 | **MISSING** | **MISSING** |
| 0.8 | 2 | 38 | −1.854×10⁰ | 8.067×10¹ |

**This is the outstanding diagnostic still needed.**

## Summary of What Is/Isn't Known

| Input to density kernels | Same? |
|---|---|
| function_values_transposed | ✓ SAME |
| gradient_values_transposed (valid elements) | ✓ SAME |
| hessian_values_transposed | ✓ SAME |
| rmm_input_cpu (density matrix) | **UNKNOWN — needs comparison** |
| point_weights_gpu | UNKNOWN (not yet diagnosed) |

The `factors_gpu` diverge → bug is upstream of `gpu_accumulate_point`.
Given functions are identical, the culprit is almost certainly `rmm_input_cpu` or `point_weights_gpu`.

## Next Diagnostic Steps

1. **Run fgm=0.0 to get `[rmm_in] grp 2` checksum** and compare with fgm=0.8 (sum=-1.854e+00, abs=8.067e+01).
   - If different → density matrix differs → investigate why initial P_{μν} would differ between fgm settings
   - If same → add `[pwt]` diagnostic for `point_weights_gpu` (download and sum)

2. **If rmm_input matches**: add diagnostic after `gpu_compute_density` kernel (before accumulate)
   to compare `partial_densities_gpu` between runs.

3. **If still stuck**: check whether density kernel reads the texture correctly by verifying
   a single-point density value against a CPU reference calculation.

## Diagnostic Code Locations

All debug code is `static int counter` guarded (`<= 3`), minimally invasive:

- **`g2g/partition.cpp`** — `#include <cstdio>` added; `[XC_FOCK]` diagnostic at the
  `compute_rmm` accumulation point in `Partition::solve()` (closed-shell else branch).

- **`g2g/cuda/iteration.cu`** — four diagnostic blocks in `PointGroupGPU::solve_closed()`:
  1. `[rmm_in]` after `get_rmm_input()` (line ~207)
  2. `[fvt]`+`[gvt]` after `cudaMemcpy2DToArrayAsync` with sync (line ~269)
  3. `[factors]` before `if (compute_rmm)` after accumulate kernel (line ~519)
  4. `[compute_functions] CACHE HIT` in `compute_functions()` (line ~1018)

**IMPORTANT**: Remove all diagnostic code before committing any optimization.

## Key Code Paths

- `g2g/cuda/iteration.cu`: `PointGroupGPU::solve_closed()` — main GPU pipeline
- `g2g/cuda/iteration.cu`: `PointGroupGPU::compute_functions()` — basis function evaluation
- `g2g/partition.cpp`: `PointGroupGPU::deallocate()` — cache invalidation (lines 314–399)
- `g2g/global_memory_pool.cpp`: `GlobalMemoryPool::tryAlloc()` — pool accounting
- `g2g/partition.h`: `PointGroupGPU` member declarations (inGlobal, rmm_input_cpu_cache, etc.)

## Hypotheses Ranked by Likelihood

1. **rmm_input_cpu differs** between fgm=0.0 and fgm=0.8 in the first iteration.
   Could happen if Coulomb pool allocation interacts with density matrix initialization.

2. **point_weights_gpu differs** — unlikely since `!is_allocated()` check ensures first upload.

3. **GPU race condition** — unlikely; stream serialization is correct (blocking streams).

4. **Texture data corruption** — unlikely; cudaMemcpy2DToArrayAsync + density kernel all on
   stream 0, which serializes with transpose streams automatically.
