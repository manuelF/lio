# Optimization: Reduce Open-Shell GGA Register Pressure via Split Kernel Calls

## Summary

The production open-shell GGA density kernel (`gpu_compute_density_opened<float,*,*,false>`)
uses **93 registers** per thread. With `DENSITY_BLOCK_SIZE=64` threads:

```
register limit: 65536 / 93 / 64 = 11 blocks per SM
thread limit:   2048 / 64 = 32 blocks per SM
→ theoretical occupancy = 11 × 64 / 2048 = 34%
```

This is much lower than the closed-shell GGA kernel (56 regs → 18 blocks → 56% occupancy).
On a latency-dominated kernel (~50% `stall_exec_dependency` from texture fetches),
higher occupancy directly improves throughput by allowing more in-flight warps to hide
memory latency.

## Root Cause

The kernel processes **alpha AND beta spins** in a single launch, using two RMM textures.
This doubles nearly every live variable:

- Accumulation scalars: `w_a, w_b, w2_a, w2_b` (4 vs 2)
- GGA vectors: `w3_a/b, ww1_a/b, ww2_a/b, w32_a/b, ww12_a/b, ww22_a/b` (12 vec3 vs 6 vec3 = 36 regs vs 18)
- Output: `partial_density_a/b, dxyz_a/b, dd1_a/b, dd2_a/b` (8 vec3 vs 4 vec3)

Total extra: ~37 registers vs the closed-shell GGA kernel (93 vs 56).

## Additional Dead Code

`gpu_compute_density_opened` has three **dead template parameters**:
- `bool compute_energy` — never used in the kernel body
- `bool compute_factor` — never used in the kernel body
- `const scalar_type* point_weights` — passed as a parameter, never read

These cause the compiler to generate 3 identical object files (one per
`<true,true,false>`, `<true,false,false>`, `<false,true,false>` instantiation),
wasting compile time and instruction cache.

## Proposed Optimization

### Option A: Split into two kernel calls per spin (Recommended)

Replace the single open-shell kernel call with two calls to the existing
`gpu_compute_density<scalar_type, /*lda=*/false>` kernel:

```cpp
// Current:
gpu_compute_density_opened<scalar_type, true, true, false><<<...>>>(
    tex_a, tex_b, point_weights, points, fv, gv, hv, m,
    rho_a, dxyz_a, dd1_a, dd2_a, rho_b, dxyz_b, dd1_b, dd2_b);

// Proposed:
gpu_compute_density<scalar_type, false><<<...>>>(
    tex_a, /*energy=*/nullptr, /*factor=*/nullptr, nullptr, points,
    fv, gv, hv, m, rho_a, dxyz_a, dd1_a, dd2_a);
gpu_compute_density<scalar_type, false><<<...>>>(
    tex_b, nullptr, nullptr, nullptr, points,
    fv, gv, hv, m, rho_b, dxyz_b, dd1_b, dd2_b);
```

**Trade-offs:**
- ✓ 56 regs → 56% occupancy (vs 34%) — 22 percentage points higher
- ✓ Removes 93-reg bottleneck; same kernel code path as closed-shell
- ✓ Can overlap with streams (tex_a launch then tex_b launch with different streams)
- ✗ Reads `function_values`, `gradient_values`, `hessian_values` **twice** from global memory
- ✗ Two kernel launch overheads (negligible for large groups, ~2 µs for small groups)
- ✗ Requires removing `gpu_compute_density_opened` entirely or keeping it for reference

**Memory bandwidth analysis:**
For M basis functions and P grid points:
- Function values: M × P × 4 bytes = 4MP bytes read twice = 8MP bytes total
- Gradient/hessian values (GGA): 4 × M × P × 4 bytes × 2 reads = 32MP bytes total
- RMM textures: M² × 4 bytes × 2 reads (unchanged) = 8M² bytes

For typical group sizes (M=64, P=100): extra bandwidth = 8×64×100 = 51 KB.
With 192 GB/s peak (GTX 1080 SX), 51 KB costs ~0.27 µs extra. Very small.

The occupancy improvement from 34% → 56% enables ~1.6× more warp slots to hide
texture fetch latency — likely a net win for all but the smallest groups.

### Option B: `__launch_bounds__` to force moderate spilling

```cpp
template <class scalar_type, bool lda>
__launch_bounds__(DENSITY_BLOCK_SIZE, 16)  // 16 blocks min → 64 regs max
__global__ void gpu_compute_density_opened(...) { ... }
```

With 16 blocks minimum: 65536 / 16 / 64 = 64 regs max → ~29 regs spilled to local memory.
Spills go through L2 (~100 cycles per access). Not recommended without profiling.

## Implementation Notes

The kernel also has dead `compute_energy`/`compute_factor` template params that should
be removed when simplifying:

```cpp
// Before: 3 template params, none used in body
template <class scalar_type, bool compute_energy, bool compute_factor, bool lda>
__global__ void gpu_compute_density_opened(cudaTextureObject_t tex1, cudaTextureObject_t tex2,
    const scalar_type* point_weights, ...)  // point_weights also unused

// After (if switching to Option A): whole function deleted
```

Iteration.cu call sites (lines 719, 737, 771) all pass `lda=false` — confirming the
LDA variant of `gpu_compute_density_opened` is never used in production.

## Impact Estimate

- Open-shell GGA density phase: **1.3–1.8× speedup** (from 34% → 56% occupancy)
- End-to-end for open-shell calculations: **10–25% improvement** (density is major phase)
- Closed-shell systems: **no change**

## Difficulty Assessment

**Low–Medium.**
- Option A: ~50 lines changed in `iteration.cu`, remove `energy_open.h` or
  deprecate it. No algorithmic change.
- Risk: Function value reads happen twice — verify that L2 cache handles this
  efficiently (likely yes for typical group sizes where fv fits in L2).

## Files to Modify (Option A)

- `g2g/cuda/iteration.cu`: Replace `gpu_compute_density_opened` calls (lines 719, 737, 771)
  with two `gpu_compute_density<scalar_type,false>` calls per launch site.
- `g2g/cuda/kernels/energy_open.h`: Keep for reference or delete.
- `g2g/cuda/iteration.cu`: Remove `#include "kernels/energy_open.h"` if deleting.
