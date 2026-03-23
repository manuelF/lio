# Optimization: Replace Volatile Shared Memory Reductions with Warp Shuffles

## Summary
`g2g/cuda/kernels/energy.h` contains `warpReduceScalar` and `warpReduceVector3`
(lines ~50–75) that use `volatile T*` shared memory to synchronize within a warp.
This is the pre-Kepler (SM < 3.5) idiom, required before warp-level intrinsics
existed. Since Kepler, `__shfl_down_sync` provides intra-warp communication directly
in registers — no shared memory required, no compiler-barrier fence, no bank-conflict
risk.

On SM 6.1 (Pascal), volatile shared memory writes create store fences that serialize
the warp pipeline. Replacing them with `__shfl_down_sync` removes these fences and
frees ~512–2048 bytes of shared memory per block (depending on scalar_type), directly
enabling **higher occupancy** on the GTX 1080's 20 SMs.

## Code Audit: Current Implementation

```cpp
// energy.h:50
template<class T>
__device__ __forceinline__ void warpReduceScalar(volatile T* sdata, int tid) {
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

// energy.h:63
template<class T>
__device__ __forceinline__ void warpReduceVector3(volatile vec_type<T,3>* sdata, int tid) {
  // Same 5-step pattern for .x, .y, .z
}
```

Call sites: `energy.h:231` (`warpReduceScalar(fj_sh, tid)`) and `energy.h:233–235`
(three `warpReduceVector3` calls for gradient components).

The shared arrays `fj_sh`, `fgj_sh`, `fh1j_sh`, `fh2j_sh` are all sized
`DENSITY_BLOCK_SIZE=64` of `scalar_type`. In LDA mode that's 64×4=256 bytes; in GGA
with 4 arrays of vec_type3 it's 64×3×4 = 768 bytes per array × 4 = ~3 KB per block.

## Proposed Replacement

```cpp
// New: register-only warp reduce (no shared memory needed)
template<class T>
__device__ __forceinline__ T warpReduceScalar(T val) {
  val += __shfl_down_sync(0xffffffffu, val, 16);
  val += __shfl_down_sync(0xffffffffu, val,  8);
  val += __shfl_down_sync(0xffffffffu, val,  4);
  val += __shfl_down_sync(0xffffffffu, val,  2);
  val += __shfl_down_sync(0xffffffffu, val,  1);
  return val;
}

template<class T>
__device__ __forceinline__ vec_type<T,3> warpReduceVector3(vec_type<T,3> v) {
  v.x = warpReduceScalar(v.x);
  v.y = warpReduceScalar(v.y);
  v.z = warpReduceScalar(v.z);
  return v;
}
```

Call site update in `energy.h`:
```cpp
// Old:
warpReduceScalar(fj_sh, tid);
if (tid == 0) partial_densities_gpu[...] = fj_sh[0];

// New:
T pd_reduced = warpReduceScalar(partial_pd_reg);  // partial_pd_reg is a register variable
if (lane_id == 0) partial_densities_gpu[...] = pd_reduced;
```

The key refactor: accumulate per-thread partial sums into **register variables**
rather than indexing into `fj_sh[tid]`, then call the register-based shuffle reduce.

## Cross-Warp Reduction (if block size > 32)
`DENSITY_BLOCK_SIZE=64` means 2 warps per block. After intra-warp reduction, lane 0
of each warp writes to a small shared buffer (only 2 entries!), then warp 0 reduces:
```cpp
__shared__ T warp_results[2];  // Only 2 entries instead of 64!
if (lane_id == 0) warp_results[warp_id] = val_reduced;
__syncthreads();
if (warp_id == 0) {
  T final = (lane_id < 2) ? warp_results[lane_id] : T(0);
  final = warpReduceScalar(final);
  if (tid == 0) output[...] = final;
}
```
This reduces shared memory from 64×4=256 bytes to 2×4=8 bytes per scalar.
For GGA with 4 vec3 fields: from ~3 KB to ~96 bytes per block.

## Impact

### Occupancy Improvement
SM 6.1 shared memory: 96 KB per SM. Current per-block shared usage ~3 KB (GGA).
With `DENSITY_BLOCK_SIZE=64` → 48 threads = 1.5 warps → 2 blocks per SM (shared limited).
After: ~96 bytes per block → 32+ blocks per SM (register-limited at ~32).
**Occupancy doubles for GGA mode.**

### Instruction Throughput
`__shfl_down_sync` on SM 6.1: 4-cycle throughput (8 ops per SM per cycle).
Volatile shared memory write+read: 10–20 cycles (L1 cache round-trip + fence).
**Per warp: 5 shuffles at ~20 cycles total vs 5 volatile r/w at ~75 cycles.**

### Estimated Kernel Speedup
- Intra-warp reduction is a small fraction of total kernel time.
- The dominant benefit is occupancy: more blocks active simultaneously hides
  latency of global memory loads.
- **Density kernel alone: 10–20% speedup.**
- **Force derivs (energy_derivs.h has same pattern): 5–15% speedup.**

## Difficulty Assessment
**Low — this is a quick win.**

- Files: `g2g/cuda/kernels/energy.h` only.
  `energy_derivs.h` does not use warp reductions (it's per-point, no reduction needed).
- No API changes, no host-code changes.
- The refactor is purely mechanical: accumulate into register variables, call
  register-based shuffle, write lane-0 result.
- Risk: **None** — `__shfl_down_sync` has been stable since CUDA 9 (mask=0xffffffff
  is correct for full-warp reductions). SM 6.1 supports it.

## Files to Modify
- `g2g/cuda/kernels/energy.h`: Replace `warpReduceScalar`/`warpReduceVector3`
  implementations and update all 4 call sites.

## Estimations
- Implementation time: **2–4 hours**.
- Density phase speedup: **10–20%**.
- End-to-end improvement: **5–12%** (density phase is significant but not dominant).
- Risk: **Negligible** — pure refactor of well-understood pattern.

## Recommendation
**Do this first.** It is the lowest-risk highest-ratio optimization in the GPU kernel
suite. Combine it with Part A of `optimize_density_texture.md`.

---

## Status: DONE (2026-03-04)

### Changes implemented
- `energy.h`: `warpReduceScalar(volatile T*, int)` / `warpReduceVector3(volatile vec_type<T,3>*, int)` → register-based `T warpReduceScalar(T)` / `vec_type<T,3> warpReduceVector3(vec_type<T,3>)`. Cross-warp step uses 2 shared-memory slots (reusing the existing bj-loop caching arrays). Reduced `__syncthreads()` in the reduction from 2 → 1. Also added `full_block` + `#pragma unroll 4` (already present in energy.h; verified correct).
- `energy_open.h`: Same shuffle treatment applied to both alpha+beta reductions simultaneously. Reduced `__syncthreads()` from 4 → 2 in the reduction. Added `full_block` optimization + `#pragma unroll 4` to the inner bj-loop (was missing before).

### Measured results (SM 6.1, float, nvprof + ptxas)
| Kernel | Variant | Registers | smem (before) | smem (after) | stall_sync |
|---|---|---|---|---|---|
| gpu_compute_density | lda=true (LDA) | 32 | 2560 bytes | **256 bytes** | ~30% |
| gpu_compute_density | lda=false (GGA) | 56 | 2560 bytes | 2560 bytes | ~35% |
| gpu_compute_density_opened | lda=true (LDA) | 40 | ~2560 bytes | **256 bytes** | ~35% |

GGA smem is unchanged (2560 bytes needed for the bj-loop caching arrays fgj_sh/fh1j_sh/fh2j_sh).
LDA smem 10× reduction → **100% theoretical occupancy** for closed-shell LDA (reg-limited: 65536/32/64=32 blocks = thread limit).

### Key insight: open-shell GGA register pressure
Production open-shell always uses `lda=false` (GGA). That variant has **93 registers** → only **34% theoretical occupancy** on SM 6.1 (vs 56% for closed-shell GGA at 56 regs). Splitting alpha/beta into two calls to `gpu_compute_density` would restore 56% occupancy at the cost of reading function values twice. Tracked in a separate TODO.
