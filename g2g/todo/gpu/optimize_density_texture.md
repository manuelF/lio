# Optimization: Replace Textures and Volatile Reductions in Density Kernel

## Summary

`gpu_compute_density` in `g2g/cuda/kernels/energy.h` uses two costly patterns that
can be improved for SM 6.1 (Pascal):

1. **Texture 2D (`tex2D<float>`) for RMM reads** — accessed via the `fetch(t,x,y)`
   macro defined in `iteration.cu` before including `energy.h`. On SM 6.1, the L1
   texture cache and the `__ldg()` read-only L1 cache are physically the same hardware.
   Textures add binding overhead in host code (cudaCreateTextureObject, CUDA array copy)
   and restrict the memory access pattern.

2. **Volatile shared memory reductions** — `warpReduceScalar` and `warpReduceVector3`
   in `energy.h` (lines ~50–75) use `volatile T* sdata` to prevent compiler reordering.
   This is the pre-Kepler idiom. Since Kepler (SM 3.5) `__shfl_down_sync` provides
   warp-level communication without shared memory, eliminating bank conflicts and
   reducing shared memory pressure (enabling higher occupancy).

## Code Audit: Current State

`energy.h:50`: `warpReduceScalar(volatile T* sdata, int tid)` — 6 unrolled writes
to volatile shared memory, each forcing a store-fence. On SM 6.1 this serializes
intra-warp communication unnecessarily.

`iteration.cu:191–205`: Full `cudaArray_t` + `cudaTextureObject_t` setup, including
`cudaMemcpy2DToArrayAsync` for every group. This staging into CUDA array format is
extra work that `__ldg` + a standard 2D-strided global pointer avoids.

`energy_derivs.h` also uses texture (`rmm_input_gpu_tex`) for the same RMM data.
Both kernels need updating together.

## Proposal

### Part A — Replace Volatile Warp Reductions with `__shfl_down_sync` (Easy, High Impact)

```cpp
// Replace warpReduceScalar:
template<class T>
__device__ __forceinline__ T warpReduceScalar(T val) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffffu, val, offset);
  return val;
}

// Replace warpReduceVector3:
template<class T>
__device__ __forceinline__ vec_type<T,3> warpReduceVector3(vec_type<T,3> v) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(0xffffffffu, v.x, offset);
    v.y += __shfl_down_sync(0xffffffffu, v.y, offset);
    v.z += __shfl_down_sync(0xffffffffu, v.z, offset);
  }
  return v;
}
```

Call sites in `energy.h:231–235`:
```cpp
scalar_type pd_val = warpReduceScalar(partial_pd);
// then: if (lane == 0) partial_densities_gpu[...] = pd_val;
```

This eliminates the shared memory allocations for `fj_sh`, `fgj_sh`, etc. — freeing
~1–4 KB of shared memory per block, which directly increases occupancy on SM 6.1
(20 SM units, 96 KB shared per SM).

**Estimated impact: 10–20% speedup for gpu_compute_density.**
See `optimize_warp_shuffle.md` for the standalone proposal on this pattern.

### Part B — Replace Texture with `__ldg` + Linear Pointer (Medium, Moderate Impact)

On SM 6.1, `__ldg(ptr)` triggers the same L1 read-only cache as textures:
```cpp
// Old: float val = tex2D<float>(rmm_tex, col, row);
// New: float val = __ldg(&rmm_ptr[row * stride + col]);
```

**Changes in iteration.cu:**
- Remove: `cudaArray_t rmm_cuArray`, `cudaTextureObject_t rmm_tex`,
  `cudaCreateTextureObject`, `cudaMemcpy2DToArrayAsync(rmm_cuArray, ...)`.
- Add: `CudaMatrix<float> rmm_input_gpu` (already computed by `get_rmm_input`);
  just keep the `cudaMemcpy` of `rmm_input_cpu → rmm_input_gpu`.
- Pass `rmm_input_gpu.data` and `rmm_input_gpu.width` (stride) to the kernel.

**Changes in energy.h:**
- Remove: `#define fetch(t,x,y) tex2D<float>(t,x,y)` and all `cudaTextureObject_t` args.
- Add: `const float* __restrict__ rmm_ptr, int rmm_stride` args.
- Access: `__ldg(&rmm_ptr[i * rmm_stride + j])`.

**Same changes apply to `energy_derivs.h`** — it uses the same texture pattern.

Access pattern analysis: RMM is accessed as `rmm[i * stride + bj + threadIdx.x]`
inside the batch loop — this IS coalesced (32 consecutive threads access 32 consecutive
columns). `__ldg` benefits from L1 caching here identically to texture.

**Estimated impact: 5–10% for gpu_compute_density, 5–10% for gpu_compute_density_derivs**
(removes texture setup overhead; bandwidth equivalent to texture for this access pattern).

## Priority: Part A First
Part A (warp shuffle) is lower risk and higher reward. Do it in isolation first.
Part B simplifies host code significantly but requires coordinated changes to both
`energy.h` and `energy_derivs.h`.

## Difficulty Assessment
- Part A (warp shuffle): **Low** — 30-line change in `energy.h`, no API changes.
- Part B (remove textures): **Medium** — touches `iteration.cu` + 2 kernel headers;
  requires careful stride passing and layout verification.

## Files to Modify
- `g2g/cuda/kernels/energy.h`: Both parts.
- `g2g/cuda/kernels/energy_derivs.h`: Part B (same texture replacement).
- `g2g/cuda/iteration.cu`: Part B (remove texture setup; ~40 lines of host code).

## Estimations
- Part A alone: **10–20% speedup for density phase**.
- Part B alone: **5–10% speedup for density + derivs phases**.
- Combined: **15–25% speedup for the density and force computation phases**.
- End-to-end: **8–15% overall** depending on the fraction of time in these kernels.
