# Optimization: gpu_compute_density_derivs — O(M²·P) Bottleneck

## Summary
`gpu_compute_density_derivs` in `g2g/cuda/kernels/energy_derivs.h` is the force
computation kernel. It is the **single most expensive kernel** for simulations with
force computation (molecular dynamics, geometry optimization).

Its complexity is O(M²·P) — quadratic in the number of basis functions M per group.
For a group with M=60 functions and P=512 points: 60²×512 = 1.84M multiply-adds,
all reading from a texture (RMM) that must be fully re-loaded per point.

By contrast, the GEMM formulation (see `optimize_density_gemm.md`) reduces force
derivs to O(M·P) by reusing the intermediate matrix Y = F·R.

## Code Audit: Current Structure

```
energy_derivs.h: outer loop over `point` (blockIdx.x * blockDim.x)
  inner loop over bi batches (DENSITY_DERIV_BATCH_SIZE=128):
    load nuc[bi..bi+128] into shared memory nuc_sh[]
    inner loop over bj batches (DENSITY_DERIV_BATCH_SIZE=128):
      load rmm[bi+i, bj..bj+128] into shared rdm_sh[] (texture)
      inner loop over functions j in batch:
        w += rdm_sh[j] * fv[bj+j] * factor  (where factor = 2 if i==j else 1)
      compute w_ii = sum_j rmm[ii,j] * fv[j]
    dd[nuc_sh[i]] -= w_ii * gxv[ii]  (atomic add to global memory)
```

Key observation: the `bj` inner loop reads one full RMM row per outer-loop `bi`
element. Total texture reads: M×M per point. For M=60: 3600 texture reads per point.

**Memory bound**: each point requires M² texture reads of 4 bytes = 14.4 KB.
At 320 GB/s L2 bandwidth (SM 6.1), theoretical limit: 320e9/(14400) = 22M points/s.
With 512 points/group: each group takes ~23 µs just in memory bandwidth.

## Proposal: Reuse Y from Density Phase (best approach)

After implementing `optimize_density_gemm.md`, Y = F·R is available on GPU with
shape P×M (npoints × group_m). The force derivs simplify dramatically:

```
For each point p:
  For each function ii:
    w_ii = Y[p, ii]    // O(1) — already computed!
    dd[nuc[ii]] -= w_ii * grad_F[p, ii]   // O(M·P) total
```

This eliminates the O(M²) inner loop entirely. The kernel becomes:
```cpp
__global__ void gpu_compute_density_derivs_from_Y(
    const float* Y,              // P×M intermediate matrix from GEMM
    const float* gxv, const float* gyv, const float* gzv,  // M×P gradient matrices
    const uint*  nuc,            // M: function→nucleus map
    float* force_x, float* force_y, float* force_z,        // N_atoms output
    int npoints, int M) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npoints) return;
  for (int ii = 0; ii < M; ++ii) {
    float w = Y[p * M + ii];    // replaces the entire bi/bj double-loop
    atomicAdd(&force_x[nuc[ii]], -w * gxv[ii * npoints + p]);
    atomicAdd(&force_y[nuc[ii]], -w * gyv[ii * npoints + p]);
    atomicAdd(&force_z[nuc[ii]], -w * gzv[ii * npoints + p]);
  }
}
```

**Complexity reduction: O(M²·P) → O(M·P).** For M=60, P=512: from 1.84M to 30.7K
operations — a **60× reduction** in work. Combined with eliminating texture reads:
each thread now reads M elements from Y (already in L1 from density phase) and M
gradient values — all sequential, coalesced access.

## Standalone Improvement (if GEMM not yet done)

Even without Y, the current kernel can be improved:

### 1. Remove texture, use `__ldg` + flat pointer
Same as `optimize_density_texture.md` Part B. Eliminates CUDA array staging.

### 2. Reduce atomic contention on force array
Multiple threads writing to the same `force[nuc[ii]]` cause atomic contention.
Improve by:
- Sorting threads by `nuc[ii]` (warp-level reordering is complex)
- Using shared memory accumulation per warp, then one atomic add per nucleus per warp
- Using `__atomic_add` in registers via `__shfl_xor_sync` within warp if all
  threads in a warp target the same nucleus (possible for compact groups)

### 3. Loop unrolling via `#pragma unroll N`
The bj inner loop with DENSITY_DERIV_BATCH_SIZE=128 can be partially unrolled
to reduce loop overhead:
```cpp
#pragma unroll 4
for (int j = 0; j < DENSITY_DERIV_BATCH_SIZE; ++j) { ... }
```

### Standalone improvement estimate: 20–40% speedup.

## Impact

| Approach | Complexity | Speedup |
|---|---|---|
| Current (texture + O(M²·P)) | Baseline | 1× |
| Remove texture + __ldg | O(M²·P) | 1.1–1.3× |
| With Y reuse (GEMM) | O(M·P) | 5–20× |

For force-computing simulations (MD, geometry opt):
- Without GEMM: **10–30% speedup** from texture removal + atomic improvement.
- With GEMM (Y reuse): **force derivs phase 5–20× faster** → **15–40% end-to-end**.

The Y reuse improvement is by far the highest-value optimization in the entire
codebase for force-computing runs. It directly reduces asymptotic complexity.

## Difficulty Assessment
**Medium** (standalone improvements) | **High** (Y reuse — requires GEMM first)

Standalone:
- Files: `energy_derivs.h`, `iteration.cu` (texture removal).
- Risk: **Low** — same texture→__ldg migration as density kernel.

Y reuse:
- Requires `optimize_density_gemm.md` to be implemented first.
- Files: `energy_derivs.h` (new kernel), `iteration.cu` (pass Y buffer pointer).
- Risk: **Medium** — factor-of-2 conventions (symmetric RMM, open-shell) must be
  verified. The current `w_ii * factor` where `factor = 2 if i==j else 1` is already
  absorbed into Y since R is symmetric. Validate against `agua` forces reference.

## Files to Modify
- `g2g/cuda/kernels/energy_derivs.h`: Kernel replacement.
- `g2g/cuda/iteration.cu`: Pass Y buffer; remove texture setup.
- `g2g/partition.h`: Y buffer is already needed for GEMM (shared with density).

## Estimations
- Standalone texture removal: **1.1–1.3× force kernel speedup**, **3–8% end-to-end**.
- Y reuse (post-GEMM): **5–20× force kernel speedup**, **15–40% end-to-end**.
- Combined with warp shuffle (optimize_warp_shuffle.md): additional 10% on density.
