# Optimization: Reformulate Density Computation as GEMM

## Summary
`gpu_compute_density` in `g2g/cuda/kernels/energy.h` computes the density
$\rho(p) = \sum_{ij} R_{ij} \phi_i(p) \phi_j(p)$ by iterating over function pairs
$(i,j)$ for each point $p$. With the current texture-based RMM access, the kernel
loads O(M²) floats from global memory per point. For typical group sizes
(M~30, P~512), this is ~900 float reads per point × 512 points = ~1.8 MB of
reads per group, with very little data reuse (each RMM row is read M times).

**Reformulation via GEMM**: Pre-compute the intermediate matrix $Y = F \cdot R$
(P×M = P×M · M×M), then compute $\rho(p) = \sum_k F_{pk} Y_{pk}$ (dot product per point).

$Y$ has arithmetic intensity $O(M)$ (each RMM element is used M times), making it
compute-bound for M≥32. The dot product is trivially memory-streaming.

## Code Audit: Current Structure

`energy.h` grid: `dim3(npoints, divUp(group_m, 2*DENSITY_BLOCK_SIZE))`.
Each block processes one point and a range of function pairs. The reduction
uses shared memory (`fj_sh[]`) and volatile warp reductions.

The RMM is stored in a CUDA array (`rmm_cuArray`) and accessed via `tex2D<float>`.
This staging (H2D → CUDA array → texture) is extra work that GEMM eliminates.

## Proposal

### Step 1 — Compute Y = F·R via cuBLAS SGEMM (requires cuda=2)

```cpp
// F_gpu: group_m × npoints (M×P row-major) = transposed function values
// R_gpu: group_m × group_m (M×M, lower triangle from get_rmm_input, padded to symmetric)
// Y_gpu: npoints × group_m (P×M, intermediate)
float alpha = 1.0f, beta = 0.0f;
cublasSgemm(cublas_handle,
            CUBLAS_OP_T,     // F stored as M×P (col-major: P×M)
            CUBLAS_OP_N,     // R stored as M×M
            npoints, group_m, group_m,
            &alpha,
            function_values_transposed.data, group_m,  // F: M×P → treat as P×M in col-major
            rmm_input_gpu.data, group_m,               // R: M×M
            &beta,
            Y_gpu.data, npoints);
```

Layout note: `function_values_transposed` is M×P (M rows of P elements). cuBLAS
col-major sees this as P×M. Adjust Op accordingly — verify against existing kernel
output before replacing.

### Step 2 — Dot Product Kernel (lightweight, replaces gpu_compute_density)

```cpp
__global__ void gpu_compute_density_from_Y(
    const float* Y, const float* F,
    const float* gxF, const float* gyF, const float* gzF,
    float* pd, float* tdx, float* tdy, float* tdz,
    int npoints, int M) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npoints) return;
  float pd_p = 0.f, tx = 0.f, ty = 0.f, tz = 0.f;
  for (int k = 0; k < M; ++k) {
    float yk = Y[p * M + k];
    float fk = F[p * M + k];  // need non-transposed F (P×M layout)
    pd_p += yk * fk;
    tx   += yk * gxF[p * M + k];
    ty   += yk * gyF[p * M + k];
    tz   += yk * gzF[p * M + k];
  }
  pd[p] = pd_p;
  if (tdx) { tdx[p] = 2.f * tx; tdy[p] = 2.f * ty; tdz[p] = 2.f * tz; }
}
```

This kernel is P-parallel, M-serial per thread. For M=30, P=512 this fits entirely
in L1/registers and runs in a handful of microseconds.

### Step 3 — Reuse Y for gpu_compute_density_derivs

The current `energy_derivs.h` also accesses RMM via texture and does O(M²·P) work.
With Y already computed, force derivs reduce to:
```
dd_a(nuc_i) -= sum_i Y[p,i] * grad_F[p,i]    // O(M·P) instead of O(M²·P)
```
This is the **largest win**: force derivs are O(M²·P) today vs O(M·P) with Y cached.

## Impact

| Phase | Current | With GEMM |
|---|---|---|
| Density (energy.h) | O(M²·P), texture-bound | O(M²·P) GEMM + O(M·P) dot = compute-bound |
| Density derivs (energy_derivs.h) | O(M²·P), texture-bound | O(M·P) with cached Y |
| Memory for RMM | CUDA array + texture | Plain CudaMatrix (simpler host code) |

- **gpu_compute_density phase alone**: 2–5× speedup (GEMM vs texture loop for M≥16).
- **gpu_compute_density_derivs**: 3–10× speedup (O(M²→M) reduction with Y reuse).
- **End-to-end**: **15–35% overall speedup** for force-computing runs.
  Energy-only runs: **10–20%**.

Note: The "5–10× density speedup" from the original file was optimistic for SM 6.1.
cuBLAS SGEMM on a GTX 1080 achieves ~8 TFLOPS for large matrices, but group sizes
(M~30, P~512) are too small to reach peak cuBLAS efficiency. Realistic gain is 2–5×
for density alone, but the energy_derivs improvement is the real prize.

## Difficulty Assessment
**High** (unchanged from original, but reason is clearer now)

Key risks:
1. **Matrix layout**: F is stored transposed in `function_values_transposed` (M×P).
   cuBLAS uses column-major. Passing the right `Op`, `LDA`, `LDC` requires careful
   derivation. Off-by-one in stride → silent numerical corruption.
2. **RMM symmetry**: `get_rmm_input` fills only the lower triangle. cuBLAS GEMM
   needs a full symmetric matrix OR use `cublasSsymm` instead of `cublasSgemm`.
   `cublasSsymm` (Symmetric Matrix-Matrix multiply) exploits half the storage and
   reads only the lower triangle: perfect fit.
3. **Y buffer lifetime**: Y must persist from density phase through force phase
   (two separate code regions in `solve_closed`). Allocate in `PointGroupGPU`.
4. **Gradient function layout**: `gX`, `gY`, `gZ` are separate M×P matrices
   (currently stored as `HostMatrix::row(point)`). They need to be P×M on GPU for
   the dot product kernel.

## Files to Modify
- `g2g/cuda/iteration.cu`: Replace `gpu_compute_density` launch + remove texture setup;
  add `cublasSsymm` call; add Y buffer allocation.
- `g2g/cuda/kernels/energy.h`: Replace with lightweight dot-product kernel.
- `g2g/cuda/kernels/energy_derivs.h`: Replace O(M²) loop with O(M) reuse of Y.
- `g2g/partition.h`: Add `CudaMatrix<float> Y_scratch` to `PointGroupGPU`.

## Estimations
- Density phase: **2–5×** speedup.
- Force derivs phase: **3–10×** speedup (O(M²→M) with Y reuse).
- End-to-end: **15–35%** for force runs; **10–20%** for energy-only.
- Implementation: 1–2 weeks including layout verification and validation.
