# Optimization: Replace Custom RMM Update with cuBLAS SYRK

## Summary
`gpu_update_rmm` in `g2g/cuda/kernels/rmm.h` manually implements a symmetric rank-k
update (SYRK): $R_{ij} \mathrel{+}= \sum_p w_p \cdot F_{pi} \cdot F_{pj}$.
The kernel uses a clever `sqrtf`-based lower-triangle block mapping and 16×16 shared
memory tiles with `functions_i_local[16][17]` (17-padded for bank-conflict avoidance).
It is well-written but cannot match the instruction-level tuning of cuBLAS SYRK,
which uses register blocking, software pipelining, and architecture-specific warp
scheduling on SM 6.1.

## Code Analysis

Current kernel config: `RMM_BLOCK_SIZE_XY=16`, grid = lower-triangle blocks via
`sqrtf`. Each block loads a 16×16 tile of F into shared memory with bank-conflict
padding, then computes the 16×16 output tile as an outer product accumulation.

This is functionally correct but the cuBLAS `cublasSsyrk` equivalent will:
- Use wider register tiles (e.g., 64×64 or 128×128 with vectorized loads)
- Pipeline global memory loads with arithmetic
- Apply warp-level predication for partial tiles

## Prerequisite
Requires building with `cuda=2` (enables cuBLAS). Check `g2g/Makefile.cuda`:
the `cuda=2` path already links `-lcublas` and defines `BLAS_GPU`.

## Proposal

1. **Weight application kernel** (new, trivial): Scale each row p of the function
   matrix by `sqrt(weight[p])`:
   ```cpp
   // New kernel: g2g/cuda/kernels/apply_weights.h
   __global__ void gpu_apply_sqrt_weights(float* F_out, const float* F_in,
                                          const float* weights, int M, int P) {
     int p = blockIdx.x * blockDim.x + threadIdx.x;
     int i = blockIdx.y;
     if (p < P) F_out[i * P + p] = F_in[i * P + p] * sqrtf(weights[p]);
   }
   ```

2. **Replace `gpu_update_rmm` launch** in `iteration.cu` with:
   ```cpp
   // Apply weights: F_w(i,p) = F(i,p) * sqrt(w(p))
   gpu_apply_sqrt_weights<<<grid, block>>>(F_weighted, function_values_transposed,
                                           point_weights_gpu, group_m, npoints);
   // SYRK: R += F_w * F_w^T  (lower triangle, N=group_m, K=npoints)
   float alpha = 1.0f, beta = 1.0f;
   cublasSsyrk(cublas_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
               group_m, npoints, &alpha,
               F_weighted, group_m,   // LDA = group_m (column-major: M×P stored as P rows of M)
               &beta, rmm_output_gpu, group_m);
   ```
   Note: The transposed function_values matrix is already M×P (M rows of P elements,
   row-major). As cuBLAS column-major this is P×M, so we use `CUBLAS_OP_T` with LDA=P
   — OR keep the existing row-major pointer and use `CUBLAS_OP_N` with LDA=M.
   Double-check the layout against the current kernel before committing.

3. **Temporary buffer**: Allocate `F_weighted` as a `CudaMatrix<float>(group_m, npoints)`
   in `PointGroupGPU` or reuse an existing scratch buffer. Size is identical to the
   existing `function_values_transposed` array.

## Impact
- **RMM phase alone**: 2–5× speedup for the `gpu_update_rmm` kernel.
- **End-to-end**: RMM update is one of the longer phases. Expect **5–15% overall**
  for typical simulations where `compute_rmm=true`.
- Scales better with system size (M² grows; cuBLAS handles this better than the
  custom 16×16 tile kernel).

## Difficulty Assessment
**Low–Medium** (requires `cuda=2`; main risk is layout alignment).

- Files to modify:
  - `g2g/cuda/iteration.cu`: Replace `gpu_update_rmm` launch with SYRK call.
  - `g2g/cuda/kernels/rmm.h`: Keep as reference/fallback with `#ifdef BLAS_GPU` guard.
  - `g2g/partition.h`: Add `CudaMatrix<float> F_weighted_scratch` to `PointGroupGPU`.
- Correctness: **Low risk** — SYRK is numerically equivalent to the custom kernel
  modulo floating-point reordering. The result is the lower-triangle only; the
  existing `add_rmm_output` scatter only reads the lower triangle, so no change needed.
- Testing: Run `agua`, `fosfato`, `Fe3H2O6` tests with `cuda=2`. Energy should match
  to within SCF convergence tolerance (~1e-6 Hartree).

## Estimations
- RMM kernel speedup: **2–5×**.
- End-to-end: **5–15%** for RMM-dominated runs.
- Implementation time: 1–2 days (weight kernel + SYRK call + validation).
