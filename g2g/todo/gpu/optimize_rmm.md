# Optimization: Replace Custom RMM Update with cuBLAS

## Summary
The current implementation of `gpu_update_rmm` manually calculates a matrix multiplication to update the Reduced Density Matrix (RMM). This involves a custom CUDA kernel with complex indexing logic (`sqrtf` for block mapping) and manual shared memory blocking. This operation is mathematically equivalent to a Symmetric Rank-k update (SYRK) or a General Matrix Multiplication (GEMM).

## Proposal
Replace the `gpu_update_rmm` kernel call with `cublasDsyrk` (for double precision) or `cublasSsyrk` (for single precision) from the cuBLAS library.

The operation is: $R_{ij} = \sum_p F_{pi} F_{pj} w_p$.
This can be formulated as $C = \alpha A^T A + \beta C$, where $A$ is the weighted function values matrix.

## Impact
*   **Performance**: `cuBLAS` is highly tuned for all GPU architectures. It will significantly outperform the custom kernel, especially for large matrices, by utilizing better tiling, register blocking, and instruction pipelining.
*   **Code Maintenance**: Reduces code complexity by removing a complex custom kernel (`g2g/cuda/kernels/rmm.h`).

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Replace kernel launch with `cublasDsyrk`.
    *   `g2g/cuda/kernels/rmm.h`: **DELETE** this file (or deprecate).
    *   `g2g/cuda/kernels/functions.h`: Need a kernel to apply weights: `F_weighted = F * sqrt(w)`.

*   **Correctness Impact**: **Low**. Standard linear algebra.
*   **Memory**: Requires temporary buffer for `F_weighted` ($P \times M$).

## Sketch of Changes
1.  **Weighting Kernel**:
    *   `apply_weights<<<...>>>(functions, weights, weighted_functions)`
    *   `val = functions[idx] * sqrt(weights[point_idx])`.

2.  **cuBLAS Call**:
    *   `cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, N, K, alpha, weighted_functions, LDA, beta, rmm_output, LDC)`.
    *   Note: `CUBLAS_OP_T` because we want $F^T \cdot F$. $N=M$, $K=P$.

## Estimations
*   Speedup: Expected 2x-5x faster for the RMM update phase.