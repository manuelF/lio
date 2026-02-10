# Optimization: Replace Manual Loops with BLAS (CPU)

## Summary
The current CPU implementation of `solve_closed` (and `solve_opened`) manually implements matrix multiplications for the Density and RMM update phases.
*   **Density**: Computes `pd += Fi * w` inside triple nested loops. Effectively $ho = \text{diag}(F \cdot R \cdot F^T)$.
*   **RMM Update**: Computes `res += fvr[point] * fvc[point] * factors(point)` inside triple nested loops. Effectively $R_{out} += F^T \cdot \text{diag}(factors) \cdot F$.

These manual loops have poor arithmetic intensity and fail to utilize SIMD instructions efficiently compared to optimized BLAS libraries (OpenBLAS, MKL, etc.).

## Proposal
Replace these manual loops with calls to Level 3 BLAS functions: `cblas_dgemm` (Matrix-Matrix Multiplication) and `cblas_dsyrk` (Symmetric Rank-k Update).

## Impact
*   **Throughput**: BLAS libraries are highly optimized for CPU caches and vector units (AVX2/AVX-512). Expected 10x-50x speedup for the dense matrix operations.
*   **Code Quality**: Simplifies the code by removing complex nested loops.
*   **Portability**: Works on any CPU architecture with a BLAS implementation.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/cpu/iteration.cpp`: This is where `PointGroupCPU::solve_closed` and `PointGroupCPU::solve_opened` are implemented. The manual loops for "Compute Density" and "Update RMM" reside here.
    *   `g2g/Makefile`: Needs to ensure BLAS libraries (OpenBLAS/MKL) are linked correctly if `cpu=1` is selected.
    *   `g2g/common.h`: Might need to include `cblas.h` or similar headers.

*   **Correctness Impact**: **Low/Medium**. Floating-point operations are not associative. Replacing a summation loop with BLAS will change the order of operations, leading to bit-level differences in results. This is generally acceptable for DFT convergence, but it might fail strict regression tests that expect bit-exactness. Validation against `agua` and `fosfato` tests is crucial.

*   **Complexity**: The main challenge is mapping the existing flat pointers or custom `HostMatrix` layouts to BLAS parameters (LDA, M, N, K) and handling potential transpositions correctly (`CblasRowMajor` vs `CblasColMajor`).

## Sketch of Changes
1.  **Density Phase (`g2g/cpu/iteration.cpp`)**:
    *   Identify the loops computing `density_matrix` contributions.
    *   Allocate a temporary buffer $Y$ (size $P \times M$).
    *   Call `cblas_dgemm`:
        *   Layout: `CblasRowMajor`
        *   TransA: `CblasNoTrans` (for F), TransB: `CblasNoTrans` (for R) -> Check actual layout of R!
        *   $Y = 1.0 \cdot F \cdot R + 0.0 \cdot Y$.
    *   Replace the inner loops with a simple contraction: `density[p] += dot_product(F[p], Y[p])`.

2.  **RMM Phase (`g2g/cpu/iteration.cpp`)**:
    *   Identify the loops updating `rmm_output`.
    *   Construct a weighted matrix $F_w$ (size $P \times M$) where $F_w(p, i) = F(p, i) \cdot \sqrt{\text{factors}(p)}$.
    *   Call `cblas_dsyrk`:
        *   Layout: `CblasRowMajor`
        *   Uplo: `CblasLower`
        *   Trans: `CblasTrans` (since we want $F_w^T \cdot F_w$)
        *   $C = 1.0 \cdot F_w^T \cdot F_w + 1.0 \cdot C$ (Accumulate into `rmm_output`).

## Estimations
*   Speedup: Massive for large groups where $M$ (functions) and $P$ (points) are large.