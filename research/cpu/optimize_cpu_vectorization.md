# Optimization: Vectorization of Basis Function Loop (CPU)

## Summary
The CPU implementation of `compute_functions` in `g2g/cpu/functions.cpp` contains:
1.  Nested loops over points and functions (s/p/d cases) with complex branching.
2.  `if (exponent > 70.0) continue;` check inside the inner contraction loop.
3.  Scalar `exp()` calls.

This structure inhibits compiler auto-vectorization (SIMD) and makes poor use of modern CPU vector units (AVX2, AVX-512). The code effectively processes one scalar value at a time, leaving 7/8 (AVX2) or 15/16 (AVX-512) vector lanes idle.

## Proposal
Refactor `compute_functions` to expose explicit SIMD parallelism:
1.  **Flatten Loop**: Iterate over points in chunks (e.g., 8/16 points) to utilize SIMD registers.
2.  **Vectorized Exp**: Use vectorized math libraries (e.g., Intel VML/SVML, SLEEF) or compiler intrinsics (`_mm256_exp_ps`) for exponential calculation.
3.  **Masked Operations**: Instead of `if (exponent > 70.0) continue;`, use SIMD masks (`_mm256_cmp_ps`) to zero out results or skip expensive calculations only if *all* lanes in a vector are out of range.
4.  **Structure of Arrays (SoA)**: Ensure input/output data structures (`function_values`, `gX`, `gY`...) use SoA layout and aligned memory for efficient loads/stores.

## Impact
*   **Throughput**: 4x-8x speedup for basis function evaluation on modern CPUs.
*   **Latency**: Reduces the time spent on one of the most expensive CPU kernels.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/cpu/functions.cpp`: Complete rewrite of the kernel loops.
    *   `g2g/cpu/common.h` or similar: Add SIMD wrapper classes or include intrinsics headers (`immintrin.h`).
    *   `g2g/partition.h`: Need SoA layout for Points to enable efficient gathering of coordinates.

*   **Correctness Impact**: **Medium**. Vector math approximations (e.g., `_mm256_exp_ps` vs `std::exp`) can have slightly lower precision (1 ulp difference). This is usually fine for Physics/ML but needs validation.
*   **Maintenance**: Intrinsics code is hard to read and maintain. Ideally, use a lightweight wrapper or `#pragma omp simd` hints if the compiler is smart enough (often not for `exp`).

## Sketch of Changes
1.  **Chunk Processing**:
    *   Outer loop: `for (int p = 0; p < n_points; p += 8)` (assuming AVX2).
    *   Load positions: `__m256 x = _mm256_load_ps(&X[p])`.

2.  **Vectorized Kernel**:
    *   Loop over shells.
    *   Compute distances squared: `d2 = (x-xn)^2 + ...`.
    *   Compute exponent: `arg = alpha * d2`.
    *   Compare: `mask = _mm256_cmp_ps(arg, threshold, _CMP_LT_OQ)`.
    *   If `_mm256_movemask_ps(mask) == 0` continue (all points pruned).
    *   Compute Exp: `val = _mm256_exp_ps(arg)` (Need an implementation of this).
    *   Blend: `val = _mm256_blendv_ps(zero, val, mask)`.
    *   Store results.

## Estimations
*   Speedup: Significant improvement for the basis function evaluation phase.