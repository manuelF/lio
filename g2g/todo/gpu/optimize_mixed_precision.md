# Optimization: Mixed Precision and Tensor Cores

## Summary
Most of the current kernels are implemented in `scalar_type` (float or double) precision. For DFT applications, especially for the far-field density calculation or integration grid points with low weight, full double precision may not be strictly necessary. Modern NVIDIA GPUs (Volta+) have Tensor Cores and dedicated hardware for half-precision (FP16/BF16) and TF32 arithmetic, which offers significantly higher throughput (2x-8x) than FP32/FP64.

## Proposal
Implement mixed precision strategies:
1.  **Reduced Precision Storage**: Store `function_values` and `density_matrix` in `half` or `bfloat16`.
2.  **Tensor Core Accumulation**: Use `wmma::` (Tensor Core instructions) for the RMM update ($F^T \cdot F$) or Density evaluation ($F \cdot R$) phases if reformulated as GEMM.
3.  **Use TF32**: Enable TensorFloat-32 (TF32) for FP32 matrix multiplications (cuBLAS). This is a simple flag/mode on Ampere+ GPUs.

## Impact
*   **Throughput**: Massive potential for speedup (2x-4x for compute-bound kernels).
*   **Memory**: Reduced memory footprint (2x smaller matrices).
*   **Accuracy**: Careful validation is required. For DFT, the grid accuracy is often limited by discretization error, so reduced precision for small contributions might be acceptable.

## Difficulty Assessment
**Medium/High**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Enable TF32 for cuBLAS (`cublasSetMathMode`).
    *   `g2g/cuda/kernels/*.h`: Explicit usage of `__half` types and `mma_sync`.
    *   `g2g/matrix.h`: Template specializations for `half` storage.

*   **Correctness Impact**: **High**. Precision loss. Needs strict validation. Energy might drift.
*   **Safe Start**: Enable TF32 (TensorFloat-32) on Ampere. It keeps FP32 range but reduces mantissa precision for multiplication. Often "free" speedup.

## Sketch of Changes
1.  **TF32**:
    *   `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);`
    *   This is the easiest first step.

2.  **FP16 Storage**:
    *   Convert `function_values` to `CudaMatrix<__half>`.
    *   Kernel `compute_functions` outputs `__half` (simple cast).
    *   GEMM uses `CUDA_R_16F` input, `CUDA_R_32F` accumulation.

## Estimations
*   Speedup: 1.5x-3x for compute-bound parts on supported hardware.