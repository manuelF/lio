# Optimization: Reformulate Density Computation as GEMM

## Summary
The current `gpu_compute_density` kernel computes the density $ho(p) = \sum_{ij} R_{ij} \phi_i(p) \phi_j(p)$ by iterating over $i, j$ for each point $p$. This approach loads the entire $R$ matrix (size $M^2$) from global memory (via texture) for *every* point $p$. This results in extremely low arithmetic intensity and is heavily memory bandwidth bound.

## Proposal
Reformulate the computation using Matrix Multiplication (GEMM).
Let $F$ be the matrix of basis function values ($P \times M$) and $R$ be the density matrix ($M \times M$).
The term $\sum_j R_{ij} \phi_j(p)$ is equivalent to the matrix-matrix multiplication $Y = F \cdot R$ (or $R \cdot F^T$).
The density at point $p$ is then the dot product of the row $p$ of $Y$ and the row $p$ of $F$: $ho(p) = \sum_k Y_{pk} F_{pk}$.

Similarly for gradients: $
abla ho = 2 \sum_{ij} R_{ij} \phi_i 
abla \phi_j = 2 \sum_k Y_{pk} (
abla F)_{pk}$.

## Impact
*   **Arithmetic Intensity**: GEMM ($F \cdot R$) has an arithmetic intensity of $O(M)$ (compute/bytes), compared to $O(1)$ for the current kernel.
*   **Performance**: `cuBLAS` GEMM is near-peak performance. We compute the heavy intermediate $Y$ once and reuse it for density, gradients, and Hessians.
*   **Memory**: Requires storing $Y$ ($P \times M$), which is manageable.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Add `cublasDgemm` calls. Allocate intermediate buffer `Y`.
    *   `g2g/cuda/kernels/energy.h`: Rewrite `gpu_compute_density` to take `Y` as input and perform only the dot product.
    *   `g2g/cuda/kernels/energy_derivs.h`: Rewrite derivative kernels similarly.

*   **Correctness Impact**: **Medium**. Reordering operations.
*   **Architecture**: We need a temp buffer of size $P \times M$ on the GPU. Check memory constraints.

## Sketch of Changes
1.  **Intermediate Calculation**:
    *   `cublasDgemm(..., A=Functions, B=RMM, C=Y)`.
    *   $Y$ now holds "Half-transformed" density vectors.

2.  **Final Contraction Kernel**:
    *   Kernel `compute_rho_from_Y<<<...>>>`:
        *   `idx = blockIdx.x * blockDim.x + threadIdx.x` (Point index).
        *   Loop `k = 0..M`:
            *   `rho += Y[idx*M + k] * Functions[idx*M + k]`.
    *   This is a simple vector dot product per point. Very fast, memory streaming.

3.  **Derivatives**:
    *   `grad_rho += Y[idx*M + k] * GradFunctions[idx*M + k]`.

## Estimations
*   Speedup: 5x-10x for the density evaluation phase.