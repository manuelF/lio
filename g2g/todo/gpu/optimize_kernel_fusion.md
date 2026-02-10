# Optimization: Fusion of Basis Function and Density Kernels

## Summary
Currently, `g2g/cuda/iteration.cu` runs a sequence of kernels: `gpu_compute_functions`, `transpose`, `gpu_compute_density`, `gpu_accumulate_point`.
The output of `gpu_compute_functions` is a large matrix ($P \times M$) stored in global memory, which is then loaded (via texture or global memory) by `gpu_compute_density`.
This intermediate data is massive and memory-bound.

## Proposal
Implement a fused kernel `gpu_compute_density_fused` that calculates basis function values on-the-fly inside the shared memory block, instead of reading them from global memory.

## Impact
*   **Memory Bandwidth**: Eliminates the write and read of the $P \times M$ matrix.
*   **Memory Capacity**: Frees up VRAM, allowing larger systems to fit on the GPU.
*   **Performance**: Trades memory bandwidth (which is often the bottleneck) for compute (re-evaluating exponentials). On modern GPUs with high FLOPS, this trade-off is often beneficial.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/cuda/kernels/energy.h`: Create a new fused kernel here.
    *   `g2g/cuda/kernels/functions.h`: Extract the basis function evaluation logic into a `__device__` function.

*   **Correctness Impact**: **High**. Complex state management in shared memory.
*   **Constraint**: Register pressure. Basis function evaluation uses many registers. Density accumulation uses registers. Doing both might cause spilling or low occupancy.

## Sketch of Changes
1.  **Refactor**: Make `eval_basis_function(...)` a reusable device function.

2.  **Fused Kernel**:
    *   Loop over tiles of the Density Matrix $R$ (size $32 \times 32$ maybe).
    *   Load $R_{tile}$ into shared memory.
    *   For each point assigned to thread:
        *   Re-compute $\phi_i(p)$ for the functions in the current tile.
        *   Accumulate density contribution.
    *   **Note**: This is tricky because basis functions are usually grouped by atoms, not arbitrary indices $i$. The $R$ matrix is dense. We might need to iterate over atoms, compute all functions for that atom, and multiply by corresponding $R$ block.

## Estimations
*   Speedup: 1.5x-2x for memory-bound scenarios.