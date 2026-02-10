# Optimization: Enhance Density Computation Kernel

## Summary
The `gpu_compute_density` kernel relies on texture objects (`tex2D`) for reading the Reduced Density Matrix (RMM). On modern GPU architectures (Volta/Ampere/Hopper), the L1 cache provides comparable or better performance for read-only data, especially with `__ldg()` or `const` restrict pointers, avoiding the overhead of texture binding and layout conversion. Additionally, the kernel uses shared memory for reduction, which can cause bank conflicts and is slower than warp shuffle primitives.

## Proposal
1.  **Remove Textures**: Switch from `cudaTextureObject_t` to standard global memory pointers (`const scalar_type* __restrict__ rmm`).
2.  **Use Warp Shuffle**: Replace the shared memory reduction loop with warp shuffle intrinsics (`__shfl_down_sync`) for the final summation within warps.
3.  **Optimize Reduction**: Reduce register pressure by accumulating directly in registers where possible.

## Impact
*   **Memory Bandwidth**: Eliminates texture cache overhead and potentially improves memory coalescing if the access pattern aligns.
*   **Latency**: Removes the texture binding/unbinding steps in the host code (`iteration.cu`).
*   **Throughput**: Warp shuffles are faster and use less shared memory, allowing higher occupancy.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Remove texture setup code.
    *   `g2g/cuda/kernels/energy.h`: Update kernel signature and implementation.

*   **Correctness Impact**: **Low**. Standard refactoring.
*   **Hardware Dependency**: Warp shuffle requires Kepler+ (which we surely support).

## Sketch of Changes
1.  **Host Code**:
    *   Remove `cudaCreateTextureObject`.
    *   Pass `rmm_input_gpu.data` directly to kernel.

2.  **Kernel**:
    *   `__global__ void gpu_compute_density(..., const double* __restrict__ rmm, ...)`
    *   Access: `val = rmm[row * stride + col];`
    *   Reduction:
        ```cpp
        // Intra-warp reduction
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        
        // First lane of each warp writes to shared memory
        if (lane_id == 0) shared_mem[warp_id] = sum;
        __syncthreads();
        
        // First warp reduces shared memory
        ```

## Estimations
*   Speedup: 10-20% for the density computation kernel.