# Optimization: Kernel Batching (Multi-Group Kernel)

## Summary
The system partitions grid points into `PointGroup` objects (spatial blocks), which are processed sequentially (or via streams in stream sharding). However, small groups may not provide enough work to saturate modern GPUs (A100 has 108 SMs!). The overhead of kernel launches and small grids limits performance.

## Proposal
Implement a "Batched" approach or "Mega-Kernel" where multiple `PointGroup`s are processed in a single kernel launch.
1.  **Indirect Access**: Pass arrays of pointers (or offsets into a monolithic buffer) to the kernel.
2.  **Persistent Threads**: Launch a persistent kernel that consumes tasks from a queue in global memory.
3.  **Cooperative Groups**: Use `cooperative_groups` to synchronize across blocks if needed.

## Impact
*   **Occupancy**: Fully saturates the GPU even with small/medium PointGroups.
*   **Latency**: Amortizes kernel launch overhead across many groups.
*   **Scalability**: Scales automatically with GPU size without needing manual stream tuning.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Major architectural change in how kernels are dispatched.
    *   `g2g/cuda/kernels/functions.h`: Must handle an extra indirection layer (group ID -> data pointer).
    *   `g2g/partition.h`: Data structures might need to be flattened into a single GPU buffer instead of `vector<CudaMatrix*>`.

*   **Correctness Impact**: **High**. Complex indexing logic is prone to off-by-one errors.
*   **Requirement**: Data for all groups in a batch must be resident on GPU.

## Sketch of Changes
1.  **Memory Layout**:
    *   Instead of `PointGroup[i].functions` being a separate allocation, use `GlobalFunctionsBuffer` and assign offsets.
    *   `int* group_offsets`: Array on GPU containing start index for each group.

2.  **Kernel**:
    *   `__global__ void batched_functions(..., int* group_offsets, ...)`
    *   `int global_idx = blockIdx.x * blockDim.x + threadIdx.x;`
    *   Determine which group `global_idx` belongs to (binary search on offsets, or assign 1 block per group).
    *   1 Block per Group is easiest: `int group_id = blockIdx.x`.
    *   Threads in block process points in that group.

## Estimations
*   Speedup: Significant (2x-5x) for systems with many small/irregular groups.