# Optimization: Stream Sharding for Point Groups

## Summary
The `Partition` class manages multiple `PointGroup` objects (cubes/spheres). Currently, these are processed sequentially on the GPU (likely on the default stream). This leaves the GPU underutilized during host-to-device transfers and CPU-side processing (like RMM preparation), and prevents small kernels from filling the GPU.

## Proposal
Implement "Stream Sharding" where the workload (list of PointGroups) is distributed across multiple CUDA streams.

## Impact
*   **Concurrency**: Allows multiple groups to be processed simultaneously.
*   **Overlap**: Perfect overlap of Data Transfer (Copy Engine) and Computation (Compute Engine). While Group A is copying data, Group B is computing.
*   **Occupancy**: Smaller groups might not fill the GPU. Running multiple small groups in parallel increases overall GPU utilization.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/partition.cpp`: The `solve` loop needs to manage a pool of streams.
    *   `g2g/cuda/iteration.cu`: `solve_closed` must accept a `cudaStream_t` argument.
    *   `g2g/common.h`: Define `MAX_STREAMS`.

*   **Correctness Impact**: **High**.
    *   **Race Conditions**: If different groups update the *same* global accumulator (e.g., `forces` or `rmm_output_global`), we need atomics or separate buffers.
    *   `rmm_output_local` is usually per-group, so that's safe.
    *   `GlobalMemoryPool` usage must be thread-safe or stream-safe (if resizing matrices concurrently).

## Sketch of Changes
1.  **Stream Pool**:
    *   `vector<cudaStream_t> streams(8);`
    *   Init in `Partition` constructor.

2.  **Loop**:
    *   `for (int i = 0; i < groups.size(); i++)`
    *   `cudaStream_t s = streams[i % 8];`
    *   `groups[i]->solve(..., s);`

3.  **Accumulation Safety**:
    *   Ensure `solve` writes to group-private buffers.
    *   The reduction to global `RMM` or `Forces` should happen either atomically or in a final synchronous pass.

## Estimations
*   Speedup: Significant improvement (1.5x - 2x) for systems with many small/medium groups.