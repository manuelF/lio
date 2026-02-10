# Optimization: Async Execution and CPU-GPU Overlap

## Summary
The current implementation (`iteration.cu`) uses heavy synchronization (`timers.xxx.start_and_sync()`) and blocking memory transfers (`cudaMemcpy`, `cudaMemcpyToArray`). This serializes CPU and GPU operations, preventing overlap. The CPU-based RMM preparation (`get_rmm_input`) is particularly expensive as it runs on the CPU while the GPU is idle, then triggers a blocking copy.

## Proposal
1.  **Remove Synchronization**: Replace blocking timer calls with non-blocking ones or remove explicit `cudaDeviceSynchronize()` calls unless strictly necessary for data dependencies.
2.  **Async Memory Transfers**: Use `cudaMemcpyAsync` with pinned memory (`cudaHostAlloc` or `cudaMallocHost`) for `point_weights_cpu` and other host-to-device transfers.
3.  **Use Streams**: Create separate CUDA streams for compute and memory operations to allow overlap.
4.  **Move RMM Prep to GPU**: Port the CPU logic `get_rmm_input` to a CUDA kernel. This will keep the data on the GPU and avoid the host round-trip.

## Impact
*   **Latency Hiding**: Overlap CPU work (if any) with GPU kernels.
*   **Throughput**: Avoids pipeline stalls due to unnecessary synchronization.
*   **Efficiency**: Reduces the overhead of blocking API calls.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: The main loop logic in `solve_closed`.
    *   `g2g/timer.h`: Timer implementation (needs to support `cudaEventRecord` without `cudaEventSynchronize` for profiling).
    *   `g2g/matrix.cpp`: Ensure memory is pinned (`cudaMallocHost`) for Async copies to work.

*   **Correctness Impact**: **High**. Race conditions are likely. If we launch a kernel reading `buffer A` while `memcpyAsync` is still writing to `buffer A` (without proper stream ordering), we get garbage data.
*   **Complexity**: Managing stream dependencies.

## Sketch of Changes
1.  **Streams**:
    *   In `PointGroupGPU`, create `cudaStream_t computeStream, copyStream`.

2.  **Timer Cleanup**:
    *   Remove `timer.sync()` calls. Use `cudaEventRecord` for timing if needed, but don't block host.

3.  **Async Copy**:
    *   `cudaMemcpyAsync(d_ptr, h_ptr, size, kind, copyStream)`.
    *   Ensure `h_ptr` is pinned.

4.  **RMM Prep (Gather/Scatter)**:
    *   This is the biggest win. Write a kernel `gather_rmm<<<...>>>` that takes `global_rmm` (on GPU) and indices, and fills `local_rmm` (on GPU).
    *   This removes the `D2H -> CPU Shuffle -> H2D` bottleneck completely.

## Estimations
*   Speedup: 10-30% overall runtime improvement depending on CPU/GPU balance.