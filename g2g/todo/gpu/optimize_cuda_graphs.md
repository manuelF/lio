# Optimization: CUDA Graphs and Launch Optimization

## Summary
The `solve_closed` function in `g2g/cuda/iteration.cu` performs a fixed sequence of operations (functions -> transpose -> density -> accumulate -> RMM update). This entire sequence is repeated many times, often for small point groups.
Each kernel launch and `cudaMemcpy` call incurs a CPU-side overhead (launch latency + driver overhead). For small groups, this overhead can dominate the actual GPU execution time.

## Proposal
Capture the `solve_closed` execution flow as a CUDA Graph (`cudaGraph_t`).
1.  **Capture**: In the first iteration, capture the stream of operations using `cudaStreamBeginCapture` / `cudaStreamEndCapture`.
2.  **Instantiate**: Create an executable graph (`cudaGraphExec_t`).
3.  **Launch**: On subsequent iterations, launch the entire graph with a single API call: `cudaGraphLaunch`.

## Impact
*   **Latency**: Removes CPU launch overhead for subsequent iterations.
*   **Optimization**: The CUDA driver can optimize the graph structure (e.g., merge kernels, optimize dependencies).
*   **Performance**: Significant for small problem sizes (small PointGroups) where kernel launch latency is comparable to kernel execution time.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/cuda/iteration.cu`: Wrap the kernel sequence in capture logic.
    *   `g2g/partition.h`: Add graph members (`cudaGraphExec_t`, etc.) to `PointGroupGPU`.

*   **Correctness Impact**: **Medium**. If pointers or sizes change between iterations (e.g., `resize()` is called), the graph becomes invalid. We must detect this and `cudaGraphExecUpdate` or re-capture.
*   **Constraint**: Arguments to kernels are baked into the graph. If we pass `&time` or dynamic scalars by value, we must use `cudaGraphExecKernelNodeSetParams` or ensure they are consistent.

## Sketch of Changes
1.  **Modify `PointGroupGPU`**:
    *   Add `cudaGraphExec_t graphExec = nullptr;`.
    *   Add `bool graph_captured = false;`.

2.  **Modify `solve_closed`**:
    ```cpp
    if (graph_captured) {
        cudaGraphLaunch(graphExec, stream);
    } else {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        
        // ... Call all kernels and Async memcpys ...
        
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&graphExec, graph, ...);
        graph_captured = true;
        cudaGraphLaunch(graphExec, stream);
    }
    ```
3.  **Invalidation**:
    *   If `points` change or `functions` are resized, set `graph_captured = false`.

## Estimations
*   Speedup: 1.2x-2x for small systems/groups.