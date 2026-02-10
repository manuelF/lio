# Optimization: GPU Arena Allocator (True Pool)

## Summary
The current `GlobalMemoryPool` only tracks available VRAM using a counter to prevent exceeding a threshold. It does *not* manage memory itself. Actual allocations are done via `cudaMalloc` and `cudaFree` inside `CudaMatrix::resize/deallocate`.
This leads to:
*   **Latency**: `cudaMalloc` is synchronous and can take milliseconds, stalling the CPU.
*   **Fragmentation**: Repeated alloc/free of small matrices fragments GPU memory.

## Proposal
Implement a true GPU memory pool/arena allocator.
1.  **Block Allocation**: At startup, allocate a large contiguous block of GPU memory (e.g., 80% of VRAM).
2.  **Sub-Allocation**: Manage allocations within this block using a fast algorithm (e.g., bump pointer for transient/per-frame memory, free-list/TLSF for long-lived objects).
3.  **Reset**: If most matrices are transient (per iteration), reset the allocator pointer after each iteration.

## Impact
*   **Performance**: Removes allocation latency from the critical path.
*   **Stability**: Avoids out-of-memory errors due to fragmentation.

## Difficulty Assessment
**Medium/High**

*   **Files to Modify**:
    *   `g2g/global_memory_pool.h` & `g2g/global_memory_pool.cpp`: Implement the allocator logic (Arena/Bump ptr).
    *   `g2g/matrix.cpp`: Modify `CudaMatrix::alloc_data` and `CudaMatrix::dealloc_data` to call the pool instead of raw CUDA API.
    *   `g2g/init.cpp`: Initialize the pool at startup.
    *   `g2g/partition.cpp` or `g2g/cuda/iteration.cu`: Reset the pool at the start of each SCF cycle if using a bump allocator.

*   **Correctness Impact**: **High**. Memory corruption if the allocator logic is buggy or if pointers are used after the pool is reset. Debugging custom allocators on GPU is painful.
*   **Risk**: Fragmentation handling if object lifetimes vary. A simple bump allocator works only if *everything* is freed at the same time.

## Sketch of Changes
1.  **Modify `GlobalMemoryPool`**:
    *   Add `void* base_ptr; size_t offset; size_t capacity;`.
    *   Init: `cudaMalloc(&base_ptr, capacity)`.
    *   `allocate(size)`: `void* ret = base_ptr + offset; offset += align(size); return ret;`.
    *   `reset()`: `offset = 0;`.
    *   Add safety checks for `offset > capacity`.

2.  **Modify `CudaMatrix`**:
    *   Inject `GlobalMemoryPool` instance.
    *   Use `pool->allocate(...)`.

3.  **Lifecycle Management**:
    *   Determine which matrices are persistent (e.g., `PointGroup` structures) vs transient (e.g., `density_matrix` calculated every step).
    *   We might need TWO pools: `StaticPool` (freelist) and `FramePool` (bump/linear).

## Estimations
*   Speedup: Significant reduction in driver overhead (milliseconds per group -> microseconds).