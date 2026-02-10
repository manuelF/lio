# Optimization: Eliminate Matrix Transposition

## Summary
The current implementation of `compute_functions` writes function values in one layout ($P \times M$), and then explicitly calls a `transpose` kernel twice (once for function values, once for gradients) to create a transposed matrix ($M \times P$). This incurs significant memory bandwidth (read entire matrix + write entire matrix). The `transpose` kernel consumes around 10-15% of total runtime in some test cases.

## Proposal
Modify the `compute_functions` kernel (`g2g/cuda/kernels/functions.h`) to write the values directly in the transposed layout (function-major, i.e., $M \times P$) or to match the layout expected by `gpu_compute_density`.

## Impact
*   **Performance**: Completely eliminates the `transpose` kernel overhead (memory bandwidth).
*   **Memory**: Saves temporary memory needed for the intermediate non-transposed matrix.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/cuda/kernels/functions.h`: Change the write index logic.
    *   `g2g/cuda/iteration.cu`: Remove the `transpose` kernel launch. Update pointer arguments.
    *   `g2g/cuda/kernels/transpose.h`: **DELETE** (or keep for generic utility).

*   **Correctness Impact**: **Low**. Just indexing changes.
*   **Performance Note**: Writing column-major ($M \times P$) with threads mapped to $P$ (points) means coalesced writes ONLY IF we transpose the thread mapping too (Threads mapped to Functions?).
    *   *Correction*: If we write $M \times P$, and $M$ is large stride, threads $tid, tid+1$ writing to `tid*stride` is NOT coalesced.
    *   **Better Approach**: Use shared memory tile to transpose in-flight inside `compute_functions`. Write out coalesced.

## Sketch of Changes
1.  **In-Kernel Transpose**:
    *   Compute block of values (e.g., 32 points x 32 functions).
    *   Store in `__shared__ float tile[32][33];`
    *   Syncthreads.
    *   Read transposed from shared memory.
    *   Write coalesced to global memory in $M \times P$ layout.

2.  **Simplified**:
    *   Just write $P \times M$ (Point-Major) and update `compute_density` (or GEMM) to accept $P \times M$ (using `CUBLAS_OP_N`). **This is the real fix.** Stop enforcing $M \times P$ layout if not needed!

## Estimations
*   Speedup: Removes the runtime of the `transpose` kernel completely.