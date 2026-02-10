# Optimization: Basis Function Screening (Shell Culling)

## Summary
The `gpu_compute_functions` kernel evaluates basis functions for every point-function pair. While there is a check `if (exponent > 70.0) continue;` inside the contraction loop, this still incurs the overhead of loading data and calculating the distance squared for every single pair. For large systems, a significant portion of these evaluations result in negligible values (far field).

## Proposal
Implement a coarse-grained screening mechanism.
1.  **Pre-computation**: Divide the grid points into spatial blocks (e.g., 4x4x4 or 8x8x8 points).
2.  **Bounding Spheres**: For each Gaussian shell, define a cutoff radius $R_{cut}$ beyond which the value is effectively zero (< $10^{-10}$).
3.  **Intersection Test**: For each grid block, determine the list of shells that overlap with the block's bounding box.
4.  **Kernel Update**: Pass this "active list" to the kernel. Threads in the block only iterate over the active shells.

## Impact
*   **Complexity Reduction**: Reduces the computational complexity from $O(N_{points} \times N_{functions})$ to $O(N_{points} \times N_{neighbors})$. This is essential for linear scaling (or near-linear) behavior.
*   **Performance**: Massive speedup for large molecules where $N_{functions}$ is large but local interaction range is constant.

## Difficulty Assessment
**High**

*   **Files to Modify**:
    *   `g2g/partition.cpp`: Host side logic to build the neighbor lists.
    *   `g2g/cuda/iteration.cu`: Copy the neighbor list to GPU (`int* d_shell_list`, `int* d_block_offsets`).
    *   `g2g/cuda/kernels/functions.h`: Update loop to iterate over indirect indices.

*   **Correctness Impact**: **Medium**. Conservative screening ($10^{-12}$) is safe. Aggressive screening ($10^{-8}$) might affect SCF convergence.
*   **Data Structure**: Efficiently packing variable-length lists on GPU is the challenge. (CSR format: `offsets` + `values`).

## Sketch of Changes
1.  **Host**:
    *   `vector<int> shell_list; vector<int> block_offsets;`
    *   For each block `b`:
        *   `block_offsets[b] = shell_list.size()`
        *   Find intersecting shells -> push to `shell_list`.
    
2.  **GPU**:
    *   `__global__ void compute_functions(..., int* active_shells, int* offsets)`
    *   `int block_id = blockIdx.x;` (Assuming 1 block per spatial block)
    *   `int start = offsets[block_id];`
    *   `int end = offsets[block_id+1];`
    *   `for (int k = start; k < end; k++) { int shell_idx = active_shells[k]; ... }`

## Estimations
*   Speedup: Proportional to system size. For large systems, can be 2x-10x or more.