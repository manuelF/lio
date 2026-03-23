# Optimization: Block Screening for CPU Basis Functions

## Summary
The current CPU implementation checks `exponent > 70.0` inside the contraction loop for *every* point-function pair. This check involves loading data (`a`, `c`) and computing `dist^2` even if the shell is far away.

## Proposal
Implement a coarse-grained screening mechanism similar to the GPU proposal.
1.  **Block Division**: Divide grid points into spatial blocks (e.g., 32 points/block or aligned with vector width).
2.  **Shell Screening**: Pre-compute a list of "active" shells for each block based on bounding box intersection.
3.  **Kernel Update**: Iterate only over the active shells for each block of points.

## Impact
*   **Complexity Reduction**: Massive reduction in the number of function evaluations for large systems (linear scaling behavior).
*   **Performance**: Avoids unnecessary memory loads and calculations for negligible interactions.
*   **SIMD Friendly**: Ensures that most iterations within a block perform useful work, improving vectorization efficiency.

## Difficulty Assessment
**Medium/High**

*   **Files to Modify**:
    *   `g2g/cpu/functions.cpp`: Rewrite the main loop structure of `PointGroupCPU::compute_functions`.
    *   `g2g/partition.h`: Might need to add structures to `PointGroupCPU` to store the active list (e.g., `vector<vector<int>> active_shells_per_block`).
    *   `g2g/partition.cpp`: Logic to compute the active list (likely during `regenerate` or at the start of `solve`).

*   **Correctness Impact**: **Medium**. If the screening radius is too aggressive (too small), we lose numerical precision. The cutoff must be chosen such that $e^{-\alpha R^2} < \epsilon_{mach}$.
*   **Complexity**: Building the neighbor list efficiently on the CPU is non-trivial. It adds a setup phase cost that must be amortized by the faster evaluation.

## Sketch of Changes
1.  **Setup Phase (in `solve` or `regenerate`)**:
    *   Group points into chunks of $N=32$.
    *   Compute AABB (Axis Aligned Bounding Box) for each chunk.
    *   For each shell: Compute cutoff radius $R_{cut}$.
    *   For each chunk: Find shells where `distance(chunk_center, shell_center) < chunk_radius + shell_radius`. Store indices in `active_list[chunk_id]`.

2.  **Kernel Modification (`g2g/cpu/functions.cpp`)**:
    *   Loop over chunks `c`.
    *   Retrieve `active_shells = active_list[c]`.
    *   Loop over points `p` in chunk `c`.
    *   Loop over `s` in `active_shells`.
    *   Evaluate functions.
    *   **Crucial**: Zero out `function_values` beforehand, as we are skipping many writes.

## Estimations
*   Speedup: Proportional to system size (2x-10x for large molecules).