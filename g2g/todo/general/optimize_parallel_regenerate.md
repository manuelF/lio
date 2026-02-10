# Optimization: Parallel Partition Regeneration

## Summary
The `regenerate` function in `g2g/regenerate_partition.cpp` is responsible for updating the grid (partitioning points into cubes/spheres) whenever atomic positions change (which happens at every step of Molecular Dynamics or Geometry Optimization).
Currently, the generation loop iterates over all atoms and their shells, and then over grid points within shells. This loop is largely serial or implicitly parallel via OMP in some places but not the structure building.
It also constructs complex nested `vector` structures (`prism`).

## Proposal
Parallelize the grid generation and point assignment using OpenMP.
1.  **Parallel Loop**: Use `#pragma omp parallel for` over atoms to compute their contributions to the grid.
2.  **Thread-Local Storage**: Each thread builds a local list of points or grid contributions to avoid race conditions when inserting into `prism`.
3.  **Merge**: Combine thread-local lists into the global `prism` structure or directly into `PointGroup`s.

## Impact
*   **Scalability**: Makes grid updates scale with CPU cores. Essential for large systems where regeneration dominates CPU time.
*   **Latency**: Reduces the turnaround time for each MD step.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/regenerate_partition.cpp`: The entire `regenerate` function and helper logic.

*   **Correctness Impact**: **Medium**. Race conditions when writing to the shared `prism` (grid of vectors) are the main risk. Order of points within a group might change, which shouldn't matter mathematically but might affect bit-exact reproducibility.

*   **Complexity**: Merging thread-local vectors is efficient but requires extra memory.

## Sketch of Changes
1.  **Analysis**:
    *   The code iterates atoms -> shells -> points.
    *   Points are added to `prism[cell_index]`.

2.  **Implementation**:
    *   Define `PrismType = vector<vector<Point>>`.
    *   `#pragma omp parallel`
    *   Create `PrismType local_prism(grid_size)`.
    *   `#pragma omp for` loop over atoms.
        *   Calculate points.
        *   `local_prism[cell].push_back(point)`.
    *   **Merge Step**:
        *   Use `#pragma omp critical` or manual offset calculation to merge `local_prism` into global `prism`.
        *   Alternatively, parallelize the merge: iterate over `cells`, each cell gathers from all thread-local prisms.

## Estimations
*   Speedup: Linear scaling with core count for the regeneration phase.