# Optimization: SoA Layout for PointGroup

## Summary
The `PointGroup` struct contains `std::vector<Point> points`, where `Point` is a struct of 48 bytes (AoS).
`Point` contains: `atom` (4B), `shell` (4B), `point` (4B), `position` (24B), `weight` (8B).
Access patterns in CPU kernels iterate over `points[i].position` (stride 48 bytes) or `points[i].weight` (stride 48 bytes), which is suboptimal for SIMD vectorization and cache utilization.

## Proposal
Convert `PointGroup` to use Structure of Arrays (SoA).
1.  **Refactor**:
    *   `vector<double3> positions;`
    *   `vector<double> weights;`
    *   `vector<uint> atoms;`
    *   (Optional) `vector<uint> shells;` `vector<uint> indices;` if needed.
2.  **Kernel Access**:
    *   `positions[i]` is contiguous. Vectorized loads can fetch 2 `double3` or 4 `float3` elements efficiently.
    *   `weights[i]` is contiguous. Vectorized accumulation is trivial.

## Impact
*   **SIMD Efficiency**: Enables efficient auto-vectorization and explicit SIMD loads.
*   **Cache Locality**: Improves memory access patterns for position-dependent and weight-dependent calculations.
*   **Struct Size**: Reduces effective structure size by eliminating padding (if any) and unused fields for specific kernels.

## Difficulty Assessment
**Medium**

*   **Files to Modify**:
    *   `g2g/partition.h`: Update `struct PointGroup`. Remove `struct Point` or keep it for temporary usage.
    *   `g2g/partition.cpp`: Update `PointGroup::add_point` and `PointGroup::remove_point`.
    *   `g2g/cpu/functions.cpp`: Update `compute_functions` to access `positions[i]`.
    *   `g2g/cpu/weight.cpp`: Update `compute_weights`.
    *   `g2g/cuda/iteration.cu`: **Critical Update**. The GPU code copies these vectors to device. If they are split, we need separate `cudaMemcpy` calls or pack them into a temp buffer (defeats the purpose) or update GPU kernels to take SoA pointers.

*   **Refactoring Scope**: This touches the core data structure, so it ripples through almost every file that touches points.

## Sketch of Changes
1.  **Modify `g2g/partition.h`**:
    *   Remove `std::vector<Point> points`.
    *   Add `std::vector<scalar_type> pos_x, pos_y, pos_z;` (Full SoA is best for SIMD) or `std::vector<double3> positions`.
    *   Add `std::vector<scalar_type> weights`.

2.  **Update CPU Kernels**:
    *   `functions.cpp`: `x = pos_x[i]; y = pos_y[i]; ...` -> perfect for AVX load.

3.  **Update GPU Data Transfer (`iteration.cu`)**:
    *   Instead of copying `Point` struct (AoS), copying `pos_x`, `pos_y`, `pos_z` requires 3 copies.
    *   However, `gpu_atom_positions` is likely `__constant__` or global.
    *   GPU Kernels usually read position in the first step.
    *   We can remove the `Point` struct definition from CUDA kernels entirely if we pass arrays.

## Estimations
*   Speedup: 10-20% for memory-bound CPU kernels due to better cache usage.